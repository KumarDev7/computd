"""
DWA Shakespeare training + text generation.

Architecture: DWA with 1-layer causal attention (RoPE) in PartA for context.
Task: character-level next-token prediction on Tiny Shakespeare.

Run:
    python scripts/train_shakespeare.py
"""
import os, sys, time
os.environ["JAX_DEBUG_NANS"]   = "False"
os.environ["JAX_LOG_COMPILES"] = "False"
XLA_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".jax_cache")
os.makedirs(XLA_CACHE_DIR, exist_ok=True)
os.environ["JAX_COMPILATION_CACHE_DIR"] = XLA_CACHE_DIR
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from configs.shakespeare import get_shakespeare_config
from src.model.dwa import DWAModel
from src.training.trainer import make_optimizer, train_step, get_phase_params, Phase1Rotator, reset_dead_vectors
from src.data.text_loader import shakespeare_loader, CharTokenizer

# ── Settings ──────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "shakespeare.txt")
BATCH_SIZE = 32
SEQ_LEN    = 64        # context window
TOTAL_STEPS = 30_000
LOG_EVERY   = 500
GEN_EVERY   = 2_000   # show generated text every N steps
SEED        = 42


# ── Text generation ────────────────────────────────────────────────────────────

def generate(model, tok: CharTokenizer, prompt: str, max_new: int = 300,
             temperature: float = 0.8, top_k: int = 40) -> str:
    """Autoregressive character generation with temperature + top-k sampling."""
    vocab = tok.vocab_size
    ids   = list(tok.encode(prompt))
    rng   = np.random.default_rng(0)

    for _ in range(max_new):
        # Use up to SEQ_LEN context
        ctx    = ids[-SEQ_LEN:]
        x_oh   = jax.nn.one_hot(np.array(ctx)[None, :], vocab)   # (1, ctx_len, vocab)
        logits = model(x_oh, use_sigmoid=True, lambda_sharp=5.0)  # (1, ctx_len, vocab)
        last   = np.array(logits[0, -1])                          # (vocab,)

        # Temperature + top-k sampling
        last = last / temperature
        if top_k > 0:
            cutoff = np.sort(last)[-top_k]
            last   = np.where(last >= cutoff, last, -1e9)
        probs = np.exp(last - np.max(last))
        probs /= probs.sum()
        nxt = rng.choice(vocab, p=probs)
        ids.append(int(nxt))

    return tok.decode(ids)


# ── Validation loss ────────────────────────────────────────────────────────────

def val_loss(model, tok, n_batches: int = 20) -> float:
    vocab    = tok.vocab_size
    val_gen  = shakespeare_loader(DATA_PATH, BATCH_SIZE, SEQ_LEN, split="val", seed=999)
    total    = 0.0
    for _ in range(n_batches):
        batch, _ = next(val_gen)
        x_oh   = jax.nn.one_hot(batch[:, :-1], vocab)
        tgt    = batch[:, 1:]
        logits = model(x_oh, use_sigmoid=True, lambda_sharp=5.0)
        lp     = jax.device_get(jax.nn.log_softmax(logits, axis=-1))
        tgt_np = jax.device_get(tgt)
        total += -float(np.mean(np.sum(np.eye(vocab)[tgt_np] * lp, axis=-1)))
    return total / n_batches


# ── Training loop ──────────────────────────────────────────────────────────────

def main():
    cfg  = get_shakespeare_config()
    rngs = nnx.Rngs(params=SEED, dropout=SEED + 1)
    model    = DWAModel(cfg, rngs=rngs)
    optimizer = make_optimizer(model, cfg)
    rng_np   = np.random.default_rng(SEED)

    # Count parameters
    params = nnx.state(model, nnx.Param)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print("=" * 70)
    print(f"  DWA Shakespeare Training")
    print(f"  Vocab: {cfg.d_input}  SeqLen: {SEQ_LEN}  Batch: {BATCH_SIZE}")
    print(f"  d_A={cfg.d_A}  d_B={cfg.d_B}  D={cfg.D}  N={cfg.N}  r={cfg.r}")
    print(f"  n_heads={cfg.n_heads}  k_max={cfg.k_max}  S={cfg.S}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Total steps: {TOTAL_STEPS:,}")
    print("=" * 70)

    data_gen = shakespeare_loader(DATA_PATH, BATCH_SIZE, SEQ_LEN, split="train", seed=SEED)
    batch0, tok = next(data_gen)
    rotator  = Phase1Rotator(cfg.N, BATCH_SIZE, cfg.k_max, seed=SEED)
    resets   = 0

    # Show initial generation (random model)
    print("\n[Step 0 — untrained model]")
    print(generate(model, tok, prompt="HAMLET:\n", max_new=200))
    print()

    best_val = float("inf")
    t_start  = time.time()

    for step in range(TOTAL_STEPS):
        batch, tok = next(data_gen)

        # Codebook reset
        if cfg.reset_interval > 0 and step > 0 and step % cfg.reset_interval == 0:
            resets += reset_dead_vectors(model, rng_np)

        use_s, lam, lent = get_phase_params(step, cfg)
        forced = rotator.next() if not use_s else rotator.dummy()
        metrics = train_step(model, optimizer, batch, use_s, lam, lent, forced)

        if step % LOG_EVERY == 0 or step == TOTAL_STEPS - 1:
            metrics = jax.device_get(metrics)
            vloss   = val_loss(model, tok)
            elapsed = time.time() - t_start
            phase   = 1 if step < cfg.phase1_end else (2 if step < cfg.phase2_end else 3)
            print(f"  step={step:>6}  phase={phase}  "
                  f"train={float(metrics['loss']):.4f}  val={vloss:.4f}  "
                  f"ppl={np.exp(vloss):.1f}  resets={resets}  t={elapsed:.0f}s")
            best_val = min(best_val, vloss)

        if (step + 1) % GEN_EVERY == 0:
            print(f"\n{'─'*70}")
            print(f"[Step {step+1}] Generated text (temp=0.8, top_k=40):")
            print(f"{'─'*70}")
            for prompt in ["HAMLET:\n", "ROMEO:\n", "First Citizen:\n"]:
                print(f"\n  Prompt: {repr(prompt)}")
                out = generate(model, tok, prompt=prompt, max_new=250,
                               temperature=0.8, top_k=40)
                print("  " + out.replace("\n", "\n  "))
            print()

    # ── Final generation ──────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  TRAINING COMPLETE  —  best val loss={best_val:.4f}  ppl={np.exp(best_val):.1f}")
    print(f"{'═'*70}")

    for temp in [0.7, 1.0]:
        print(f"\n── Temperature={temp} ──")
        for prompt in ["HAMLET:\n", "ROMEO:\n", "KING HENRY:\n"]:
            print(f"\nPrompt: {repr(prompt)}")
            print(generate(model, tok, prompt=prompt, max_new=400, temperature=temp, top_k=40))


if __name__ == "__main__":
    main()
