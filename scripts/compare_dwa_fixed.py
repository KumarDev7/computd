"""
compare_dwa_fixed.py
Trains DWA-v2 for 20K steps on Shakespeare and compares against
pre-captured DWA-v1 and Dense-Small results.
"""
import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.shakespeare_v2 import get_shakespeare_v2_config
from src.data.text_loader import shakespeare_loader, CharTokenizer
from src.model.dwa import DWAModel
from src.training.trainer import (
    make_optimizer, train_step, get_phase_params,
    Phase1Rotator, reset_dead_vectors,
)

# ─── Pre-captured results ──────────────────────────────────────────────────────

V1_STEPS  = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 19999]
V1_PPLS   = [58.19, 6.28, 5.96, 5.78, 5.69, 5.62, 5.55, 5.52, 5.50, 5.48, 5.48]
V1_PARAMS = 2_936_396

DENSE_STEPS  = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 19999]
DENSE_PPLS   = [72.95, 6.12, 5.54, 5.29, 5.20, 5.11, 5.05, 5.02, 5.01, 5.03, 4.95]
DENSE_PARAMS = 313_728

# ─── Training config ──────────────────────────────────────────────────────────

BATCH     = 32
SEQ       = 64
STEPS     = 20_000
SEED      = 42
DATA_PATH = "data/shakespeare.txt"
LOG_EVERY = 500
EVAL_EVERY = 2_000


# ─── Evaluation ───────────────────────────────────────────────────────────────

def eval_ppl(model, cfg, val_iter, n_batches=8):
    vocab = cfg.d_input
    loss_sum = tok_count = 0
    for _ in range(n_batches):
        batch, _ = next(val_iter)
        x_oh   = jax.nn.one_hot(batch[:, :-1], vocab)
        logits = jax.device_get(model(x_oh, use_sigmoid=True, lambda_sharp=5.0))
        tgts   = jax.device_get(batch[:, 1:])
        lp     = np.array(jax.nn.log_softmax(jnp.array(logits), axis=-1))
        loss_sum  += -float(np.sum(np.eye(vocab)[tgts] * lp))
        tok_count += tgts.size
    return float(np.exp(loss_sum / tok_count))


# ─── Generation ───────────────────────────────────────────────────────────────

def generate(model, tok, cfg, prompt="HAMLET:\n", steps=200, temp=0.8, top_k=40, seed=0):
    ids = list(tok.encode(prompt))
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        x = jnp.array([ids[-cfg.max_seq_len:]])
        x_oh = jax.nn.one_hot(x, cfg.d_input)
        logits = jax.device_get(model(x_oh, use_sigmoid=True, lambda_sharp=5.0))
        logits = logits[0, -1] / temp
        top_idx = np.argpartition(-logits, top_k)[:top_k]
        probs = np.exp(logits[top_idx] - logits[top_idx].max())
        probs /= probs.sum()
        next_id = rng.choice(top_idx, p=probs)
        ids.append(int(next_id))
    return tok.decode(ids)


# ─── Param count ──────────────────────────────────────────────────────────────

def count_params(model):
    graphdef, state = nnx.split(model)
    leaves = jax.tree_util.tree_leaves(state)
    return sum(x.size for x in leaves)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70, flush=True)
    print("DWA Fixed Architecture Benchmark", flush=True)
    print("=" * 70, flush=True)

    # Pre-captured results header
    print("\n--- Pre-captured results ---", flush=True)
    print(f"{'step':>8}  {'DWA-v1 PPL':>12}  {'Dense-Small PPL':>16}", flush=True)
    for s, v1, ds in zip(V1_STEPS, V1_PPLS, DENSE_PPLS):
        print(f"{s:>8}  {v1:>12.2f}  {ds:>16.2f}", flush=True)
    print(f"\nDWA-v1 params:    {V1_PARAMS:,}", flush=True)
    print(f"Dense-Small params: {DENSE_PARAMS:,}", flush=True)

    # Build data iterators
    print(f"\nLoading data from {DATA_PATH} ...", flush=True)
    train_iter = shakespeare_loader(DATA_PATH, BATCH, SEQ, split="train", seed=SEED)
    val_iter   = shakespeare_loader(DATA_PATH, BATCH, SEQ, split="val",   seed=SEED + 1)

    # Grab tokenizer from first batch
    _, tok = next(shakespeare_loader(DATA_PATH, 1, SEQ, split="train", seed=0))

    # Build model
    cfg  = get_shakespeare_v2_config()
    rngs = nnx.Rngs(params=SEED, dropout=SEED + 1)
    model = DWAModel(cfg, rngs)
    optimizer = make_optimizer(model, cfg)

    n_params = count_params(model)
    print(f"\nDWA-v2 params: {n_params:,}", flush=True)
    print(f"Config: d_A={cfg.d_A}, d_B={cfg.d_B}, D={cfg.D}, r={cfg.r}, "
          f"N={cfg.N}, k_max={cfg.k_max}, n_layers={cfg.n_assembly_layers}, "
          f"n_heads={cfg.n_heads}", flush=True)

    # Training
    print("\n--- DWA-v2 Training ---", flush=True)
    print(f"{'step':>8}  {'val_ppl':>8}  {'gamma_0':>8}  {'dead%':>7}", flush=True)

    rng_np    = np.random.default_rng(SEED)
    rotator   = Phase1Rotator(cfg.N, BATCH, cfg.k_max, seed=SEED)
    v2_steps  = []
    v2_ppls   = []

    for step, (batch, _) in zip(range(STEPS), train_iter):
        # Codebook reset
        if cfg.reset_interval > 0 and step > 0 and step % cfg.reset_interval == 0:
            reset_dead_vectors(model, rng_np)

        use_sigmoid, lambda_sharp, lambda_entropy_eff = get_phase_params(step, cfg)

        forced_idx = rotator.next() if not use_sigmoid else rotator.dummy()

        train_step(
            model, optimizer, batch,
            use_sigmoid, lambda_sharp, lambda_entropy_eff, forced_idx,
        )

        # Log every LOG_EVERY steps
        if step % LOG_EVERY == 0 or step == STEPS - 1:
            val_ppl = eval_ppl(model, cfg, val_iter)
            gamma_0 = float(jax.device_get(model.blocks[0].assembler.gamma.value))
            ema     = np.array(jax.device_get(model.pool.ema_usage.value))
            dead_pct = 100.0 * np.mean(ema < cfg.reset_threshold)

            print(f"step={step:05d}  val_ppl={val_ppl:.3f}  "
                  f"gamma_0={gamma_0:.4f}  dead%={dead_pct:.1f}%", flush=True)

            if step % EVAL_EVERY == 0 or step == STEPS - 1:
                v2_steps.append(step)
                v2_ppls.append(val_ppl)

    # Final comparison table
    print("\n" + "=" * 70, flush=True)
    print("Final Comparison Table", flush=True)
    print("=" * 70, flush=True)
    print(f"{'step':>8}  {'DWA-v1':>8}  {'Dense-Sm':>9}  {'DWA-v2':>8}", flush=True)

    # Align rows by step (v2 evaluates at EVAL_EVERY intervals)
    v2_map = dict(zip(v2_steps, v2_ppls))
    all_steps = sorted(set(V1_STEPS) | set(v2_steps))
    for s in all_steps:
        v1  = V1_PPLS[V1_STEPS.index(s)]   if s in V1_STEPS   else float("nan")
        ds  = DENSE_PPLS[DENSE_STEPS.index(s)] if s in DENSE_STEPS else float("nan")
        v2  = v2_map.get(s, float("nan"))
        print(f"{s:>8}  {v1:>8.2f}  {ds:>9.2f}  {v2:>8.2f}", flush=True)

    final_v2  = v2_ppls[-1] if v2_ppls else float("nan")
    final_v1  = V1_PPLS[-1]
    final_ds  = DENSE_PPLS[-1]
    print(f"\nFinal PPL — DWA-v1: {final_v1:.2f}  Dense-Small: {final_ds:.2f}  "
          f"DWA-v2: {final_v2:.2f}", flush=True)
    delta_v1 = final_v2 - final_v1
    delta_ds = final_v2 - final_ds
    print(f"DWA-v2 vs DWA-v1:    {delta_v1:+.3f} PPL", flush=True)
    print(f"DWA-v2 vs Dense-Sm:  {delta_ds:+.3f} PPL", flush=True)
    print(f"\nParams — DWA-v2: {n_params:,}  Dense-Small: {DENSE_PARAMS:,}  "
          f"ratio: {n_params / DENSE_PARAMS:.1f}x", flush=True)

    # Generated text sample
    print("\n--- Generated text (DWA-v2, temp=0.8, top_k=40) ---", flush=True)
    sample = generate(model, tok, cfg, prompt="HAMLET:\n", steps=200, temp=0.8, top_k=40)
    print(sample, flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
