"""
DWA vs Dense LM Benchmark on Tiny Shakespeare.

Trains three models under identical conditions and compares:
  - DWA          (2.9M params — pool=90%, dense=10%)
  - Dense-Large  (~2.9M params — standard GPT, matched total size)
  - Dense-Small  (~300K params — matched to DWA's non-pool compute)

Produces:
  - Val-loss learning curves
  - Steps-to-threshold table (when each model first reaches PPL ≤ X)
  - Final PPL + generated text comparison
"""
import os, sys, time
os.environ["JAX_DEBUG_NANS"]   = "False"
os.environ["JAX_LOG_COMPILES"] = "False"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax, jax.numpy as jnp
import optax
from flax import nnx

from configs.shakespeare import get_shakespeare_config
from src.model.dwa import DWAModel
from src.model.dense_lm import DenseLM, make_dense_configs
from src.training.trainer import (make_optimizer, make_train_step,
                                    get_phase_params, Phase1Rotator,
                                    reset_dead_vectors)
from src.data.text_loader import shakespeare_loader, CharTokenizer

# ── Settings ──────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "shakespeare.txt")
BATCH_SIZE  = 32
SEQ_LEN     = 64
TOTAL_STEPS = 20_000
LOG_EVERY   = 500
SEED        = 42
VOCAB       = 65

PPL_THRESHOLDS = [20, 10, 7, 6, 5.5]


# ── Dense training helpers ─────────────────────────────────────────────────────

@nnx.jit
def dense_train_step(model: nnx.Module, optimizer: nnx.Optimizer,
                     batch: jax.Array) -> jax.Array:
    vocab = batch.shape[0]  # unused placeholder

    def loss_fn(m):
        x, y    = batch[:, :-1], batch[:, 1:]
        logits  = m(x)
        log_p   = jax.nn.log_softmax(logits, axis=-1)
        targets = jax.nn.one_hot(y, logits.shape[-1])
        return -jnp.mean(jnp.sum(targets * log_p, axis=-1))

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def make_dense_optimizer(model: nnx.Module, lr: float = 3e-4) -> nnx.Optimizer:
    tx = optax.adamw(lr, weight_decay=1e-2)
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


# ── Shared eval ───────────────────────────────────────────────────────────────

def compute_val_loss(model, model_type: str, tok: CharTokenizer,
                     n_batches: int = 15) -> float:
    val_gen = shakespeare_loader(DATA_PATH, BATCH_SIZE, SEQ_LEN,
                                 split="val", seed=999)
    total = 0.0
    for _ in range(n_batches):
        batch, _ = next(val_gen)
        if model_type == "dwa":
            x_oh   = jax.nn.one_hot(batch[:, :-1], VOCAB)
            logits = model(x_oh, use_sigmoid=True, lambda_sharp=5.0)
        else:
            logits = model(batch[:, :-1])
        y   = jax.device_get(batch[:, 1:])
        lp  = jax.device_get(jax.nn.log_softmax(logits, axis=-1))
        total += -float(np.mean(np.sum(np.eye(VOCAB)[y] * lp, axis=-1)))
    return total / n_batches


def generate_text(model, model_type: str, tok: CharTokenizer,
                  prompt: str, max_new: int = 300,
                  temperature: float = 0.8, top_k: int = 40) -> str:
    ids = list(tok.encode(prompt))
    rng = np.random.default_rng(7)
    for _ in range(max_new):
        ctx  = ids[-SEQ_LEN:]
        ctx_arr = np.array(ctx)[None, :]
        if model_type == "dwa":
            x_oh   = jax.nn.one_hot(ctx_arr, VOCAB)
            logits = np.array(model(x_oh, use_sigmoid=True, lambda_sharp=5.0))
        else:
            logits = np.array(model(jnp.array(ctx_arr)))
        last = logits[0, -1] / temperature
        if top_k > 0:
            cutoff = np.sort(last)[-top_k]
            last   = np.where(last >= cutoff, last, -1e9)
        probs = np.exp(last - np.max(last)); probs /= probs.sum()
        ids.append(int(rng.choice(VOCAB, p=probs)))
    return tok.decode(ids)


# ── Per-model training run ────────────────────────────────────────────────────

def run_model(name: str, model, model_type: str, optimizer, tok,
              data_gen, rotator=None, rng_np=None, cfg=None,
              graphdef=None, state=None, opt_graphdef=None, opt_state=None,
              step_phase1=None, step_phase2=None):
    """Train one model and return (steps[], val_ppls[], thresholds{})."""
    steps_log, ppl_log = [], []
    thresholds = {t: None for t in PPL_THRESHOLDS}
    resets = 0
    t0 = time.time()

    print(f"\n{'─'*65}")
    print(f"  Training: {name}")
    print(f"{'─'*65}")

    for step in range(TOTAL_STEPS):
        batch, _ = next(data_gen)

        if model_type == "dwa":
            if cfg.reset_interval > 0 and step > 0 and step % cfg.reset_interval == 0:
                resets += reset_dead_vectors(model, rng_np)
            use_s, lam, lent = get_phase_params(step, cfg)
            step_fn = step_phase2 if use_s else step_phase1
            forced = rotator.next() if not use_s else rotator.dummy()
            metrics, state, opt_state = step_fn(
                graphdef, state, opt_graphdef, opt_state, batch,
                jnp.float32(lam), jnp.float32(lent), forced
            )
        else:
            dense_train_step(model, optimizer, batch)

        if step % LOG_EVERY == 0 or step == TOTAL_STEPS - 1:
            vl  = compute_val_loss(model, model_type, tok)
            ppl = float(np.exp(vl))
            steps_log.append(step)
            ppl_log.append(ppl)
            elapsed = time.time() - t0
            phase_s = ""
            if model_type == "dwa":
                p = 1 if step < cfg.phase1_end else (2 if step < cfg.phase2_end else 3)
                phase_s = f"  ph={p}"
            print(f"  step={step:>6}{phase_s}  val_ppl={ppl:>6.2f}  "
                  f"resets={resets}  t={elapsed:.0f}s")

            for thresh in PPL_THRESHOLDS:
                if thresholds[thresh] is None and ppl <= thresh:
                    thresholds[thresh] = step

    return steps_log, ppl_log, thresholds


# ── Main benchmark ────────────────────────────────────────────────────────────

def main():
    # Build shared tokeniser
    gen0, tok = next(iter([next(
        shakespeare_loader(DATA_PATH, 1, 8, seed=0)
    )])), None
    # Re-open properly
    _gen = shakespeare_loader(DATA_PATH, BATCH_SIZE, SEQ_LEN, split="train", seed=SEED)
    _, tok = next(_gen)

    print("=" * 65)
    print("  DWA vs Dense LM — Shakespeare Benchmark")
    print(f"  Steps={TOTAL_STEPS}  Batch={BATCH_SIZE}  SeqLen={SEQ_LEN}")
    print("=" * 65)

    results = {}

    # ── 1. DWA ────────────────────────────────────────────────────────────────
    cfg      = get_shakespeare_config()
    rngs_dwa = nnx.Rngs(params=SEED, dropout=SEED+1)
    dwa      = DWAModel(cfg, rngs=rngs_dwa)
    opt_dwa  = make_optimizer(dwa, cfg)
    rot      = Phase1Rotator(cfg.N, BATCH_SIZE, cfg.k_max, seed=SEED)
    rng_np   = np.random.default_rng(SEED)
    data_dwa = shakespeare_loader(DATA_PATH, BATCH_SIZE, SEQ_LEN,
                                  split="train", seed=SEED)

    graphdef, state = nnx.split(dwa, nnx.Param)
    opt_graphdef, opt_state = nnx.split(opt_dwa, nnx.Param)
    step_phase1 = make_train_step(cfg, use_sigmoid=False)
    step_phase2 = make_train_step(cfg, use_sigmoid=True)

    n_dwa = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(dwa, nnx.Param)))
    print(f"\n  DWA params:          {n_dwa:>10,}")

    steps, ppls, thresh = run_model(
        "DWA (pool=90%)", dwa, "dwa", opt_dwa, tok,
        data_dwa, rotator=rot, rng_np=rng_np, cfg=cfg,
        graphdef=graphdef, state=state, opt_graphdef=opt_graphdef, opt_state=opt_state,
        step_phase1=step_phase1, step_phase2=step_phase2)

    nnx.update(dwa, state)
    nnx.update(opt_dwa, opt_state)
    results["DWA"] = dict(model=dwa, steps=steps, ppls=ppls,
                          thresh=thresh, n_params=n_dwa, type="dwa")

    # ── 2. Dense-Large ────────────────────────────────────────────────────────
    dcfgs    = make_dense_configs(VOCAB, max_seq_len=256)
    rngs_dl  = nnx.Rngs(params=SEED, dropout=SEED+1)
    dl       = DenseLM(**dcfgs["large"], rngs=rngs_dl)
    opt_dl   = make_dense_optimizer(dl, lr=3e-4)
    data_dl  = shakespeare_loader(DATA_PATH, BATCH_SIZE, SEQ_LEN,
                                  split="train", seed=SEED)

    n_dl = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(dl, nnx.Param)))
    print(f"  Dense-Large params:  {n_dl:>10,}  (target ≈ {n_dwa:,})")

    steps, ppls, thresh = run_model(
        f"Dense-Large ({n_dl/1e6:.2f}M)", dl, "dense", opt_dl, tok, data_dl)
    results["Dense-Large"] = dict(model=dl, steps=steps, ppls=ppls,
                                  thresh=thresh, n_params=n_dl, type="dense")

    # ── 3. Dense-Small ────────────────────────────────────────────────────────
    rngs_ds = nnx.Rngs(params=SEED, dropout=SEED+1)
    ds      = DenseLM(**dcfgs["small"], rngs=rngs_ds)
    opt_ds  = make_dense_optimizer(ds, lr=3e-4)
    data_ds = shakespeare_loader(DATA_PATH, BATCH_SIZE, SEQ_LEN,
                                 split="train", seed=SEED)

    n_ds = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(ds, nnx.Param)))
    dwa_dense_only = n_dwa - (cfg.N * cfg.D)  # pool removed
    print(f"  Dense-Small params:  {n_ds:>10,}  (target ≈ DWA non-pool {dwa_dense_only:,})")

    steps, ppls, thresh = run_model(
        f"Dense-Small ({n_ds/1e3:.0f}K)", ds, "dense", opt_ds, tok, data_ds)
    results["Dense-Small"] = dict(model=ds, steps=steps, ppls=ppls,
                                   thresh=thresh, n_params=n_ds, type="dense")

    # ── Results table ─────────────────────────────────────────────────────────
    print(f"\n\n{'═'*65}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'═'*65}")

    # Learning curve table
    all_steps = results["DWA"]["steps"]
    header = f"  {'step':>6}  {'DWA':>8}  {'Dense-L':>8}  {'Dense-S':>8}"
    print(f"\n  Val PPL learning curve:")
    print(f"  {'─'*50}")
    print(header)
    print(f"  {'─'*50}")
    for i, s in enumerate(all_steps):
        row = f"  {s:>6}"
        for name in ["DWA", "Dense-Large", "Dense-Small"]:
            r = results[name]
            # Find closest step
            idx = min(range(len(r["steps"])), key=lambda j: abs(r["steps"][j]-s))
            if abs(r["steps"][idx] - s) <= LOG_EVERY:
                row += f"  {r['ppls'][idx]:>8.2f}"
            else:
                row += f"  {'—':>8}"
        print(row)

    # Threshold table
    print(f"\n  Steps to first reach PPL threshold:")
    print(f"  {'─'*55}")
    print(f"  {'PPL≤':>6}  {'DWA':>8}  {'Dense-L':>10}  {'Dense-S':>10}")
    print(f"  {'─'*55}")
    for t in PPL_THRESHOLDS:
        row = f"  {t:>6}"
        for name in ["DWA", "Dense-Large", "Dense-Small"]:
            v = results[name]["thresh"][t]
            row += f"  {str(v) if v is not None else 'never':>10}"
        print(row)

    # Final summary
    print(f"\n  Final results at step {TOTAL_STEPS}:")
    print(f"  {'─'*55}")
    dwa_final = results["DWA"]["ppls"][-1]
    for name, r in results.items():
        final_ppl = r["ppls"][-1]
        delta     = (final_ppl - dwa_final) / dwa_final * 100
        sign      = "+" if delta >= 0 else ""
        print(f"  {name:<15}  params={r['n_params']:>10,}  "
              f"PPL={final_ppl:.3f}  vs_DWA={sign}{delta:.1f}%")

    # Extra-steps-to-parity analysis
    print(f"\n  Extra steps Dense-Large needs to match DWA's quality:")
    for t in PPL_THRESHOLDS:
        dwa_s  = results["DWA"]["thresh"][t]
        dl_s   = results["Dense-Large"]["thresh"][t]
        if dwa_s is not None and dl_s is not None:
            extra = dl_s - dwa_s
            mult  = dl_s / max(dwa_s, 1)
            print(f"    PPL≤{t}: DWA={dwa_s}  Dense-L={dl_s}  "
                  f"extra={extra:+d}  ({mult:.1f}x steps)")
        elif dwa_s is not None:
            print(f"    PPL≤{t}: DWA={dwa_s}  Dense-L=never reached in {TOTAL_STEPS} steps")

    # ── Generated text comparison ─────────────────────────────────────────────
    print(f"\n\n{'═'*65}")
    print(f"  GENERATED TEXT COMPARISON  (temp=0.8  top_k=40  len=250)")
    print(f"{'═'*65}")
    prompt = "HAMLET:\n"
    for name, r in results.items():
        print(f"\n  ── {name} (PPL={r['ppls'][-1]:.2f}) ──")
        text = generate_text(r["model"], r["type"], tok, prompt,
                             max_new=250, temperature=0.8, top_k=40)
        print("  " + text.replace("\n", "\n  "))

    # Second prompt
    prompt2 = "ROMEO:\n"
    print(f"\n\n  Prompt: {repr(prompt2)}")
    for name, r in results.items():
        print(f"\n  ── {name} ──")
        text = generate_text(r["model"], r["type"], tok, prompt2,
                             max_new=200, temperature=0.8, top_k=40)
        print("  " + text.replace("\n", "\n  "))

    print(f"\n{'═'*65}")
    print(f"  VERDICT")
    print(f"{'═'*65}")
    best = min(results.items(), key=lambda x: x[1]["ppls"][-1])
    print(f"\n  Best model: {best[0]} (PPL={best[1]['ppls'][-1]:.3f})")
    dwa_p, dl_p = results["DWA"]["ppls"][-1], results["Dense-Large"]["ppls"][-1]
    if dwa_p < dl_p:
        gap_pct = (dl_p - dwa_p) / dl_p * 100
        print(f"  DWA beats Dense-Large by {gap_pct:.1f}% PPL with same param count.")
        print(f"  DWA's pool mechanism provides better capacity utilisation.")
    else:
        gap_pct = (dwa_p - dl_p) / dwa_p * 100
        print(f"  Dense-Large beats DWA by {gap_pct:.1f}% PPL.")

        # Find how many extra steps DWA needs to match dense final PPL
        target = dl_p
        crossover = None
        for i, (s, p) in enumerate(zip(results["DWA"]["steps"], results["DWA"]["ppls"])):
            if p <= target:
                crossover = s
                break
        if crossover is not None:
            dl_final_step = results["Dense-Large"]["steps"][-1]
            print(f"  DWA reaches Dense-Large's final quality at step ~{crossover}  "
                  f"({crossover/dl_final_step:.1f}x of Dense's budget).")
        else:
            print(f"  DWA never reaches Dense-Large's final PPL={target:.3f} in {TOTAL_STEPS} steps.")
            print(f"  Consider more steps or larger pool.")


if __name__ == "__main__":
    main()
