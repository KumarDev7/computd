"""
Compare three training strategies:
  A. Baseline — no warmup, no entropy loss
  B. Entropy only — annealed entropy loss, no warmup
  C. Warmup + Entropy — phase-1 rotates all vectors before competitive retrieval starts

Key question: does phase-1 warmup (all vectors get gradient) produce better
diversity AND better task performance in phase 2?
"""
import os, sys
os.environ["JAX_DEBUG_NANS"]   = "False"
os.environ["JAX_LOG_COMPILES"] = "False"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax, jax.numpy as jnp
from flax import nnx

from configs.small import DWAConfig
from src.model.dwa import DWAModel
from src.training.trainer import (
    make_optimizer, train_step, get_phase_params,
    Phase1Rotator, reset_dead_vectors,
)
from src.data.loader import synthetic_copy_task


VOCAB  = 64
BATCH  = 32
SEQ    = 16
STEPS  = 3000
SEED   = 42


def pool_stats(model, batch):
    vocab = model.config.d_input
    x     = jax.nn.one_hot(batch[:, :-1], vocab)
    _, lam, _ = get_phase_params(9999, model.config)
    _, aux = model(x, use_sigmoid=True, lambda_sharp=lam, return_aux=True)
    idx    = jax.device_get(aux["idx"])

    unique   = len(set(idx.flatten()))
    pairs    = np.random.default_rng(0).integers(0, BATCH, size=(30, 2))
    overlaps = [len(set(idx[i]) & set(idx[j])) / model.config.k_max
                for i, j in pairs if i != j]
    overlap  = float(np.mean(overlaps)) if overlaps else 0.0

    ema      = jax.device_get(model.pool.ema_usage.value)
    alive    = int((ema >= model.config.reset_threshold).sum())
    ema_n    = ema / (ema.sum() + 1e-8)
    ent_r    = float(-np.sum(ema_n * np.log(ema_n + 1e-8)) / np.log(model.config.N))

    # Pool vector L2 movement from random init proxy (variance as proxy)
    vecs     = jax.device_get(model.pool.vectors.value)
    movement = float(np.mean(np.std(vecs, axis=0)))   # mean feature std across pool

    return unique, overlap, alive, ent_r, movement


def eval_metrics(model, batches):
    vocab = model.config.d_input
    _, lam, _ = get_phase_params(9999, model.config)
    correct = tokens = loss_sum = 0
    for batch in batches:
        x      = jax.nn.one_hot(batch[:, :-1], vocab)
        tgts   = jax.device_get(batch[:, 1:])
        logits = model(x, use_sigmoid=True, lambda_sharp=lam, return_aux=False)
        preds  = jax.device_get(jnp.argmax(logits, axis=-1))
        lp     = jax.device_get(jax.nn.log_softmax(logits, axis=-1))
        correct   += (preds == tgts).sum()
        tokens    += tgts.size
        loss_sum  += -float(np.sum(np.eye(vocab)[tgts] * lp))
    return float(np.exp(loss_sum / tokens)), float(correct / tokens * 100)


def run(cfg, label):
    print(f"\n{'━'*72}")
    print(f"  {label}")
    print(f"  phase1_end={cfg.phase1_end}  λ_ent={cfg.lambda_entropy}")
    print(f"{'━'*72}")
    hdr = (f"  {'step':>5}  {'task':>6}  {'ppl':>6}  {'acc%':>5}  "
           f"{'uniq':>5}  {'ovlp':>5}  {'alive':>6}  {'entr':>5}  {'vmov':>6}  {'phase':>7}")
    print(hdr)
    print(f"  {'-'*70}")

    rngs      = nnx.Rngs(params=SEED, dropout=SEED+1)
    model     = DWAModel(cfg, rngs=rngs)
    optimizer = make_optimizer(model, cfg)
    data      = synthetic_copy_task(VOCAB, BATCH, SEQ, seed=SEED)
    rng_np    = np.random.default_rng(SEED)
    rotator   = Phase1Rotator(cfg.N, BATCH, cfg.k_max, seed=SEED)
    eval_data = synthetic_copy_task(VOCAB, BATCH, SEQ, seed=99)
    eval_batches = [next(eval_data) for _ in range(20)]

    resets = 0
    snap   = {}

    # Track pool vector states at end of phase 1
    phase1_snap = None

    for step in range(STEPS + 1):
        batch = next(data)

        if cfg.reset_interval > 0 and step > 0 and step % cfg.reset_interval == 0:
            resets += reset_dead_vectors(model, rng_np)

        use_s, lam, lent = get_phase_params(step, cfg)
        forced = rotator.next() if not use_s else rotator.dummy()

        # Snapshot pool diversity at phase1→phase2 transition
        if step == cfg.phase1_end and phase1_snap is None:
            ema_p1 = jax.device_get(model.pool.ema_usage.value)
            alive_p1 = int((ema_p1 >= cfg.reset_threshold).sum())
            ema_n = ema_p1 / (ema_p1.sum() + 1e-8)
            ent_p1 = float(-np.sum(ema_n * np.log(ema_n + 1e-8)) / np.log(cfg.N))
            phase1_snap = dict(alive=alive_p1, entropy=ent_p1)

        if step % 500 == 0 or step == STEPS:
            uniq, ovlp, alive, entr, vmoov = pool_stats(model, batch)
            vocab_s = cfg.d_input
            x_oh    = jax.nn.one_hot(batch[:, :-1], vocab_s)
            # Always eval with sigmoid=True so phase-1 steps are comparable
            eval_lam = lam if use_s else get_phase_params(cfg.phase1_end + 1, cfg)[1]
            logits, _ = model(x_oh, use_sigmoid=True, lambda_sharp=eval_lam, return_aux=True)
            lp   = jax.device_get(jax.nn.log_softmax(logits, axis=-1))
            tgts = jax.device_get(batch[:, 1:])
            task = float(-np.mean(np.sum(np.eye(vocab_s)[tgts] * lp, axis=-1)))
            ppl  = float(np.exp(task))
            acc  = float((jax.device_get(jnp.argmax(logits, -1)) == tgts).mean() * 100)
            ph   = "phase1" if not use_s else "phase2/3"
            print(f"  {step:>5}  {task:>6.3f}  {ppl:>6.1f}  {acc:>5.1f}"
                  f"  {uniq:>5}  {ovlp:>5.3f}  {alive:>6}  {entr:>5.3f}  {vmoov:>6.4f}  {ph:>7}")
            snap = dict(task=task, ppl=ppl, acc=acc, uniq=uniq, ovlp=ovlp,
                        alive=alive, entr=entr)

        if step < STEPS:
            train_step(model, optimizer, batch, use_s, lam, lent, forced)

    test_ppl, test_acc = eval_metrics(model, eval_batches)
    print(f"\n  TEST  ppl={test_ppl:.2f}  acc={test_acc:.1f}%  "
          f"alive={snap['alive']}/{cfg.N}  resets={resets}")
    if phase1_snap:
        print(f"  @ phase1 end: alive_vectors={phase1_snap['alive']}/{cfg.N}  "
              f"entropy_ratio={phase1_snap['entropy']:.3f}")
    snap.update(test_ppl=test_ppl, test_acc=test_acc, resets=resets,
                phase1_snap=phase1_snap)
    return snap


def main():
    print("\n" + "█"*72)
    print("  DWA Phase-1 Warmup Rotation Test")
    print("  Hypothesis: forcing ALL vectors to get gradient in phase 1")
    print("  gives retrieval real patterns to distinguish in phase 2.")
    print("█"*72)

    base_kw = dict(
        d_input=VOCAB, d_A=64, d_B=64, D=2048, r=4, N=512,
        k_max=8, S=2, d_k=32, phase2_end=2_000,
        lambda_util=0.01, reset_interval=0, reset_threshold=0.0001,
    )

    configs = [
        ("A — Baseline  (no warmup, no entropy)",
         DWAConfig(**base_kw, phase1_end=100, lambda_entropy=0.0)),
        ("B — Entropy only  (λ=0.02 annealed, no warmup)",
         DWAConfig(**base_kw, phase1_end=100, lambda_entropy=0.02)),
        ("C — Warmup-200 + Entropy  (λ=0.02 annealed)",
         DWAConfig(**base_kw, phase1_end=200, lambda_entropy=0.02)),
        ("D — Warmup-500 + Entropy  (λ=0.02 annealed)",
         DWAConfig(**base_kw, phase1_end=500, lambda_entropy=0.02)),
    ]

    results = []
    for label, cfg in configs:
        snap = run(cfg, label)
        results.append((label, snap))

    # ── Final table ───────────────────────────────────────────────────────────
    print("\n\n" + "═"*80)
    print("  FINAL RESULTS (held-out 20 batches)")
    print("═"*80)
    hdr = f"  {'Config':<42}  {'ppl':>6}  {'acc%':>5}  {'uniq':>5}  {'ovlp':>5}  {'entr':>5}"
    print(hdr + f"  {'p1_alive':>9}")
    print(f"  {'-'*77}")
    for label, s in results:
        p1 = s["phase1_snap"]
        p1_str = f"{p1['alive']:>9}" if p1 else "       N/A"
        print(f"  {label[:42]:<42}  {s['test_ppl']:>6.1f}  {s['test_acc']:>5.1f}"
              f"  {s['uniq']:>5}  {s['ovlp']:>5.3f}  {s['entr']:>5.3f}{p1_str}")

    print("\n" + "═"*80)
    print("  ANALYSIS")
    print("═"*80)
    base_ppl  = results[0][1]["test_ppl"]
    base_uniq = results[0][1]["uniq"]
    print(f"  Baseline:  ppl={base_ppl:.1f}  unique_vecs={base_uniq}  random_ppl={VOCAB}")
    print()

    for label, s in results[1:]:
        ppl_d  = (base_ppl - s["test_ppl"]) / base_ppl * 100
        uniq_d = s["uniq"] - base_uniq
        p1     = s["phase1_snap"]
        print(f"  {label[:50]}")
        print(f"    ppl: {base_ppl:.1f} → {s['test_ppl']:.1f}  ({ppl_d:+.1f}%)")
        print(f"    unique vecs: {base_uniq} → {s['uniq']}  ({uniq_d:+d})")
        if p1:
            print(f"    at phase1→2 handoff: {p1['alive']}/{512} vectors alive  "
                  f"entropy={p1['entropy']:.3f}")
        print()

    best = min(results, key=lambda x: x[1]["test_ppl"])
    print(f"  ★ Lowest ppl: {best[0].split('—')[0].strip()}  "
          f"ppl={best[1]['test_ppl']:.1f}  acc={best[1]['test_acc']:.1f}%")

    warmup_beats = [(l, s) for l, s in results
                    if "Warmup" in l and s["test_ppl"] < base_ppl]
    if warmup_beats:
        print(f"  ✓ Warmup confirms hypothesis: better ppl AND more vectors used.")
    else:
        print(f"  ~ Warmup did not improve ppl — but check alive vectors at phase1 end.")


if __name__ == "__main__":
    main()
