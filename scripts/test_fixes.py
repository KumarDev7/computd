"""
Sweep: Baseline vs Annealed Entropy Loss (no codebook reset disruption).
Annealing: λ_ent starts at max in early phase-2, tapers to 0.3× by end of phase-2,
           then 0.1× in phase-3 — strong early to prevent collapse, relaxes for specialization.
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
from src.training.trainer import make_optimizer, train_step, get_phase_params, reset_dead_vectors
from src.data.loader import synthetic_copy_task


VOCAB  = 64
BATCH  = 32
SEQ    = 16
STEPS  = 3000
SEED   = 42


def pool_stats(model, batch):
    vocab = model.config.d_input
    x     = jax.nn.one_hot(batch[:, :-1], vocab)
    use_s, lam, lent = get_phase_params(9999, model.config)
    _, aux = model(x, use_sigmoid=use_s, lambda_sharp=lam, return_aux=True)
    idx    = jax.device_get(aux["idx"])

    unique   = len(set(idx.flatten()))
    pairs    = np.random.default_rng(0).integers(0, BATCH, size=(30, 2))
    overlaps = [len(set(idx[i]) & set(idx[j])) / model.config.k_max
                for i, j in pairs if i != j]
    overlap  = float(np.mean(overlaps)) if overlaps else 0.0

    ema      = jax.device_get(model.pool.ema_usage.value)
    util_pct = float((ema >= model.config.reset_threshold).mean() * 100)
    ema_n    = ema / (ema.sum() + 1e-8)
    ent_r    = float(-np.sum(ema_n * np.log(ema_n + 1e-8)) / np.log(model.config.N))

    return unique, overlap, util_pct, ent_r


def eval_metrics(model, batches):
    vocab  = model.config.d_input
    use_s, lam, _ = get_phase_params(9999, model.config)
    correct = tokens = loss_sum = 0
    for batch in batches:
        x      = jax.nn.one_hot(batch[:, :-1], vocab)
        tgts   = jax.device_get(batch[:, 1:])
        logits = model(x, use_sigmoid=use_s, lambda_sharp=lam, return_aux=False)
        preds  = jax.device_get(jnp.argmax(logits, axis=-1))
        lp     = jax.device_get(jax.nn.log_softmax(logits, axis=-1))
        correct   += (preds == tgts).sum()
        tokens    += tgts.size
        loss_sum  += -float(np.sum(np.eye(vocab)[tgts] * lp))
    return float(np.exp(loss_sum / tokens)), float(correct / tokens * 100)


def run(cfg, label):
    print(f"\n{'─'*72}")
    print(f"  {label}")
    print(f"  λ_ent(base)={cfg.lambda_entropy}  reset_interval={cfg.reset_interval}")
    print(f"{'─'*72}")
    hdr = f"  {'step':>5}  {'task':>6}  {'ppl':>6}  {'acc%':>5}  {'uniq':>5}  {'ovlp':>5}  {'util%':>6}  {'entr':>5}  {'λ_ent':>6}"
    print(hdr)
    print(f"  {'-'*70}")

    rngs      = nnx.Rngs(params=SEED, dropout=SEED+1)
    model     = DWAModel(cfg, rngs=rngs)
    optimizer = make_optimizer(model, cfg)
    data      = synthetic_copy_task(VOCAB, BATCH, SEQ, seed=SEED)
    rng_np    = np.random.default_rng(SEED)
    eval_data = synthetic_copy_task(VOCAB, BATCH, SEQ, seed=99)
    eval_batches = [next(eval_data) for _ in range(20)]

    resets = 0
    snap   = {}

    for step in range(STEPS + 1):
        batch = next(data)

        if cfg.reset_interval > 0 and step > 0 and step % cfg.reset_interval == 0:
            resets += reset_dead_vectors(model, rng_np)

        use_s, lam, lent = get_phase_params(step, cfg)

        if step % 500 == 0 or step == STEPS:
            uniq, ovlp, util, entr = pool_stats(model, batch)
            vocab_s = cfg.d_input
            x_oh    = jax.nn.one_hot(batch[:, :-1], vocab_s)
            logits, _ = model(x_oh, use_sigmoid=use_s, lambda_sharp=lam, return_aux=True)
            lp   = jax.device_get(jax.nn.log_softmax(logits, axis=-1))
            tgts = jax.device_get(batch[:, 1:])
            task = float(-np.mean(np.sum(np.eye(vocab_s)[tgts] * lp, axis=-1)))
            ppl  = float(np.exp(task))
            acc  = float((jax.device_get(jnp.argmax(logits, -1)) == tgts).mean() * 100)
            print(f"  {step:>5}  {task:>6.3f}  {ppl:>6.1f}  {acc:>5.1f}"
                  f"  {uniq:>5}  {ovlp:>5.3f}  {util:>6.1f}  {entr:>5.3f}  {lent:>6.4f}")
            snap = dict(task=task, ppl=ppl, acc=acc, uniq=uniq, ovlp=ovlp,
                        util=util, entr=entr)

        if step < STEPS:
            train_step(model, optimizer, batch, use_s, lam, lent)

    test_ppl, test_acc = eval_metrics(model, eval_batches)
    ema   = jax.device_get(model.pool.ema_usage.value)
    alive = int((ema >= cfg.reset_threshold).sum())
    print(f"\n  TEST  ppl={test_ppl:.2f}  acc={test_acc:.1f}%  "
          f"alive={alive}/{cfg.N}  resets={resets}")
    snap.update(test_ppl=test_ppl, test_acc=test_acc, alive=alive, resets=resets)
    return snap


def main():
    print("\n" + "█"*72)
    print("  DWA Annealed Entropy Loss — Sweep")
    print("  (No disruptive codebook reset. Entropy tapers as model matures.)")
    print("█"*72)

    base_kw = dict(
        d_input=VOCAB, d_A=64, d_B=64, D=2048, r=4, N=512,
        k_max=8, S=2, d_k=32, phase1_end=100, phase2_end=1_000,
        lambda_util=0.01, reset_interval=0, reset_threshold=0.0001,
    )

    configs = [
        ("Baseline  (λ_ent=0,     no anneal)",
         DWAConfig(**base_kw, lambda_entropy=0.0)),
        ("Gentle    (λ_ent=0.01,  annealed)",
         DWAConfig(**base_kw, lambda_entropy=0.01)),
        ("Moderate  (λ_ent=0.02,  annealed)",
         DWAConfig(**base_kw, lambda_entropy=0.02)),
        ("Strong    (λ_ent=0.05,  annealed)",
         DWAConfig(**base_kw, lambda_entropy=0.05)),
        ("Moderate+ reset/2000",
         DWAConfig(**{**base_kw, "reset_interval": 2000}, lambda_entropy=0.02)),
    ]

    results = []
    for label, cfg in configs:
        snap = run(cfg, label)
        results.append((label, snap))

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n\n" + "═"*80)
    print("  FINAL COMPARISON  (test on held-out 20 batches)")
    print("═"*80)
    hdr = f"  {'Config':<38}  {'ppl':>6}  {'acc%':>5}  {'uniq':>5}  {'ovlp':>5}  {'entr':>5}  {'alive':>6}"
    print(hdr)
    print(f"  {'-'*73}")
    for label, s in results:
        print(f"  {label[:38]:<38}  {s['test_ppl']:>6.1f}  {s['test_acc']:>5.1f}"
              f"  {s['uniq']:>5}  {s['ovlp']:>5.3f}  {s['entr']:>5.3f}  {s['alive']:>6}")

    print("\n" + "═"*80)
    print("  VERDICT")
    print("═"*80)
    base_ppl  = results[0][1]["test_ppl"]
    base_uniq = results[0][1]["uniq"]
    print(f"  Random baseline ppl = {VOCAB}")
    print(f"  Baseline:  ppl={base_ppl:.1f}  uniq={base_uniq}")
    print()

    # Best: lowest ppl that also has > 2× baseline unique vectors
    candidates = [(l, s) for l, s in results if s["uniq"] > base_uniq * 2]
    if candidates:
        best = min(candidates, key=lambda x: x[1]["test_ppl"])
        print(f"  ★ Best balanced config: {best[0]}")
        print(f"    ppl={best[1]['test_ppl']:.1f}  acc={best[1]['test_acc']:.1f}%"
              f"  uniq={best[1]['uniq']}  entr={best[1]['entr']:.3f}")
        ppl_chg  = (base_ppl - best[1]['test_ppl']) / base_ppl * 100
        uniq_chg = best[1]['uniq'] - base_uniq
        print(f"    PPL change: {ppl_chg:+.1f}%   Unique vectors: {base_uniq} → {best[1]['uniq']} ({uniq_chg:+d})")
        if best[1]['test_ppl'] <= base_ppl:
            print("  ✓ FIXED: better or equal task performance WITH more diverse selection.")
        else:
            print("  ~ PARTIAL: more diverse but small ppl cost — acceptable tradeoff.")
    else:
        best_ppl = min(results, key=lambda x: x[1]["test_ppl"])
        print(f"  ★ Lowest ppl: {best_ppl[0]}  ppl={best_ppl[1]['test_ppl']:.1f}")
        print("  No config achieved 2× baseline unique vectors without ppl cost.")


if __name__ == "__main__":
    main()
