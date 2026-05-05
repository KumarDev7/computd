"""
Benchmark Dense-Large and Dense-Small against pre-captured DWA results.
DWA was already trained for 20K steps; results are embedded below.
"""
import os, sys, time
os.environ["JAX_DEBUG_NANS"]   = "False"
os.environ["JAX_LOG_COMPILES"] = "False"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax, jax.numpy as jnp
from flax import nnx

from src.model.dense_lm import DenseLM, make_dense_configs
from scripts.benchmark_vs_dense import (
    dense_train_step, make_dense_optimizer,
    compute_val_loss, generate_text, PPL_THRESHOLDS
)
from src.data.text_loader import shakespeare_loader

DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "shakespeare.txt")
BATCH_SIZE  = 32
SEQ_LEN     = 64
TOTAL_STEPS = 20_000
LOG_EVERY   = 500
SEED        = 42
VOCAB       = 65

# ── DWA results from previous 20K-step run ────────────────────────────────────
DWA_STEPS = [0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,
             6500,7000,7500,8000,8500,9000,9500,10000,10500,11000,11500,
             12000,12500,13000,13500,14000,14500,15000,15500,16000,16500,
             17000,17500,18000,18500,19000,19500,19999]
DWA_PPLS  = [58.19,7.34,6.75,6.49,6.28,6.18,6.13,5.99,5.96,5.84,5.83,5.79,
             5.78,5.80,5.69,5.68,5.69,5.64,5.68,5.65,5.62,5.58,5.57,5.59,
             5.55,5.55,5.56,5.54,5.52,5.52,5.49,5.55,5.50,5.47,5.48,5.46,
             5.48,5.46,5.48,5.44,5.48]
DWA_PARAMS = 2_936_396


def run_dense(name, model, optimizer, tok, data_gen):
    steps_log, ppl_log = [], []
    thresholds = {t: None for t in PPL_THRESHOLDS}
    t0 = time.time()

    print(f"\n{'─'*65}")
    print(f"  Training: {name}")
    print(f"{'─'*65}")

    for step in range(TOTAL_STEPS):
        batch, _ = next(data_gen)
        dense_train_step(model, optimizer, batch)

        if step % LOG_EVERY == 0 or step == TOTAL_STEPS - 1:
            vl  = compute_val_loss(model, "dense", tok, n_batches=15)
            ppl = float(np.exp(vl))
            steps_log.append(step)
            ppl_log.append(ppl)
            print(f"  step={step:>6}  val_ppl={ppl:>6.2f}  t={time.time()-t0:.0f}s")
            for thresh in PPL_THRESHOLDS:
                if thresholds[thresh] is None and ppl <= thresh:
                    thresholds[thresh] = step

    return steps_log, ppl_log, thresholds


def print_report(results):
    print(f"\n\n{'═'*65}")
    print(f"  BENCHMARK RESULTS  —  DWA vs Dense (20K steps each)")
    print(f"{'═'*65}")

    # Learning curve (sampled at every 2K steps for readability)
    sample_steps = [s for s in DWA_STEPS if s % 2000 == 0 or s == 19999]
    header = f"  {'step':>6}  {'DWA(2.9M)':>10}  {'Dense-L(2.9M)':>14}  {'Dense-S(300K)':>14}"
    print(f"\n  Val PPL every 2K steps:")
    print(f"  {'─'*58}")
    print(header)
    print(f"  {'─'*58}")
    for s in sample_steps:
        # DWA
        di = min(range(len(DWA_STEPS)), key=lambda j: abs(DWA_STEPS[j]-s))
        dwa_ppl = f"{DWA_PPLS[di]:>10.2f}"
        row = f"  {s:>6}  {dwa_ppl}"
        for name in ["Dense-Large", "Dense-Small"]:
            r = results[name]
            idx = min(range(len(r["steps"])), key=lambda j: abs(r["steps"][j]-s))
            if abs(r["steps"][idx] - s) <= LOG_EVERY:
                row += f"  {r['ppls'][idx]:>14.2f}"
            else:
                row += f"  {'—':>14}"
        print(row)

    # Threshold table
    dwa_thresh = {}
    for t in PPL_THRESHOLDS:
        for i, (s, p) in enumerate(zip(DWA_STEPS, DWA_PPLS)):
            if p <= t:
                dwa_thresh[t] = s; break
        else:
            dwa_thresh[t] = None

    print(f"\n  Steps to first reach PPL threshold:")
    print(f"  {'─'*60}")
    print(f"  {'PPL≤':>6}  {'DWA':>8}  {'Dense-L':>10}  {'Dense-S':>10}  {'DL/DWA':>8}")
    print(f"  {'─'*60}")
    for t in PPL_THRESHOLDS:
        dwa_s = dwa_thresh[t]
        dl_s  = results["Dense-Large"]["thresh"][t]
        ds_s  = results["Dense-Small"]["thresh"][t]
        ratio = f"{dl_s/max(dwa_s,1):.1f}x" if dwa_s and dl_s else "—"
        print(f"  {t:>6}  {str(dwa_s) if dwa_s else 'never':>8}  "
              f"{str(dl_s) if dl_s else 'never':>10}  "
              f"{str(ds_s) if ds_s else 'never':>10}  {ratio:>8}")

    # Final summary
    dwa_final = DWA_PPLS[-1]
    print(f"\n  Final PPL at step {TOTAL_STEPS}:")
    print(f"  {'─'*55}")
    rows = [("DWA", DWA_PARAMS, dwa_final, "dwa")]
    for name, r in results.items():
        rows.append((name, r["n_params"], r["ppls"][-1], r["type"]))
    for name, n_p, ppl, _ in rows:
        delta = (ppl - dwa_final) / dwa_final * 100
        sign  = "+" if delta >= 0 else ""
        bar   = "▓" * int(ppl) + "░" * max(0, int(dwa_final) - int(ppl))
        print(f"  {name:<15}  {n_p:>10,}p  PPL={ppl:.3f}  ({sign}{delta:.1f}% vs DWA)")

    # Generated text
    print(f"\n\n{'═'*65}")
    print(f"  GENERATED TEXT  (prompt='HAMLET:\\n'  temp=0.8  top_k=40)")
    print(f"{'═'*65}")
    gen = shakespeare_loader(DATA_PATH, 1, 8, seed=0)
    _, tok = next(gen)
    for name, r in results.items():
        print(f"\n  ── {name} (PPL={r['ppls'][-1]:.2f}) ──")
        text = generate_text(r["model"], "dense", tok,
                             "HAMLET:\n", max_new=300, temperature=0.8, top_k=40)
        print("  " + text.replace("\n", "\n  "))

    # Verdict
    print(f"\n{'═'*65}")
    print(f"  VERDICT")
    print(f"{'═'*65}")
    dl_final = results["Dense-Large"]["ppls"][-1]
    ds_final = results["Dense-Small"]["ppls"][-1]
    gap_dl = (dl_final - dwa_final) / dwa_final * 100

    print(f"\n  DWA (2.9M):          PPL = {dwa_final:.3f}")
    print(f"  Dense-Large (2.9M):  PPL = {dl_final:.3f}  ({'+' if gap_dl>=0 else ''}{gap_dl:.1f}% vs DWA)")
    print(f"  Dense-Small (300K):  PPL = {ds_final:.3f}")

    if dwa_final < dl_final:
        print(f"\n  ★ DWA wins at equal param count by {abs(gap_dl):.1f}% better PPL.")
        print(f"    Pool mechanism gives better capacity utilisation than dense weights.")
    else:
        # How many extra steps does DWA need?
        target = dl_final
        dwa_cross = next((s for s,p in zip(DWA_STEPS, DWA_PPLS) if p <= target), None)
        dl_step   = TOTAL_STEPS
        if dwa_cross:
            print(f"\n  Dense-Large wins by {abs(gap_dl):.1f}% PPL at equal params.")
            print(f"  DWA reaches Dense-Large's final quality at step ~{dwa_cross} "
                  f"({dwa_cross/dl_step:.2f}x the training budget).")
        else:
            print(f"\n  Dense-Large wins by {abs(gap_dl):.1f}% PPL.")
            print(f"  DWA did not reach Dense-Large's final PPL={target:.3f} in {TOTAL_STEPS} steps.")

    # Extra steps Dense-Large needs to match DWA (if DWA wins)
    if dwa_final < dl_final:
        for t in [dwa_final + 0.01, dwa_final + 0.1, dwa_final + 0.3]:
            dl_reach = results["Dense-Large"]["thresh"].get(round(t, 2))
            # Find first step where Dense-L PPL <= t
            dl_reach = next(
                (s for s,p in zip(results["Dense-Large"]["steps"],
                                  results["Dense-Large"]["ppls"]) if p <= t), None)
            dwa_reach = next((s for s,p in zip(DWA_STEPS, DWA_PPLS) if p <= t), None)
            if dl_reach and dwa_reach:
                print(f"  To reach PPL≤{t:.2f}: DWA needs {dwa_reach} steps, "
                      f"Dense-L needs {dl_reach} steps ({dl_reach/max(dwa_reach,1):.1f}x more)")


def main():
    gen = shakespeare_loader(DATA_PATH, BATCH_SIZE, SEQ_LEN, split="train", seed=SEED)
    _, tok = next(gen)

    print("=" * 65)
    print("  Dense Model Benchmark  (DWA results pre-captured)")
    print(f"  Steps={TOTAL_STEPS}  Batch={BATCH_SIZE}  SeqLen={SEQ_LEN}")
    print("=" * 65)
    print(f"\n  DWA (pre-captured): {DWA_PARAMS:,} params  final PPL={DWA_PPLS[-1]:.3f}")

    results = {}
    dcfgs   = make_dense_configs(VOCAB)

    for key, label in [("large", "Dense-Large"), ("small", "Dense-Small")]:
        rngs  = nnx.Rngs(params=SEED, dropout=SEED+1)
        model = DenseLM(**dcfgs[key], rngs=rngs)
        n     = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
        opt   = make_dense_optimizer(model)
        data  = shakespeare_loader(DATA_PATH, BATCH_SIZE, SEQ_LEN, split="train", seed=SEED)
        print(f"\n  {label}: {n:,} params")

        steps, ppls, thresh = run_dense(label, model, opt, tok, data)
        results[label] = dict(model=model, steps=steps, ppls=ppls,
                              thresh=thresh, n_params=n, type="dense")

    print_report(results)


if __name__ == "__main__":
    main()
