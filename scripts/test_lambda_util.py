"""
test_lambda_util.py

Compares lambda_util values [0.01, 0.05, 0.1, 0.3] over 2000 training steps
to diagnose vector selection collapse in the DWA model.

Run from project root:
  cd /home/dev/ml_model/computd
  JAX_DEBUG_NANS=False JAX_LOG_COMPILES=False python scripts/test_lambda_util.py 2>&1
"""

import os
import sys

os.environ["JAX_DEBUG_NANS"] = "False"
os.environ["JAX_LOG_COMPILES"] = "False"
sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from configs.small import DWAConfig
from src.model.dwa import DWAModel
from src.training.trainer import make_optimizer, make_train_step, get_phase_params
from src.data.loader import synthetic_copy_task

# ── Constants ──────────────────────────────────────────────────────────────────
LAMBDA_UTIL_VALUES = [0.01, 0.05, 0.1, 0.3]
TOTAL_STEPS = 2000
CHECKPOINT_STEPS = [0, 200, 500, 1000, 1500, 2000]
EVAL_BATCHES = 10
BATCH_SIZE = 32
SEQ_LEN = 32
VOCAB_SIZE = 64   # DWAConfig.d_input default
SEED = 42


# ── Metrics helpers ────────────────────────────────────────────────────────────

def compute_perplexity(task_loss: float) -> float:
    return float(np.exp(min(task_loss, 20.0)))


def compute_top1_acc(model, batch, use_sigmoid, lambda_sharp):
    """Top-1 accuracy over batch[:, 1:] targets."""
    x_onehot = jax.nn.one_hot(batch[:, :-1], VOCAB_SIZE)
    logits = model(x_onehot, use_sigmoid=use_sigmoid, lambda_sharp=lambda_sharp)
    preds = jnp.argmax(logits, axis=-1)          # (B, T-1)
    targets = batch[:, 1:]
    acc = jnp.mean(preds == targets)
    return float(jax.device_get(acc))


def compute_diversity_metrics(model, batch, use_sigmoid, lambda_sharp):
    """
    Returns:
      unique_vecs  - number of distinct pool vectors selected in this batch
      mean_overlap - mean pairwise Jaccard overlap across 20 random pairs
      pool_util_pct - % of vectors with EMA > 0.001
      entropy_ratio - EMA entropy ratio (0=collapsed, 1=fully spread)
    """
    x_onehot = jax.nn.one_hot(batch[:, :-1], VOCAB_SIZE)
    _, aux = model(
        x_onehot,
        use_sigmoid=use_sigmoid,
        lambda_sharp=lambda_sharp,
        return_aux=True,
    )
    idx = jax.device_get(aux["idx"])   # (B, k_max) — but model is per-token
    # idx shape could be (B, T-1, k_max) if retrieval is per token; flatten safely
    idx_flat_per_sample = idx.reshape(BATCH_SIZE, -1)  # (B, k_max * T or k_max)

    unique_vecs = len(set(idx_flat_per_sample.flatten().tolist()))

    # Pairwise overlap over 20 random pairs (i, j) from batch dimension
    rng = np.random.default_rng(0)
    n_pairs = min(20, BATCH_SIZE * (BATCH_SIZE - 1) // 2)
    pairs = rng.choice(BATCH_SIZE, size=(n_pairs, 2), replace=True)
    k_max = idx_flat_per_sample.shape[1]
    overlaps = []
    for i, j in pairs:
        if i == j:
            continue
        si = set(idx_flat_per_sample[i].tolist())
        sj = set(idx_flat_per_sample[j].tolist())
        overlap = len(si & sj) / max(k_max, 1)
        overlaps.append(overlap)
    mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0

    # Pool EMA stats
    ema = jax.device_get(model.pool.ema_usage.value)  # (512,)
    pool_util_pct = float(np.mean(ema > 0.001) * 100.0)

    # EMA entropy ratio
    ema_norm = ema / (ema.sum() + 1e-8)
    entropy = -np.sum(ema_norm * np.log(ema_norm + 1e-8))
    max_entropy = np.log(model.config.N)
    entropy_ratio = float(entropy / max_entropy)

    return unique_vecs, mean_overlap, pool_util_pct, entropy_ratio


def eval_on_held_out(model, data_iter, n_batches, use_sigmoid, lambda_sharp):
    """Compute mean task loss and top-1 accuracy over n_batches eval batches."""
    total_loss = 0.0
    total_acc = 0.0
    for _ in range(n_batches):
        batch = next(data_iter)
        x_onehot = jax.nn.one_hot(batch[:, :-1], VOCAB_SIZE)
        logits, aux = model(
            x_onehot,
            use_sigmoid=use_sigmoid,
            lambda_sharp=lambda_sharp,
            return_aux=True,
        )
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        targets = batch[:, 1:]
        task_loss = -jnp.mean(
            jnp.sum(jax.nn.one_hot(targets, VOCAB_SIZE) * log_probs, axis=-1)
        )
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == targets)
        total_loss += float(jax.device_get(task_loss))
        total_acc += float(jax.device_get(acc))
    return total_loss / n_batches, total_acc / n_batches


# ── Per-run storage ────────────────────────────────────────────────────────────

class RunResult:
    def __init__(self, lambda_util):
        self.lambda_util = lambda_util
        # Checkpoint records: list of dicts
        self.checkpoints = []
        # Final eval
        self.final_ppl = None
        self.final_acc = None


# ── Main training loop ─────────────────────────────────────────────────────────

def run_experiment(lambda_util_val: float) -> RunResult:
    print(f"\n{'='*70}")
    print(f"  Starting run: lambda_util = {lambda_util_val}")
    print(f"{'='*70}")

    cfg = DWAConfig(lambda_util=lambda_util_val)
    rngs = nnx.Rngs(params=SEED, dropout=SEED + 1)
    model = DWAModel(cfg, rngs=rngs)
    optimizer = make_optimizer(model, cfg)

    graphdef, state = nnx.split(model, nnx.Param)
    opt_graphdef, opt_state = nnx.split(optimizer, nnx.Param)
    step_phase1 = make_train_step(cfg, use_sigmoid=False)
    step_phase2 = make_train_step(cfg, use_sigmoid=True)

    data_iter = synthetic_copy_task(
        vocab_size=VOCAB_SIZE,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        seed=SEED,
    )

    result = RunResult(lambda_util_val)
    checkpoint_set = set(CHECKPOINT_STEPS)

    for step in range(TOTAL_STEPS + 1):
        if step in checkpoint_set:
            use_sigmoid, lambda_sharp = get_phase_params(step, cfg)

            # Get a fresh eval batch for diversity metrics
            eval_batch = next(data_iter)
            x_onehot = jax.nn.one_hot(eval_batch[:, :-1], VOCAB_SIZE)
            logits, aux = model(
                x_onehot,
                use_sigmoid=use_sigmoid,
                lambda_sharp=lambda_sharp,
                return_aux=True,
            )
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            targets = eval_batch[:, 1:]
            task_loss = float(jax.device_get(
                -jnp.mean(jnp.sum(jax.nn.one_hot(targets, VOCAB_SIZE) * log_probs, axis=-1))
            ))
            ppl = compute_perplexity(task_loss)
            acc = compute_top1_acc(model, eval_batch, use_sigmoid, lambda_sharp)
            unique_vecs, mean_overlap, pool_util_pct, entropy_ratio = compute_diversity_metrics(
                model, eval_batch, use_sigmoid, lambda_sharp
            )

            rec = {
                "step": step,
                "task_loss": task_loss,
                "ppl": ppl,
                "top1_acc": acc,
                "unique_vecs": unique_vecs,
                "mean_overlap": mean_overlap,
                "pool_util_pct": pool_util_pct,
                "entropy_ratio": entropy_ratio,
            }
            result.checkpoints.append(rec)

            print(
                f"  step={step:4d} | loss={task_loss:.4f} ppl={ppl:6.2f} "
                f"acc={acc:.3f} | unique={unique_vecs:3d}/512 "
                f"overlap={mean_overlap:.3f} util={pool_util_pct:5.1f}% "
                f"ent={entropy_ratio:.3f}"
            )

        if step < TOTAL_STEPS:
            use_sigmoid, lambda_sharp = get_phase_params(step, cfg)
            step_fn = step_phase2 if use_sigmoid else step_phase1
            batch = next(data_iter)
            forced_idx = jnp.zeros((BATCH_SIZE, cfg.k_max), dtype=jnp.int32)
            metrics, state, opt_state = step_fn(
                graphdef, state, opt_graphdef, opt_state, batch,
                jnp.float32(lambda_sharp), jnp.float32(0.0), forced_idx
            )

    nnx.update(model, state)
    nnx.update(optimizer, opt_state)
    # Final held-out evaluation
    eval_iter = synthetic_copy_task(
        vocab_size=VOCAB_SIZE,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        seed=SEED + 999,   # different seed for held-out
    )
    use_sigmoid, lambda_sharp = get_phase_params(TOTAL_STEPS, cfg)
    final_loss, final_acc = eval_on_held_out(
        model, eval_iter, EVAL_BATCHES, use_sigmoid, lambda_sharp
    )
    result.final_ppl = compute_perplexity(final_loss)
    result.final_acc = final_acc

    print(
        f"  >> FINAL EVAL (held-out): ppl={result.final_ppl:.2f}  acc={result.final_acc:.3f}"
    )
    return result


# ── Table printing ─────────────────────────────────────────────────────────────

def print_comparison_table(results: list[RunResult]):
    lam_vals = [r.lambda_util for r in results]

    print("\n")
    print("=" * 110)
    print("  COMPARISON TABLE: lambda_util effect on DWA vector selection")
    print("=" * 110)

    # Header
    col_w = 22
    hdr = f"{'metric':<28}" + "".join(f"{'λ='+str(v):<{col_w}}" for v in lam_vals)
    print(hdr)
    print("-" * 110)

    metrics_order = [
        ("task_loss",     "Task Loss"),
        ("ppl",           "Perplexity"),
        ("top1_acc",      "Top-1 Acc"),
        ("unique_vecs",   "Unique Vecs /512"),
        ("mean_overlap",  "Mean Overlap"),
        ("pool_util_pct", "Pool Util %"),
        ("entropy_ratio", "EMA Entropy Ratio"),
    ]

    for step in CHECKPOINT_STEPS:
        print(f"\n  -- Step {step} --")
        for key, label in metrics_order:
            row = f"  {label:<26}"
            for r in results:
                cp = next((c for c in r.checkpoints if c["step"] == step), None)
                if cp is None:
                    row += f"{'N/A':<{col_w}}"
                else:
                    v = cp[key]
                    if isinstance(v, float):
                        row += f"{v:<{col_w}.4f}"
                    else:
                        row += f"{v:<{col_w}}"
            print(row)

    # Final eval row
    print("\n  -- Final Held-Out Eval --")
    for label, attr in [("Perplexity", "final_ppl"), ("Top-1 Acc", "final_acc")]:
        row = f"  {label:<26}"
        for r in results:
            v = getattr(r, attr)
            row += f"{v:<{col_w}.4f}"
        print(row)

    print("=" * 110)


def print_verdict(results: list[RunResult]):
    print("\n")
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)

    # Find best diversity (highest entropy at step 2000)
    best_ent = max(results, key=lambda r: r.checkpoints[-1]["entropy_ratio"])
    best_task = min(results, key=lambda r: r.checkpoints[-1]["task_loss"])
    best_util = max(results, key=lambda r: r.checkpoints[-1]["pool_util_pct"])

    print(f"\n  [Diversity] Best EMA entropy at step 2000: lambda_util={best_ent.lambda_util}")
    print(f"    entropy_ratio = {best_ent.checkpoints[-1]['entropy_ratio']:.4f} "
          f"(unique_vecs={best_ent.checkpoints[-1]['unique_vecs']}, "
          f"pool_util={best_ent.checkpoints[-1]['pool_util_pct']:.1f}%)")

    print(f"\n  [Task Quality] Best task loss at step 2000: lambda_util={best_task.lambda_util}")
    print(f"    task_loss = {best_task.checkpoints[-1]['task_loss']:.4f} "
          f"(ppl={best_task.checkpoints[-1]['ppl']:.2f})")

    print(f"\n  [Pool Utilization] Best pool util at step 2000: lambda_util={best_util.lambda_util}")
    print(f"    pool_util = {best_util.checkpoints[-1]['pool_util_pct']:.1f}%")

    # Does higher lambda_util hurt task performance?
    losses_2000 = [(r.lambda_util, r.checkpoints[-1]["task_loss"]) for r in results]
    losses_2000_sorted = sorted(losses_2000, key=lambda x: x[0])
    task_monotone_up = all(
        losses_2000_sorted[i][1] <= losses_2000_sorted[i+1][1]
        for i in range(len(losses_2000_sorted)-1)
    )
    task_monotone_down = all(
        losses_2000_sorted[i][1] >= losses_2000_sorted[i+1][1]
        for i in range(len(losses_2000_sorted)-1)
    )
    print("\n  [Trade-off Analysis]")
    if task_monotone_up:
        print("    Higher lambda_util HURTS task performance monotonically.")
        print("    There is a clear diversity-vs-quality tradeoff.")
    elif task_monotone_down:
        print("    Higher lambda_util HELPS task performance monotonically.")
        print("    The utilization loss regularizes effectively without quality cost.")
    else:
        print("    Non-monotonic relationship: optimal lambda_util is in the middle range.")
    for lam, loss in losses_2000_sorted:
        print(f"      lambda={lam}: task_loss={loss:.4f}")

    # Recommended value: best balance score = entropy_ratio / task_loss
    # (high diversity, low loss)
    scores = []
    for r in results:
        cp = r.checkpoints[-1]
        # Balance: maximize entropy and minimize task loss
        # Normalize: task_loss closer to 0 = better, entropy_ratio closer to 1 = better
        task_loss_norm = cp["task_loss"]
        entropy = cp["entropy_ratio"]
        # Composite: penalize collapsed vectors and high task loss
        balance = entropy / (task_loss_norm + 1e-4)
        scores.append((r.lambda_util, balance, entropy, task_loss_norm))

    scores.sort(key=lambda x: x[1], reverse=True)
    best_lam, best_score, best_ent_v, best_task_v = scores[0]

    print(f"\n  [Recommendation]")
    print(f"    Best balanced lambda_util = {best_lam}")
    print(f"      entropy_ratio={best_ent_v:.4f}, task_loss={best_task_v:.4f}, "
          f"balance_score={best_score:.4f}")

    # Collapse diagnosis
    print(f"\n  [Collapse Diagnosis]")
    baseline = results[0]  # lambda=0.01
    final_uniq = baseline.checkpoints[-1]["unique_vecs"]
    if final_uniq < 50:
        print(f"    CONFIRMED collapse at lambda_util=0.01: only {final_uniq}/512 vectors active.")
    else:
        print(f"    No severe collapse at lambda_util=0.01: {final_uniq}/512 vectors active.")

    for r in results:
        uniq = r.checkpoints[-1]["unique_vecs"]
        ent = r.checkpoints[-1]["entropy_ratio"]
        util = r.checkpoints[-1]["pool_util_pct"]
        print(f"    lambda={r.lambda_util}: unique={uniq}/512, "
              f"pool_util={util:.1f}%, entropy={ent:.4f}")

    print("\n  [Final Answer]")
    print(f"    Recommended lambda_util = {best_lam}")
    print("=" * 70)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("DWA lambda_util sweep")
    print(f"  TOTAL_STEPS={TOTAL_STEPS}, BATCH_SIZE={BATCH_SIZE}, "
          f"SEQ_LEN={SEQ_LEN}, VOCAB_SIZE={VOCAB_SIZE}")
    print(f"  lambda_util values: {LAMBDA_UTIL_VALUES}")
    print(f"  Checkpoint steps: {CHECKPOINT_STEPS}")
    print(f"  JAX devices: {jax.devices()}")

    results = []
    for lam in LAMBDA_UTIL_VALUES:
        r = run_experiment(lam)
        results.append(r)

    print_comparison_table(results)
    print_verdict(results)


if __name__ == "__main__":
    main()
