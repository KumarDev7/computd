"""
DWA Accuracy Verification — proper token-by-token evaluation on two tasks:
  1. Copy task (repeating pattern) — tests basic learning
  2. Bigram task (deterministic transitions) — tests input-dependent retrieval

Fixes applied:
  - Per-position z queries (each token independently selects pool vectors)
  - Shared bigram transition table between train and test
  - No positional encoding (clean one-hot input)
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
from src.training.trainer import make_optimizer, make_train_step, get_phase_params, Phase1Rotator
from src.data.loader import synthetic_copy_task, bigram_task, make_bigram_table, random_token_batches


VOCAB  = 64
BATCH  = 32
SEQ    = 16
STEPS  = 5000
SEED   = 42


# ─── Train + eval helpers ─────────────────────────────────────────────────────

def train_and_eval(task_name, data_fn, eval_data_fn=None):
    """Full train→eval pipeline for one task."""
    if eval_data_fn is None:
        eval_data_fn = data_fn

    cfg = DWAConfig(
        d_input=VOCAB, d_A=64, d_B=64, D=2048, r=4, N=512,
        k_max=8, S=2, d_k=32,
        phase1_end=100, phase2_end=1_500,
        lambda_entropy=0.02,
    )

    rngs      = nnx.Rngs(params=SEED, dropout=SEED+1)
    model     = DWAModel(cfg, rngs=rngs)
    optimizer = make_optimizer(model, cfg)
    data      = data_fn(VOCAB, BATCH, SEQ, seed=SEED)
    rotator   = Phase1Rotator(cfg.N, BATCH, cfg.k_max, seed=SEED)

    graphdef, state = nnx.split(model)
    opt_graphdef, opt_state = nnx.split(optimizer)
    step_phase1 = make_train_step(cfg, use_sigmoid=False)
    step_phase2 = make_train_step(cfg, use_sigmoid=True)

    # Pre-generate held-out test data using the eval generator (same transition table)
    test_data = [next(eval_data_fn(VOCAB, BATCH, SEQ, seed=SEED + i + 1)) for i in range(20)]

    print(f"\n{'━'*72}")
    print(f"  TRAINING ON: {task_name}")
    print(f"{'━'*72}")

    for step in range(STEPS):
        batch = next(data)
        use_s, lam, lent = get_phase_params(step, cfg)
        step_fn = step_phase2 if use_s else step_phase1
        forced = rotator.next() if not use_s else rotator.dummy()
        metrics, state, opt_state = step_fn(
            graphdef, state, opt_graphdef, opt_state, batch,
            jnp.float32(lam), jnp.float32(lent), forced
        )

        if step % 1000 == 0 or step == STEPS - 1:
            vocab = cfg.d_input
            x_oh   = jax.nn.one_hot(batch[:, :-1], vocab)
            logits = model(x_oh, use_sigmoid=True, lambda_sharp=5.0, return_aux=False)
            tgts  = batch[:, 1:]
            preds = jnp.argmax(logits, axis=-1)
            acc   = float(jax.device_get((preds == tgts).mean() * 100))
            lp    = jax.device_get(jax.nn.log_softmax(logits, axis=-1))
            loss  = -float(np.mean(np.sum(np.eye(vocab)[jax.device_get(tgts)] * lp, axis=-1)))
            print(f"  step={step:>5}  loss={loss:.4f}  acc={acc:.1f}%")

    nnx.update(model, state)
    nnx.update(optimizer, opt_state)
    results = full_eval(model, test_data, cfg)
    ret     = check_retrieval(model, test_data, cfg)

    return model, cfg, results, ret


def full_eval(model, batches, cfg):
    vocab = cfg.d_input
    all_preds, all_targets = [], []
    all_top5, all_top10 = [], []
    loss_sum = tok_count = 0

    for batch in batches:
        x       = jax.nn.one_hot(batch[:, :-1], vocab)
        targets = jax.device_get(batch[:, 1:])
        logits  = jax.device_get(model(x, use_sigmoid=True, lambda_sharp=5.0, return_aux=False))

        preds = np.argmax(logits, axis=-1)
        all_preds.append(preds)
        all_targets.append(targets)

        top5  = np.argpartition(-logits, 5, axis=-1)[:, :, :5]
        top10 = np.argpartition(-logits, 10, axis=-1)[:, :, :10]
        all_top5.append(np.any(top5 == targets[:,:,None], axis=-1))
        all_top10.append(np.any(top10 == targets[:,:,None], axis=-1))

        lp = jax.device_get(jax.nn.log_softmax(jnp.array(logits), axis=-1))
        loss_sum += -float(np.sum(np.eye(vocab)[targets] * lp))
        tok_count += targets.size

    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    per_pos = (preds == targets).mean(axis=0) * 100

    return {
        "top1":    float((preds == targets).mean() * 100),
        "top5":    float(np.concatenate(all_top5).mean() * 100),
        "top10":   float(np.concatenate(all_top10).mean() * 100),
        "ppl":     float(np.exp(loss_sum / tok_count)),
        "per_pos": per_pos,
        "preds":   preds,
        "targets": targets,
    }


def check_retrieval(model, batches, cfg):
    vocab = cfg.d_input
    all_idx, all_alpha = [], []
    for batch in batches[:5]:
        x = jax.nn.one_hot(batch[:, :-1], vocab)
        _, aux = model(x, use_sigmoid=True, lambda_sharp=5.0, return_aux=True)
        all_idx.append(jax.device_get(aux["idx"]))
        all_alpha.append(jax.device_get(aux["alpha"]))

    # Flatten to (total_positions, k_max) — handles both (batch, k_max) and (batch, seq, k_max)
    idx_raw   = np.concatenate(all_idx, axis=0)
    alpha_raw = np.concatenate(all_alpha, axis=0)
    idx_2d   = idx_raw.reshape(-1, idx_raw.shape[-1])
    alpha_2d = alpha_raw.reshape(-1, alpha_raw.shape[-1])

    sets   = [frozenset(row) for row in idx_2d]
    unique = len(set(sets))
    total  = len(sets)

    rng = np.random.default_rng(0)
    jaccards = []
    for _ in range(500):
        i, j = rng.integers(0, total, size=2)
        if i != j:
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            jaccards.append(inter / union if union else 0)

    cos_vals = []
    sample = min(64, alpha_2d.shape[0])
    for i in range(sample):
        for j in range(i+1, sample):
            a, b = alpha_2d[i], alpha_2d[j]
            cos_vals.append(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    return {
        "jaccard":   float(np.mean(jaccards)),
        "unique":    unique,
        "total":     total,
        "alpha_cos": float(np.mean(cos_vals)) if cos_vals else 1.0,
    }


def print_results(task_name, res, ret, rand_baseline):
    print(f"\n  {task_name} Results:")
    print(f"  {'─'*50}")
    print(f"  Random baseline:       {rand_baseline:.2f}%")
    print(f"  Top-1 accuracy:        {res['top1']:.2f}%  ({res['top1']/rand_baseline:.1f}x random)")
    print(f"  Top-5 recall:          {res['top5']:.2f}%")
    print(f"  Top-10 recall:         {res['top10']:.2f}%")
    print(f"  Perplexity:            {res['ppl']:.2f}  (random={VOCAB})")
    print(f"  PPL improvement:       {(VOCAB-res['ppl'])/VOCAB*100:.1f}%")
    print(f"\n  Retrieval: Jaccard={ret['jaccard']:.3f}  unique={ret['unique']}/{ret['total']}  α_cos={ret['alpha_cos']:.3f}")
    dyn = "DYNAMIC ✓" if ret['jaccard'] < 0.3 else "PARTIAL ~" if ret['jaccard'] < 0.7 else "FIXED ✗"
    asm = "DYNAMIC" if ret['alpha_cos'] < 0.5 else "PARTIAL" if ret['alpha_cos'] < 0.9 else "FIXED"
    print(f"  Retrieval: {dyn}    Assembly: {asm}")

    print(f"\n  Sample predictions (ground-truth → predicted):")
    for ex in range(min(8, res['preds'].shape[0])):
        gt   = res['targets'][ex].tolist()
        pred = res['preds'][ex].tolist()
        ok   = sum(1 for g, p in zip(gt, pred) if g == p)
        marks = " ".join("✓" if g==p else "·" for g, p in zip(gt, pred))
        gt_s  = " ".join(f"{g:>2}" for g in gt)
        pr_s  = " ".join(f"{p:>2}" for p in pred)
        print(f"    [{gt_s}] → [{pr_s}]  {ok}/{len(gt)}")
        print(f"     {marks}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  DWA ACCURACY VERIFICATION (per-position retrieval)")
    print("  Testing on two tasks to isolate: learning vs retrieval dynamics")
    print("=" * 72)

    rand_baseline = 100.0 / VOCAB

    # ── Task 1: Copy task ────────────────────────────────────────────────────
    model1, cfg1, res1, ret1 = train_and_eval(
        "COPY TASK (repeating pattern)", synthetic_copy_task
    )
    print_results("Copy Task", res1, ret1, rand_baseline)

    # ── Task 2: Bigram task — shared transition table for train+test ─────────
    table = make_bigram_table(VOCAB, seed=42)
    bigram_fn = lambda v, b, s, seed: bigram_task(v, b, s, seed=seed, transition_table=table)
    model2, cfg2, res2, ret2 = train_and_eval(
        "BIGRAM TASK (deterministic A→B, shared table)", bigram_fn, eval_data_fn=bigram_fn
    )
    print_results("Bigram Task", res2, ret2, rand_baseline)

    # ── Per-position accuracy comparison ─────────────────────────────────────
    print(f"\n{'━'*72}")
    print(f"  PER-POSITION ACCURACY COMPARISON")
    print(f"{'━'*72}")
    print(f"  {'pos':>3}  {'copy':>8}  {'bigram':>8}")
    print(f"  {'-'*25}")
    for pos in range(len(res1['per_pos'])):
        c = res1['per_pos'][pos]
        b = res2['per_pos'][pos] if pos < len(res2['per_pos']) else 0
        print(f"  {pos:>3}  {c:>7.1f}%  {b:>7.1f}%")

    # ── Shift-1 baseline ──────────────────────────────────────────────────────
    print(f"\n{'━'*72}")
    print(f"  TRIVIAL BASELINE COMPARISON")
    print(f"{'━'*72}")
    for name, fn in [("Copy", synthetic_copy_task), ("Bigram", bigram_fn)]:
        batches = [next(fn(VOCAB, BATCH, SEQ, seed=SEED + i)) for i in range(10)]
        shift_ok = shift_total = 0
        for b in batches:
            t = np.array(b)
            shift_ok    += (t[:, 2:] == t[:, 1:-1]).sum()
            shift_total += t[:, 2:].size
        shift_acc = shift_ok / shift_total * 100
        model_acc = res1['top1'] if name == "Copy" else res2['top1']
        print(f"  {name}:  shift-1 baseline={shift_acc:.1f}%  model={model_acc:.1f}%  "
              f"{'beats ✓' if model_acc > shift_acc else 'fails ✗'}")

    # ── Final verdict ────────────────────────────────────────────────────────
    print(f"\n{'█'*72}")
    print(f"  VERDICT")
    print(f"{'█'*72}")

    copy_learning   = res1['top1'] > rand_baseline * 3
    bigram_learning = res2['top1'] > rand_baseline * 3
    copy_dynamic    = ret1['jaccard'] < 0.5
    bigram_dynamic  = ret2['jaccard'] < 0.5

    print(f"\n  Copy task:")
    print(f"    Learning:  {'YES ✓' if copy_learning else 'NO ✗'}  (acc={res1['top1']:.1f}%  ppl={res1['ppl']:.1f})")
    print(f"    Dynamic:   {'YES ✓' if copy_dynamic else 'NO ✗'}  (Jaccard={ret1['jaccard']:.3f})")

    print(f"\n  Bigram task:")
    print(f"    Learning:  {'YES ✓' if bigram_learning else 'NO ✗'}  (acc={res2['top1']:.1f}%  ppl={res2['ppl']:.1f})")
    print(f"    Dynamic:   {'YES ✓' if bigram_dynamic else 'NO ✗'}  (Jaccard={ret2['jaccard']:.3f})")

    if bigram_learning and bigram_dynamic:
        print(f"\n  ★ FULL SUCCESS — model learns AND retrieval is input-dependent.")
    elif bigram_learning:
        print(f"\n  ~ PARTIAL — model learns but retrieval still partially collapses.")
    elif copy_learning:
        print(f"\n  ~ Model learns on copy task but struggles with bigram.")
    else:
        print(f"\n  ✗ Model is not learning effectively.")


if __name__ == "__main__":
    main()
