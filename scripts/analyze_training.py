"""
DWA Model Comprehensive Training Analysis
==========================================
Trains from scratch for 3000 steps and evaluates:
  1. Is task loss actually decreasing vs random baseline?
  2. Is vector selection dynamic (different inputs → different top-k)?
  3. Are pool vectors learning (gradients flowing, movement from init)?
  4. Real output quality: top-1 accuracy on copy task
  5. Training capacity: convergence speed and best perplexity
"""

import os
import sys

# Silence JAX debug noise
os.environ.setdefault("JAX_DEBUG_NANS", "False")
os.environ.setdefault("JAX_LOG_COMPILES", "False")

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.small import DWAConfig
from src.model.dwa import DWAModel
from src.training.trainer import make_optimizer, make_train_step, get_phase_params
from src.data.loader import synthetic_copy_task

# ── Config ────────────────────────────────────────────────────────────────────
TOTAL_STEPS = 3000
LOG_EVERY   = 100
BATCH_SIZE  = 32
SEQ_LEN     = 16
VOCAB_SIZE  = 64
EVAL_BATCHES = 20
RANDOM_BASELINE_PPL = float(VOCAB_SIZE)   # log(64) nats ≈ 4.158 task loss
RANDOM_BASELINE_LOSS = float(np.log(VOCAB_SIZE))

print("=" * 70)
print("DWA Model Training Analysis")
print("=" * 70)
print(f"Config: D={2048}, d_A=d_B={64}, r={4}, N={512}, k_max={8}, S={2}")
print(f"Training: {TOTAL_STEPS} steps, batch={BATCH_SIZE}, seq={SEQ_LEN}, vocab={VOCAB_SIZE}")
print(f"Random baseline loss: {RANDOM_BASELINE_LOSS:.4f} nats  (ppl={RANDOM_BASELINE_PPL:.1f})")
print(f"JAX devices: {jax.devices()}")
print()

# ── Initialize model ──────────────────────────────────────────────────────────
config = DWAConfig()
rngs   = nnx.Rngs(params=0, dropout=1)
model  = DWAModel(config, rngs=rngs)
optimizer = make_optimizer(model, config)

# Split model and optimizer state for make_train_step
graphdef, state = nnx.split(model)
opt_graphdef, opt_state = nnx.split(optimizer)

# Create step functions for both phases
step_phase1 = make_train_step(config, use_sigmoid=False)
step_phase2 = make_train_step(config, use_sigmoid=True)

# Save initial pool vectors (numpy copy for movement tracking)
pool_init = np.array(jax.device_get(model.pool.vectors.value))  # (N, D)

# ── Data ──────────────────────────────────────────────────────────────────────
train_iter = synthetic_copy_task(VOCAB_SIZE, BATCH_SIZE, SEQ_LEN, seed=42)
test_iter  = synthetic_copy_task(VOCAB_SIZE, BATCH_SIZE, SEQ_LEN, seed=9999)

# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_accuracy(logits, targets):
    """top-1 accuracy: fraction of tokens where argmax == target."""
    preds = jnp.argmax(logits, axis=-1)
    return float(jnp.mean(preds == targets))


def measure_dynamic_selection(idx_batch):
    """
    idx_batch: (batch, k_max) numpy array of selected vector indices.
    Returns:
      n_unique: number of unique vector indices across all examples in batch
      mean_overlap: mean |idx_a ∩ idx_b| / k_max over random pairs
    """
    idx_np = np.array(idx_batch)
    batch_size, k_max = idx_np.shape

    # Count unique vectors used
    n_unique = int(len(np.unique(idx_np)))

    # Estimate mean pairwise overlap on up to 200 random pairs
    max_pairs = batch_size * (batch_size - 1) // 2
    n_pairs = min(200, max_pairs)
    rng = np.random.default_rng(0)
    if batch_size >= 2:
        all_pairs = [(a, b) for a in range(batch_size) for b in range(a + 1, batch_size)]
        chosen = rng.choice(len(all_pairs), size=n_pairs, replace=False)
        pairs = [all_pairs[i] for i in chosen]
    else:
        pairs = [(0, 0)]
    overlaps = []
    for a, b in pairs:
        set_a = set(idx_np[a].tolist())
        set_b = set(idx_np[b].tolist())
        overlap = len(set_a & set_b) / k_max
        overlaps.append(overlap)
    mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0

    return n_unique, mean_overlap


def pool_l2_movement(current_vectors_np):
    """Mean L2 distance each pool vector has moved from initialization."""
    delta = current_vectors_np - pool_init  # (N, D)
    return float(np.mean(np.linalg.norm(delta, axis=-1)))

# ── Training loop with checkpointing ─────────────────────────────────────────
print(f"{'Step':>6}  {'TaskLoss':>9}  {'PPL':>8}  {'Acc%':>6}  "
      f"{'UniqueVecs':>10}  {'Overlap':>8}  {'PoolMov':>8}  "
      f"{'Gamma':>7}  {'Lambda':>7}")
print("-" * 90)

log_records = []

for step, batch in zip(range(TOTAL_STEPS), train_iter):
    use_sigmoid, lambda_sharp = get_phase_params(step, config)
    step_fn = step_phase2 if use_sigmoid else step_phase1

    # Training step
    forced_idx = jnp.zeros((BATCH_SIZE, config.k_max), dtype=jnp.int32)
    metrics, state, opt_state = step_fn(
        graphdef, state, opt_graphdef, opt_state, batch,
        jnp.float32(lambda_sharp), jnp.float32(0.0), forced_idx
    )
    metrics = jax.device_get(metrics)

    if step % LOG_EVERY == 0:
        task_loss = float(metrics["task"])
        ppl       = float(np.exp(task_loss))

        # Extra forward pass to get aux (idx) for dynamic selection measurement
        x_input   = jax.nn.one_hot(batch[:, :-1], VOCAB_SIZE)
        logits, aux = model(
            x_input,
            use_sigmoid=use_sigmoid,
            lambda_sharp=lambda_sharp,
            return_aux=True,
        )
        logits_np = jax.device_get(logits)
        idx_np    = jax.device_get(aux["idx"])

        # top-1 accuracy on current batch
        targets_np = np.array(jax.device_get(batch[:, 1:]))
        acc        = compute_accuracy(jnp.array(logits_np), jnp.array(targets_np))

        # Dynamic selection metrics
        n_unique, mean_overlap = measure_dynamic_selection(idx_np)

        # Pool movement from init
        pool_now = np.array(jax.device_get(model.pool.vectors.value))
        movement = pool_l2_movement(pool_now)

        # Gamma (LoRA scale)
        gamma_val = float(jax.device_get(model.assembler.gamma.value))

        record = dict(
            step=step,
            task_loss=task_loss,
            ppl=ppl,
            acc=acc * 100,
            n_unique=n_unique,
            mean_overlap=mean_overlap,
            pool_movement=movement,
            gamma=gamma_val,
            lambda_sharp=lambda_sharp,
        )
        log_records.append(record)

        print(f"{step:>6}  {task_loss:>9.4f}  {ppl:>8.2f}  {acc*100:>5.1f}%  "
              f"{n_unique:>10}  {mean_overlap:>8.4f}  {movement:>8.6f}  "
              f"{gamma_val:>7.4f}  {lambda_sharp:>7.2f}")

print("-" * 90)
print()

# Restore model and optimizer state after training
nnx.update(model, state)
nnx.update(optimizer, opt_state)

# ── Final evaluation on held-out test data ────────────────────────────────────
print("=" * 70)
print("FINAL EVALUATION  (20 held-out test batches)")
print("=" * 70)

# Use final phase params
use_sigmoid_final, lambda_sharp_final = get_phase_params(TOTAL_STEPS - 1, config)

test_losses, test_accs = [], []
example_batches = []

for i, test_batch in zip(range(EVAL_BATCHES), test_iter):
    x_input = jax.nn.one_hot(test_batch[:, :-1], VOCAB_SIZE)
    logits, aux = model(
        x_input,
        use_sigmoid=use_sigmoid_final,
        lambda_sharp=lambda_sharp_final,
        return_aux=True,
    )
    logits_np  = jax.device_get(logits)
    targets_np = np.array(jax.device_get(test_batch[:, 1:]))

    # task loss
    log_probs  = jax.nn.log_softmax(jnp.array(logits_np), axis=-1)
    tgt_oh     = jax.nn.one_hot(jnp.array(targets_np), VOCAB_SIZE)
    t_loss     = float(-jnp.mean(jnp.sum(tgt_oh * log_probs, axis=-1)))
    test_losses.append(t_loss)

    acc = compute_accuracy(jnp.array(logits_np), jnp.array(targets_np))
    test_accs.append(acc)

    if i == 0:
        # Save first batch for examples
        example_logits  = logits_np
        example_targets = targets_np
        example_batch   = np.array(jax.device_get(test_batch))

test_loss = float(np.mean(test_losses))
test_ppl  = float(np.exp(test_loss))
test_acc  = float(np.mean(test_accs)) * 100

print(f"Test task loss:  {test_loss:.4f}")
print(f"Test perplexity: {test_ppl:.2f}  (random baseline = {RANDOM_BASELINE_PPL:.1f})")
print(f"Test accuracy:   {test_acc:.1f}%")
improvement_ratio = (RANDOM_BASELINE_PPL - test_ppl) / RANDOM_BASELINE_PPL * 100
print(f"Improvement over random: {improvement_ratio:.1f}%  "
      f"({'better' if improvement_ratio > 0 else 'WORSE than random'})")
print()

# ── 5 example predictions ─────────────────────────────────────────────────────
print("5 Example Predictions (first test batch, first 5 examples):")
print(f"  {'Ex':>3}  {'Input seq (first 8 tokens)':>28}  "
      f"{'Target':>8}  {'Predicted':>10}  {'Correct':>7}")
print("-" * 70)
for ex in range(5):
    input_seq  = example_batch[ex, :-1][:8].tolist()
    target_seq = example_targets[ex, :8].tolist()
    pred_seq   = np.argmax(example_logits[ex, :8], axis=-1).tolist()
    n_correct  = sum(t == p for t, p in zip(target_seq, pred_seq))
    print(f"  {ex:>3}  {str(input_seq):>28}  "
          f"{str(target_seq[:4]):>8}  {str(pred_seq[:4]):>10}  {n_correct}/8")
print()

# ── Pool utilization ──────────────────────────────────────────────────────────
print("Pool Utilization:")
ema_usage = np.array(jax.device_get(model.pool.ema_usage.value))  # (N,)
n_active  = int(np.sum(ema_usage > 0.001))
pct_active = n_active / config.N * 100
print(f"  Vectors with EMA usage > 0.001: {n_active} / {config.N}  ({pct_active:.1f}%)")
print(f"  Max EMA usage:  {float(np.max(ema_usage)):.6f}")
print(f"  Mean EMA usage: {float(np.mean(ema_usage)):.6f}")
print(f"  Std EMA usage:  {float(np.std(ema_usage)):.6f}")
print()

# ── Pool movement from initialization ────────────────────────────────────────
pool_final = np.array(jax.device_get(model.pool.vectors.value))
delta      = pool_final - pool_init
per_vec_l2 = np.linalg.norm(delta, axis=-1)  # (N,)
print("Pool Vector Movement from Initialization:")
print(f"  Mean L2 movement:   {float(np.mean(per_vec_l2)):.6f}")
print(f"  Max L2 movement:    {float(np.max(per_vec_l2)):.6f}")
print(f"  Min L2 movement:    {float(np.min(per_vec_l2)):.6f}")
print(f"  Std L2 movement:    {float(np.std(per_vec_l2)):.6f}")
print(f"  Vectors moved > 0.01: {int(np.sum(per_vec_l2 > 0.01))} / {config.N}")
print()

# ── Dynamic selection final snapshot ─────────────────────────────────────────
print("Dynamic Selection (final training checkpoint):")
last = log_records[-1]
print(f"  Unique vectors used in last batch: {last['n_unique']} / {config.N}")
print(f"  Mean pairwise overlap:             {last['mean_overlap']:.4f}  "
      f"({'dynamic (low overlap)' if last['mean_overlap'] < 0.5 else 'fixed (high overlap)'})")
print()

# ── Final gamma and lambda ────────────────────────────────────────────────────
gamma_final = float(jax.device_get(model.assembler.gamma.value))
print(f"Final gamma (LoRA scale): {gamma_final:.6f}")
print(f"Final lambda_sharp:       {lambda_sharp_final:.2f}")
print()

# ── Summary table ─────────────────────────────────────────────────────────────
print("=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

# Best and final metrics from training log
if log_records:
    best_rec   = min(log_records, key=lambda r: r["task_loss"])
    first_rec  = log_records[0]
    final_rec  = log_records[-1]

    print(f"{'Metric':<38}  {'Value':>12}")
    print("-" * 54)
    print(f"{'Random baseline loss (log 64)':<38}  {RANDOM_BASELINE_LOSS:>12.4f}")
    print(f"{'Random baseline perplexity':<38}  {RANDOM_BASELINE_PPL:>12.1f}")
    print(f"{'Initial task loss (step 0)':<38}  {first_rec['task_loss']:>12.4f}")
    print(f"{'Initial perplexity':<38}  {first_rec['ppl']:>12.2f}")
    print(f"{'Best task loss (training)':<38}  {best_rec['task_loss']:>12.4f}")
    print(f"{'Best perplexity (training)':<38}  {best_rec['ppl']:>12.2f}")
    print(f"{'Best step':<38}  {best_rec['step']:>12}")
    print(f"{'Final task loss (step ~{final_rec[\"step\"]})':<38}  {final_rec['task_loss']:>12.4f}")
    print(f"{'Final training perplexity':<38}  {final_rec['ppl']:>12.2f}")
    print(f"{'Test task loss (held-out)':<38}  {test_loss:>12.4f}")
    print(f"{'Test perplexity (held-out)':<38}  {test_ppl:>12.2f}")
    print(f"{'Test accuracy':<38}  {test_acc:>11.1f}%")
    print(f"{'Improvement over random (%)':<38}  {improvement_ratio:>11.1f}%")
    print(f"{'Unique vectors last batch':<38}  {last['n_unique']:>9} / 512")
    print(f"{'Mean pairwise overlap (last)':<38}  {last['mean_overlap']:>12.4f}")
    print(f"{'Pool mean L2 movement':<38}  {float(np.mean(per_vec_l2)):>12.6f}")
    print(f"{'Pool vectors moved >0.01':<38}  {int(np.sum(per_vec_l2 > 0.01)):>9} / 512")
    print(f"{'Pool active vectors (EMA>0.001)':<38}  {n_active:>9} / 512")
    print(f"{'Final gamma':<38}  {gamma_final:>12.6f}")
    print(f"{'Final lambda_sharp':<38}  {lambda_sharp_final:>12.2f}")
    print("-" * 54)

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

# 1. Is the model learning?
loss_drop = first_rec['task_loss'] - final_rec['task_loss']
beats_random = test_loss < RANDOM_BASELINE_LOSS
print(f"1. IS THE MODEL LEARNING?")
print(f"   Loss drop from init to final: {loss_drop:+.4f}")
print(f"   Test loss ({test_loss:.4f}) vs random ({RANDOM_BASELINE_LOSS:.4f}): "
      f"{'BETTER - model IS learning' if beats_random else 'WORSE or equal - model NOT learning'}")
print()

# 2. Dynamic selection?
is_dynamic = last['n_unique'] > config.k_max * 4 and last['mean_overlap'] < 0.5
print(f"2. IS SELECTION DYNAMIC?")
print(f"   Unique vectors used: {last['n_unique']}/512 "
      f"({'good diversity' if last['n_unique'] > 50 else 'low diversity - selection may be fixed'})")
print(f"   Mean pairwise overlap: {last['mean_overlap']:.4f} "
      f"({'LOW = dynamic' if last['mean_overlap'] < 0.5 else 'HIGH = fixed/collapsed'})")
print(f"   Verdict: {'DYNAMIC selection' if is_dynamic else 'FIXED / collapsed selection'}")
print()

# 3. Pool vectors learning?
vecs_moving = int(np.sum(per_vec_l2 > 0.01))
print(f"3. ARE POOL VECTORS LEARNING?")
print(f"   Vectors that moved >0.01 L2: {vecs_moving}/512")
print(f"   Mean movement: {float(np.mean(per_vec_l2)):.6f}")
if float(np.mean(per_vec_l2)) > 0.001:
    print(f"   Verdict: YES - gradients flowing, vectors moving from init")
else:
    print(f"   Verdict: MINIMAL movement - gradients may not be reaching pool")
print()

# 4. Output quality?
print(f"4. OUTPUT QUALITY (copy task top-1 accuracy)?")
print(f"   Top-1 accuracy: {test_acc:.1f}%")
print(f"   Random chance:  {100.0/VOCAB_SIZE:.1f}%  (1/{VOCAB_SIZE})")
if test_acc > 100.0 / VOCAB_SIZE * 2:
    print(f"   Verdict: ABOVE chance - model has learned the copy pattern")
else:
    print(f"   Verdict: NEAR CHANCE - model has not learned the copy pattern")
print()

# 5. Convergence?
print(f"5. TRAINING CAPACITY & CONVERGENCE?")
print(f"   Best perplexity reached: {best_rec['ppl']:.2f} at step {best_rec['step']}")
print(f"   Random perplexity: {RANDOM_BASELINE_PPL:.1f}")
print(f"   Perplexity improvement: {improvement_ratio:.1f}%")
if improvement_ratio > 20:
    print(f"   Verdict: STRONG learning signal - model is converging well")
elif improvement_ratio > 5:
    print(f"   Verdict: MODERATE learning - model is improving but slowly")
elif improvement_ratio > 0:
    print(f"   Verdict: WEAK but positive learning")
else:
    print(f"   Verdict: NO learning or diverging")
print()
print("=" * 70)
