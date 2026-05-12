"""
Diagnostic: JIT-compile each DWA component independently to find the bottleneck.

Tests:
  1. PartA (embedding + projection)
  2. CausalSelfAttention
  3. MultiAspectRetrieval (seq-level, phase1 forced_idx)
  4. MultiAspectRetrieval (hybrid_forward)
  5. WeightAssembler (hard mode, seq-level idx)
  6. WeightAssembler (hybrid mode, per-position idx)
  7. DWABlock (full block, hybrid)
  8. DWAModel full forward
  9. Full train_step (loss + grad + optimizer update)
"""
import os, sys, time
import numpy as np

os.environ.setdefault('JAX_ENABLE_X64', 'False')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'true')
os.environ.setdefault('JAX_DEBUG_NANS', 'False')
os.environ.setdefault('JAX_LOG_COMPILES', 'True')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from flax import nnx

from configs.large_1b import get_1b_config
from src.model.dwa import DWAModel
from src.training.sharding import make_mesh, shard_model, shard_optimizer
from src.training.trainer import make_optimizer, loss_fn, compute_aux_losses

BATCH = 8
SEQ = 1024

def main():
    print(f'[env] platform={jax.devices()[0].platform}  devices={len(jax.devices())}')
    cfg = get_1b_config()
    cfg.max_seq_len = SEQ
    mesh = make_mesh()
    print(f'[mesh] {mesh}')

    rngs = nnx.Rngs(params=0, dropout=1)
    print('[init] creating DWAModel...')
    model = DWAModel(cfg, rngs=rngs, mesh=mesh)

    with mesh:
        shard_model(model, mesh)
    optimizer = make_optimizer(model, cfg)
    shard_optimizer(optimizer, model, mesh)
    print('[init] model+optimizer ready')

    dummy_tokens = jnp.ones((BATCH, SEQ + 1), dtype=jnp.int32)

    # ===== TEST: full train_step (phase 1 - forced_idx) =====
    print('\n' + '='*60)
    print('[test] FULL train_step (phase1, hybrid=True, forced_idx)')
    print('='*60)

    from src.training.trainer import get_phase_params, Phase1Rotator
    rot = Phase1Rotator(cfg.N, BATCH, cfg.k_max, seed=0)
    forced_idx = rot.next()

    @nnx.jit
    def train_step_p1(model, optimizer, batch, lambda_sharp, lambda_entropy_eff, forced_idx):
        use_sigmoid = False
        soft = False
        hybrid = True
        token_ids = batch
        grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)
        (total_loss, (metrics, aux)), grads = grad_fn(
            model, batch, use_sigmoid, lambda_sharp, lambda_entropy_eff, forced_idx, soft, hybrid, token_ids
        )
        optimizer.update(model, grads)
        return total_loss, metrics

    t0 = time.time()
    print(f'  [jit] compiling train_step_p1...')
    loss, metrics = train_step_p1(model, optimizer, dummy_tokens, 1.0, 0.0, forced_idx)
    loss.block_until_ready()
    t1 = time.time()
    print(f'  [jit] train_step_p1 compiled+executed in {t1-t0:.1f}s  loss={float(loss):.4f}')

    # ===== TEST: phase2 train_step (sigmoid, hybrid) =====
    print('\n' + '='*60)
    print('[test] FULL train_step (phase2, hybrid=True, sigmoid)')
    print('='*60)

    forced_idx_dummy = jnp.zeros((BATCH, cfg.k_max), dtype=jnp.int32)

    @nnx.jit
    def train_step_p2(model, optimizer, batch, lambda_sharp, lambda_entropy_eff, forced_idx):
        use_sigmoid = True
        soft = False
        hybrid = True
        token_ids = batch
        grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)
        (total_loss, (metrics, aux)), grads = grad_fn(
            model, batch, use_sigmoid, lambda_sharp, lambda_entropy_eff, forced_idx, soft, hybrid, token_ids
        )
        optimizer.update(model, grads)
        return total_loss, metrics

    t0 = time.time()
    print(f'  [jit] compiling train_step_p2...')
    loss2, metrics2 = train_step_p2(model, optimizer, dummy_tokens, 5.0, 0.02, forced_idx_dummy)
    loss2.block_until_ready()
    t1 = time.time()
    print(f'  [jit] train_step_p2 compiled+executed in {t1-t0:.1f}s  loss={float(loss2):.4f}')

    # ===== TEST: step 2 runtime (already compiled) =====
    print('\n' + '='*60)
    print('[test] Step 2 (already-compiled phase1)')
    print('='*60)

    forced_idx2 = rot.next()
    t0 = time.time()
    loss3 = train_step_p1(model, optimizer, dummy_tokens, 1.0, 0.0, forced_idx2)
    loss3.block_until_ready()
    t1 = time.time()
    print(f'  [exec] step 2 (cached JIT) in {t1-t0:.3f}s  loss={float(loss3):.4f}')

    print('\n[done] all diagnostics passed')


if __name__ == '__main__':
    main()