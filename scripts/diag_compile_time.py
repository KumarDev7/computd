"""
Measure the HLO graph size (operation count) of each DWA component to find what's
exploding compilation time. Each component is forward-pass only, no grad.

Root cause hypothesis: The 12x unrolled DWABlock with hybrid retrieval creates
a massive HLO graph. Each block does:
  - hybrid_forward: (S, batch, seq, N) similarity GEMM + per-position top_k
  - assembly: gather + low-rank decompose + einsum
With 12 blocks + jax.remat, the backward graph doubles the operation count.

Possible fixes:
  1. Scan over blocks instead of Python for-loop (jax.lax.scan)
  2. Reduce block count for faster compile-test
  3. Use lax.cond for phase switching instead of separate JIT functions
  4. Check if remat is re-tracing (not just re-computing)
"""
import os, sys, time
import numpy as np

os.environ['JAX_ENABLE_X64'] = 'False'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['JAX_DEBUG_NANS'] = 'False'
os.environ['JAX_LOG_COMPILES'] = 'False'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from flax import nnx

from configs.large_1b import get_1b_config
from src.model.dwa import DWAModel
from src.training.sharding import make_mesh, shard_model

BATCH = 8
SEQ = 1024


def measure_hlo(fn_name, fn, *args, **kwargs):
    """Trace a function and count HLO operations."""
    lowered = fn.lower(*args, **kwargs)
    hlo_text = lowered.as_text()
    op_count = hlo_text.count(' = ')
    total_lines = len(hlo_text.split('\n'))
    return op_count, total_lines, len(hlo_text)


def main():
    print(f'[env] platform={jax.devices()[0].platform}  devices={len(jax.devices())}')

    cfg = get_1b_config()
    cfg.max_seq_len = SEQ
    mesh = make_mesh()

    rngs = nnx.Rngs(params=0, dropout=1)
    model = DWAModel(cfg, rngs=rngs, mesh=mesh)
    with mesh:
        shard_model(model, mesh)
    print(f'[model] {sum(x.size for x in jax.tree.leaves(nnx.split(model)[1].filter(nnx.Param))):,} params')

    dummy_tokens = jnp.ones((BATCH, SEQ + 1), dtype=jnp.int32)

    # ===== FORWARD-ONLY (no grad) =====
    print('\n' + '='*70)
    print('FORWARD PASS ONLY (no gradient) — measuring HLO graph size')
    print('='*70)

    @nnx.jit
    def forward_only(model, batch, use_sigmoid, lambda_sharp, lambda_entropy_eff, forced_idx, soft, hybrid, token_ids):
        logits, aux = model(
            None, use_sigmoid=use_sigmoid, lambda_sharp=lambda_sharp,
            forced_idx=forced_idx, return_aux=True, soft=soft, hybrid=hybrid,
            token_ids=token_ids,
        )
        return logits

    forced_idx = jnp.zeros((BATCH, cfg.k_max), dtype=jnp.int32)

    print('\n[1] Compiling forward-only (phase1: use_sigmoid=False, hybrid=True)...')
    t0 = time.time()

    print('\n[1] Compiling forward-only (phase1: use_sigmoid=False, hybrid=True)...')
    t0 = time.time()
    try:
        result = forward_only(model, dummy_tokens, False, 1.0, 0.0, forced_idx, False, True, dummy_tokens)
        result.block_until_ready()
        t1 = time.time()
        print(f'  Forward-only (phase1) compiled+executed in {t1-t0:.1f}s')
    except Exception as e:
        print(f'  Forward-only (phase1) FAILED: {e}')

    # ===== FORWARD + BACKWARD (full grad) =====
    print('\n' + '='*70)
    print('FORWARD + BACKWARD (with gradient) — measuring HLO graph size')
    print('='*70)

    from src.training.trainer import make_optimizer, loss_fn
    optimizer = make_optimizer(model, cfg)
    from src.training.sharding import shard_optimizer
    shard_optimizer(optimizer, model, mesh)

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
        return total_loss

    print('\n[2] Compiling train_step (phase1: backward+optimizer update)...')
    t0 = time.time()
    try:
        loss = train_step_p1(model, optimizer, dummy_tokens, 1.0, 0.0, forced_idx)
        loss.block_until_ready()
        t1 = time.time()
        print(f'  train_step (phase1) compiled+executed in {t1-t0:.1f}s')
    except Exception as e:
        print(f'  train_step (phase1) FAILED: {type(e).__name__}: {e}')

    # ===== SECOND STEP (cached JIT) =====
    print('\n[3] Second step (cached JIT, should be fast)...')
    forced_idx2 = jnp.zeros((BATCH, cfg.k_max), dtype=jnp.int32)
    t0 = time.time()
    try:
        loss2 = train_step_p1(model, optimizer, dummy_tokens, 1.0, 0.0, forced_idx2)
        loss2.block_until_ready()
        t1 = time.time()
        print(f'  Step 2 (cached) executed in {t1-t0:.3f}s')
    except Exception as e:
        print(f'  Step 2 FAILED: {type(e).__name__}: {e}')

    # ===== KEY METRICS =====
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    print(f'  Model: 1B params, N={cfg.N}, d_A={cfg.d_A}, d_B={cfg.d_B}, D={cfg.D}')
    print(f'  Blocks: {cfg.n_assembly_layers}, k_max={cfg.k_max}, S={cfg.S}, d_k={cfg.d_k}')
    print(f'  Batch: {BATCH}, Seq: {SEQ}')
    print(f'  Hybrid mode: {cfg.hybrid_train}')
    print(f'  Key dimension for compilation: each DWABlock has hybrid_forward')
    print(f'    - similarity GEMM: (S={cfg.S}, batch={BATCH}, seq={SEQ}, N={cfg.N})')
    print(f'    - per-position top_k over N={cfg.N} entries')
    print(f'    - 12 blocks unrolled in Python = 12x the HLO ops')
    print(f'    - jax.remat doubles: forward saved, recomputed in backward')


if __name__ == '__main__':
    main()