"""
Diagnostic: Identify exactly what causes the 3-step recompilation and 30+ min compile.

Strategy:
  1. Use make_train_step (from train_tpu_1b.py) which uses @jax.jit + split/merge
  2. Enable JAX_LOG_COMPILES=True to see every recompilation
  3. Add jax.debug.print inside train_step to trace which code paths execute
  4. Test step 0, 1, 2, 3 to see if any trigger recompilation
  5. Specifically test the phase boundary (step 499→500)
  
  Also test with @nnx.jit vs @jax.jit to compare.
"""
import os, sys, time
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
from src.training.trainer import make_optimizer, loss_fn, compute_aux_losses
from src.training.trainer import get_phase_params, Phase1Rotator
from src.training.sharding import make_mesh, shard_model, shard_optimizer

BATCH = 8
SEQ = 1024


def make_train_step_debug(cfg, use_sigmoid: bool):
    """Same as train_tpu_1b.py make_train_step but with jax.debug.print tracing."""
    use_embed = getattr(cfg, 'use_embedding', False)
    soft = getattr(cfg, 'soft_train', False)
    hybrid = getattr(cfg, 'hybrid_train', False)

    @jax.jit
    def train_step(graphdef, state, opt_graphdef, opt_state, batch,
                   lambda_sharp, lambda_entropy_eff, forced_idx):
        model = nnx.merge(graphdef, state)
        optimizer = nnx.merge(opt_graphdef, opt_state)
        token_ids = batch if use_embed else None

        # DEBUG: trace which path we take
        jax.debug.print("DEBUG train_step: use_sigmoid={} soft={} hybrid={}", 
                        use_sigmoid, soft, hybrid)

        grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)
        (total_loss, (metrics, aux)), grads = grad_fn(
            model, batch, use_sigmoid, lambda_sharp, lambda_entropy_eff, forced_idx, soft, hybrid, token_ids
        )
        optimizer.update(model, grads)

        N = model.pool.N
        alpha_all = aux.get('alpha_all', [aux['alpha']])
        idx_all = aux.get('idx_all', [aux['idx']])
        n_blocks = len(alpha_all)

        # DEBUG: trace alpha/idx shapes
        jax.debug.print("DEBUG: n_blocks={} alpha[0].shape={} idx[0].ndim={}", 
                        n_blocks, alpha_all[0].shape, idx_all[0].ndim)

        if idx_all[0] is None:
            jax.debug.print("DEBUG: EMA path=SOFT (idx is None)")
            all_alpha = jnp.stack(alpha_all)
            alpha_sum = all_alpha.mean(axis=(0, 1, 2))
        elif idx_all[0].ndim == 3 and idx_all[0].shape[1] > 1:
            jax.debug.print("DEBUG: EMA path=HYBRID (idx.ndim=3, shape[1]>1)")
            all_idx = jnp.stack(idx_all)
            all_alpha = jnp.stack(alpha_all)
            alpha_sum = jnp.zeros(N).at[all_idx.reshape(-1)].add(
                all_alpha.reshape(-1) / (all_idx.size)
            )
        else:
            jax.debug.print("DEBUG: EMA path=HARD (else)")
            all_idx = jnp.stack(idx_all)
            all_alpha = jnp.stack(alpha_all)
            if all_idx.ndim == 3:
                all_alpha = all_alpha.sum(axis=2)
                denom = batch.shape[0] * n_blocks * alpha_all[0].shape[1]
            else:
                denom = batch.shape[0] * n_blocks
            alpha_sum = jnp.zeros(N).at[all_idx.reshape(-1)].add(
                all_alpha.reshape(-1) / denom
            )

        model.pool.update_ema(alpha_sum, cfg.beta_ema)
        metrics['loss'] = total_loss
        _, new_state = nnx.split(model)
        _, new_opt_state = nnx.split(optimizer)
        return metrics, new_state, new_opt_state

    return train_step


def main():
    print(f'[env] platform={jax.devices()[0].platform}  devices={len(jax.devices())}')
    print(f'[env] jax={jax.__version__}')

    cfg = get_1b_config()
    cfg.max_seq_len = SEQ
    mesh = make_mesh()

    rngs = nnx.Rngs(params=0, dropout=1)
    print('[model] initializing...')
    model = DWAModel(cfg, rngs=rngs, mesh=mesh)

    n_params = sum(x.size for x in jax.tree.leaves(nnx.split(model)[1].filter(nnx.Param)))
    print(f'[model] params: {n_params:,}')

    with mesh:
        shard_model(model, mesh)
    print('[sharding] done')

    optimizer = make_optimizer(model, cfg)
    shard_optimizer(optimizer, model, mesh)
    print('[optimizer] sharded')

    print()
    print('='*70)
    print('TEST: @jax.jit with split/merge (train_tpu_1b pattern)')
    print('='*70)

    step_fn = make_train_step_debug(cfg, use_sigmoid=False)
    graphdef, state = nnx.split(model)
    opt_graphdef, opt_state = nnx.split(optimizer)

    # Step 0 — first compilation
    rot = Phase1Rotator(cfg.N, BATCH, cfg.k_max, seed=0)
    forced_idx = rot.next()
    dummy_batch = jnp.ones((BATCH, SEQ + 1), dtype=jnp.int32)

    print('\n--- Step 0 (phase1, use_sigmoid=False) ---')
    t0 = time.time()
    metrics, state, opt_state = step_fn(
        graphdef, state, opt_graphdef, opt_state, dummy_batch,
        1.0, 0.0, forced_idx
    )
    # Block until computation is done
    _ = jax.device_get(metrics)
    t1 = time.time()
    print(f'  Step 0: {t1-t0:.1f}s')

    # Steps 1-3 — should be cached
    for i in range(1, 4):
        t0 = time.time()
        forced_idx = rot.next()
        metrics, state, opt_state = step_fn(
            graphdef, state, opt_graphdef, opt_state, dummy_batch,
            1.0, 0.0, forced_idx
        )
        _ = jax.device_get(metrics)
        t1 = time.time()
        print(f'  Step {i}: {t1-t0:.3f}s')

    print()
    print('='*70)
    print('TEST: Phase boundary — Phase2 (use_sigmoid=True)')
    print('='*70)

    step_fn_p2 = make_train_step_debug(cfg, use_sigmoid=True)

    print('\n--- Phase2 Step 0 (use_sigmoid=True) ---')
    t0 = time.time()
    forced_idx_dummy = jnp.zeros((BATCH, cfg.k_max), dtype=jnp.int32)
    metrics, state, opt_state = step_fn_p2(
        graphdef, state, opt_graphdef, opt_state, dummy_batch,
        5.0, 0.02, forced_idx_dummy
    )
    _ = jax.device_get(metrics)
    t1 = time.time()
    print(f'  Phase2 Step 0: {t1-t0:.1f}s')

    print('\n[done]')


if __name__ == '__main__':
    main()