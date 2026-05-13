"""Test jax.remat + nnx.value_and_grad interaction with DWA blocks.

This is the exact path used in the training loop:
  for block in self.blocks:
      def _run_block(h, _b=block):
          return _b(h, pool_vecs, ...)
      h, alpha, idx, sims, alpha_raw = jax.remat(_run_block)(h)
"""
import os
os.environ.setdefault('JAX_ENABLE_X64', 'False')

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from configs.small import DWAConfig
from src.model.dwa import DWAModel
from src.training.trainer import loss_fn, make_optimizer

cfg = DWAConfig(
    d_input=64, d_A=64, d_B=64, D=2048, r=4,
    N=128, k_max=4, S=2, d_k=16,
    n_heads=16, max_seq_len=32, n_assembly_layers=2,
    hybrid_train=True, use_embedding=False,
)

def test_remat_forward():
    rngs = nnx.Rngs(params=0)
    model = DWAModel(cfg, rngs=rngs)
    B, T = 2, 8
    x = jax.random.normal(jax.random.PRNGKey(1), (B, T, cfg.d_input))

    print("Test 1: Forward with jax.remat (n_heads=16, hybrid=True)...")
    try:
        logits, aux = model(x, use_sigmoid=True, lambda_sharp=1.0, return_aux=True, hybrid=True)
        print(f"  OK! logits={logits.shape} alpha={aux['alpha'].shape} idx={aux['idx'].shape}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return

    print("\nTest 2: nnx.value_and_grad + loss_fn + remat (2 blocks, n_heads=16)...")
    try:
        batch = jax.random.randint(jax.random.PRNGKey(2), (B, T + 1), 0, cfg.d_input)
        forced_idx = jnp.zeros((B, cfg.k_max), dtype=jnp.int32)
        grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)
        result = grad_fn(model, batch, True, 1.0, 0.02, forced_idx, False, True, None)
        (total_loss, aux_data), grads = result
        metrics, aux = aux_data
        print(f"  OK! loss={float(total_loss):.4f}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()

    print("\nTest 3: Full train_step (2 blocks, n_heads=16, hybrid)...")
    try:
        optimizer = make_optimizer(model, cfg)
        graphdef, state = nnx.split(model)
        opt_graphdef, opt_state = nnx.split(optimizer)

        use_embed = getattr(cfg, 'use_embedding', False)
        soft = getattr(cfg, 'soft_train', False)
        hybrid = getattr(cfg, 'hybrid_train', False)

        @jax.jit
        def _train_step(graphdef, state, opt_graphdef, opt_state, batch,
                       lambda_sharp, lambda_entropy_eff, forced_idx):
            model = nnx.merge(graphdef, state)
            optimizer = nnx.merge(opt_graphdef, opt_state)
            token_ids = batch if use_embed else None

            grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)
            (total_loss, (metrics, aux)), grads = grad_fn(
                model, batch, True, lambda_sharp, lambda_entropy_eff, forced_idx, soft, hybrid, token_ids
            )
            optimizer.update(model, grads)

            N = model.pool.N
            alpha_all = aux.get('alpha_all', [aux['alpha']])
            idx_all = aux.get('idx_all', [aux['idx']])

            all_alpha = jnp.stack(alpha_all)
            all_idx = jnp.stack(idx_all)
            n_blocks = len(alpha_all)
            batch_size = batch.shape[0]

            if idx_all[0] is None:
                alpha_sum = all_alpha.mean(axis=(0, 1, 2))
            elif idx_all[0].ndim == 3 and idx_all[0].shape[1] > 1:
                alpha_sum = jnp.zeros(N, dtype=jnp.float32).at[all_idx.reshape(-1)].add(
                    all_alpha.reshape(-1) / all_idx.size
                )
            else:
                if all_idx.ndim == 3:
                    all_alpha_summed = all_alpha.sum(axis=2)
                    denom = batch_size * n_blocks * alpha_all[0].shape[1]
                else:
                    all_alpha_summed = all_alpha
                    denom = batch_size * n_blocks
                alpha_sum = jnp.zeros(N, dtype=jnp.float32).at[all_idx.reshape(-1)].add(
                    all_alpha_summed.reshape(-1) / denom
                )

            model.pool.update_ema(alpha_sum, cfg.beta_ema)
            metrics['loss'] = total_loss

            _, new_state = nnx.split(model)
            _, new_opt_state = nnx.split(optimizer)
            return metrics, new_state, new_opt_state

        m, state, opt_state = _train_step(
            graphdef, state, opt_graphdef, opt_state, batch,
            jnp.float32(5.0), jnp.float32(0.02), jnp.zeros((B, cfg.k_max), dtype=jnp.int32)
        )
        m = jax.device_get(m)
        print(f"  OK! loss={float(m['loss']):.4f}")

        # Second step (phase1, hard mode)
        print("\nTest 4: Phase 1 hard mode step (use_sigmoid=False)...")
        @jax.jit
        def _train_step_p1(graphdef, state, opt_graphdef, opt_state, batch,
                          lambda_sharp, lambda_entropy_eff, forced_idx):
            model = nnx.merge(graphdef, state)
            optimizer = nnx.merge(opt_graphdef, opt_state)
            token_ids = batch if use_embed else None

            grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)
            (total_loss, (metrics, aux)), grads = grad_fn(
                model, batch, False, lambda_sharp, lambda_entropy_eff, forced_idx, soft, hybrid, token_ids
            )
            optimizer.update(model, grads)

            N = model.pool.N
            alpha_all = aux.get('alpha_all', [aux['alpha']])
            idx_all = aux.get('idx_all', [aux['idx']])
            all_alpha = jnp.stack(alpha_all)
            all_idx = jnp.stack(idx_all)

            # Phase 1 hard mode: idx_all[0] is (batch, k_max) — ndim==2
            batch_size = batch.shape[0]
            n_blocks = len(alpha_all)

            if idx_all[0] is None:
                alpha_sum = all_alpha.mean(axis=(0, 1, 2))
            elif idx_all[0].ndim == 3 and idx_all[0].shape[1] > 1:
                alpha_sum = jnp.zeros(N, dtype=jnp.float32).at[all_idx.reshape(-1)].add(
                    all_alpha.reshape(-1) / all_idx.size
                )
            else:
                if all_idx.ndim == 3:
                    all_alpha_summed = all_alpha.sum(axis=2)
                    denom = batch_size * n_blocks * alpha_all[0].shape[1]
                else:
                    all_alpha_summed = all_alpha
                    denom = batch_size * n_blocks
                alpha_sum = jnp.zeros(N, dtype=jnp.float32).at[all_idx.reshape(-1)].add(
                    all_alpha_summed.reshape(-1) / denom
                )

            model.pool.update_ema(alpha_sum, cfg.beta_ema)
            metrics['loss'] = total_loss

            _, new_state = nnx.split(model)
            _, new_opt_state = nnx.split(optimizer)
            return metrics, new_state, new_opt_state

        m2, state, opt_state = _train_step_p1(
            graphdef, state, opt_graphdef, opt_state, batch,
            jnp.float32(1.0), jnp.float32(0.0), jnp.zeros((B, cfg.k_max), dtype=jnp.int32)
        )
        m2 = jax.device_get(m2)
        print(f"  OK! loss={float(m2['loss']):.4f}")

    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()


if __name__ == '__main__':
    test_remat_forward()