"""Test with 1B-scale config (single device, no mesh) to reproduce shape bug.

Uses 1B dimensions but tiny N and seq to fit in memory on a single device.
"""
import os
os.environ.setdefault('JAX_ENABLE_X64', 'False')
os.environ.setdefault('XLA_FLAGS', '')

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from configs.small import DWAConfig
from src.model.dwa import DWAModel
from src.model.retrieval import MultiAspectRetrieval
from src.training.trainer import loss_fn

cfg_1b_scale = DWAConfig(
    d_input=64, d_A=1280, d_B=1280, D=1280*8*2 + 1280, r=8,
    N=512, k_max=16, S=8, d_k=64,
    n_heads=16, max_seq_len=32, n_assembly_layers=2,
    hybrid_train=True, use_embedding=False,
    lambda_entropy=0.02, phase1_end=100, phase2_end=1000,
)

def test_1b_scale_grad():
    rngs = nnx.Rngs(params=0)
    model = DWAModel(cfg_1b_scale, rngs=rngs)

    B, T = 2, 8
    x = jax.random.normal(jax.random.PRNGKey(1), (B, T, cfg_1b_scale.d_input))
    
    print("Test 1: Forward (hybrid=True, use_sigmoid=True)...")
    try:
        logits, aux = model(
            x, use_sigmoid=True, lambda_sharp=1.0, return_aux=True,
            soft=False, hybrid=True,
        )
        print(f"  OK! logits={logits.shape} alpha={aux['alpha'].shape} idx={aux['idx'].shape}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return

    print("\nTest 2: loss_fn + nnx.value_and_grad (hybrid=True)...")
    try:
        batch = jax.random.randint(jax.random.PRNGKey(2), (B, T + 1), 0, cfg_1b_scale.d_input)
        forced_idx = jnp.zeros((B, cfg_1b_scale.k_max), dtype=jnp.int32)
        grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)
        result = grad_fn(model, batch, True, 1.0, 0.02, forced_idx, False, True, None)
        (total_loss, aux_data), grads = result
        metrics, aux = aux_data
        print(f"  OK! loss={float(total_loss):.4f}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()

    print("\nTest 3: loss_fn + nnx.value_and_grad (phase1, hard mode)...")
    try:
        result = grad_fn(model, batch, False, 1.0, 0.0, forced_idx, False, False, None)
        (total_loss, aux_data), grads = result
        metrics, aux = aux_data
        print(f"  OK! loss={float(total_loss):.4f}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()

    print("\nTest 4: train_step (make_train_step closure pattern)...")
    try:
        from src.training.trainer import make_optimizer, get_phase_params, Phase1Rotator
        
        optimizer = make_optimizer(model, cfg_1b_scale)
        graphdef, state = nnx.split(model)
        opt_graphdef, opt_state = nnx.split(optimizer)
        
        # Emulate make_train_step pattern
        use_embed = getattr(cfg_1b_scale, 'use_embedding', False)
        soft = getattr(cfg_1b_scale, 'soft_train', False)
        hybrid = getattr(cfg_1b_scale, 'hybrid_train', False)

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

            # Hybrid mode: alpha is (batch, seq, k_max), idx is (batch, seq, k_max)
            # idx_all[0].ndim == 3, idx_all[0].shape[1] > 1 => path 2 in EMA
            if idx_all[0] is None:
                alpha_sum = all_alpha.mean(axis=(0, 1, 2))
            elif idx_all[0].ndim == 3 and idx_all[0].shape[1] > 1:
                alpha_sum = jnp.zeros(N, dtype=jnp.float32).at[all_idx.reshape(-1)].add(
                    all_alpha.reshape(-1) / all_idx.size
                )
            else:
                all_idx = jnp.stack(idx_all)
                all_alpha = jnp.stack(alpha_all)
                if all_idx.ndim == 3:
                    all_alpha = all_alpha.sum(axis=2)
                    denom = batch_size * n_blocks * alpha_all[0].shape[1]
                else:
                    denom = batch_size * n_blocks
                alpha_sum = jnp.zeros(N, dtype=jnp.float32).at[all_idx.reshape(-1)].add(
                    all_alpha.reshape(-1) / denom
                )

            model.pool.update_ema(alpha_sum, cfg_1b_scale.beta_ema)
            metrics['loss'] = total_loss

            _, new_state = nnx.split(model)
            _, new_opt_state = nnx.split(optimizer)
            return metrics, new_state, new_opt_state

        _rot = Phase1Rotator(cfg_1b_scale.N, B, cfg_1b_scale.k_max, seed=0)
        forced_idx = _rot.dummy()
        m, state, opt_state = _train_step(
            graphdef, state, opt_graphdef, opt_state, batch,
            jnp.float32(5.0), jnp.float32(0.02), forced_idx
        )
        m = jax.device_get(m)
        print(f"  OK! loss={float(m['loss']):.4f}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()


def test_retrieval_1b_scale():
    rngs = nnx.Rngs(params=42)
    retrieval = MultiAspectRetrieval(D=cfg_1b_scale.D, d_A=1280, S=8, d_k=64, N=512, rngs=rngs)
    
    B, seq = 2, 8
    z_3d = jax.random.normal(rngs.params(), (B, seq, 1280))
    pool = jax.random.normal(rngs.params(), (512, cfg_1b_scale.D))

    print("\nTest 5: hybrid_forward (1B scale) under grad...")
    def loss_hybrid(retrieval_mod):
        alpha, top_idx, sims, alpha_raw = retrieval_mod.hybrid_forward(
            z_3d, pool, k_max=16, use_sigmoid=True
        )
        return alpha.sum()
    try:
        grad_fn = nnx.value_and_grad(loss_hybrid, argnums=nnx.DiffState(0, nnx.Param))
        (val, grads) = grad_fn(retrieval)
        print(f"  OK! val={float(val):.4f}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()


if __name__ == '__main__':
    test_1b_scale_grad()
    test_retrieval_1b_scale()