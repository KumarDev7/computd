"""Minimal reproduction of MultiAspectRetrieval shape error under nnx.value_and_grad.

Tests all three retrieval paths:
  1. Hard mode (use_sigmoid=False) — the __call__ method
  2. Sigmoid mode (use_sigmoid=True) — the __call__ method
  3. Hybrid mode — hybrid_forward method

Goal: find which path breaks under nnx.value_and_grad.
"""
import os
os.environ.setdefault('JAX_ENABLE_X64', 'False')
os.environ.setdefault('XLA_FLAGS', '')

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from configs.small import DWAConfig
from src.model.dwa import DWAModel, _retrieve_per_position
from src.model.retrieval import MultiAspectRetrieval
from src.training.trainer import loss_fn

cfg = DWAConfig(
    d_input=64, d_A=64, d_B=64, D=2048, r=4,
    N=128, k_max=4, S=2, d_k=16,
    n_heads=0, max_seq_len=32, n_assembly_layers=1,
    hybrid_train=True, use_embedding=False,
)

def test_hybrid_forward_under_grad():
    rngs = nnx.Rngs(params=0)
    model = DWAModel(cfg, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adamw(1e-4), wrt=nnx.Param)

    B, T = 2, 8
    x = jax.random.normal(jax.random.PRNGKey(1), (B, T, cfg.d_input))

    # --- Test 1: Forward pass only (no grad) ---
    print("Test 1: Hybrid forward (no grad)...")
    try:
        logits, aux = model(
            x, use_sigmoid=True, lambda_sharp=1.0, return_aux=True,
            soft=False, hybrid=True,
        )
        print(f"  OK! logits shape: {logits.shape}")
        print(f"  alpha shape: {aux['alpha'].shape}")
        print(f"  idx shape: {aux['idx'].shape}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # --- Test 2: loss_fn under nnx.value_and_grad (hybrid mode) ---
    print("\nTest 2: loss_fn + nnx.value_and_grad (hybrid=True, use_sigmoid=True)...")
    try:
        grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)
        batch = jax.random.randint(jax.random.PRNGKey(2), (B, T + 1), 0, cfg.d_input)
        forced_idx = jnp.zeros((B, cfg.k_max), dtype=jnp.int32)
        result = grad_fn(
            model, batch, True, 1.0, 0.02, forced_idx, False, True, None
        )
        # nnx.value_and_grad with has_aux returns ((value, aux), grads)
        (total_loss, aux_data), grads = result
        metrics, aux = aux_data
        print(f"  OK! loss={float(total_loss):.4f}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # --- Test 3: Phase 1 under grad (use_sigmoid=False, forced_idx) ---
    print("\nTest 3: loss_fn + nnx.value_and_grad (use_sigmoid=False, forced_idx)...")
    try:
        grad_fn2 = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)
        forced_idx = jnp.zeros((B, cfg.k_max), dtype=jnp.int32)
        result = grad_fn2(
            model, batch, False, 1.0, 0.0, forced_idx, False, False, None
        )
        (total_loss, aux_data), grads = result
        metrics, aux = aux_data
        print(f"  OK! loss={float(total_loss):.4f}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # --- Test 4: Sigmoid mode under grad (use_sigmoid=True, no hybrid) ---
    print("\nTest 4: loss_fn + nnx.value_and_grad (use_sigmoid=True, no hybrid)...")
    try:
        grad_fn3 = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)
        result = grad_fn3(
            model, batch, True, 1.0, 0.02, forced_idx, False, False, None
        )
        (total_loss, aux_data), grads = result
        metrics, aux = aux_data
        print(f"  OK! loss={float(total_loss):.4f}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def test_retrieval_standalone():
    """Test MultiAspectRetrieval standalone under gradient."""
    print("\n=== Standalone MultiAspectRetrieval tests ===")
    rngs = nnx.Rngs(params=42)
    retrieval = MultiAspectRetrieval(D=2048, d_A=64, S=2, d_k=16, N=128, rngs=rngs)
    
    B = 2
    z_2d = jax.random.normal(rngs.params(), (B, 64))
    z_3d = jax.random.normal(rngs.params(), (B, 8, 64))  # (batch, seq, d_A)
    vectors = retrieval.W_K.value  # just to get the pool
    pool = jax.random.normal(rngs.params(), (128, 2048))

    # --- Test 5: __call__ with 2D input under grad ---
    print("\nTest 5: __call__ (2D, sigmoid=True) under grad...")
    def loss_2d(retrieval_mod):
        alpha, idx, sims, alpha_raw = retrieval_mod(z_2d, pool, k_max=4, use_sigmoid=True)
        return alpha.sum()
    try:
        grad_fn = nnx.value_and_grad(loss_2d, argnums=nnx.DiffState(0, nnx.Param))
        (val, grads) = grad_fn(retrieval)
        print(f"  OK! val={float(val):.4f}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

    # --- Test 6: hybrid_forward under grad ---
    print("\nTest 6: hybrid_forward (3D) under grad...")
    def loss_hybrid(retrieval_mod):
        alpha, top_idx, sims, alpha_raw = retrieval_mod.hybrid_forward(
            z_3d, pool, k_max=4, use_sigmoid=True
        )
        return alpha.sum()
    try:
        grad_fn = nnx.value_and_grad(loss_hybrid, argnums=nnx.DiffState(0, nnx.Param))
        (val, grads) = grad_fn(retrieval)
        print(f"  OK! val={float(val):.4f}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # --- Test 7: __call__ with 3D input (sequence-level) under grad ---
    print("\nTest 7: __call__ (3D z, sigmoid=True) under grad...")
    def loss_3d(retrieval_mod):
        # z.ndim == 3 goes through mean(z, axis=1) -> 2D, then __call__
        z_mean = z_3d.mean(axis=1)
        alpha, idx, sims, alpha_raw = retrieval_mod(
            z_mean, pool, k_max=4, use_sigmoid=True
        )
        return alpha.sum()
    try:
        grad_fn = nnx.value_and_grad(loss_3d, argnums=nnx.DiffState(0, nnx.Param))
        (val, grads) = grad_fn(retrieval)
        print(f"  OK! val={float(val):.4f}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")


if __name__ == '__main__':
    import optax
    test_hybrid_forward_under_grad()
    test_retrieval_standalone()