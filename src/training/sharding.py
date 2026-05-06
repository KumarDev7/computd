"""
Multi-chip TPU sharding utilities for the 7B DWA model.

Mesh layout
-----------
All 8 TPU cores form a single 1-D tensor-parallel axis ('tp').

Sharding strategy
-----------------
Pool vectors (N, D)          → P('tp', None)   — split N across chips
                                Each chip holds (N//8, D).
                                Similarity GEMM runs locally; cross-chip
                                all_gather + top_k gives global top-k.

W_K  (S, d_k, D)             → replicated      — small enough (285 MB total)
W_Q  (S, d_k, d_A)           → replicated      — tiny
All PartA/PartB MLP           → column-parallel on the hidden dimension
  fc1.kernel (d_in, hidden)  → P(None, 'tp')   column parallel
  fc1.bias   (hidden,)       → P('tp',)
  fc2.kernel (hidden, d_out) → P('tp', None)   row parallel
  fc2.bias   (d_out,)        → replicated

Activations h (B, S, d_model) → replicated    — small vs. pool
"""

from __future__ import annotations
import functools
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax import nnx


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

def make_mesh() -> Mesh:
    """Create a 1-D TP mesh over all available TPU devices."""
    devices = jax.devices()
    return Mesh(np.array(devices), ('tp',))


# ---------------------------------------------------------------------------
# Parameter sharding map
# ---------------------------------------------------------------------------

def _sharding_for_param(path_str: str, arr: jax.Array, mesh: Mesh) -> NamedSharding:
    """
    Returns the NamedSharding for a single parameter array based on its path
    in the model state tree.
    """
    ndim = arr.ndim

    # ---- Pool vectors (N, D) — shard N across 'tp' ----
    if 'pool/vectors' in path_str:
        return NamedSharding(mesh, P('tp', None))

    # ---- Retrieval W_K (S, d_k, D) — shard D across 'tp' ----
    #  W_K is the memory bottleneck: (16, 128, 69632) = 570 MB per block
    if 'W_K' in path_str and 'kernel' in path_str and ndim == 3:
        return NamedSharding(mesh, P(None, None, 'tp'))

    # ---- PartA / PartB MLP column-row parallelism ----
    # fc1 column parallel: (d_in, hidden) → P(None, 'tp')
    if ('part_a' in path_str or 'part_b' in path_str) and 'fc1' in path_str:
        if 'kernel' in path_str and ndim == 2:
            return NamedSharding(mesh, P(None, 'tp'))
        if 'bias' in path_str and ndim == 1:
            return NamedSharding(mesh, P('tp',))

    # fc2 row parallel: (hidden, d_out) → P('tp', None)
    if ('part_a' in path_str or 'part_b' in path_str) and 'fc2' in path_str:
        if 'kernel' in path_str and ndim == 2:
            return NamedSharding(mesh, P('tp', None))
        # bias replicated

    # DWABlock assembly W_base, gamma — replicate
    # Retrieval W_Q, W_K, aspect_logits, tau — replicate
    # Everything else replicated
    return NamedSharding(mesh, P(*([None] * ndim)))


def shard_model(model: nnx.Module, mesh: Mesh) -> None:
    """
    Shards model parameters in-place using jax.device_put.
    Also stores mesh on model for use by pallas_hybrid_forward.
    Call once after model creation, before JIT-compiling train_step.

    Skips params already correctly sharded (e.g. pool/W_K init-sharded).
    """
    model._mesh = mesh
    graphdef, state = nnx.split(model)

    def _shard_leaf(path, leaf):
        if not isinstance(leaf, jax.Array):
            return leaf
        path_str = '/'.join(
            str(p.key) if hasattr(p, 'key') else str(p) for p in path
        )
        target = _sharding_for_param(path_str, leaf, mesh)
        # Skip params already on the correct sharding (init-sharded pool/W_K)
        if hasattr(leaf, 'sharding') and leaf.sharding is not None:
            try:
                if leaf.sharding == target:
                    return leaf
            except Exception:
                pass
        # Skip SingleDeviceSharding params > 100 MB — these must be init-sharded
        # or they'll OOM during device_put (too large to replicate to all devices)
        if leaf.size * leaf.dtype.itemsize > 100_000_000:  # > 100 MB
            from jax.sharding import SingleDeviceSharding
            if isinstance(leaf.sharding, SingleDeviceSharding):
                # Not yet sharded but too large for device_put — skip with warning
                import warnings
                warnings.warn(f'Skipping sharding of large param {path_str} '
                              f'({leaf.size * leaf.dtype.itemsize / 1e6:.0f} MB) '
                              f'on single device. Init-shard it or reduce size.')
                return leaf
        return jax.device_put(leaf, target)

    sharded_state = jax.tree_util.tree_map_with_path(_shard_leaf, state)
    nnx.update(model, sharded_state)


# ---------------------------------------------------------------------------
# Sharding constraints for activations (used inside JIT-compiled functions)
# ---------------------------------------------------------------------------

def constrain_replicated(x: jax.Array, mesh: Mesh) -> jax.Array:
    """Mark an activation as replicated across all TP devices."""
    return jax.lax.with_sharding_constraint(
        x, NamedSharding(mesh, P(*([None] * x.ndim)))
    )


def constrain_pool_sharded(x: jax.Array, mesh: Mesh) -> jax.Array:
    """Mark a (N, D) tensor as sharded along N."""
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P('tp', None)))
