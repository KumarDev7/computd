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
    if 'W_K' in path_str and ndim == 3:
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

    # ---- Attention in DWABlocks: column-row parallel ----
    # qkv kernel: (d_model, 3*d_model) → column parallel P(None, 'tp')
    if 'attn/qkv/kernel' in path_str and ndim == 2:
        return NamedSharding(mesh, P(None, 'tp'))
    # proj kernel: (d_model, d_model) → row parallel P('tp', None)
    if 'attn/proj/kernel' in path_str and ndim == 2:
        return NamedSharding(mesh, P('tp', None))

    # ---- Query projection in DWABlocks ----
    # query_proj kernel: (d_model, d_model) → column parallel P(None, 'tp')
    if 'query_proj/kernel' in path_str and ndim == 2:
        return NamedSharding(mesh, P(None, 'tp'))

    # Everything else replicated
    return NamedSharding(mesh, P(*([None] * ndim)))


def shard_model(model: nnx.Module, mesh: Mesh) -> None:
    """
    Shards model parameters in-place using jax.device_put.
    Also stores mesh on model for use by pallas_hybrid_forward.
    Call once after model creation, before JIT-compiling train_step.

    Skips params already correctly sharded (e.g. pool/W_K/MLP init-sharded).
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
        # Skip params already on the correct sharding
        if hasattr(leaf, 'sharding') and leaf.sharding is not None:
            try:
                if leaf.sharding == target:
                    return leaf
            except Exception:
                pass
        return jax.device_put(leaf, target)

    sharded_state = jax.tree_util.tree_map_with_path(_shard_leaf, state)
    nnx.update(model, sharded_state)


def shard_optimizer(optimizer: nnx.Optimizer, model: nnx.Module, mesh: Mesh) -> None:
    """
    Shard optimizer state (Adam m, v) to match parameter sharding.
    Must be called after shard_model().
    """
    graphdef, model_state = nnx.split(model)
    param_state = model_state.filter(nnx.Param)
    opt_state = optimizer.opt_state

    def _shard_opt_leaf(leaf):
        if not isinstance(leaf, jax.Array):
            return leaf
        # Optimizer state follows the same sharding as its parameter
        # device_put with replicated sharding for small arrays,
        # or find the matching param sharding
        from jax.sharding import SingleDeviceSharding
        if isinstance(leaf.sharding, SingleDeviceSharding):
            # Move off device 0 — replicate by default (safe for small arrays)
            replicated = NamedSharding(mesh, P(*([None] * leaf.ndim)))
            return jax.device_put(leaf, replicated)
        return leaf

    sharded_opt_state = jax.tree_util.tree_map(_shard_opt_leaf, opt_state)
    optimizer.opt_state = sharded_opt_state


# ---------------------------------------------------------------------------
# Sharding constraints for activations (used inside JIT-compiled functions)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Init-sharded parameter creation (avoids OOM on device 0)
# ---------------------------------------------------------------------------

def init_sharded_param(
    shape: tuple[int, ...],
    sharding: NamedSharding,
    rng_key: jax.Array,
    scale: float = 0.02,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """
    Create a sharded parameter directly on all devices — never materialises the
    full array on a single device.  Uses jax.make_array_from_callback so each
    device only allocates its own shard.
    """
    def _callback(shard_indices):
        offset = 0
        local_shape = []
        for i, (sl, full_dim) in enumerate(zip(shard_indices, shape)):
            if isinstance(sl, slice):
                start = sl.start if sl.start is not None else 0
                stop = sl.stop if sl.stop is not None else full_dim
                offset += start * (10 ** i)
                local_shape.append(stop - start)
            else:
                offset += sl * (10 ** i)
                local_shape.append(1)
        shard_key = jax.random.fold_in(rng_key, offset)
        return jax.random.normal(shard_key, tuple(local_shape), dtype=dtype) * scale

    return jax.make_array_from_callback(shape, sharding, _callback)


def make_linear_sharded(
    d_in: int,
    d_out: int,
    rngs: nnx.Rngs,
    sharding: NamedSharding | None = None,
    use_bias: bool = True,
    kernel_sharding: NamedSharding | None = None,
    bias_sharding: NamedSharding | None = None,
) -> nnx.Linear:
    """
    Create an nnx.Linear with an init-sharded kernel.

    If sharding is provided, the kernel is created via make_array_from_callback
    so it never resides on a single device.  Bias is always small and left
    replicated (or follows kernel row-parallel convention if bias_sharding set).

    Two usage patterns:
      1. sharding=single_sharding → both kernel and bias use this sharding spec
         (useful for replicated linesrs).
      2. kernel_sharding + bias_sharding → fine-grained per-parameter control
         (useful for TP column/row parallel).
    """
    if kernel_sharding is None and sharding is not None:
        kernel_sharding = sharding
    if bias_sharding is None and sharding is not None:
        bias_sharding = sharding

    lin = nnx.Linear(d_in, d_out, use_bias=use_bias, rngs=rngs)

    if kernel_sharding is not None:
        kernel_key = rngs.params()
        lin.kernel.value = init_sharded_param(
            (d_in, d_out), kernel_sharding, kernel_key,
            scale=jnp.sqrt(2.0 / d_in),   # Kaiming-ish init matching nnx default
        )

    if use_bias and bias_sharding is not None:
        # Bias is small enough to replicate; create replicated zero bias
        lin.bias.value = init_sharded_param(
            (d_out,), bias_sharding, rngs.params(), scale=0.0,
        )

    return lin


def constrain_replicated(x: jax.Array, mesh: Mesh) -> jax.Array:
    """Mark an activation as replicated across all TP devices."""
    return jax.lax.with_sharding_constraint(
        x, NamedSharding(mesh, P(*([None] * x.ndim)))
    )


def constrain_pool_sharded(x: jax.Array, mesh: Mesh) -> jax.Array:
    """Mark a (N, D) tensor as sharded along N."""
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P('tp', None)))
