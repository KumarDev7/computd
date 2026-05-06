"""
Pallas tiled similarity GEMM kernel for TPU.

Design
------
Problem: hybrid_forward in retrieval.py computes:
    sims (B,T,N) → gate (B,T,N) → alpha_raw (B,T,N) → top_k → (B,T,k)
At 7B scale (N=32K, B=4, T=2048) this chain creates ~3 GB of intermediate
HBM tensors (one per op) before top_k discards them.

Solution: Pallas 2-D grid kernel that fuses GEMM + sigmoid gate + exp into a
single pass over (BT//128, N//128) tiles.  Each (128,128) tile is computed
entirely in VMEM, and only the single fused-score array is written to HBM
(replacing 3 separate intermediate writes from the plain-JAX path).

With 8-way TP sharding (N per chip = N//8):
    Full sims (B,T,N)  = 1 GB on a single chip  → never created here
    Per-chip fused      = (BT, N//8) ≈ 128 MB   → one write, then top_k

Mosaic tile-size rules (TPU v5e):
    Block last dim      must be divisible by 128 or equal full dim
    Block second-to-last must be divisible by 8  or equal full dim

We use TILE_BT=128, TILE_N=128, d_k=128 — all satisfy both rules.

Runtime scalars (lambda_sharp, tau, T)
---------------------------------------
These are passed as (1,) float32 inputs so the kernel can be compiled once
per use_sigmoid value without recompiling every time the learned tau shifts.

API
---
tiled_topk_fused(queries, keys, k_max, lambda_sharp, tau, T, use_sigmoid)
    queries : (BT, d_k)   — pre-projected, aspect-weighted
    keys    : (N_local, d_k)
    lambda_sharp, tau, T  : Python float or 0-d JAX array
    returns   top_vals (BT, k_max), top_idx (BT, k_max)
"""

from __future__ import annotations
import functools
import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

TILE_BT = 128   # query tile   — MXU-optimal, Mosaic-compliant
TILE_N  = 128   # key tile     — MXU-optimal, Mosaic-compliant


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------

def _make_fused_kernel(use_sigmoid: bool):
    """
    Returns a Pallas kernel that fuses GEMM + sigmoid-gate + exp in VMEM.

    use_sigmoid is a compile-time flag (only 2 kernels ever compiled).
    lambda_sharp, tau, T are runtime scalars passed as (1,) array inputs —
    the kernel is NOT recompiled when these values change during training.
    """

    def kernel(q_ref, k_ref, ls_ref, tau_ref, T_ref, out_ref):
        q    = q_ref[...]             # (TILE_BT, d_k) — loaded into VMEM by BlockSpec
        k    = k_ref[...]             # (TILE_N,  d_k) — loaded into VMEM by BlockSpec
        ls   = ls_ref[0]              # scalar: lambda_sharp
        tau  = tau_ref[0]             # scalar: learned threshold
        T    = T_ref[0]               # scalar: temperature
        sims = jnp.dot(q, k.T)       # (TILE_BT, TILE_N) — stays in VMEM, never written to HBM

        if use_sigmoid:
            score = jax.nn.sigmoid(ls * (sims - tau)) * jnp.exp(sims / T)
        else:
            score = jnp.exp(sims / T)

        # One HBM write: fused score (replaces 3 separate writes sims/gate/exp in JAX)
        out_ref[...] = score

    return kernel


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=('k_max', 'use_sigmoid', 'tile_bt', 'tile_n'))
def tiled_topk_fused(
    queries     : jax.Array,         # (BT, d_k)       pre-projected, aspect-weighted queries
    keys        : jax.Array,         # (N_local, d_k)  pre-projected, aspect-weighted keys (local shard)
    k_max       : int,
    lambda_sharp,                    # Python float or 0-d JAX array — concrete at JIT trace time
    tau         : jax.Array,         # 0-d JAX array — learned parameter, abstract in JIT
    T,                               # Python float or 0-d JAX array
    use_sigmoid : bool,              # static: specialises the kernel body (only 2 variants)
    tile_bt     : int = TILE_BT,
    tile_n      : int = TILE_N,
) -> tuple[jax.Array, jax.Array]:
    """
    Fused tiled GEMM + gate/exp + top_k on a single (possibly sharded) pool shard.

    Never materialises a separate sims, gate, or alpha_raw array — only the
    final fused score (BT, N_local) is written to HBM before top_k discards it.

    Returns
    -------
    top_vals : (BT, k_max)   float32
    top_idx  : (BT, k_max)   int32   — indices into [0, N_local)
    """
    BT, d_k   = queries.shape
    N_local   = keys.shape[0]

    # Pack runtime scalars as (1,) float32 for Pallas input
    ls   = jnp.asarray(lambda_sharp, dtype=jnp.float32).reshape(1)
    tau_ = jnp.asarray(tau,          dtype=jnp.float32).reshape(1)
    T_   = jnp.asarray(T,            dtype=jnp.float32).reshape(1)

    # Pad to multiples of tile sizes (Mosaic requirement)
    pad_bt = (-BT)      % tile_bt
    pad_n  = (-N_local) % tile_n
    if pad_bt > 0:
        queries = jnp.pad(queries, ((0, pad_bt), (0, 0)))
    if pad_n > 0:
        keys = jnp.pad(keys, ((0, pad_n), (0, 0)))

    BT_p = BT + pad_bt
    N_p  = N_local + pad_n

    kern = _make_fused_kernel(bool(use_sigmoid))

    (fused_scores,) = pl.pallas_call(
        kern,
        out_shape=[jax.ShapeDtypeStruct((BT_p, N_p), jnp.float32)],
        grid=(BT_p // tile_bt, N_p // tile_n),
        in_specs=[
            pl.BlockSpec((tile_bt, d_k), lambda bt, nt: (bt, 0)),
            pl.BlockSpec((tile_n,  d_k), lambda bt, nt: (nt, 0)),
            pl.no_block_spec,   # lambda_sharp (1,) — replicated across all tiles
            pl.no_block_spec,   # tau          (1,)
            pl.no_block_spec,   # T            (1,)
        ],
        out_specs=[
            pl.BlockSpec((tile_bt, tile_n), lambda bt, nt: (bt, nt)),
        ],
    )(queries, keys, ls, tau_, T_)

    # Mask padding before top_k so padded positions never win
    if pad_n > 0:
        fused_scores = fused_scores[:BT_p, :N_local]
    else:
        fused_scores = fused_scores[:BT_p, :]

    top_vals, top_idx = jax.lax.top_k(fused_scores[:BT], k_max)
    return top_vals, top_idx


# ---------------------------------------------------------------------------
# Interpret-mode fallback (GPU / CPU — no Mosaic)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=('k_max', 'use_sigmoid'))
def tiled_topk_fallback(
    queries     : jax.Array,
    keys        : jax.Array,
    k_max       : int,
    lambda_sharp,
    tau         : jax.Array,
    T,
    use_sigmoid : bool,
) -> tuple[jax.Array, jax.Array]:
    """Plain-JAX equivalent — used when Pallas is unavailable (CPU/GPU dev)."""
    sims = jnp.dot(queries, keys.T)
    if use_sigmoid:
        score = jax.nn.sigmoid(lambda_sharp * (sims - tau)) * jnp.exp(sims / T)
    else:
        score = jnp.exp(sims / T)
    return jax.lax.top_k(score, k_max)
