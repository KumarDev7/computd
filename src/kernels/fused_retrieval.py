"""
Pallas-fused aspect-weighted similarity kernel.

Fuses: sims_w[b,t,n] = Σ_s w[s] * Q[s,b,t,:] · K[s,n,:]
without materializing the (S, B, T, N) intermediate.

At 7B (S=8, B=4, T=2048, N=32768):
  Naive two-einsum path: (S,B,T,N) peak ≈ 4 GB in HBM
  This module:           (B,T,N)   peak ≈ 0.5 GB        → 8× reduction

Backends:
  TPU  → Pallas Mosaic kernel (tiles (BT, N) in VMEM, MXU-saturated)
  GPU  → per-aspect einsum loop (same asymptotic peak, no Pallas needed)
  CPU  → same per-aspect loop
"""

import functools
import jax
import jax.numpy as jnp

# ── Optional Pallas imports ────────────────────────────────────────────────
_PALLAS_OK = False
_PALLAS_TPU_OK = False

try:
    from jax.experimental import pallas as pl
    _PALLAS_OK = True
except ImportError:
    pass

try:
    from jax.experimental.pallas import tpu as plp
    _PALLAS_TPU_OK = True
except ImportError:
    pass

# ── Block sizes ────────────────────────────────────────────────────────────
# v5e MXU tile: 128×128. BT=128 aligns batch*seq tiles to MXU rows.
# N=512 amortises HBM-to-VMEM latency per key tile.
_TPU_BLOCK_BT = 128
_TPU_BLOCK_N  = 512


# ── Fallback: per-aspect einsum (GPU / CPU / no Pallas) ───────────────────

def _per_aspect_sim(q_flat: jax.Array, keys: jax.Array, w: jax.Array) -> jax.Array:
    """
    Accumulate Σ_s w[s] * Q[s] @ K[s].T  without (S, BT, N) peak.
    S is a Python int → loop unrolls at trace time (S ≤ 16 in all configs).

    q_flat : (S, BT, d_k)
    keys   : (S, N,  d_k)
    w      : (S,)
    return : (BT, N)
    """
    S = q_flat.shape[0]
    BT, N = q_flat.shape[1], keys.shape[1]
    acc = jnp.zeros((BT, N), dtype=q_flat.dtype)
    for s in range(S):
        acc = acc + w[s] * jnp.dot(q_flat[s], keys[s].T)
    return acc


# ── Pallas kernel body ─────────────────────────────────────────────────────

def _sim_kernel_body(queries_ref, keys_ref, w_ref, out_ref):
    """
    One kernel program handles tile (block_bt, block_n).

    queries_ref : (S, block_bt, d_k) — loaded from HBM for this tile
    keys_ref    : (S, block_n,  d_k)
    w_ref       : (S,)               — replicated for every tile
    out_ref     : (block_bt, block_n)— write-back slot in HBM

    Accumulates S dot-products in VMEM without forming (S, block_bt, block_n).
    """
    q = queries_ref[...]   # (S, block_bt, d_k)
    k = keys_ref[...]      # (S, block_n,  d_k)
    w = w_ref[...]         # (S,)

    S = q.shape[0]
    acc = jnp.zeros((q.shape[1], k.shape[1]), dtype=jnp.float32)
    for s in range(S):
        acc = acc + w[s] * jnp.dot(q[s], k[s].T)
    out_ref[...] = acc


# ── TPU kernel factory (cached by shape to avoid recompilation) ────────────

@functools.lru_cache(maxsize=32)
def _build_tpu_kernel(S: int, BT_p: int, N_p: int, d_k: int, block_bt: int, block_n: int):
    """
    Build + cache a Pallas Mosaic pallas_call for static shapes.
    BT_p and N_p are already padded to multiples of block sizes.
    """
    return pl.pallas_call(
        _sim_kernel_body,
        out_shape=jax.ShapeDtypeStruct((BT_p, N_p), jnp.float32),
        grid_spec=plp.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(BT_p // block_bt, N_p // block_n),
            in_specs=[
                pl.BlockSpec(lambda i, j: (0, i, 0), (S, block_bt, d_k)),
                pl.BlockSpec(lambda i, j: (0, j, 0), (S, block_n,  d_k)),
                pl.BlockSpec(lambda i, j: (0,),       (S,)),
            ],
            out_specs=pl.BlockSpec(lambda i, j: (i, j), (block_bt, block_n)),
        ),
    )


# ── Public API ─────────────────────────────────────────────────────────────

def fused_weighted_similarity(
    queries: jax.Array,   # (S, batch, seq, d_k) — normalized query projections
    keys:    jax.Array,   # (S, N, d_k)           — normalized key projections
    w:       jax.Array,   # (S,)                   — softmax aspect weights
) -> jax.Array:           # (batch, seq, N)         — weighted similarity scores
    """
    Compute Σ_s w[s] * Q[s] @ K[s].T without materializing (S, B, T, N).

    TPU: Pallas Mosaic kernel tiles computation in VMEM — only (block_bt, block_n)
         lives in VMEM at once; result (BT, N) written back tile-by-tile.
    GPU/CPU: per-aspect einsum loop — peak HBM is (BT, N), not (S, BT, N).

    Drop-in replacement for the two-einsum sequence in hybrid_forward / soft_forward:
        sims_s = einsum('sbtk,snk->sbtn', queries, keys)   ← 4 GB at 7B
        sims_w = einsum('s,sbtn->btn',    w,       sims_s) ← same tensor
    """
    S, batch, seq, d_k = queries.shape
    N = keys.shape[1]
    BT = batch * seq

    q_flat = queries.reshape(S, BT, d_k)

    platform = jax.devices()[0].platform

    if _PALLAS_OK and _PALLAS_TPU_OK and platform == 'tpu':
        block_bt = min(_TPU_BLOCK_BT, BT)
        block_n  = min(_TPU_BLOCK_N,  N)

        pad_bt = (-BT) % block_bt
        pad_n  = (-N)  % block_n

        q_p = jnp.pad(q_flat, [(0, 0), (0, pad_bt), (0, 0)]) if pad_bt else q_flat
        k_p = jnp.pad(keys,   [(0, 0), (0, pad_n),  (0, 0)]) if pad_n  else keys

        kernel   = _build_tpu_kernel(S, BT + pad_bt, N + pad_n, d_k, block_bt, block_n)
        sims_flat = kernel(q_p, k_p, w)          # (BT+pad, N+pad)
        sims_flat = sims_flat[:BT, :N]           # trim padding
    else:
        sims_flat = _per_aspect_sim(q_flat, keys, w)   # (BT, N)

    return sims_flat.reshape(batch, seq, N)
