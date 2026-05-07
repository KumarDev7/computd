"""
Fused GEMM → sigmoid-gate → streaming top_k kernel.

Eliminates the (B, T, N) HBM intermediate by fusing the aspect-weighted
similarity, sigmoid gate, and top-k selection into a single pass over N-tiles.

At 7B (S=16, B=8, T=2048, N=32768, k_max=32):
  Previous pipeline:     sims_w (B,T,N) + gate (B,T,N) + alpha_raw (B,T,N)
                         ≈ 3 × 0.5 GB = 1.5 GB peak HBM per layer
  This module:           Only (BT, tile_n) + (BT, k_max) in VMEM per tile
                         ≈ 0 HBM for (B,T,N) intermediates

Backends:
  TPU  → Pallas Mosaic kernel (streaming top-k in VMEM, MXU-saturated)
  GPU  → JAX-level tiled loop (lax.scan over N-tiles, same memory savings)
  CPU  → same JAX-level tiled loop

Outputs:
  top_vals:      (batch, seq, k_max)  — top-k gated scores
  top_idx:       (batch, seq, k_max)  — global N-indices of top-k
  sims_seq:      (batch, N)           — seq-mean similarities (aux losses)
  alpha_raw_seq: (batch, N)           — seq-mean raw scores (aux losses)

The aux tensors (batch, N) are still full-N but written tile-by-tile;
they never expand to (B, T, N).
"""

import functools
import jax
import jax.numpy as jnp
from jax import lax

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

# ── Tile sizes ─────────────────────────────────────────────────────────────
# BT tile: rows of the (batch*seq) dimension per Pallas grid cell.
# N tile: how many pool vectors processed per streaming step.
#   Larger → more MXU utilization, but uses more VMEM for the tile.
#   At k_max=32, the top-k merge per N-tile is cheap.
_TPU_BLOCK_BT = 128     # MXU-aligned
_TPU_BLOCK_N  = 512     # Amortise VMEM load latency
_JAX_TILE_N   = 1024    # JAX-level tiling for GPU/CPU fallback


# ── Streaming top-k merge utility ─────────────────────────────────────────

def _merge_topk(
    heap_vals: jax.Array,    # (BT, k_max) — current best values
    heap_idx:  jax.Array,    # (BT, k_max) — current best indices
    new_vals:  jax.Array,    # (BT, tile_n) — candidate values from this tile
    new_idx:   jax.Array,    # (BT, tile_n) — candidate global indices
    k_max:     int,
) -> tuple[jax.Array, jax.Array]:
    """
    Merge current top-k heap with new candidates, keeping only the top k_max.

    Concatenates heap + candidates along the last axis, then takes top_k.
    The concat is at most (BT, k_max + tile_n) — small in VMEM.
    """
    merged_vals = jnp.concatenate([heap_vals, new_vals], axis=-1)  # (BT, k_max + tile_n)
    merged_idx  = jnp.concatenate([heap_idx,  new_idx],  axis=-1)

    # top_k on last axis
    top_vals, sel = jax.lax.top_k(merged_vals, k_max)  # (BT, k_max)
    top_idx = jnp.take_along_axis(merged_idx, sel, axis=-1)

    return top_vals, top_idx


# ── Pallas kernel body: fused GEMM + gate + tile-local top-k ──────────────

def _fused_topk_kernel_body(
    queries_ref, keys_ref, w_ref, gate_params_ref,
    top_vals_ref, top_idx_ref, sims_mean_ref, alpha_raw_mean_ref,
):
    """
    One Pallas program instance processes tile (block_bt, block_n).

    Computes aspect-weighted sims for this tile, applies sigmoid gate,
    writes tile-local results to output refs.  The outer loop merges
    tile-local top-k into the global running heap.

    queries_ref     : (S, block_bt, d_k)
    keys_ref        : (S, block_n,  d_k)
    w_ref           : (S,)
    gate_params_ref : (3,)  — [lambda_sharp, tau_scalar, T_inv]
    top_vals_ref    : (block_bt, block_n) — write: gated scores for this tile
    top_idx_ref     : () — unused (indices computed outside)
    sims_mean_ref   : (block_bt, block_n) — write: raw sims for this tile
    alpha_raw_mean_ref : () — unused (computed outside)
    """
    q = queries_ref[...]   # (S, block_bt, d_k)
    k = keys_ref[...]      # (S, block_n,  d_k)
    w = w_ref[...]         # (S,)
    gp = gate_params_ref[...]  # (3,)

    lambda_sharp = gp[0]
    tau_scalar   = gp[1]
    T_inv        = gp[2]

    S = q.shape[0]

    # Aspect-weighted similarity: Σ_s w[s] * Q[s] @ K[s].T
    acc = jnp.zeros((q.shape[1], k.shape[1]), dtype=jnp.float32)
    for s in range(S):
        acc = acc + w[s] * jnp.dot(q[s], k[s].T)

    # Write raw sims (for seq-mean aux computation outside)
    sims_mean_ref[...] = acc

    # Sigmoid gate + exp scoring
    gate = jax.nn.sigmoid(lambda_sharp * (acc - tau_scalar))
    alpha_raw = gate * jnp.exp(acc * T_inv)

    top_vals_ref[...] = alpha_raw


# ── TPU kernel factory ─────────────────────────────────────────────────────

@functools.lru_cache(maxsize=32)
def _build_fused_topk_tpu_kernel(
    S: int, BT_p: int, N_tile: int, d_k: int,
    block_bt: int,
):
    """
    Build a Pallas kernel that processes one N-tile (block_n = N_tile).
    Grid is only over BT (single column of N per call).

    Returns raw sims and gated alpha_raw for the tile — the streaming
    top-k merge happens at the JAX level outside pallas_call.
    """
    # Single-tile kernel: grid only over BT dimension
    return pl.pallas_call(
        _fused_topk_kernel_body,
        out_shape=[
            jax.ShapeDtypeStruct((BT_p, N_tile), jnp.float32),   # alpha_raw tile
            jax.ShapeDtypeStruct((), jnp.float32),                # unused idx placeholder
            jax.ShapeDtypeStruct((BT_p, N_tile), jnp.float32),   # sims tile
            jax.ShapeDtypeStruct((), jnp.float32),                # unused placeholder
        ],
        grid_spec=plp.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(BT_p // block_bt,),
            in_specs=[
                pl.BlockSpec(lambda i: (0, i, 0), (S, block_bt, d_k)),   # queries
                pl.BlockSpec(lambda i: (0, 0, 0), (S, N_tile, d_k)),     # keys (full tile)
                pl.BlockSpec(lambda i: (0,),       (S,)),                 # w
                pl.BlockSpec(lambda i: (0,),       (3,)),                 # gate_params
            ],
            out_specs=[
                pl.BlockSpec(lambda i: (i, 0), (block_bt, N_tile)),  # alpha_raw tile
                pl.BlockSpec(lambda i: (),     ()),                   # unused
                pl.BlockSpec(lambda i: (i, 0), (block_bt, N_tile)),  # sims tile
                pl.BlockSpec(lambda i: (),     ()),                   # unused
            ],
        ),
    )


# ── JAX-level tiled fallback (GPU/CPU) ────────────────────────────────────

def _per_aspect_sim_tile(
    q_flat: jax.Array,    # (S, BT, d_k)
    k_tile: jax.Array,    # (S, tile_n, d_k)
    w:      jax.Array,    # (S,)
) -> jax.Array:           # (BT, tile_n)
    """Compute weighted similarity for one N-tile without (S,BT,tile_n) peak."""
    S = q_flat.shape[0]
    BT = q_flat.shape[1]
    tile_n = k_tile.shape[1]
    acc = jnp.zeros((BT, tile_n), dtype=q_flat.dtype)
    for s in range(S):
        acc = acc + w[s] * jnp.dot(q_flat[s], k_tile[s].T)
    return acc


# ── Public API ─────────────────────────────────────────────────────────────

def fused_topk_retrieval(
    queries:      jax.Array,   # (S, batch, seq, d_k) — normalized query projections
    keys:         jax.Array,   # (S, N, d_k)          — normalized key projections
    w:            jax.Array,   # (S,)                  — softmax aspect weights
    k_max:        int,         # number of top-k to keep
    lambda_sharp: float | jax.Array,  # sigmoid sharpness
    tau_scalar:   float | jax.Array,  # sigmoid threshold
    T:            float | jax.Array,  # temperature for exp scoring
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Fused aspect-weighted GEMM → sigmoid gate → streaming top-k.

    Never materializes (B, T, N) in HBM.  Processes N in tiles, maintaining
    a running top-k heap of size k_max per (batch, seq) position.

    Returns:
      top_vals:      (batch, seq, k_max)  — top-k gated alpha_raw scores
      top_idx:       (batch, seq, k_max)  — global N-indices of top-k
      sims_seq:      (batch, N)           — seq-mean raw sims (for aux losses)
      alpha_raw_seq: (batch, N)           — seq-mean alpha_raw (for aux losses)
    """
    S, batch, seq, d_k = queries.shape
    N = keys.shape[1]
    BT = batch * seq

    q_flat = queries.reshape(S, BT, d_k)

    platform = jax.devices()[0].platform

    # Ensure scalar params are arrays for tracing
    lambda_sharp = jnp.asarray(lambda_sharp, dtype=jnp.float32)
    tau_scalar   = jnp.asarray(tau_scalar,   dtype=jnp.float32)
    T_arr        = jnp.asarray(T,            dtype=jnp.float32)
    T_inv        = 1.0 / T_arr

    # Determine tile size
    if _PALLAS_OK and _PALLAS_TPU_OK and platform == 'tpu':
        tile_n = min(_TPU_BLOCK_N, N)
    else:
        tile_n = min(_JAX_TILE_N, N)

    # Pad N to multiple of tile_n
    pad_n = (-N) % tile_n
    N_padded = N + pad_n
    if pad_n > 0:
        keys = jnp.pad(keys, [(0, 0), (0, pad_n), (0, 0)])  # (S, N_padded, d_k)

    n_tiles = N_padded // tile_n

    # ── Initialize streaming top-k heap ────────────────────────────────
    # Start with -inf values so any real score beats the initial heap.
    heap_vals = jnp.full((BT, k_max), -jnp.inf, dtype=jnp.float32)
    heap_idx  = jnp.zeros((BT, k_max), dtype=jnp.int32)

    # Determine tile-local top-k size: if tile_n < k_max (rare), take all
    tile_k = min(k_max, tile_n)  # Python int — static for top_k

    # Pre-compute tile-local index template (reused every iteration)
    # (1, tile_n) → broadcast to (BT, tile_n) inside the loop
    tile_idx_template = jnp.arange(tile_n, dtype=jnp.int32)[None, :]  # (1, tile_n)

    # Aux output accumulators: (batch, N_padded) written tile-by-tile.
    # Each tile reduces (BT, tile_n) → (batch, tile_n) via seq-mean,
    # then writes to the correct N-offset. Never expands to (B, T, N).
    sims_seq_acc      = jnp.zeros((batch, N_padded), dtype=jnp.float32)
    alpha_raw_seq_acc = jnp.zeros((batch, N_padded), dtype=jnp.float32)

    # ── Tiled streaming loop ───────────────────────────────────────────
    # Use lax.fori_loop to avoid Python unrolling (N could be 32k+).

    use_pallas = _PALLAS_OK and _PALLAS_TPU_OK and platform == 'tpu'

    if use_pallas:
        block_bt = min(_TPU_BLOCK_BT, BT)
        pad_bt = (-BT) % block_bt
        BT_p = BT + pad_bt
        q_p = jnp.pad(q_flat, [(0, 0), (0, pad_bt), (0, 0)]) if pad_bt else q_flat
        gate_params = jnp.array([lambda_sharp, tau_scalar, T_inv], dtype=jnp.float32)

    def _tile_step(tile_idx, carry):
        heap_v, heap_i, sims_acc, ar_acc = carry
        n_start = tile_idx * tile_n

        # Extract key tile using lax.dynamic_slice (no negative-index check)
        k_tile = lax.dynamic_slice(keys, (0, n_start, 0), (S, tile_n, d_k))  # (S, tile_n, d_k)

        if use_pallas:
            kernel = _build_fused_topk_tpu_kernel(S, BT_p, tile_n, d_k, block_bt)
            alpha_raw_tile_p, _, sims_tile_p, _ = kernel(q_p, k_tile, w, gate_params)
            # Trim BT padding
            alpha_raw_tile = alpha_raw_tile_p[:BT]  # (BT, tile_n)
            sims_tile      = sims_tile_p[:BT]
        else:
            # JAX-level: compute weighted sims for this tile
            sims_tile = _per_aspect_sim_tile(q_flat, k_tile, w)  # (BT, tile_n)

            # Sigmoid gate + exp
            gate_tile      = jax.nn.sigmoid(lambda_sharp * (sims_tile - tau_scalar))
            alpha_raw_tile = gate_tile * jnp.exp(sims_tile * T_inv)  # (BT, tile_n)

        # ── Tile-local top-k for merge ─────────────────────────────────
        # Take top-k from this tile (tile_n values → tile_k candidates)
        # This prevents the merge concat from growing unboundedly.
        tile_top_vals, tile_top_sel = jax.lax.top_k(alpha_raw_tile, tile_k)  # (BT, tile_k)
        tile_top_idx = jnp.take_along_axis(
            jnp.broadcast_to(tile_idx_template, (BT, tile_n)),
            tile_top_sel, axis=-1
        ) + n_start  # Global N-indices

        # Pad tile_top if tile_k < k_max (when tile_n < k_max, rare)
        pad_candidates_v = tile_top_vals
        pad_candidates_i = tile_top_idx
        if tile_k < k_max:
            pad_k = k_max - tile_k
            pad_candidates_v = jnp.pad(tile_top_vals, [(0, 0), (0, pad_k)],
                                        constant_values=-jnp.inf)
            pad_candidates_i = jnp.pad(tile_top_idx,  [(0, 0), (0, pad_k)],
                                        constant_values=0)

        # ── Merge with running heap ────────────────────────────────────
        heap_v, heap_i = _merge_topk(heap_v, heap_i, pad_candidates_v, pad_candidates_i, k_max)

        # ── Accumulate aux tensors: seq-mean ───────────────────────────
        # sims_tile: (BT, tile_n) → reduce to (batch, tile_n) via mean over seq
        sims_bt      = sims_tile.reshape(batch, seq, tile_n)
        ar_bt        = alpha_raw_tile.reshape(batch, seq, tile_n)
        sims_mean_t  = sims_bt.mean(axis=1)      # (batch, tile_n)
        ar_mean_t    = ar_bt.mean(axis=1)         # (batch, tile_n)

        # Write to accumulator at correct N-offset using dynamic_update_slice
        sims_acc = lax.dynamic_update_slice(sims_acc, sims_mean_t, (0, n_start))
        ar_acc   = lax.dynamic_update_slice(ar_acc,   ar_mean_t,   (0, n_start))

        return (heap_v, heap_i, sims_acc, ar_acc)

    # Run the streaming loop
    init_carry = (heap_vals, heap_idx, sims_seq_acc, alpha_raw_seq_acc)
    heap_vals, heap_idx, sims_seq_acc, alpha_raw_seq_acc = lax.fori_loop(
        0, n_tiles, _tile_step, init_carry
    )

    # ── Trim N-padding from aux tensors ────────────────────────────────
    sims_seq      = sims_seq_acc[:, :N]       # (batch, N)
    alpha_raw_seq = alpha_raw_seq_acc[:, :N]  # (batch, N)

    # ── Reshape outputs ────────────────────────────────────────────────
    top_vals = heap_vals.reshape(batch, seq, k_max)
    top_idx  = heap_idx.reshape(batch, seq, k_max)

    return top_vals, top_idx, sims_seq, alpha_raw_seq

