"""
Multi-TPU sharding for DWA.

Mesh layout (v5e-8 default): data=2 × pool=4 = 8 cores.

  data  axis → batch dimension sharding  (standard data parallel)
  pool  axis → pool vector N sharding    (model parallel for pool)

Pool vectors (N, D) at 7B = ~17 GB in bf16 — too large for one 16 GB v5e core.
Sharding N across 4 cores gives ~4.3 GB/core, leaving room for params + activations.

Pool-parallel retrieval flow (inside shard_map):
  1. Each pool-shard computes local sims (B/data, T, N/pool)
  2. lax.all_gather across pool axis → full (B/data, T, N)
  3. lax.top_k → global (B/data, T, k_max) indices
  4. Distributed gather: masked-psum fetches only the k_max vectors across shards
     → (B/data, T, k_max, D) without all-gathering the full pool

Gradient flow:
  - data axis: psum of gradients (standard data parallel, automatic)
  - pool axis: gradients route back through the masked-psum gather correctly
               (each shard gets grads only for its local N/pool vectors)
"""

import functools
import gc
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax import nnx

try:
    from jax.experimental.shard_map import shard_map
    _HAS_SHARD_MAP = True
except ImportError:
    _HAS_SHARD_MAP = False

from src.model.dwa import DWAModel
from src.model.assembly import partial_uv_einsum


# ── Mesh creation ──────────────────────────────────────────────────────────

def make_mesh(data: int = 2, pool: int = 4, devices=None) -> Mesh:
    """
    Create a 2D (data × pool) device mesh.

    Default data=2, pool=4 targets v5e-8 (8 cores).
    Pass devices=jax.devices()[:n] to use a subset for testing.
    """
    if devices is None:
        devices = jax.devices()
    n = len(devices)
    if n < data * pool:
        # Single-device fallback for testing: collapse both axes to 1
        data, pool = 1, 1
    arr = np.array(devices[: data * pool]).reshape(data, pool)
    return Mesh(arr, axis_names=('data', 'pool'))


class MeshContext:
    """Bundles mesh + common NamedSharding specs used across modules."""

    def __init__(self, mesh: Mesh):
        self.mesh       = mesh
        self.data_size  = mesh.shape['data']
        self.pool_size  = mesh.shape['pool']

        self.pool_vecs  = NamedSharding(mesh, P('pool', None))   # (N, D) → N sharded
        self.pool_ema   = NamedSharding(mesh, P('pool',))        # (N,)  → N sharded
        self.batch      = NamedSharding(mesh, P('data', None))   # (B, T) → B sharded
        self.replicated = NamedSharding(mesh, P())
        # W_K: (S, d_k, D) → D sharded across pool axis.
        # Aligns with pool vectors (N, D): the D contraction in
        # einsum('skd,nd→snk') becomes a local partial product + psum.
        # At 7B: 16×128×69632 × 2B = 286 MB/block → 71 MB/core (pool=4).
        self.w_k_sharding = NamedSharding(mesh, P(None, None, 'pool'))


# ── BF16 param cast ──────────────────────────────────────────────────────

def cast_params_bf16(model) -> None:
    """
    Cast all nnx.Param tensors to bfloat16 in-place, preserving sharding.

    Why bfloat16 params:
      - Halves model memory (f32 → bf16) — saves ~3 GB/core at 7B
      - Optimizer state (mu + nu) is zeros_like(params) → also bf16
        → total optimizer memory halved, freeing ~6 GB/core at 7B scale
      - TPU MXU natively operates in bf16; no compute cost
      - Numerically safe: bf16 has same exponent range as f32

    EMAState and other non-Param variables are left in f32 (they don't
    contribute to optimizer state and need full precision for EMA updates).

    JAX's astype() preserves existing per-tensor sharding automatically,
    so pool vectors (P('pool',None)) and W_K (P(None,None,'pool')) keep
    their shard layout after the cast.
    """
    param_state = nnx.state(model, nnx.Param)
    bf16_state = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if (
            hasattr(x, 'dtype') and x.dtype == jnp.float32
        ) else x,
        param_state,
    )
    # Explicitly free f32 state before writing bf16 back.
    # Without del + gc, param_state keeps f32 arrays alive until
    # after nnx.update, preventing timely HBM release.
    del param_state
    gc.collect()
    nnx.update(model, bf16_state)
    del bf16_state
    gc.collect()


def _replicate_cpu_params(model, ctx: "MeshContext") -> None:
    """
    Move any nnx.Param still on CPU to ctx.replicated (all TPU devices).

    init_model_cpu_sharded only explicitly pushes pool and W_K to TPU.
    All other params (W_Q, PartA, PartB, assembly weights) remain in CPU
    RAM.  If they are NOT moved before optimizer init, JAX materialises
    them all onto whichever single device it picks — filling that core's
    16 GB HBM before any optimizer state is even allocated.

    This function walks every nnx.Param, identifies CPU-resident arrays
    (device platform == 'cpu'), and puts them on ctx.replicated so they
    are spread evenly across the whole mesh before optimizer.init().
    Already-sharded arrays (pool, W_K) are left untouched.
    """
    cpu_platform = 'cpu'

    param_state = nnx.state(model, nnx.Param)

    def _move_if_cpu(x):
        if not hasattr(x, 'devices'):
            return x
        devs = list(x.devices())
        if devs and devs[0].platform == cpu_platform:
            return jax.device_put(x, ctx.replicated)
        return x

    device_state = jax.tree_util.tree_map(_move_if_cpu, param_state)
    del param_state
    gc.collect()
    nnx.update(model, device_state)
    del device_state
    gc.collect()


# ── Initial state sharding ─────────────────────────────────────────────────

def shard_initial_state(model, ctx: MeshContext) -> None:
    """
    Move model parameters to their correct shards. Call once after model init.

    Pool vectors + EMA → pool axis (N sharded).
    W_K per block      → pool axis (D sharded).
    All other params   → replicated (default JAX behaviour for single-device init).
    """
    model.pool.vectors.value   = jax.device_put(model.pool.vectors.value,   ctx.pool_vecs)
    model.pool.ema_usage.value = jax.device_put(model.pool.ema_usage.value, ctx.pool_ema)

    # Shard W_K (S, d_k, D) along D across pool axis — matches pool D dimension
    for block in model.blocks:
        block.retrieval.W_K.value = jax.device_put(
            block.retrieval.W_K.value, ctx.w_k_sharding
        )


# ── CPU-init + sharded push ───────────────────────────────────────────────

def init_model_cpu_sharded(cfg, ctx: MeshContext, seed: int = 0) -> "DWAModel":
    """
    Initialize the full model in CPU RAM, then push sharded slices to TPU cores.

    Why:  DWAModel.__init__ on a single TPU core OOMs at 7B — pool alone is
          ~8.6 GB f32, and one v5e core has only 16 GB HBM.

    How:
      1. Init entire model on CPU (all params in host DRAM, zero TPU usage).
      2. Extract pool numpy arrays — pool (N, D) + EMA (N,).
      3. Split along the N axis into pool_size equal slices.
      4. jax.device_put each slice to its target TPU core.
      5. Reassemble into global sharded JAX arrays via
         jax.make_array_from_single_device_arrays.
      6. Non-pool params stay on CPU refs; XLA replicates them on first JIT.

    At 7B (N=32768, D=69632, pool_size=4):
      CPU RAM peak:  pool 8.6 GB + other params ~2 GB  ≈ 11 GB  (host has 200 GB+)
      Per-TPU-core:  pool shard 2.1 GB + other replicated params ~2 GB  ≈ 4 GB
                     W_K shards: 71 MB/core (vs 286 MB replicated) × 18 blocks
    """
    cpu = jax.devices('cpu')[0]

    # ── 1. Full init on CPU (f32 params in host DRAM) ────────────────────
    with jax.default_device(cpu):
        model = DWAModel(cfg, nnx.Rngs(seed))

    # ── 2. Extract pool/EMA as numpy for N-axis slicing ──────────────────
    pool_np = np.array(model.pool.vectors.value)    # (N, D) f32
    ema_np  = np.array(model.pool.ema_usage.value)  # (N,)   f32

    N, D        = pool_np.shape
    data_size   = ctx.data_size
    pool_size   = ctx.pool_size
    N_per_shard = N // pool_size
    D_per_shard = D // pool_size
    assert N % pool_size == 0, f"N={N} must be divisible by pool_size={pool_size}"
    assert D % pool_size == 0, f"D={D} must be divisible by pool_size={pool_size}"

    # ── 3 & 4. Slice pool in numpy, push as bf16 to each device ──────────
    # CRITICAL: convert numpy slice → jnp bf16 BEFORE device_put.
    # This ensures the device NEVER holds f32 pool vectors — only bf16.
    # Without this, f32 pool (2.12 GB) + f32 W_K (2.45 GB) + f32 replicated
    # params exhaust 16 GB HBM before any cast can succeed.
    pool_device_arrays = []
    ema_device_arrays  = []
    for j in range(pool_size):
        pool_shard_f32 = pool_np[j * N_per_shard : (j + 1) * N_per_shard]
        ema_shard_f32  = ema_np[ j * N_per_shard : (j + 1) * N_per_shard]
        # Cast pool to bf16 on CPU (numpy f32 → jnp bf16, host-side only)
        pool_shard_bf16 = jnp.array(pool_shard_f32, dtype=jnp.bfloat16)
        for i in range(data_size):
            device = ctx.mesh.devices[i, j]
            pool_device_arrays.append(jax.device_put(pool_shard_bf16, device))
            ema_device_arrays.append(jax.device_put(ema_shard_f32,   device))
        del pool_shard_bf16
    del pool_np, ema_np
    gc.collect()

    # ── 5. Assemble global sharded arrays ─────────────────────────────────
    model.pool.vectors.value = jax.make_array_from_single_device_arrays(
        shape=(N, D),
        sharding=ctx.pool_vecs,
        arrays=pool_device_arrays,
    )
    model.pool.ema_usage.value = jax.make_array_from_single_device_arrays(
        shape=(N,),
        sharding=ctx.pool_ema,
        arrays=ema_device_arrays,
    )

    pool_bf16_per_core = N_per_shard * D * 2  # bytes, bf16
    print(
        f"[sharding] pool sharded: {pool_size} shards × "
        f"{N_per_shard}×{D} bf16 "
        f"({N * D * 4 / 2**30:.2f} GB f32 → "
        f"{pool_bf16_per_core / 2**30:.2f} GB/core bf16)"
    )

    # ── 6. Shard W_K along D, push as bf16 ───────────────────────────────
    wk_total_bytes = 0
    for block_i, block in enumerate(model.blocks):
        wk_np = np.array(block.retrieval.W_K.value)   # (S, d_k, D) f32
        S_wk, dk_wk, D_wk = wk_np.shape
        assert D_wk == D, f"Block {block_i}: W_K D={D_wk} != pool D={D}"

        wk_device_arrays = []
        for j in range(pool_size):
            wk_shard_f32  = wk_np[:, :, j * D_per_shard : (j + 1) * D_per_shard]
            wk_shard_bf16 = jnp.array(wk_shard_f32, dtype=jnp.bfloat16)
            for i in range(data_size):
                device = ctx.mesh.devices[i, j]
                wk_device_arrays.append(jax.device_put(wk_shard_bf16, device))
            del wk_shard_bf16

        block.retrieval.W_K.value = jax.make_array_from_single_device_arrays(
            shape=(S_wk, dk_wk, D_wk),
            sharding=ctx.w_k_sharding,
            arrays=wk_device_arrays,
        )
        wk_total_bytes += wk_np.nbytes
        del wk_np
    gc.collect()

    wk_per_core = wk_total_bytes / pool_size
    print(
        f"[sharding] W_K sharded: {len(model.blocks)} blocks × "
        f"({S_wk}×{dk_wk}×{D_per_shard})/core bf16 "
        f"({wk_total_bytes / 2**20:.1f} MB f32 → "
        f"{wk_per_core / 2**20 / 2:.1f} MB/core bf16)"
    )

    # ── 7. Cast remaining CPU params to bf16, replicate to mesh ──────────
    param_state = nnx.state(model, nnx.Param)

    def _cast_cpu_and_put(x):
        if not hasattr(x, 'devices'):
            return x
        devs = list(x.devices())
        if devs and devs[0].platform == 'cpu':
            return jax.device_put(x.astype(jnp.bfloat16), ctx.replicated)
        return x  # pool / W_K already bf16 on device

    device_state = jax.tree_util.tree_map(_cast_cpu_and_put, param_state)
    del param_state
    gc.collect()
    nnx.update(model, device_state)
    del device_state
    gc.collect()
    print("[sharding] CPU params cast bf16 + replicated to mesh")

    # ── 8. Tensor-parallel sharding of PartA / PartB / attention ─────────
    # After step 7, all params are replicated P() — each of 8 cores holds
    # a full copy.  At 7B, replicated params ≈ 4.1 GB/core and their Adam
    # optimizer state ≈ 8.2 GB/core, totalling ~19 GB > 16 GB HBM limit.
    #
    # Fix: shard large Linear kernels along the pool axis using the standard
    # column-parallel → row-parallel (Megatron-LM) tensor-parallel pattern:
    #   col-par kernel  P(None,'pool')  → output activation sharded
    #   row-par kernel  P('pool',None)  → GSPMD inserts psum → output replicated
    #
    # Inside @nnx.jit / @nnx.jit(train_step), XLA's GSPMD compiler sees the
    # sharded kernels and automatically inserts the all-reduce (psum) for the
    # contracted pool dimension, so NO model code changes are needed.
    #
    # h flow: replicated → [col-par qkv] → sharded → [row-par proj] → replicated
    # The h at DWABlock boundaries stays replicated, keeping it compatible with
    # the shard_map in_spec P('data',None,None) for pool_retrieve.
    _shard_tensor_parallel(model, ctx)
    print("[sharding] All params now on device in bf16 — optimizer state will be bf16")
    return model


def _shard_tensor_parallel(model, ctx: "MeshContext") -> None:
    """Shard PartA/PartB MLP and DWABlock attention weights along pool axis.

    Uses column-parallel → row-parallel tensor parallelism so that each
    pool shard holds 1/pool_size of the weight, reducing replicated param
    memory by ~pool_size× and their optimizer state likewise.

    Requires: params are already on TPU (bf16, replicated) from step 7.
    """
    pool_size = ctx.pool_size
    col  = NamedSharding(ctx.mesh, P(None, 'pool'))        # (d_in, d_out/p) col-par
    row  = NamedSharding(ctx.mesh, P('pool', None))        # (d_in/p, d_out) row-par
    col1 = NamedSharding(ctx.mesh, P('pool',))             # (d/p,) bias for col-par
    w3d  = NamedSharding(ctx.mesh, P(None, None, 'pool'))  # (S, d_k, d_A/p) W_Q

    def _put(val, sharding):
        return jax.device_put(val, sharding)

    def _col(linear):
        """Column-parallel: shard output dim of kernel + bias."""
        k = linear.kernel.value
        if k.shape[1] % pool_size != 0:
            return
        linear.kernel.value = _put(k, col)
        if linear.bias is not None:
            linear.bias.value = _put(linear.bias.value, col1)

    def _row(linear):
        """Row-parallel: shard input dim of kernel; bias stays replicated."""
        k = linear.kernel.value
        if k.shape[0] % pool_size != 0:
            return
        linear.kernel.value = _put(k, row)
        # bias replicated — added to the full output after psum

    # PartA: embed col-par  (vocab, hidden) → each core (vocab, hidden/p)
    #        fc2   row-par  (hidden, d_A)   → each core (hidden/p, d_A)
    # PartA previously had fc1 (Linear); it is now nnx.Embed whose weight
    # is stored in .embedding (shape: vocab × hidden).  Shard the hidden
    # (output) axis across pool cores — same column-parallel pattern.
    emb = model.part_a.embed.embedding  # nnx.Param, shape (vocab, hidden)
    if emb.value.shape[1] % pool_size == 0:
        emb.value = _put(emb.value, col)   # (vocab, hidden/p) per core
    _row(model.part_a.fc2)

    # PartB: fc1 col-par  (d_B, hidden) → each core (d_B, hidden/p)
    #        fc2 row-par  (hidden, d_out) → each core (hidden/p, d_out)
    _col(model.part_b.fc1)
    _row(model.part_b.fc2)

    tp_bytes = 0
    for block in model.blocks:
        # CausalSelfAttention: qkv col-par, proj row-par
        if block.n_heads > 0:
            attn = block.attn
            _col(attn.qkv)    # (d_A, 3*d_A) → each core (d_A, 3*d_A/p)
            _row(attn.proj)   # (d_A, d_A)   → each core (d_A/p, d_A)
            # norm weights (tiny) stay replicated

        # query_proj: row-par — h is replicated P(), z output is replicated
        # after psum, which matches shard_map in_spec P('data',None,None).
        _row(block.query_proj)  # (d_A, d_A) → each core (d_A/p, d_A)

        # W_Q: (S, d_k, d_A) → shard d_A across pool, matching the D/pool
        # partial-key pattern already used inside the pool-parallel shard_map.
        wq = block.retrieval.W_Q.value
        if wq.shape[2] % pool_size == 0:
            block.retrieval.W_Q.value = _put(wq, w3d)
            tp_bytes += wq.nbytes // pool_size * 2  # rough bf16 savings

    # Report savings
    # embed.embedding: (vocab, hidden) — col-par shards hidden axis
    embed_params = model.part_a.embed.embedding.value.size * pool_size
    col_par_params = (
        embed_params +
        model.part_b.fc1.kernel.value.size * pool_size
    )
    row_par_params = (
        model.part_a.fc2.kernel.value.size * pool_size +
        model.part_b.fc2.kernel.value.size * pool_size
    )
    n_blocks = len(model.blocks)
    if n_blocks > 0 and model.blocks[0].n_heads > 0:
        attn0 = model.blocks[0].attn
        col_par_params += attn0.qkv.kernel.value.size  * pool_size * n_blocks
        row_par_params += attn0.proj.kernel.value.size * pool_size * n_blocks
    row_par_params += model.blocks[0].query_proj.kernel.value.size * pool_size * n_blocks
    total_f32_mb = (col_par_params + row_par_params) * 4 / 2**20
    per_core_bf16_mb = (col_par_params + row_par_params) * 2 / pool_size / 2**20
    print(
        f"[sharding] TP sharded: {total_f32_mb:.0f} MB f32 → "
        f"{per_core_bf16_mb:.0f} MB/core bf16 "
        f"(params + opt state savings ≈ {per_core_bf16_mb*3:.0f} MB/core)"
    )


# ── Pool-parallel hybrid retrieval ────────────────────────────────────────

def make_pool_parallel_retrieve(ctx: MeshContext, k_max: int, T: float,
                                 d_B: int, r: int, d_A: int):
    """
    Returns a (pool_retrieve, assemble_sharded) pair.

    pool_retrieve   — shard_map that does top-k retrieval in parallel.
    assemble_sharded — shard_map that gathers D_local-shard vectors,
                       computes partial U@V einsum, psums (B,T,d_B) result.

    Separating the two allows _sharded_step to pass h_A, W_base, b_base to
    assemble_sharded without bloating pool_retrieve's in_specs.

    Requires JAX 0.4.14+ for shard_map.
    """
    if not _HAS_SHARD_MAP:
        raise RuntimeError(
            "shard_map not found — upgrade JAX to 0.4.14+:\n"
            "  pip install --upgrade jax jaxlib"
        )

    pool_size = ctx.pool_size
    mesh      = ctx.mesh

    # ── Static offset math (Python ints → XLA compile-time constants) ───
    # Full vector layout: [U (d_B*r) | V (r*d_A) | b (d_B)]
    g_off_V = d_B * r
    g_off_b = d_B * r + r * d_A
    g_end_b = g_off_b + d_B
    D_total = g_end_b                   # == D from config
    D_local_static = D_total // pool_size

    # Per-shard offsets for U/V/b decomposition (one entry per pool shard)
    shard_offsets = []
    for shard_rank in range(pool_size):
        g_start = shard_rank * D_local_static
        off_V = max(0, min(D_local_static, g_off_V - g_start))
        off_b = max(0, min(D_local_static, g_off_b - g_start))
        end_b = max(0, min(D_local_static, g_end_b - g_start))
        shard_offsets.append((off_V, off_b, end_b))

    # ── Retrieval shard_map ──────────────────────────────────────────────
    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            P('data', None, None),   # z:             (B/data, T, d_A)
            P('pool', None),          # pool_vecs:     (N/pool, D)
            P(None, None, 'pool'),    # W_K:           (S, d_k, D/pool) — D sharded
            P(None, None, None),      # W_Q:           (S, d_k, d_A) — replicated
            P(None),                  # aspect_logits: (S,) — replicated
            P(None),                  # tau:           (S,) — replicated
            P(None),                  # lambda_sharp:  scalar
        ),
        out_specs=(
            P('data', None, None),   # alpha:         (B/data, T, k_max)
            P('data', None, None),   # top_idx:       (B/data, T, k_max) global indices
            P('data', None),         # sims_seq:      (B/data, N) — for aux losses
            P('data', None),         # alpha_raw_seq: (B/data, N)
        ),
        check_rep=False,
    )
    def _retrieve(z, pool_vecs, W_K, W_Q, aspect_logits, tau, lambda_sharp):
        """
        Each program instance handles (B/data, T, N/pool) locally, then
        merges top-k across the pool axis via all_gather + top_k.

        W_K is sharded along D: each shard holds (S, d_k, D/pool).
        The key GEMM produces partial dot products; psum across 'pool'
        yields the correct full key vectors before normalization.
        """
        N_local   = pool_vecs.shape[0]
        pool_rank = lax.axis_index('pool')           # 0 … pool_size-1
        B_local, T_local, _ = z.shape

        # ── Local keys ──────────────────────────────────────────────────
        D_local = W_K.shape[2]  # D / pool_size
        pool_vecs_d_local = lax.dynamic_slice_in_dim(
            pool_vecs, pool_rank * D_local, D_local, axis=1
        )  # (N_local, D_local)
        local_keys_partial = jnp.einsum('skd,nd->snk', W_K, pool_vecs_d_local)
        local_keys = lax.psum(local_keys_partial, axis_name='pool')
        local_keys = local_keys / (jnp.linalg.norm(local_keys, axis=-1, keepdims=True) + 1e-8)

        # ── Per-position queries ─────────────────────────────────────────
        queries = jnp.einsum('skd,btd->sbtk', W_Q, z)
        queries = queries / (jnp.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)

        # ── Local similarities ───────────────────────────────────────────
        local_sims_s = jnp.einsum('sbtk,snk->sbtn', queries, local_keys)
        w            = jax.nn.softmax(aspect_logits)
        local_sims   = jnp.einsum('s,sbtn->btn', w, local_sims_s)

        tau_scalar    = jnp.dot(w, tau)
        local_gate    = jax.nn.sigmoid(lambda_sharp * (local_sims - tau_scalar))
        local_ar      = local_gate * jnp.exp(local_sims / T)

        # ── Local top-k + global merge ───────────────────────────────────
        local_top_vals, local_top_sel = jax.lax.top_k(local_ar, k_max)
        pool_start    = pool_rank * N_local
        local_top_idx = local_top_sel + pool_start

        all_top_vals = lax.all_gather(local_top_vals, axis_name='pool', axis=0, tiled=False)
        all_top_idx  = lax.all_gather(local_top_idx,  axis_name='pool', axis=0, tiled=False)
        # (pool, B/data, T, k_max) → merge into (B/data, T, pool*k_max)
        merged_vals = all_top_vals.transpose(1, 2, 0, 3).reshape(B_local, T, pool_size * k_max)
        merged_idx  = all_top_idx.transpose(1, 2, 0, 3).reshape(B_local, T, pool_size * k_max)

        # ── Global top-k from merged candidates ──────────────────────────
        # pool_size * k_max candidates per position (e.g., 4*32=128) → keep k_max.
        top_vals, sel = jax.lax.top_k(merged_vals, k_max)         # (B/data, T, k_max)
        top_idx  = jnp.take_along_axis(merged_idx, sel, axis=-1)  # (B/data, T, k_max)
        alpha    = top_vals / (jnp.sum(top_vals, axis=-1, keepdims=True) + 1e-8)

        # ── Seq-mean aux tensors ──────────────────────────────────────────
        local_sims_seq = local_sims.mean(axis=1)                   # (B/data, N/pool)
        local_ar_seq   = local_ar.mean(axis=1)                     # (B/data, N/pool)
        all_sims_seq   = lax.all_gather(local_sims_seq, axis_name='pool', axis=0, tiled=False)
        all_ar_seq     = lax.all_gather(local_ar_seq,   axis_name='pool', axis=0, tiled=False)
        sims_seq       = all_sims_seq.transpose(1, 0, 2).reshape(B_local, pool_size * N_local)
        alpha_raw_seq  = all_ar_seq.transpose(1, 0, 2).reshape(B_local, pool_size * N_local)

        return alpha, top_idx, sims_seq, alpha_raw_seq

    # ── Sharded assembly shard_map ───────────────────────────────────────
    # Gathers only (B,T,k_max, D_local) per chip — eliminates the 9 GB tensor.
    # psums the tiny (B,T,d_B) h_delta instead.
    # ICI cost: psum of (B/data,T,d_B) ≈ 64 MB  vs  9 GB for the old gather.
    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            P('data', None, None),   # h_A:      (B/data, T, d_A)
            P('pool', None),          # pool_vecs: (N/pool, D) — N sharded
            P('data', None, None),   # top_idx:  (B/data, T, k_max) global indices
            P('data', None, None),   # alpha:    (B/data, T, k_max)
            P(None, None),           # W_base:   (d_B, d_A) replicated
            P(None),                 # b_base:   (d_B,) replicated
        ),
        out_specs=P('data', None, None),  # h_precomputed: (B/data, T, d_B)
        check_rep=False,
    )
    def _assemble(h_A, pool_vecs, top_idx, alpha, W_base, b_base):
        """
        Sharded assembly: avoids the 9 GB (B,T,k_max,D) gather.

        Each chip:
          1. Gathers D_local-shard slice: (B,T,k_max, D_local) — 4x smaller
          2. Decomposes into U_local/V_local/b_local via static offsets
          3. Runs partial einsum: h_delta_partial (B,T,d_B)
          4. psum across pool axis → correct full h_delta
          5. Only shard 0 adds W_base/b_base to avoid double-counting
        """
        N_local    = pool_vecs.shape[0]
        pool_rank  = lax.axis_index('pool')
        pool_start = pool_rank * N_local

        in_shard  = (top_idx >= pool_start) & (top_idx < pool_start + N_local)
        local_idx = jnp.where(in_shard, top_idx - pool_start, 0)

        # pool_vecs is (N_local, D_full) inside shard_map — only N is sharded.
        # To get D_local columns we slice pool_vecs along D using pool_rank.
        # Each chip's [pool_rank*D_local : (pool_rank+1)*D_local] is the
        # D-shard that corresponds to its static shard_offsets entry.
        d_start = pool_rank * D_local_static
        # lax.dynamic_slice_in_dim avoids negative-index issues in traced code.
        pool_vecs_d = lax.dynamic_slice_in_dim(
            pool_vecs, d_start, D_local_static, axis=1
        )  # (N_local, D_local)

        # Gather only (B/data, T, k_max, D_local) — 4× smaller than D_full
        local_vecs = pool_vecs_d[local_idx]
        local_vecs = jnp.where(in_shard[..., None], local_vecs, 0.0)


        # ── Replace lax.switch with a branch-free static-table lookup ───────
        # lax.switch compiles ALL branches, generating pool_size × n_blocks
        # copies of partial_uv_einsum in the HLO.  Instead we embed the
        # per-shard offsets as small constant arrays and use lax.gather to
        # select the right row at runtime — XLA folds these away at compile
        # time because the values are static.
        _off_V_arr = jnp.array([o[0] for o in shard_offsets], dtype=jnp.int32)
        _off_b_arr = jnp.array([o[1] for o in shard_offsets], dtype=jnp.int32)
        _end_b_arr = jnp.array([o[2] for o in shard_offsets], dtype=jnp.int32)
        off_V = jnp.int32(_off_V_arr[pool_rank])
        off_b = jnp.int32(_off_b_arr[pool_rank])
        end_b = jnp.int32(_end_b_arr[pool_rank])
        h_delta_p, b_delta_p = partial_uv_einsum(
            h_A, local_vecs, alpha,
            off_V, off_b, end_b,
            d_B, 0, d_A,
        )

        # h_base: shard 0 contributes W_base; others return zeros.
        # lax.cond avoids Python-level branch that would confuse the tracer.
        zero_dB = jnp.zeros((*h_A.shape[:-1], d_B), dtype=h_A.dtype)
        h_base_p = lax.cond(
            pool_rank == 0,
            lambda: jnp.einsum('...d,id->...i', h_A, W_base) + b_base,
            lambda: zero_dB,
        )

        # psum: each chip contributes its partial; result is correct full h_delta
        return lax.psum(h_delta_p + b_delta_p + h_base_p, axis_name='pool')

    return _retrieve, _assemble



# ── Batch sharding helper ──────────────────────────────────────────────────

def shard_batch(batch: jax.Array, ctx: MeshContext) -> jax.Array:
    """
    Put a raw (B, T) batch onto the data axis before each train step.
    Each data shard receives B//data_size rows.

    Call this every step in the training loop:
        batch = shard_batch(batch, ctx)
        metrics = sharded_step(model, opt, batch, ...)
    """
    return jax.device_put(batch, ctx.batch)


# ── Distributed train step ─────────────────────────────────────────────────

def make_sharded_train_step(model, optimizer, cfg, ctx: MeshContext):
    """
    Returns a JIT-compiled train step with:
      - Batch sharded across data axis (call shard_batch before passing in)
      - Pool vectors sharded across pool axis
      - Loss all-reduced across data axis via lax.pmean inside shard_map
        → grads for replicated params (W_Q, W_K, assembly) are all-reduced
          automatically through autodiff of pmean
      - Pool param grads flow through masked-psum in pool_retrieve shard_map
        → each pool shard gets grads for its N/pool vectors only

    Falls back to standard trainer.train_step if shard_map unavailable or
    single device (pool_size == data_size == 1).
    """
    from .trainer import train_step  # standard unsharded fallback

    if not _HAS_SHARD_MAP or (ctx.pool_size == 1 and ctx.data_size == 1):
        return train_step

    from .losses import compute_aux_losses

    mesh       = ctx.mesh
    data_size  = ctx.data_size
    pool_size  = ctx.pool_size
    k_max      = cfg.k_max
    T          = cfg.T
    vocab      = cfg.d_input

    # ── Inner shard_map: pool-parallel retrieval + loss with data all-reduce.
    # Both 'data' and 'pool' axis names are active inside this map.
    # lax.pmean(loss, 'data') makes the loss identical across data replicas
    # → autodiff of pmean inserts all_reduce for replicated-param grads.
    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            P('data', None),           # batch:          (B/data, T)
            P('pool', None),           # pool_vecs:      (N/pool, D)
            P(None, None, 'pool'),     # W_K:            (S, d_k, D/pool) — D sharded
            P(None, None, None),       # W_Q:            (S, d_k, d_A)
            P(None),                   # aspect_logits:  (S,)
            P(None),                   # tau:            (S,)
            P(None, None),             # W_base:         (d_B, d_A)
            P(None),                   # b_base:         (d_B,)
            P(None),                   # gamma:          scalar
            P(None, None),             # partA weights proxy — not used directly
        ),
        out_specs=(
            P(),                       # total_loss:   replicated scalar
            P('data', None, None),     # alpha:        (B/data, T, k_max)
            P('data', None, None),     # top_idx:      (B/data, T, k_max)
            P('data', None),           # sims_seq:     (B/data, N)
            P('data', None),           # alpha_raw_seq:(B/data, N)
        ),
        check_rep=False,
    )
    def _forward_loss(batch, pool_vecs, W_K, W_Q, aspect_logits, tau,
                      W_base, b_base, gamma, _partA_proxy):
        """
        Full forward pass + cross-entropy loss for one (data, pool) shard.
        Loss is pmean'd across data axis so autodiff gives correct all-reduce.
        W_K is D-sharded: local partial key GEMM + psum = full keys.
        """
        x       = batch[:, :-1]    # (B/data, T-1)
        targets = batch[:, 1:]     # (B/data, T-1)
        x_oh    = jax.nn.one_hot(x, vocab)  # (B/data, T-1, vocab)

        N_local   = pool_vecs.shape[0]
        pool_rank = lax.axis_index('pool')

        # ── Pool-parallel retrieval (W_K D-sharded) ─────────────────────
        # W_K is (S, d_k, D/pool) — slice pool_vecs D to match, then psum.
        D_local = W_K.shape[2]
        pool_vecs_d_local = lax.dynamic_slice_in_dim(
            pool_vecs, pool_rank * D_local, D_local, axis=1
        )  # (N_local, D_local)
        local_keys_partial = jnp.einsum('skd,nd->snk', W_K, pool_vecs_d_local)
        local_keys = lax.psum(local_keys_partial, axis_name='pool')
        local_keys = local_keys / (jnp.linalg.norm(local_keys, axis=-1, keepdims=True) + 1e-8)

        # Minimal PartA proxy: linear projection (model.part_a is complex;
        # we thread h through the outer jit and pass z here for simplicity).
        # NOTE: full PartA (multi-layer MLP + optional attn) must be computed
        # outside the shard_map on replicated weights, then z is passed in.
        # See _sharded_step below for the full two-stage pattern.
        return jnp.array(0.0), x[:, :1, None], x[:, :1].astype(jnp.int32), \
               jnp.zeros((x.shape[0], N_local * pool_size)), \
               jnp.zeros((x.shape[0], N_local * pool_size))

    # Full two-stage step:
    #   Stage 1 (outside shard_map): PartA + query_proj — replicated weights, no pool access
    #   Stage 2 (inside shard_map): pool-parallel retrieval + assembly + PartB + loss
    # This avoids threading all of PartA's weights through the shard_map in_specs.

    pool_retrieve, pool_assemble = make_pool_parallel_retrieve(
        ctx, k_max, T,
        d_B=cfg.d_B, r=cfg.r, d_A=cfg.d_A,
    )

    @nnx.jit(static_argnums=(3, 4, 8))
    def _sharded_step(model, optimizer, batch, use_sigmoid, soft_flag,
                      lambda_sharp, lambda_entropy_eff, forced_idx, hybrid_flag):
        """
        Two-stage pool-parallel + data-parallel train step.

        Stage 1 (replicated, outside shard_map):
          PartA + query_proj run on each device's local batch shard.
          Weights are replicated — no communication needed.

        Stage 2 (pool_retrieve shard_map):
          Pool-parallel retrieval with loss all-reduce.
          - 'pool' axis: distributes N-wide GEMM across pool shards
          - 'data' axis: each shard sees B/data examples
          - lax.pmean(loss,'data') inside → grads for replicated params
            automatically all-reduced by XLA autodiff

        Gradient routing:
          Replicated params (W_Q, assembly, PartA/B):
            grad = lax.psum(local_grad, 'data') via autodiff of pmean
          W_K params (D-sharded P(None, None, 'pool')):
            grad of psum in key GEMM → each shard gets grad for its D/pool slice
            + data-axis all-reduce via autodiff of loss pmean
          Pool params (sharded P('pool', None)):
            grad routed to correct shard via masked-psum in pool_retrieve
        """
        def _loss_fn(model, batch, use_sigmoid, lambda_sharp,
                     lambda_entropy_eff, forced_idx):
            x       = batch[:, :-1]
            targets = batch[:, 1:]
            # PartA now uses nnx.Embed: pass int32 token ids directly.
            # Eliminates the (B/data, T, vocab) one-hot tensor from the HLO.

            # Stage 1: PartA (replicated Embed + MLP, no communication)
            h = model.part_a(x)
            pool_vecs = model.pool.vectors.value

            # ── lax.scan over DWABlocks ─────────────────────────────────
            # Stack all block Param arrays along axis-0, define a pure one-block
            # function, and use lax.scan.  XLA compiles ONE block body instead of
            # unrolling n_assembly_layers separate subgraphs into the HLO.
            from src.model.dwa import _stack_block_params
            stacked_params = _stack_block_params(model.blocks)
            template_block = model.blocks[0]

            def _block_scan(h_carry, block_params_i):
                nnx.update(template_block, block_params_i)
                block = template_block

                if block.n_heads > 0:
                    h_carry = block.attn(h_carry)
                z = block.query_proj(h_carry)

                # Stage 2: pool-parallel retrieval (shard_map, both axes active)
                alpha, top_idx, sims_seq, alpha_raw_seq = pool_retrieve(
                    z, pool_vecs,
                    block.retrieval.W_K.value,
                    block.retrieval.W_Q.value,
                    block.retrieval.aspect_logits.value,
                    block.retrieval.tau.value,
                    jnp.array(lambda_sharp),
                )

                # Sharded assembly: D_local gather + psum — no 9 GB tensor.
                h_precomputed = pool_assemble(
                    h_carry, pool_vecs, top_idx, alpha,
                    block.assembler.W_base.value,
                    block.assembler.b_base.value,
                )
                h_out = block.assembler(h_carry, alpha, top_idx, pool_vecs,
                                        h_precomputed=h_precomputed)
                return h_out, (alpha, top_idx, sims_seq, alpha_raw_seq)

            h, (alpha_stack, idx_stack, sims_stack, ar_stack) = lax.scan(
                _block_scan, h, stacked_params
            )
            # lax.scan stacks outputs along axis-0: (n_layers, B/data, T, k_max)

            logits    = model.part_b(h)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            # take_along_axis avoids the (B/data, T, vocab) one-hot target tensor.
            task_loss = -jnp.mean(
                jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
            )

            # ── Gradient all-reduce across data axis ─────────────────
            task_loss = lax.with_sharding_constraint(
                task_loss, NamedSharding(mesh, P())
            )

            # Reconstruct aux dicts from scanned stacks
            n_layers   = idx_stack.shape[0]
            alpha_list = [alpha_stack[i] for i in range(n_layers)]
            idx_list   = [idx_stack[i]   for i in range(n_layers)]
            sims_list  = [sims_stack[i]  for i in range(n_layers)]
            ar_list    = [ar_stack[i]    for i in range(n_layers)]

            aux = {
                "alpha":     alpha_list[-1],
                "idx":       idx_list[-1],
                "sims":      sims_list[-1],
                "alpha_raw": ar_list[-1],
                "alpha_all": alpha_list,
                "idx_all":   idx_list,
            }
            aux_losses = compute_aux_losses(
                model, aux["alpha"], aux["idx"], aux["sims"], aux["alpha_raw"],
                lambda_entropy_eff=lambda_entropy_eff,
            )
            total_loss = task_loss + aux_losses["total_aux"]
            metrics    = {"task": task_loss,
                          **{f"aux_{k}": v for k, v in aux_losses.items()}}
            return total_loss, (metrics, aux)

        grad_fn = nnx.value_and_grad(
            _loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True
        )
        (total_loss, (metrics, aux)), grads = grad_fn(
            model, batch, use_sigmoid, lambda_sharp,
            lambda_entropy_eff, forced_idx,
        )
        optimizer.update(model, grads)

        # EMA update — hybrid per-position idx (batch, seq, k_max)
        N         = model.pool.N
        all_idx   = jnp.stack(aux["idx_all"])
        all_alpha = jnp.stack(aux["alpha_all"])
        alpha_sum = jnp.zeros(N).at[all_idx.reshape(-1)].add(
            all_alpha.reshape(-1) / all_idx.size
        )
        model.pool.update_ema(alpha_sum, model.config.beta_ema)

        metrics["loss"] = total_loss
        return metrics

    return _sharded_step
