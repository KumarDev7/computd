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


# ── Initial state sharding ─────────────────────────────────────────────────

def shard_initial_state(model, ctx: MeshContext) -> None:
    """
    Move model parameters to their correct shards. Call once after model init.

    Pool vectors + EMA → pool axis (N sharded).
    All other params   → replicated (default JAX behaviour for single-device init).
    """
    model.pool.vectors.value   = jax.device_put(model.pool.vectors.value,   ctx.pool_vecs)
    model.pool.ema_usage.value = jax.device_put(model.pool.ema_usage.value, ctx.pool_ema)


# ── Pool-parallel hybrid retrieval ────────────────────────────────────────

def make_pool_parallel_retrieve(ctx: MeshContext, k_max: int, T: float):
    """
    Returns a pool-parallel retrieval function built for the given mesh.

    The returned function has the same signature as retrieval_module.hybrid_forward
    but uses shard_map to distribute the N-wide GEMM across pool shards.

    Requires JAX 0.4.14+ for shard_map.
    """
    if not _HAS_SHARD_MAP:
        raise RuntimeError(
            "shard_map not found — upgrade JAX to 0.4.14+:\n"
            "  pip install --upgrade jax jaxlib"
        )

    pool_size = ctx.pool_size
    mesh      = ctx.mesh

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            P('data', None, None),   # z:             (B/data, T, d_A)
            P('pool', None),          # pool_vecs:     (N/pool, D)
            P(None, None, None),      # W_K:           (S, d_k, D) — replicated
            P(None, None, None),      # W_Q:           (S, d_k, d_A) — replicated
            P(None),                  # aspect_logits: (S,) — replicated
            P(None),                  # tau:           (S,) — replicated
            P(None),                  # lambda_sharp:  scalar
        ),
        out_specs=(
            P('data', None, None),   # alpha:       (B/data, T, k_max)
            P('data', None, None),   # top_idx:     (B/data, T, k_max)  global indices
            P('data', None),         # sims_seq:    (B/data, N) — for aux losses
            P('data', None),         # alpha_raw_seq:(B/data, N)
            P('data', None, None, None),  # gathered_vecs: (B/data, T, k_max, D)
        ),
        check_rep=False,
    )
    def _retrieve(z, pool_vecs, W_K, W_Q, aspect_logits, tau, lambda_sharp):
        """
        Each program instance handles (B/data, T, N/pool) locally, then
        merges top-k across the pool axis via all_gather + top_k.
        """
        N_local   = pool_vecs.shape[0]
        pool_rank = lax.axis_index('pool')           # 0 … pool_size-1
        B_local, T, d_A = z.shape

        # ── Local keys for this pool shard ──────────────────────────────
        local_keys = jnp.einsum('skd,nd->snk', W_K, pool_vecs)   # (S, N/pool, d_k)
        local_keys = local_keys / (jnp.linalg.norm(local_keys, axis=-1, keepdims=True) + 1e-8)

        # ── Per-position queries ─────────────────────────────────────────
        queries = jnp.einsum('skd,btd->sbtk', W_Q, z)             # (S, B/data, T, d_k)
        queries = queries / (jnp.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)

        # ── Local similarities: (S, B/data, T, N/pool) → (B/data, T, N/pool)
        local_sims_s = jnp.einsum('sbtk,snk->sbtn', queries, local_keys)
        w            = jax.nn.softmax(aspect_logits)
        local_sims   = jnp.einsum('s,sbtn->btn', w, local_sims_s)  # (B/data, T, N/pool)

        # ── All-gather across pool axis → full (B/data, T, N) ───────────
        # all_gather axis=0 prepends a pool-size leading dim
        all_sims = lax.all_gather(local_sims, axis_name='pool', axis=0, tiled=False)
        # all_sims: (pool, B/data, T, N/pool) → (B/data, T, N)
        sims_full = all_sims.transpose(1, 2, 0, 3).reshape(B_local, T, pool_size * N_local)

        # ── Sigmoid gate + alpha_raw over full N ────────────────────────
        tau_scalar  = jnp.dot(w, tau)
        gate        = jax.nn.sigmoid(lambda_sharp * (sims_full - tau_scalar))
        alpha_raw   = gate * jnp.exp(sims_full / T)               # (B/data, T, N)

        # ── Global top-k ─────────────────────────────────────────────────
        top_vals, top_idx = jax.lax.top_k(alpha_raw, k_max)       # (B/data, T, k_max)
        alpha = top_vals / (jnp.sum(top_vals, axis=-1, keepdims=True) + 1e-8)

        # ── Distributed gather of pool vectors ────────────────────────────
        # Each shard contributes its local slice; masked-psum merges all shards.
        # Cost per step: k_max * D floats communicated — tiny even at 7B.
        pool_start    = pool_rank * N_local
        in_shard      = (top_idx >= pool_start) & (top_idx < pool_start + N_local)
        local_idx     = jnp.where(in_shard, top_idx - pool_start, 0)
        local_vecs    = pool_vecs[local_idx]                        # (B/data, T, k_max, D)
        local_vecs    = jnp.where(in_shard[..., None], local_vecs, 0.0)
        gathered_vecs = lax.psum(local_vecs, axis_name='pool')     # (B/data, T, k_max, D)

        # ── Seq-mean aux tensors ──────────────────────────────────────────
        sims_seq      = sims_full.mean(axis=1)                     # (B/data, N)
        alpha_raw_seq = alpha_raw.mean(axis=1)                     # (B/data, N)

        return alpha, top_idx, sims_seq, alpha_raw_seq, gathered_vecs

    return _retrieve


# ── Distributed train step ─────────────────────────────────────────────────

def make_sharded_train_step(model, optimizer, cfg, ctx: MeshContext):
    """
    Returns a JIT-compiled train step with batch sharded across data axis
    and pool vectors sharded across pool axis.

    The returned callable has the same signature as trainer.train_step.
    If shard_map is unavailable or pool_size == 1, falls back to unsharded step.
    """
    from .trainer import train_step  # standard unsharded step

    if not _HAS_SHARD_MAP or ctx.pool_size == 1:
        # Fallback: standard data-parallel via nnx.jit (pool replicated)
        return train_step

    # Build pool-parallel retrieval once (captures mesh + k_max + T)
    pool_retrieve = make_pool_parallel_retrieve(ctx, cfg.k_max, cfg.T)

    soft   = getattr(cfg, 'soft_train',   False)
    hybrid = getattr(cfg, 'hybrid_train', False)

    from .losses import compute_aux_losses
    import optax

    @nnx.jit(static_argnums=(3, 4, 8))
    def _sharded_step(model, optimizer, batch, use_sigmoid, soft_flag,
                      lambda_sharp, lambda_entropy_eff, forced_idx, hybrid_flag):
        """Pool-parallel train step. Mirrors trainer.train_step signature."""

        def _loss(model, batch, use_sigmoid, lambda_sharp, lambda_entropy_eff,
                  forced_idx, soft_flag, hybrid_flag):
            x       = batch[:, :-1]
            targets = batch[:, 1:]
            vocab   = model.config.d_input
            x_oh    = jax.nn.one_hot(x, vocab)

            # ── Part A ────────────────────────────────────────────────────
            h = model.part_a(x_oh)
            pool_vecs = model.pool.vectors.value

            alpha_list, idx_list, sims_list, alpha_raw_list = [], [], [], []

            for block in model.blocks:
                if block.n_heads > 0:
                    h = block.attn(h)
                z = block.query_proj(h)

                # Pool-parallel hybrid retrieval via shard_map
                alpha, top_idx, sims_seq, alpha_raw_seq, gathered_vecs = pool_retrieve(
                    z, pool_vecs,
                    block.retrieval.W_K.value,
                    block.retrieval.W_Q.value,
                    block.retrieval.aspect_logits.value,
                    block.retrieval.tau.value,
                    jnp.array(lambda_sharp),
                )

                # Assembly with pre-gathered vectors (no second gather from sharded pool)
                h = block.assembler(h, alpha, top_idx, pool_vecs,
                                    pre_gathered_vecs=gathered_vecs)

                alpha_list.append(alpha)
                idx_list.append(top_idx)
                sims_list.append(sims_seq)
                alpha_raw_list.append(alpha_raw_seq)

            logits    = model.part_b(h)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            task_loss = -jnp.mean(
                jnp.sum(jax.nn.one_hot(targets, vocab) * log_probs, axis=-1)
            )

            aux = {
                "alpha":      alpha_list[-1],
                "idx":        idx_list[-1],
                "sims":       sims_list[-1],
                "alpha_raw":  alpha_raw_list[-1],
                "alpha_all":  alpha_list,
                "idx_all":    idx_list,
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
            _loss, argnums=nnx.DiffState(0, nnx.Param), has_aux=True
        )
        (total_loss, (metrics, aux)), grads = grad_fn(
            model, batch, use_sigmoid, lambda_sharp, lambda_entropy_eff,
            forced_idx, soft_flag, hybrid_flag,
        )
        optimizer.update(model, grads)

        # EMA update — hybrid mode, per-position idx (batch, seq, k_max)
        N         = model.pool.N
        alpha_all = aux["alpha_all"]
        idx_all   = aux["idx_all"]
        all_idx   = jnp.stack(idx_all)
        all_alpha = jnp.stack(alpha_all)
        alpha_sum = jnp.zeros(N).at[all_idx.reshape(-1)].add(
            all_alpha.reshape(-1) / all_idx.size
        )
        model.pool.update_ema(alpha_sum, model.config.beta_ema)

        metrics["loss"] = total_loss
        return metrics

    return _sharded_step
