import jax
import jax.numpy as jnp
from flax import nnx

from .parts import PartA, PartB, CausalSelfAttention
from .pool import VectorPool
from .retrieval import MultiAspectRetrieval
from .assembly import WeightAssembler


def _retrieve_per_position(z, retrieval_module, pool_vectors, k_max, T, lambda_sharp,
                            use_sigmoid, forced_idx, hybrid=False):
    """
    Sequence-level retrieval (3D input) or direct retrieval (2D input).

    3D path — sequence-level retrieval (fast):
      • One retrieval query per sequence (mean of per-position z).
      • idx: (batch, k_max) — shared across all positions in the sequence.
      • Gather: (batch, k_max, D) — 256× smaller than the old per-position gather.
      • alpha: (batch, seq, k_max) — per-position weights computed cheaply against
        only the k selected vectors (no N-wide scan per position).

    2D path — direct retrieval, unchanged.

    hybrid=True — per-position retrieval with full GEMM compute + top-k keep.
      • alpha: (batch, seq, k_max), idx: (batch, seq, k_max)
      • Full similarity GEMM computed, only top-k kept for assembly.
    """
    if hybrid and z.ndim == 3:
        # Hybrid mode: full GEMM compute over all N, keep only top-k per position
        alpha, top_idx, sims, alpha_raw = retrieval_module.hybrid_forward(
            z, pool_vectors, k_max, T, lambda_sharp, use_sigmoid
        )
        return alpha, top_idx, sims, alpha_raw

    if z.ndim == 3:
        batch, seq, _ = z.shape

        # Sequence-level query: mean over positions → one retrieval per sequence
        z_seq = z.mean(axis=1)  # (batch, d_A)

        alpha_seq, idx, sims, alpha_raw = retrieval_module(
            z=z_seq,
            vectors=pool_vectors,
            k_max=k_max,
            T=T,
            lambda_sharp=lambda_sharp,
            use_sigmoid=use_sigmoid,
            forced_idx=forced_idx,   # (batch, k_max) — same shape, no expansion needed
        )
        # idx: (batch, k_max), sims: (batch, N)

        if use_sigmoid:
            # Per-position alpha: re-score each position against only the k retrieved vectors.
            # Tiny gather (batch, k_max, D) instead of (batch, seq, k_max, D).
            selected_vecs = pool_vectors[idx]                               # (batch, k_max, D)
            alpha = retrieval_module.per_position_alpha(z, selected_vecs)  # (batch, seq, k_max)
        else:
            # Phase-1 warmup: uniform alpha, broadcast across positions
            alpha = jnp.broadcast_to(alpha_seq[:, None, :], (batch, seq, k_max))

        return alpha, idx, sims, alpha_raw
    else:
        return retrieval_module(
            z=z,
            vectors=pool_vectors,
            k_max=k_max,
            T=T,
            lambda_sharp=lambda_sharp,
            use_sigmoid=use_sigmoid,
            forced_idx=forced_idx,
        )


class DWABlock(nnx.Module):
    """
    One DWA assembly block: optional causal attention → query projection →
    per-position retrieval → weight assembly.
    """

    def __init__(self, d_model: int, r: int, S: int, d_k: int, N: int, D: int,
                 n_heads: int, max_seq_len: int, rngs: nnx.Rngs, wk_sharding=None):
        self.n_heads = n_heads
        if n_heads > 0:
            self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, rngs)
        self.query_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.retrieval  = MultiAspectRetrieval(D=D, d_A=d_model, S=S, d_k=d_k, N=N, rngs=rngs,
                                               wk_sharding=wk_sharding)
        self.assembler  = WeightAssembler(d_model, d_model, r, rngs=rngs)

    def __call__(self, h, pool_vectors, k_max, T, lambda_sharp, use_sigmoid,
                 forced_idx=None, soft=False, hybrid=False, pallas=False,
                 tp_axis: str | None = None, mesh=None):
        if self.n_heads > 0:
            h = self.attn(h)
        z = self.query_proj(h)

        if soft and z.ndim == 3:
            # Soft mode: all N vectors, pure GEMMs, no gather, no top-k.
            alpha, sims, alpha_raw = self.retrieval.soft_forward(
                z, pool_vectors, T, lambda_sharp, use_sigmoid
            )
            idx = None  # no discrete selection in soft mode
        elif pallas and z.ndim == 3:
            # Pallas hybrid: fused GEMM+gate+exp Pallas kernel + multi-chip all_gather.
            alpha, idx, sims, alpha_raw = self.retrieval.pallas_hybrid_forward(
                z, pool_vectors, k_max, T, lambda_sharp, use_sigmoid, tp_axis=tp_axis, mesh=mesh
            )
        elif hybrid and z.ndim == 3:
            # Pure-JAX hybrid mode: full GEMM compute, keep only top-k.
            alpha, idx, sims, alpha_raw = _retrieve_per_position(
                z, self.retrieval, pool_vectors, k_max, T, lambda_sharp,
                use_sigmoid, forced_idx, hybrid=True
            )
        else:
            alpha, idx, sims, alpha_raw = _retrieve_per_position(
                z, self.retrieval, pool_vectors, k_max, T, lambda_sharp, use_sigmoid, forced_idx
            )

        h_out = self.assembler(h, alpha, idx, pool_vectors)
        return h_out, alpha, idx, sims, alpha_raw


class DWAModel(nnx.Module):
    """
    Dynamic Weight Assembly model with multi-block support.

    Architecture:
      x → PartA → h_A
          [DWABlock_0 → ... → DWABlock_n-1]  (each: query_proj → retrieval → assembly)
          h → PartB → logits
    """

    def __init__(self, config, rngs: nnx.Rngs, mesh=None):
        n_layers    = getattr(config, 'n_assembly_layers', 1)
        n_heads     = getattr(config, 'n_heads', 0)
        max_seq_len = getattr(config, 'max_seq_len', 256)

        self.config = config
        self._mesh  = mesh   # also set by shard_model() if constructed later

        # Pre-compute shardings if mesh is available — avoids OOM on init
        pool_sharding = None
        wk_sharding   = None
        if mesh is not None:
            from jax.sharding import NamedSharding, PartitionSpec as P
            pool_sharding = NamedSharding(mesh, P('tp', None))
            wk_sharding   = NamedSharding(mesh, P(None, None, 'tp'))

        self.part_a = PartA(config.d_input, config.d_A, n_heads=0, rngs=rngs)  # MLP only
        self.pool   = VectorPool(config.N, config.D, rngs=rngs, sharding=pool_sharding)
        self.blocks = nnx.List([
            DWABlock(config.d_A, config.r, config.S, config.d_k, config.N, config.D,
                     n_heads, max_seq_len, rngs, wk_sharding=wk_sharding)
            for _ in range(n_layers)
        ])
        self.part_b = PartB(config.d_B, config.d_input, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        use_sigmoid: bool = False,
        lambda_sharp: float = 1.0,
        return_aux: bool = False,
        forced_idx: jax.Array | None = None,  # (batch, k_max) for phase-1 warmup
        soft: bool = False,                    # True → soft dense pool (TPU training)
        hybrid: bool = False,                  # True → full JAX GEMM compute, top-k keep
        pallas: bool = False,                  # True → Pallas fused kernel + multi-chip TP
        tp_axis: str | None = None,            # mesh axis name for all_gather ('tp' or None)
        mesh=None,                             # jax.sharding.Mesh — captured, not traced
    ):
        cfg       = self.config
        pool_vecs = self.pool.vectors.value
        # Resolve mesh: explicit arg > stored _mesh > None
        mesh = mesh if mesh is not None else self._mesh

        h = self.part_a(x)  # returns h_A directly (no tuple)

        alpha_list, idx_list, sims_list, alpha_raw_list = [], [], [], []
        for block in self.blocks:
            h, alpha, idx, sims, alpha_raw = block(
                h, pool_vecs, cfg.k_max, cfg.T, lambda_sharp, use_sigmoid, forced_idx,
                soft=soft, hybrid=hybrid, pallas=pallas, tp_axis=tp_axis, mesh=mesh,
            )
            alpha_list.append(alpha)
            idx_list.append(idx)
            sims_list.append(sims)
            alpha_raw_list.append(alpha_raw)

        logits = self.part_b(h)

        if return_aux:
            return logits, {
                "alpha":     alpha_list[-1],       # last block (backward compat for losses)
                "idx":       idx_list[-1],
                "sims":      sims_list[-1],
                "alpha_raw": alpha_raw_list[-1],
                "alpha_all": alpha_list,            # all blocks (for EMA update)
                "idx_all":   idx_list,
                "h_A":       h,
            }
        return logits
