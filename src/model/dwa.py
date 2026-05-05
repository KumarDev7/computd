import jax
import jax.numpy as jnp
from flax import nnx

from .parts import PartA, PartB, CausalSelfAttention
from .pool import VectorPool
from .retrieval import MultiAspectRetrieval
from .assembly import WeightAssembler


def _retrieve_per_position(z, retrieval_module, pool_vectors, k_max, T, lambda_sharp,
                            use_sigmoid, forced_idx):
    """
    Wraps MultiAspectRetrieval to handle per-position (3D) inputs.

    If z is 3D (batch, seq, d_A): flattens to (batch*seq, d_A), runs retrieval,
    reshapes back. forced_idx is cyclically rotated per position so that across
    seq positions the full pool is covered.

    Returns alpha, idx, sims, alpha_raw with shape (batch, seq, ...) when 3D input,
    or (batch, ...) when 2D input.
    """
    if z.ndim == 3:
        batch, seq, d_A = z.shape
        N = pool_vectors.shape[0]
        z_flat = z.reshape(batch * seq, d_A)

        if forced_idx is not None:
            # Per-position cyclic rotation using repeat+tile (avoids XLA layout issues).
            # Each position t gets forced_idx + t (mod N) so the full pool is
            # covered in one phase-1 step (seq=64 × k_max=8 = 512 = N).
            repeated = jnp.repeat(forced_idx, seq, axis=0)               # (batch*seq, k_max)
            offsets  = jnp.tile(jnp.arange(seq, dtype=jnp.int32), batch)[:, None]  # (batch*seq, 1)
            forced_flat = (repeated + offsets) % N
        else:
            forced_flat = None

        alpha_f, idx_f, sims_f, alpha_raw_f = retrieval_module(
            z=z_flat,
            vectors=pool_vectors,
            k_max=k_max,
            T=T,
            lambda_sharp=lambda_sharp,
            use_sigmoid=use_sigmoid,
            forced_idx=forced_flat,
        )

        alpha     = alpha_f.reshape(batch, seq, k_max)
        idx       = idx_f.reshape(batch, seq, k_max)
        sims      = sims_f.reshape(batch, seq, N)
        alpha_raw = alpha_raw_f.reshape(batch, seq, N)
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
                 n_heads: int, max_seq_len: int, rngs: nnx.Rngs):
        self.n_heads = n_heads
        if n_heads > 0:
            self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, rngs)
        self.query_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.retrieval  = MultiAspectRetrieval(D=D, d_A=d_model, S=S, d_k=d_k, N=N, rngs=rngs)
        self.assembler  = WeightAssembler(d_model, d_model, r, rngs=rngs)

    def __call__(self, h, pool_vectors, k_max, T, lambda_sharp, use_sigmoid, forced_idx=None):
        if self.n_heads > 0:
            h = self.attn(h)
        z = self.query_proj(h)
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

    def __init__(self, config, rngs: nnx.Rngs):
        n_layers    = getattr(config, 'n_assembly_layers', 1)
        n_heads     = getattr(config, 'n_heads', 0)
        max_seq_len = getattr(config, 'max_seq_len', 256)

        self.config = config
        self.part_a = PartA(config.d_input, config.d_A, n_heads=0, rngs=rngs)  # MLP only
        self.pool   = VectorPool(config.N, config.D, rngs=rngs)
        self.blocks = nnx.List([
            DWABlock(config.d_A, config.r, config.S, config.d_k, config.N, config.D,
                     n_heads, max_seq_len, rngs)
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
    ):
        cfg       = self.config
        pool_vecs = self.pool.vectors.value

        h = self.part_a(x)  # returns h_A directly (no tuple)

        alpha_list, idx_list, sims_list, alpha_raw_list = [], [], [], []
        for block in self.blocks:
            h, alpha, idx, sims, alpha_raw = block(
                h, pool_vecs, cfg.k_max, cfg.T, lambda_sharp, use_sigmoid, forced_idx
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
