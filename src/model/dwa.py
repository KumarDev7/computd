import jax
import jax.numpy as jnp
from jax import lax
from flax import nnx

from .parts import PartA, PartB, CausalSelfAttention
from .pool import VectorPool
from .retrieval import MultiAspectRetrieval
from .assembly import WeightAssembler


def _retrieve_per_position(z, retrieval_module, pool_vectors, k_max, T, lambda_sharp,
                            use_sigmoid, forced_idx, hybrid=False):
    """
    Sequence-level retrieval (3D input) or direct retrieval (2D input).
    (Unchanged — see original docstring.)
    """
    if hybrid and z.ndim == 3:
        alpha, top_idx, sims, alpha_raw = retrieval_module.hybrid_forward(
            z, pool_vectors, k_max, T, lambda_sharp, use_sigmoid
        )
        return alpha, top_idx, sims, alpha_raw

    if z.ndim == 3:
        batch, seq, _ = z.shape
        z_seq = z.mean(axis=1)
        alpha_seq, idx, sims, alpha_raw = retrieval_module(
            z=z_seq, vectors=pool_vectors, k_max=k_max, T=T,
            lambda_sharp=lambda_sharp, use_sigmoid=use_sigmoid, forced_idx=forced_idx,
        )
        if use_sigmoid:
            selected_vecs = pool_vectors[idx]
            alpha = retrieval_module.per_position_alpha(z, selected_vecs)
        else:
            alpha = jnp.broadcast_to(alpha_seq[:, None, :], (batch, seq, k_max))
        return alpha, idx, sims, alpha_raw
    else:
        return retrieval_module(
            z=z, vectors=pool_vectors, k_max=k_max, T=T,
            lambda_sharp=lambda_sharp, use_sigmoid=use_sigmoid, forced_idx=forced_idx,
        )


class DWABlock(nnx.Module):
    """One DWA assembly block: optional causal attention → query projection →
    per-position retrieval → weight assembly."""

    def __init__(self, d_model: int, r: int, S: int, d_k: int, N: int, D: int,
                 n_heads: int, max_seq_len: int, rngs: nnx.Rngs):
        self.n_heads = n_heads
        if n_heads > 0:
            self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, rngs)
        self.query_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.retrieval  = MultiAspectRetrieval(D=D, d_A=d_model, S=S, d_k=d_k, N=N, rngs=rngs)
        self.assembler  = WeightAssembler(d_model, d_model, r, rngs=rngs)

    def __call__(self, h, pool_vectors, k_max, T, lambda_sharp, use_sigmoid,
                 forced_idx=None, soft=False, hybrid=False, pre_gathered_vecs=None):
        if self.n_heads > 0:
            h = self.attn(h)
        z = self.query_proj(h)

        if soft and z.ndim == 3:
            alpha, sims, alpha_raw = self.retrieval.soft_forward(
                z, pool_vectors, T, lambda_sharp, use_sigmoid
            )
            idx = None
        elif hybrid and z.ndim == 3:
            alpha, idx, sims, alpha_raw = _retrieve_per_position(
                z, self.retrieval, pool_vectors, k_max, T, lambda_sharp,
                use_sigmoid, forced_idx, hybrid=True
            )
        else:
            alpha, idx, sims, alpha_raw = _retrieve_per_position(
                z, self.retrieval, pool_vectors, k_max, T, lambda_sharp, use_sigmoid, forced_idx
            )

        h_out = self.assembler(h, alpha, idx, pool_vectors,
                               pre_gathered_vecs=pre_gathered_vecs)
        return h_out, alpha, idx, sims, alpha_raw


# ── lax.scan helpers ──────────────────────────────────────────────────────────

def _stack_block_params(blocks):
    """Stack nnx.Param arrays from all blocks along a new leading axis.

    Returns stacked_params: a pytree of arrays with shape (n_blocks, ...)
    for use with lax.scan / nnx.scan.

    Buffer variables (_freqs, _mask) are identical across blocks and are
    NOT stacked — they stay as replicated constants.
    """
    all_param_states = [nnx.state(b, nnx.Param) for b in blocks]
    stacked = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0), *all_param_states
    )
    return stacked


class DWAModel(nnx.Module):
    """
    Dynamic Weight Assembly model with multi-block support.

    Architecture:
      x (int32 tokens) → PartA (embed+MLP) → h_A
          [DWABlock_0 → ... → DWABlock_n-1]  (each: query_proj → retrieval → assembly)
          h → PartB → logits

    Input: integer token ids, NOT one-hot vectors.
    PartA uses nnx.Embed internally to avoid materialising the (B,T,vocab)
    one-hot tensor in HBM.
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

        # ── Stacked module for nnx.scan ───────────────────────────────────
        # nnx.scan requires the module's Param arrays to have a leading axis.
        # We create this by vmapping the constructor over n_layers Rngs,
        # which produces a DWABlock whose every Param has shape (n_layers, ...).
        # nnx.scan then slices out one layer per step.
        #
        # After building self.blocks the normal way (for sharding / inspection),
        # we initialise stacked_blocks separately and copy the per-layer weights
        # into it, so both objects are numerically identical.
        if n_layers > 1:
            @nnx.split_rngs(splits=n_layers)
            @nnx.vmap(in_axes=0, out_axes=0)
            def _make_stacked(rngs_i: nnx.Rngs):
                return DWABlock(config.d_A, config.r, config.S, config.d_k,
                                config.N, config.D, n_heads, max_seq_len, rngs_i)

            self.stacked_blocks = _make_stacked(rngs)
            # Copy per-layer Params from self.blocks → self.stacked_blocks
            stacked_params = _stack_block_params(self.blocks)
            nnx.update(self.stacked_blocks, stacked_params)
        else:
            self.stacked_blocks = None  # single-layer: no scan needed

    def __call__(
        self,
        x: jax.Array,                          # (batch, seq) int32 token ids
        use_sigmoid: bool = False,
        lambda_sharp: float = 1.0,
        return_aux: bool = False,
        forced_idx: jax.Array | None = None,   # (batch, k_max) for phase-1 warmup
        soft: bool = False,
        hybrid: bool = False,
    ):
        cfg       = self.config
        pool_vecs = self.pool.vectors.value

        h = self.part_a(x)   # (batch, seq, d_A) — embed+MLP, no one_hot

        # ── lax.scan over blocks ──────────────────────────────────────────
        # Stack all block Param arrays along axis-0, then scan a pure function
        # over them.  XLA compiles a single block body and repeats it n_layers
        # times, reducing HLO graph size ~n_layers× vs the unrolled Python loop.
        n_layers = len(self.blocks)

        if self.stacked_blocks is None:
            # Single layer: no scan overhead
            h, alpha, idx, sims, alpha_raw = self.blocks[0](
                h, pool_vecs, cfg.k_max, cfg.T, lambda_sharp, use_sigmoid,
                forced_idx, soft=soft, hybrid=hybrid,
            )
            alpha_list    = [alpha]
            idx_list      = [idx]
            sims_list     = [sims]
            alpha_raw_list = [alpha_raw]
        else:
            # nnx.scan: compiles ONE block body, repeats it n_layers times
            # via lax.scan.  Reduces HLO graph size ~n_layers×.
            # The function must return (carry, output):
            #   carry  → h_carry   (passed to next step)
            #   output → per-step (alpha, idx, sims, ar) stacked along axis 0
            @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
            def _fwd(h_carry, block):
                h_out, alpha, idx, sims, ar = block(
                    h_carry, pool_vecs, cfg.k_max, cfg.T, lambda_sharp,
                    use_sigmoid, forced_idx, soft=soft, hybrid=hybrid,
                )
                return h_out, (alpha, idx, sims, ar)

            h, (alpha_stack, idx_stack, sims_stack, ar_stack) = _fwd(
                h, self.stacked_blocks
            )
            # Outputs are stacked (n_layers, ...) — convert to per-layer lists.
            alpha_list     = [alpha_stack[i]  for i in range(n_layers)]
            idx_list       = [idx_stack[i]    for i in range(n_layers)]
            sims_list      = [sims_stack[i]   for i in range(n_layers)]
            alpha_raw_list = [ar_stack[i]     for i in range(n_layers)]

        logits = self.part_b(h)

        if return_aux:
            return logits, {
                "alpha":     alpha_list[-1],
                "idx":       idx_list[-1],
                "sims":      sims_list[-1],
                "alpha_raw": alpha_raw_list[-1],
                "alpha_all": alpha_list,
                "idx_all":   idx_list,
                "h_A":       h,
            }
        return logits
