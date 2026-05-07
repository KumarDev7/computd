import jax
import jax.numpy as jnp
from flax import nnx

from src.kernels.fused_retrieval import fused_weighted_similarity


class MultiAspectRetrieval(nnx.Module):
    """
    Multi-aspect sigmoid-gated retrieval over the vector pool.

    Phase 1 (use_sigmoid=False, forced_idx provided):
      - Use externally supplied forced_idx with uniform α = 1/k_max.
      - Every vector gets gradient over the course of phase 1 (rotating batches).
      - Similarity scores are still computed for aux losses and W_K gradient.

    Phase 2+ (use_sigmoid=True):
      - Normal similarity → sigmoid gate → top-k → normalized α.
      - Now retrieval has real content to discriminate between.
    """

    def __init__(self, D: int, d_A: int, S: int, d_k: int, N: int, rngs: nnx.Rngs):
        self.S = S
        self.d_k = d_k
        self.N = N

        self.W_Q = nnx.Param(
            jax.random.normal(rngs.params(), (S, d_k, d_A)) * (d_A ** -0.5)
        )
        self.W_K = nnx.Param(
            jax.random.normal(rngs.params(), (S, d_k, D)) * (D ** -0.5)
        )
        self.aspect_logits = nnx.Param(jnp.zeros(S))
        self.tau = nnx.Param(jnp.zeros(S))

    def __call__(
        self,
        z: jax.Array,              # (batch, d_A)
        vectors: jax.Array,        # (N, D)
        k_max: int,
        T: float = 1.0,
        lambda_sharp: float = 1.0,
        use_sigmoid: bool = True,
        forced_idx: jax.Array | None = None,  # (batch, k_max) — phase 1 only
    ):
        batch = z.shape[0]

        # Always compute full similarities — needed for aux losses and W_K gradients
        keys    = jnp.einsum('skd,nd->snk', self.W_K.value, vectors)
        keys    = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)
        queries = jnp.einsum('skd,bd->sbk', self.W_Q.value, z)
        queries = queries / (jnp.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)

        aspect_sims = jnp.einsum('sbk,snk->sbn', queries, keys)  # (S, batch, N)
        w    = jax.nn.softmax(self.aspect_logits.value)           # (S,)
        sims = jnp.einsum('s,sbn->bn', w, aspect_sims)           # (batch, N)

        if use_sigmoid:
            tau      = jnp.dot(w, self.tau.value)
            gate     = jax.nn.sigmoid(lambda_sharp * (sims - tau))
            alpha_raw = gate * jnp.exp(sims / T)
            top_vals, top_idx = jax.lax.top_k(alpha_raw, k_max)
            alpha = top_vals / (jnp.sum(top_vals, axis=-1, keepdims=True) + 1e-8)
            return alpha, top_idx, sims, alpha_raw
        else:
            alpha_raw = jnp.exp(sims / T)
            if forced_idx is not None:
                # Phase-1 warmup: use externally supplied rotation indices, uniform α
                idx   = forced_idx
                alpha = jnp.full((batch, k_max), 1.0 / k_max)
            else:
                # Phase-1 eval / fallback: plain softmax top-k (no sigmoid gate)
                top_vals, idx = jax.lax.top_k(alpha_raw, k_max)
                alpha = top_vals / (jnp.sum(top_vals, axis=-1, keepdims=True) + 1e-8)
            return alpha, idx, sims, alpha_raw

    def soft_forward(
        self,
        z: jax.Array,        # (batch, seq, d_A)
        vectors: jax.Array,  # (N, D)
        T: float = 1.0,
        lambda_sharp: float = 1.0,
        use_sigmoid: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Soft retrieval: alpha over ALL N vectors — no top-k, no gather, pure GEMMs.
        TPU-optimal: every op is a dense matmul the MXU can saturate.

        Returns:
          alpha:         (batch, seq, N) — per-position soft weights over all N
          sims_seq:      (batch, N)     — seq-mean sims  (for aux losses)
          alpha_raw_seq: (batch, N)     — seq-mean raw scores
        """
        # Keys for all N vectors: (S, N, d_k)
        keys = jnp.einsum('skd,nd->snk', self.W_K.value, vectors)
        keys = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)

        # Per-position queries: (S, batch, seq, d_k)
        queries = jnp.einsum('skd,btd->sbtk', self.W_Q.value, z)
        queries = queries / (jnp.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)

        # Fused: Σ_s w[s]*Q[s]@K[s].T → (batch, seq, N) without (S,B,T,N) peak.
        w      = jax.nn.softmax(self.aspect_logits.value)
        sims_w = fused_weighted_similarity(queries, keys, w)       # (batch, seq, N)

        if use_sigmoid:
            tau       = jnp.dot(w, self.tau.value)
            gate      = jax.nn.sigmoid(lambda_sharp * (sims_w - tau))
            alpha_raw = gate * jnp.exp(sims_w / T)
        else:
            alpha_raw = jnp.exp(sims_w / T)

        alpha = alpha_raw / (alpha_raw.sum(axis=-1, keepdims=True) + 1e-8)

        # Collapse seq dim for aux-loss compatibility (diversity_loss, entropy_loss)
        return alpha, sims_w.mean(axis=1), alpha_raw.mean(axis=1)

    def hybrid_forward(
        self,
        z: jax.Array,        # (batch, seq, d_A)
        vectors: jax.Array,  # (N, D)
        k_max: int,
        T: float = 1.0,
        lambda_sharp: float = 1.0,
        use_sigmoid: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Hybrid retrieval: compute ALL N dot products (GEMM-saturated, like soft)
        but keep only top-k for assembly (memory-efficient, like hard).

        TPU-optimal: the full similarity GEMM saturates the MXU, then top_k
        immediately discards the (B, seq, N) intermediate — only (B, seq, k_max)
        is materialized for assembly and backprop.

        Returns:
          alpha:         (batch, seq, k_max) — top-k weights (tiny, same as hard)
          top_idx:       (batch, seq, k_max) — indices of top-k vectors
          sims_seq:      (batch, N)          — seq-mean sims (for aux losses)
          alpha_raw_seq: (batch, N)          — seq-mean raw scores (for aux losses)
        """
        # Keys for all N vectors: (S, N, d_k)
        keys = jnp.einsum('skd,nd->snk', self.W_K.value, vectors)
        keys = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)

        # Per-position queries: (S, batch, seq, d_k)
        queries = jnp.einsum('skd,btd->sbtk', self.W_Q.value, z)
        queries = queries / (jnp.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)

        # Fused: Σ_s w[s]*Q[s]@K[s].T → (batch, seq, N) without (S,B,T,N) peak.
        # TPU: Pallas Mosaic kernel (tiles in VMEM).  GPU/CPU: per-aspect loop.
        w      = jax.nn.softmax(self.aspect_logits.value)
        sims_w = fused_weighted_similarity(queries, keys, w)       # (batch, seq, N)

        if use_sigmoid:
            tau       = jnp.dot(w, self.tau.value)
            gate      = jax.nn.sigmoid(lambda_sharp * (sims_w - tau))
            alpha_raw = gate * jnp.exp(sims_w / T)
        else:
            alpha_raw = jnp.exp(sims_w / T)

        # Top-k: full GEMM computed, but only keep k_max per position
        top_vals, top_idx = jax.lax.top_k(alpha_raw, k_max)       # (batch, seq, k_max)
        alpha = top_vals / (jnp.sum(top_vals, axis=-1, keepdims=True) + 1e-8)

        # Collapse seq dim for aux-loss compatibility
        return alpha, top_idx, sims_w.mean(axis=1), alpha_raw.mean(axis=1)

    def per_position_alpha(
        self,
        z: jax.Array,              # (batch, seq, d_A)
        selected_vecs: jax.Array,  # (batch, k_max, D) — already-gathered k vectors
    ) -> jax.Array:                # (batch, seq, k_max)
        """
        Per-position weights against the k pre-selected vectors.
        Cost: O(batch × seq × k × d_k) — no N-wide similarity scan.
        Called after sequence-level retrieval determines which k vectors to use.
        """
        # Keys for only the k selected vectors: (S, batch, k_max, d_k)
        keys = jnp.einsum('sdc,bkc->sbkd', self.W_K.value, selected_vecs)
        keys = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)
        # Per-position queries: (S, batch, seq, d_k)
        queries = jnp.einsum('sdc,btc->sbtd', self.W_Q.value, z)
        queries = queries / (jnp.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)
        # Similarities: (S, batch, seq, k_max) — no N dim!
        sims = jnp.einsum('sbtd,sbkd->sbtk', queries, keys)
        w    = jax.nn.softmax(self.aspect_logits.value)
        return jax.nn.softmax(jnp.einsum('s,sbtk->btk', w, sims), axis=-1)
