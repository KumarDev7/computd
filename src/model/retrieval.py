import jax
import jax.numpy as jnp
from flax import nnx


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
