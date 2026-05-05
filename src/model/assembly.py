import jax
import jax.numpy as jnp
from flax import nnx


class WeightAssembler(nnx.Module):
    """
    Assembles the dynamic middle-layer weight matrix from retrieved pool vectors.

    Each vector v_i ∈ ℝ^D is split into:
      U_i ∈ ℝ^(d_B × r)   — left factor   [0 : d_B*r]
      V_i ∈ ℝ^(r × d_A)   — right factor  [d_B*r : d_B*r + r*d_A]
      b_i ∈ ℝ^d_B          — bias contrib  [d_B*r + r*d_A : d_B*r + r*d_A + d_B]

    Assembly:
      W = W_base + Σ_i α_i * (U_i @ V_i)
      b = b_base + Σ_i α_i * b_i

    Middle layer:
      h_mid = LayerNorm(h_A + γ * h_A @ W^T + b)

    Supports per-position retrieval: alpha/idx can be (batch, k_max) or
    (batch, seq, k_max) — einsums use ... notation throughout.
    """

    def __init__(self, d_A: int, d_B: int, r: int, rngs: nnx.Rngs):
        self.d_A = d_A
        self.d_B = d_B
        self.r = r

        # Offsets for vector slicing
        self._off_V = d_B * r
        self._off_b = d_B * r + r * d_A
        self._end_b = d_B * r + r * d_A + d_B

        # Base weight and bias — zero init forces model to learn from pool from the start
        self.W_base = nnx.Param(jnp.zeros((d_B, d_A)))
        self.b_base = nnx.Param(jnp.zeros(d_B))

        # LoRA-style residual scale — init 1.0 so pool contribution is immediately active
        self.gamma = nnx.Param(jnp.array(1.0))

        # LayerNorm after assembly
        self.norm = nnx.LayerNorm(d_B, rngs=rngs)

    def assemble(
        self,
        alpha: jax.Array,    # (..., k_max) — (batch, k_max) or (batch, seq, k_max)
        idx: jax.Array,      # (..., k_max)
        vectors: jax.Array,  # (N, D)
    ):
        """
        Returns W_assembled (..., d_B, d_A) and b_assembled (..., d_B).
        """
        d_B, d_A, r = self.d_B, self.d_A, self.r

        # Gather top-k vectors: (..., k_max, D)
        selected = vectors[idx]

        # Split into factors
        U = selected[..., : self._off_V].reshape(*selected.shape[:-1], d_B, r)
        V = selected[..., self._off_V : self._off_b].reshape(*selected.shape[:-1], r, d_A)
        b_vecs = selected[..., self._off_b : self._end_b]  # (..., k_max, d_B)

        # Low-rank products: (..., k_max, d_B, d_A)
        UV = jnp.einsum('...ir,...rj->...ij', U, V)

        # Weighted sum over k_max: (..., d_B, d_A)
        W_delta = jnp.einsum('...k,...kij->...ij', alpha, UV)
        b_delta = jnp.einsum('...k,...kj->...j', alpha, b_vecs)

        return self.W_base.value + W_delta, self.b_base.value + b_delta

    def __call__(
        self,
        h_A: jax.Array,     # (batch, [seq,] d_A)
        alpha: jax.Array,   # (batch, [seq,] k_max)
        idx: jax.Array,     # (batch, [seq,] k_max)
        vectors: jax.Array, # (N, D)
    ) -> jax.Array:
        """Returns h_mid (batch, [seq,] d_B) after middle layer."""
        W, b = self.assemble(alpha, idx, vectors)
        gamma = self.gamma.value

        # h_A: (..., d_A), W: (..., d_B, d_A) → (..., d_B)
        transformed = jnp.einsum('...d,...id->...i', h_A, W) + b

        residual = h_A if self.d_A == self.d_B else jnp.zeros((*h_A.shape[:-1], self.d_B))
        return self.norm(residual + gamma * transformed)
