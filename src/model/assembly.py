import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P


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
        self.W_base = nnx.Param(jnp.zeros((d_B, d_A), dtype=jnp.bfloat16))
        self.b_base = nnx.Param(jnp.zeros(d_B, dtype=jnp.bfloat16))

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
        idx: jax.Array,     # (batch, k_max) seq-level  OR  (batch, seq, k_max) per-position
        vectors: jax.Array, # (N, D)
        mesh=None,
    ) -> jax.Array:
        """Returns h_mid (batch, [seq,] d_B) after middle layer."""
        gamma    = self.gamma.value
        residual = h_A if self.d_A == self.d_B else jnp.zeros((*h_A.shape[:-1], self.d_B))

        if idx is None:
            # ── Soft mode (TPU training): alpha is (batch, seq, N) over ALL N vectors.
            # No gather — decompose the pool in-place, all ops are GEMMs.
            N      = vectors.shape[0]
            U_pool = vectors[:, :self._off_V].reshape(N, self.d_B, self.r)
            V_pool = vectors[:, self._off_V:self._off_b].reshape(N, self.r, self.d_A)
            b_pool = vectors[:, self._off_b:self._end_b]             # (N, d_B)

            hV      = jnp.einsum('bsd,nrd->bsnr', h_A, V_pool)      # (b, s, N, r)
            hV_a    = hV * alpha[:, :, :, None]                       # (b, s, N, r)
            h_delta = jnp.einsum('bsnr,nir->bsi', hV_a, U_pool)      # (b, s, d_B)
            b_delta = jnp.einsum('bsn,nj->bsj', alpha, b_pool)       # (b, s, d_B)
            h_base  = jnp.einsum('bsd,id->bsi', h_A, self.W_base.value) + self.b_base.value
        else:
            # ── Hard / Hybrid mode: gather the selected vectors, then assemble.
            # Hard: idx is (batch, k_max) seq-level or (batch, seq, k_max) per-position.
            # Hybrid: idx is always (batch, seq, k_max) per-position.
            selected = vectors[idx]   # (batch, k_max, D)  or  (batch, seq, k_max, D)
            # Shard selected across TP to avoid materializing the full (B,S,K,D) on one chip.
            if mesh is not None and selected.ndim >= 3:
                ndim = selected.ndim
                spec = [None] * ndim
                spec[0] = 'tp'  # batch dimension
                selected = jax.lax.with_sharding_constraint(
                    selected, NamedSharding(mesh, P(*spec))
                )
            U      = selected[..., :self._off_V].reshape(*selected.shape[:-1], self.d_B, self.r)
            V      = selected[..., self._off_V:self._off_b].reshape(*selected.shape[:-1], self.r, self.d_A)
            b_vecs = selected[..., self._off_b:self._end_b]

            if idx.ndim == 2:
                # Sequence-level idx (hard mode): U/V/b have no seq dim; alpha has it.
                hV      = jnp.einsum('bsd,bkrd->bskr', h_A, V)
                h_delta = jnp.einsum('bskr,bkir->bsi', hV * alpha[:, :, :, None], U)
                b_delta = jnp.einsum('bsk,bkj->bsj', alpha, b_vecs)
                h_base  = jnp.einsum('bsd,id->bsi', h_A, self.W_base.value) + self.b_base.value
            else:
                # Per-position idx (hard per-pos or hybrid): all tensors share (batch, seq) leading dims.
                hV      = jnp.einsum('...d,...krd->...kr', h_A, V)
                h_delta = jnp.einsum('...kr,...kir->...i', hV * alpha[..., None], U)
                b_delta = jnp.einsum('...k,...kj->...j', alpha, b_vecs)
                h_base  = jnp.einsum('...d,id->...i', h_A, self.W_base.value) + self.b_base.value

        return self.norm(residual + gamma * (h_base + h_delta + b_delta))
