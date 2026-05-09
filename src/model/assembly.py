import jax
import jax.numpy as jnp
from flax import nnx


# ── Shard-aware assembly helpers ───────────────────────────────────────────

def partial_uv_einsum(
    h_A:           jax.Array,  # (B, T, d_A)
    local_vecs:    jax.Array,  # (B, T, k_max, D_local)
    alpha:         jax.Array,  # (B, T, k_max)
    off_V_local:   int,        # start of V block in local shard
    off_b_local:   int,        # start of b block in local shard
    end_b_local:   int,        # end of b block in local shard
    d_B:           int,
    r_local:       int,        # number of rank columns on this shard
    d_A:           int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Compute partial h_delta, b_delta, and h_base contributions from a
    local D-shard of the selected pool vectors.

    The full D vector is split as  [U | V | b]  where
        U  :  d_B × r  floats  (r = total rank, e.g. 8)
        V  :  r  × d_A floats
        b  :  d_B       floats

    Across pool_size shards each shard receives exactly D_local = D/pool_size
    contiguous elements.  However U, V, b together may not align to pool shard
    boundaries.  To handle the general case each shard clips its view to the
    [off_V_local, off_b_local, end_b_local] offsets that the caller has
    pre-computed for it.

    This function is designed to be called inside shard_map — its outputs are
    (B_local, T, d_B) tensors that the caller psums across the pool axis to
    get the correct full-rank result.

    Returns:
        h_delta_partial : (B, T, d_B)  — partial weighted low-rank product
        b_delta_partial : (B, T, d_B)  — partial weighted bias sum
        (Caller must psum both across pool axis, then add W_base / b_base.)
    """
    # Split local shard into U / V / b portions (may have zero length)
    U_local = local_vecs[..., :off_V_local]            # (B,T,k,  d_B*r_U_local)
    V_local = local_vecs[..., off_V_local:off_b_local] # (B,T,k,  r_V_local*d_A)
    b_local = local_vecs[..., off_b_local:end_b_local] # (B,T,k,  d_B_b_local)

    # Reshape factors (trailing dims may be 0 — einsums on empty axes are 0)
    r_U = U_local.shape[-1] // d_B if d_B > 0 else 0   # rank cols in U shard
    r_V = V_local.shape[-1] // d_A if d_A > 0 else 0   # rank rows in V shard

    # Partial h_delta: sum_k alpha_k * (h_A @ V_k.T) @ U_k.T
    # Only non-zero when this shard contains V columns.
    if r_V > 0:
        V_r = V_local.reshape(*local_vecs.shape[:-1], r_V, d_A)  # (...,k,r_V,d_A)
        hV  = jnp.einsum('...d,...krd->...kr', h_A, V_r)         # (...,k,r_V)
        # Weight by alpha
        hVa = hV * alpha[..., None]                               # (...,k,r_V)
    else:
        hVa = jnp.zeros((*h_A.shape[:-1], local_vecs.shape[-3], 0), dtype=h_A.dtype)

    # Partial h_delta from U shard
    if r_U > 0:
        U_r = U_local.reshape(*local_vecs.shape[:-1], d_B, r_U)  # (...,k,d_B,r_U)
        # We need h_A @ V.T columns that correspond to this U shard's rank range
        # Because U and V may not align on the same shard, h_delta_partial is the
        # partial sum of (hVa @ U) for the rank columns present on this shard.
        # When U and V land on different shards this term will be zero on one side
        # and the psum will still yield the correct result.
        h_delta_partial = jnp.einsum('...kr,...kir->...i', hVa, U_r)  # (...,d_B)
    else:
        h_delta_partial = jnp.zeros((*h_A.shape[:-1], d_B), dtype=h_A.dtype)

    # Partial b_delta
    if b_local.shape[-1] > 0:
        b_delta_partial = jnp.einsum('...k,...ki->...i', alpha, b_local)  # (...,d_B)
    else:
        b_delta_partial = jnp.zeros((*h_A.shape[:-1], d_B), dtype=h_A.dtype)

    return h_delta_partial, b_delta_partial


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

    def partial_shard_offsets(self, D_local: int, shard_rank: int) -> tuple[int, int, int, int]:
        """
        Compute [off_V, off_b, end_b] offsets within a local D-shard.

        The full vector layout is:  [U (d_B*r) | V (r*d_A) | b (d_B)]
        With pool_size shards each shard holds D_local = D/pool_size
        contiguous elements starting at global offset shard_rank*D_local.

        Returns (off_V_local, off_b_local, end_b_local, d_B_b_local) where:
          off_V_local  — start of V block inside this local shard (clamped to [0,D_local])
          off_b_local  — start of b block inside this local shard
          end_b_local  — end   of b block inside this local shard
        All values are Python ints (static at trace time).
        """
        g_start = shard_rank * D_local
        g_end   = g_start + D_local

        # Global boundaries of U, V, b regions
        g_off_V = self._off_V
        g_off_b = self._off_b
        g_end_b = self._end_b

        # Clamp global boundaries to this shard's local window
        off_V_local  = max(0, min(D_local, g_off_V - g_start))
        off_b_local  = max(0, min(D_local, g_off_b - g_start))
        end_b_local  = max(0, min(D_local, g_end_b - g_start))

        return off_V_local, off_b_local, end_b_local

    def __call__(
        self,
        h_A: jax.Array,                        # (batch, [seq,] d_A)
        alpha: jax.Array,                       # (batch, [seq,] k_max)
        idx: jax.Array,                         # (batch, k_max) or (batch, seq, k_max)
        vectors: jax.Array,                     # (N, D)
        pre_gathered_vecs: jax.Array | None = None,  # (batch, seq, k_max, D) — pool-parallel
        h_precomputed: jax.Array | None = None,      # (batch, seq, d_B)       — sharded-assembly
    ) -> jax.Array:
        """Returns h_mid (batch, [seq,] d_B) after middle layer."""
        gamma    = self.gamma.value
        residual = h_A if self.d_A == self.d_B else jnp.zeros((*h_A.shape[:-1], self.d_B))

        # ── Sharded-assembly fast path ────────────────────────────────────
        # pool_assemble already ran inside shard_map and psummed
        # (h_delta + b_delta + h_base) into a (B,T,d_B) tensor.  Skip the
        # expensive gather + einsum entirely — just apply norm + gamma.
        if h_precomputed is not None:
            return self.norm(residual + gamma * h_precomputed)

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
            # ── Hard / Hybrid / Pool-parallel mode ───────────────────────────
            # pre_gathered_vecs: provided by pool-parallel shard_map (avoids
            #   a second distributed gather from the sharded pool).
            # Otherwise: standard gather from the (possibly replicated) pool.
            if pre_gathered_vecs is not None:
                selected = pre_gathered_vecs   # (batch, seq, k_max, D) — already fetched
            else:
                selected = vectors[idx]   # (batch, k_max, D) or (batch, seq, k_max, D)
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
