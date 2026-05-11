import jax
import jax.numpy as jnp
from flax import nnx


class Buffer(nnx.Variable):
    """Non-trainable static buffer (RoPE freqs, causal masks, etc.).

    Registered as an nnx.Variable so NNX passes it as a traced input
    to @nnx.jit rather than closing over a concrete JAX array.  This
    prevents XLA from embedding large constant tensors (e.g. a 2048×2048
    causal mask) directly into the HLO graph, which bloats compile time
    and triggers re-compilation whenever the Python closure changes.

    Buffer values are intentionally excluded from nnx.Param scans and
    optimizer state by virtue of not being nnx.Param subclasses.
    """
    pass


# ─── RoPE helpers ─────────────────────────────────────────────────────────────

def _rope_freqs(d_head: int, max_seq: int) -> jax.Array:
    """(max_seq, d_head/2) rotation frequencies."""
    theta = 1.0 / (10000.0 ** (jnp.arange(0, d_head, 2) / d_head))
    pos   = jnp.arange(max_seq)
    return jnp.outer(pos, theta)   # (max_seq, d_head/2)


def _apply_rope(x: jax.Array, freqs: jax.Array) -> jax.Array:
    """x: (..., seq, d_head)  freqs: (max_seq, d_head/2)"""
    seq = x.shape[-2]
    f   = freqs[:seq]                    # (seq, d_head/2)
    x1, x2 = x[..., ::2], x[..., 1::2]  # even / odd dims
    rot = jnp.concatenate(
        [x1 * jnp.cos(f) - x2 * jnp.sin(f),
         x1 * jnp.sin(f) + x2 * jnp.cos(f)], axis=-1
    )
    return rot


# ─── Causal Self-Attention ────────────────────────────────────────────────────

class CausalSelfAttention(nnx.Module):
    """1-layer causal MHA with RoPE — gives PartA cross-position context."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 256, rngs: nnx.Rngs = None):
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5

        self.qkv  = nnx.Linear(d_model, 3 * d_model, use_bias=False, rngs=rngs)
        self.proj = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.norm = nnx.LayerNorm(d_model, rngs=rngs)

        # Store as Buffer (nnx.Variable subclass) so NNX passes them as
        # traced inputs to @nnx.jit instead of closing over concrete arrays.
        # This removes 2×12 = 24 large HLO constants from the compiled graph.
        self._freqs = Buffer(_rope_freqs(self.d_head, max_seq_len))
        self._mask  = Buffer(jnp.tril(jnp.ones((max_seq_len, max_seq_len), dtype=bool)))

    def __call__(self, x: jax.Array) -> jax.Array:
        """x: (batch, seq, d_model) → (batch, seq, d_model)"""
        B, T, C = x.shape
        H, D    = self.n_heads, self.d_head

        qkv = self.qkv(self.norm(x))                    # (B, T, 3C)
        q, k, v = jnp.split(qkv, 3, axis=-1)            # each (B, T, C)
        q = q.reshape(B, T, H, D).transpose(0, 2, 1, 3) # (B, H, T, D)
        k = k.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, D).transpose(0, 2, 1, 3)

        # RoPE on Q and K
        q = _apply_rope(q, self._freqs.value)
        k = _apply_rope(k, self._freqs.value)

        # Scaled dot-product with causal mask (mask pre-built in __init__)
        attn = jnp.einsum('bhsd,bhtd->bhst', q, k) * self.scale
        attn = jnp.where(self._mask.value[:T, :T], attn, -1e9)
        attn = jax.nn.softmax(attn, axis=-1)

        out = jnp.einsum('bhst,bhtd->bhsd', attn, v)    # (B, H, T, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return x + self.proj(out)                        # residual


# ─── PartA ────────────────────────────────────────────────────────────────────

class PartA(nnx.Module):
    """
    Encoder half: maps integer token ids → h_A (d_A) via embedding + MLP.

    Uses nnx.Embed for the first projection instead of Linear + one_hot.
    This eliminates the (B, T, d_input) one-hot tensor (e.g. 2 GB at
    vocab=65000, B=16, T=512) from the HLO graph entirely.
    """

    def __init__(self, d_input: int, d_A: int, n_heads: int = 0,
                 max_seq_len: int = 256, rngs: nnx.Rngs = None):
        hidden = d_A * 4
        # Embed replaces norm_in + fc1: token id → hidden directly.
        # Same parameter count as Linear(d_input, hidden, use_bias=False).
        self.embed    = nnx.Embed(num_embeddings=d_input, features=hidden, rngs=rngs)
        self.fc2      = nnx.Linear(hidden, d_A, rngs=rngs)
        self.norm_out = nnx.LayerNorm(d_A, rngs=rngs)

        self.attn = (CausalSelfAttention(d_A, n_heads, max_seq_len, rngs)
                     if n_heads > 0 else None)

    def __call__(self, x: jax.Array):
        """
        x: (batch, seq) int32 token ids
        Returns h_A (batch, seq, d_A).
        """
        h   = jax.nn.gelu(self.embed(x))     # (B, T, hidden)
        h_A = self.norm_out(self.fc2(h))

        if self.attn is not None and h_A.ndim == 3:
            h_A = self.attn(h_A)

        return h_A


# ─── PartB ────────────────────────────────────────────────────────────────────

class PartB(nnx.Module):
    """
    Decoder half: maps h_mid (d_B) → output logits (d_output).
    2-layer MLP with GELU + pre-norm.
    """

    def __init__(self, d_B: int, d_output: int, rngs: nnx.Rngs):
        hidden = d_B * 4
        self.norm_in = nnx.LayerNorm(d_B, rngs=rngs)
        self.fc1 = nnx.Linear(d_B, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, d_output, rngs=rngs)

    def __call__(self, h_mid: jax.Array) -> jax.Array:
        h = jax.nn.gelu(self.fc1(self.norm_in(h_mid)))
        return self.fc2(h)
