import jax
import jax.numpy as jnp
from flax import nnx


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

        # Pre-compute RoPE frequencies and causal mask (static — not parameters)
        self._freqs = _rope_freqs(self.d_head, max_seq_len)
        self._mask  = jnp.tril(jnp.ones((max_seq_len, max_seq_len), dtype=bool))

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
        q = _apply_rope(q, self._freqs)
        k = _apply_rope(k, self._freqs)

        # Scaled dot-product with causal mask (mask pre-built in __init__)
        attn = jnp.einsum('bhsd,bhtd->bhst', q, k) * self.scale
        attn = jnp.where(self._mask[:T, :T], attn, -1e9)
        attn = jax.nn.softmax(attn, axis=-1)

        out = jnp.einsum('bhst,bhtd->bhsd', attn, v)    # (B, H, T, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return x + self.proj(out)                        # residual


# ─── PartA ────────────────────────────────────────────────────────────────────

class PartA(nnx.Module):
    """
    Encoder half: maps input features → h_A (d_A) and produces per-position query z.
    2-layer MLP with GELU + pre-norm, plus optional causal self-attention (n_heads>0)
    for cross-position context, plus a projection head for z.
    """

    def __init__(self, d_input: int, d_A: int, n_heads: int = 0,
                 max_seq_len: int = 256, rngs: nnx.Rngs = None):
        hidden = d_A * 4
        self.norm_in  = nnx.LayerNorm(d_input, rngs=rngs)
        self.fc1      = nnx.Linear(d_input, hidden, rngs=rngs)
        self.fc2      = nnx.Linear(hidden, d_A, rngs=rngs)
        self.norm_out = nnx.LayerNorm(d_A, rngs=rngs)

        self.attn = (CausalSelfAttention(d_A, n_heads, max_seq_len, rngs)
                     if n_heads > 0 else None)

    def __call__(self, x: jax.Array):
        """
        x: (batch, [seq,] d_input) — token one-hots
        Returns h_A (batch, [seq,] d_A).
        """
        h   = jax.nn.gelu(self.fc1(self.norm_in(x)))
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
