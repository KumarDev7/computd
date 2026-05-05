"""
Standard GPT-style autoregressive LM for benchmarking against DWA.
Uses pre-norm transformer blocks with RoPE — same positional encoding as DWA's PartA.
"""
import jax
import jax.numpy as jnp
from flax import nnx
from .parts import _rope_freqs, _apply_rope


class TransformerBlock(nnx.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 max_seq_len: int, rngs: nnx.Rngs):
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5

        self.ln1  = nnx.LayerNorm(d_model, rngs=rngs)
        self.qkv  = nnx.Linear(d_model, 3 * d_model, use_bias=False, rngs=rngs)
        self.o    = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)

        self.ln2  = nnx.LayerNorm(d_model, rngs=rngs)
        self.fc1  = nnx.Linear(d_model, d_ff, rngs=rngs)
        self.fc2  = nnx.Linear(d_ff, d_model, rngs=rngs)

        self._freqs = _rope_freqs(self.d_head, max_seq_len)

    def __call__(self, x: jax.Array) -> jax.Array:
        B, T, C = x.shape
        H, D = self.n_heads, self.d_head

        # Causal self-attention (pre-norm)
        h = self.ln1(x)
        qkv = self.qkv(h)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        q = _apply_rope(q, self._freqs)
        k = _apply_rope(k, self._freqs)
        attn = jnp.einsum('bhsd,bhtd->bhst', q, k) * self.scale
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        attn = jnp.where(mask, attn, -1e9)
        attn = jax.nn.softmax(attn, axis=-1)
        out  = jnp.einsum('bhst,bhtd->bhsd', attn, v)
        out  = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        x    = x + self.o(out)

        # FFN (pre-norm)
        x = x + self.fc2(jax.nn.gelu(self.fc1(self.ln2(x))))
        return x


class DenseLM(nnx.Module):
    """
    GPT-style dense language model.
    d_model, n_layers, n_heads, d_ff are the knobs for size matching.
    """

    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, d_ff: int, max_seq_len: int = 256,
                 rngs: nnx.Rngs = None):
        self.embed = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.blocks = nnx.List([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len, rngs)
            for _ in range(n_layers)
        ])
        self.ln_f  = nnx.LayerNorm(d_model, rngs=rngs)
        self.head  = nnx.Linear(d_model, vocab_size, use_bias=False, rngs=rngs)

    def __call__(self, x_ids: jax.Array) -> jax.Array:
        """x_ids: (batch, seq) int32 → logits (batch, seq, vocab)"""
        h = self.embed(x_ids)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))


def make_dense_configs(vocab_size: int = 65, max_seq_len: int = 256):
    """
    Return two config dicts:
      'large'  — matched to DWA's total param count (~2.9M)
      'small'  — matched to DWA's non-pool param count (~300K)
    """
    return {
        "large": dict(vocab_size=vocab_size, d_model=256, n_layers=4,
                      n_heads=4, d_ff=896, max_seq_len=max_seq_len),
        "small": dict(vocab_size=vocab_size, d_model=128, n_layers=2,
                      n_heads=4, d_ff=320, max_seq_len=max_seq_len),
    }
