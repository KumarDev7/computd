import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from src.training.sharding import init_sharded_param


# ─── RoPE helpers ─────────────────────────────────────────────────────────────

def _rope_freqs(d_head: int, max_seq: int) -> jax.Array:
    theta = 1.0 / (10000.0 ** (jnp.arange(0, d_head, 2) / d_head))
    pos   = jnp.arange(max_seq)
    return jnp.outer(pos, theta)


def _apply_rope(x: jax.Array, freqs: jax.Array) -> jax.Array:
    seq = x.shape[-2]
    f   = freqs[:seq]
    x1, x2 = x[..., ::2], x[..., 1::2]
    rot = jnp.concatenate(
        [x1 * jnp.cos(f) - x2 * jnp.sin(f),
         x1 * jnp.sin(f) + x2 * jnp.cos(f)], axis=-1
    )
    return rot


# ─── Causal Self-Attention ────────────────────────────────────────────────────

class CausalSelfAttention(nnx.Module):

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 256,
                 rngs: nnx.Rngs = None, mesh: Mesh = None):
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5

        rng_key = rngs.params()
        if mesh is not None:
            self.qkv = nnx.Linear(d_model, 3 * d_model, use_bias=False, rngs=rngs)
            self.qkv.kernel.value = init_sharded_param(
                (d_model, 3 * d_model),
                NamedSharding(mesh, P(None, 'tp')),
                rng_key, scale=jnp.sqrt(2.0 / d_model), dtype=jnp.bfloat16,
            )
            self.proj = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
            self.proj.kernel.value = init_sharded_param(
                (d_model, d_model),
                NamedSharding(mesh, P('tp', None)),
                rng_key, scale=jnp.sqrt(2.0 / d_model), dtype=jnp.bfloat16,
            )
        else:
            self.qkv  = nnx.Linear(d_model, 3 * d_model, use_bias=False, param_dtype=jnp.bfloat16, rngs=rngs)
            self.proj = nnx.Linear(d_model, d_model, use_bias=False, param_dtype=jnp.bfloat16, rngs=rngs)

        self.norm = nnx.LayerNorm(d_model, rngs=rngs)
        self._freqs = _rope_freqs(self.d_head, max_seq_len)
        self._mask  = jnp.tril(jnp.ones((max_seq_len, max_seq_len), dtype=jnp.float32))

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

    def __init__(self, d_input: int, d_A: int, n_heads: int = 0,
                 max_seq_len: int = 256, rngs: nnx.Rngs = None, mesh: Mesh = None,
                 use_embedding: bool = True):
        self.use_embedding = use_embedding
        hidden = d_A * 4

        if use_embedding:
            self.embed = nnx.Embed(d_input, d_A, rngs=rngs)
            self.proj = nnx.Linear(d_A, d_A, use_bias=False, param_dtype=jnp.bfloat16, rngs=rngs)
            self.norm_out = nnx.LayerNorm(d_A, rngs=rngs)
        else:
            self.norm_in  = nnx.LayerNorm(d_input, rngs=rngs)
            self.norm_out = nnx.LayerNorm(d_A, rngs=rngs)

            if mesh is not None:
                self.fc1 = nnx.Linear(d_input, hidden, rngs=rngs)
                self.fc1.kernel.value = init_sharded_param(
                    (d_input, hidden),
                    NamedSharding(mesh, P(None, 'tp')),
                    rngs.params(), scale=jnp.sqrt(2.0 / d_input), dtype=jnp.bfloat16,
                )
                self.fc1.bias.value = init_sharded_param(
                    (hidden,),
                    NamedSharding(mesh, P('tp',)),
                    rngs.params(), scale=0.0, dtype=jnp.bfloat16,
                )
                self.fc2 = nnx.Linear(hidden, d_A, rngs=rngs)
                self.fc2.kernel.value = init_sharded_param(
                    (hidden, d_A),
                    NamedSharding(mesh, P('tp', None)),
                    rngs.params(), scale=jnp.sqrt(2.0 / hidden), dtype=jnp.bfloat16,
                )
            else:
                self.fc1 = nnx.Linear(d_input, hidden, param_dtype=jnp.bfloat16, rngs=rngs)
                self.fc2 = nnx.Linear(hidden, d_A, param_dtype=jnp.bfloat16, rngs=rngs)

        self.attn = (CausalSelfAttention(d_A, n_heads, max_seq_len, rngs, mesh=mesh)
                     if n_heads > 0 else None)

    def __call__(self, x: jax.Array, token_ids: jax.Array | None = None):
        if self.use_embedding and token_ids is not None:
            h_A = self.proj(self.embed(token_ids))
            h_A = self.norm_out(h_A)
        elif self.use_embedding:
            h_A = self.proj(self.embed(x.argmax(axis=-1) if x.ndim == 3 else x))
            h_A = self.norm_out(h_A)
        else:
            h   = jax.nn.gelu(self.fc1(self.norm_in(x)))
            h_A = self.norm_out(self.fc2(h))

        if self.attn is not None and h_A.ndim == 3:
            h_A = self.attn(h_A)

        return h_A


class PartB(nnx.Module):

    def __init__(self, d_B: int, d_output: int, rngs: nnx.Rngs, mesh: Mesh = None):
        hidden = d_B * 4
        self.norm_in = nnx.LayerNorm(d_B, rngs=rngs)

        if mesh is not None:
            self.fc1 = nnx.Linear(d_B, hidden, rngs=rngs)
            self.fc1.kernel.value = init_sharded_param(
                (d_B, hidden),
                NamedSharding(mesh, P(None, 'tp')),
                rngs.params(), scale=jnp.sqrt(2.0 / d_B), dtype=jnp.bfloat16,
            )
            self.fc1.bias.value = init_sharded_param(
                (hidden,),
                NamedSharding(mesh, P('tp',)),
                rngs.params(), scale=0.0, dtype=jnp.bfloat16,
            )
            self.fc2 = nnx.Linear(hidden, d_output, rngs=rngs)
            self.fc2.kernel.value = init_sharded_param(
                (hidden, d_output),
                NamedSharding(mesh, P('tp', None)),
                rngs.params(), scale=jnp.sqrt(2.0 / hidden), dtype=jnp.bfloat16,
            )
        else:
            self.fc1 = nnx.Linear(d_B, hidden, param_dtype=jnp.bfloat16, rngs=rngs)
            self.fc2 = nnx.Linear(hidden, d_output, param_dtype=jnp.bfloat16, rngs=rngs)

    def __call__(self, h_mid: jax.Array) -> jax.Array:
        h = jax.nn.gelu(self.fc1(self.norm_in(h_mid)))
        return self.fc2(h)
