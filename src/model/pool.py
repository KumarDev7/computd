import jax
import jax.numpy as jnp
from flax import nnx


class EMAState(nnx.Variable):
    """Non-trainable EMA variable."""
    pass


class VectorPool(nnx.Module):
    """
    Pool of N learnable vectors, each of dim D.
    Each vector encodes: U_i (d_B×r), V_i (r×d_A), b_i (d_B), + key material.

    Gradient flows both from retrieval (key projection) and assembly (U,V,b).
    """

    def __init__(self, N: int, D: int, rngs: nnx.Rngs):
        key = rngs.params()
        self.vectors = nnx.Param(jax.random.normal(key, (N, D)) * 0.02)
        # EMA of mean α per vector across batches — used in utilization loss
        self.ema_usage = EMAState(jnp.zeros(N))

    @property
    def N(self) -> int:
        return self.vectors.value.shape[0]

    @property
    def D(self) -> int:
        return self.vectors.value.shape[1]

    def update_ema(self, alpha_sum: jax.Array, beta: float = 0.99) -> None:
        """alpha_sum: (N,) — sum of α values over batch for this step."""
        self.ema_usage.value = (
            beta * self.ema_usage.value + (1 - beta) * alpha_sum
        )
