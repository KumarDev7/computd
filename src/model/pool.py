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

    def __init__(self, N: int, D: int, rngs: nnx.Rngs, sharding=None):
        key = rngs.params()
        if sharding is not None:
            # Create directly sharded — avoids materialising the full (N, D) on host.
            # jax.make_array_from_callback gives each device a shard_index map.
            n_devices = sharding.mesh.size
            per_device_n = N // n_devices
            def _data_callback(shard_indices):
                # shard_indices is a tuple of slice objects per dimension
                # For P('tp', None): dim0 is sharded, dim1 is full
                # Figure out which device index this is from the dim0 slice start
                sl0 = shard_indices[0]
                device_idx = sl0.start // per_device_n
                shard_key = jax.random.fold_in(key, device_idx)
                local_shape = (sl0.stop - sl0.start, D)
                return jax.random.normal(shard_key, local_shape, dtype=jnp.bfloat16) * 0.02
            vectors = jax.make_array_from_callback(
                (N, D), sharding, _data_callback
            )
        else:
            vectors = jax.random.normal(key, (N, D), dtype=jnp.bfloat16) * 0.02
        self.vectors = nnx.Param(vectors)
        # EMA of mean α per vector across batches — used in utilization loss
        self.ema_usage = EMAState(jnp.zeros(N, dtype=jnp.float32))

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
