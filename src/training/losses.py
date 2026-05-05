import jax
import jax.numpy as jnp


def utilization_loss(ema_usage: jax.Array, beta: float = 10.0) -> jax.Array:
    """
    Prevent dead vectors.
    Uses SUM (not mean) so each dead vector contributes its full penalty
    rather than being diluted by the 97% of other dead vectors.
    Normalized by N so magnitude stays comparable across pool sizes.
    """
    N = ema_usage.shape[0]
    safe = jnp.clip(ema_usage, 1e-8, None)
    per_vector = -jnp.log1p(-jnp.exp(-beta * safe) + 1e-8)
    return jnp.sum(per_vector) / N


def selection_entropy_loss(alpha_raw: jax.Array) -> jax.Array:
    """
    Maximize entropy of the batch-averaged selection distribution over all N
    pool vectors. Shapes: (batch, N) or (batch, seq, N) — flattened to 2D.
    Returns a loss in [0, 1]:  0 = fully spread (ideal), 1 = fully collapsed.
    """
    N = alpha_raw.shape[-1]
    flat = alpha_raw.reshape(-1, N)          # (batch*seq, N)
    per_example = jax.nn.softmax(flat, axis=-1)
    batch_dist  = jnp.mean(per_example, axis=0)   # (N,)

    entropy     = -jnp.sum(batch_dist * jnp.log(batch_dist + 1e-8))
    max_entropy = jnp.log(jnp.array(N, dtype=jnp.float32))
    return 1.0 - entropy / max_entropy


def diversity_loss(sims: jax.Array, alpha: jax.Array, idx: jax.Array) -> jax.Array:
    """
    Prevent key collapse: penalize high average similarity among retrieved vectors.
    Handles (batch, N/k_max) or (batch, seq, N/k_max) by flattening to 2D.
    """
    N = sims.shape[-1]
    k = idx.shape[-1]
    sims_2d  = sims.reshape(-1, N)
    idx_2d   = idx.reshape(-1, k)
    n        = sims_2d.shape[0]
    retrieved_sims = sims_2d[jnp.arange(n)[:, None], idx_2d]  # (n, k_max)
    return jnp.mean(jnp.mean(retrieved_sims, axis=-1) ** 2)


def norm_loss(model) -> jax.Array:
    """Sum W_base Frobenius norms over all assembly blocks."""
    if hasattr(model, 'blocks'):
        return sum(jnp.sum(block.assembler.W_base.value ** 2) for block in model.blocks)
    return jnp.sum(model.assembler.W_base.value ** 2)


def sparsity_loss(alpha: jax.Array) -> jax.Array:
    """Weight entropy within top-k: encourages sparse selection. Works for any leading dims."""
    safe = jnp.clip(alpha, 1e-8, None)
    return jnp.mean(-jnp.sum(safe * jnp.log(safe), axis=-1))


def compute_aux_losses(
    model,
    alpha: jax.Array,
    idx: jax.Array,
    sims: jax.Array,
    alpha_raw: jax.Array,
    lambda_entropy_eff: float = 0.0,
):
    cfg = model.config
    ema = model.pool.ema_usage.value

    l_util    = utilization_loss(ema)                    * cfg.lambda_util
    l_entropy = selection_entropy_loss(alpha_raw)         * lambda_entropy_eff
    l_div     = diversity_loss(sims, alpha, idx)          * cfg.lambda_div
    l_norm    = norm_loss(model)                           * cfg.lambda_norm
    l_sparse  = sparsity_loss(alpha)                      * cfg.lambda_sparse

    total = l_util + l_entropy + l_div + l_norm + l_sparse
    return {
        "util":       l_util,
        "entropy":    l_entropy,
        "div":        l_div,
        "norm":       l_norm,
        "sparse":     l_sparse,
        "total_aux":  total,
    }
