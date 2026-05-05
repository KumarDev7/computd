import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from .losses import compute_aux_losses


# ─── Phase helpers ────────────────────────────────────────────────────────────

def get_phase_params(step: int, config) -> tuple[bool, float, float]:
    """
    Returns (use_sigmoid, lambda_sharp, lambda_entropy_eff) — all Python scalars.

    lambda_entropy is annealed:
      Phase 1: 0.0          (warmup rotation, no entropy pressure)
      Phase 2: max → 0.3×   (strong early to prevent premature collapse, then relax)
      Phase 3: 0.1×          (just enough to prevent extinction)
    """
    base_ent = config.lambda_entropy
    if step < config.phase1_end:
        return False, 1.0, 0.0
    elif step < config.phase2_end:
        t = (step - config.phase1_end) / max(config.phase2_end - config.phase1_end, 1)
        lam = 1.0 + 4.0 * t
        ent = base_ent * (1.0 - 0.7 * t)
        return True, lam, ent
    else:
        t = min((step - config.phase2_end) / 10_000, 1.0)
        return True, 5.0 + 5.0 * t, base_ent * 0.1


# ─── Phase-1 vector rotation schedule ────────────────────────────────────────

class Phase1Rotator:
    """
    Generates forced_idx for phase-1 warmup. Each step assigns a different
    set of vectors across the batch so every vector gets gradient signal
    before phase 2 begins.

    Strategy: build a large cyclic permutation of all N indices, then slide a
    window of size (batch × k_max) forward by that amount each step.
    With N=512, batch=32, k_max=8 → 256 vectors covered per step,
    all 512 visited every ~2 steps.
    """

    def __init__(self, N: int, batch: int, k_max: int, seed: int = 0):
        self.N     = N
        self.batch = batch
        self.k_max = k_max
        self._rng  = np.random.default_rng(seed)
        # Large tiled permutation so we never run out of indices
        base_perm  = self._rng.permutation(N).astype(np.int32)
        repeats    = (batch * k_max * 20_000) // N + 2
        self._perm = np.tile(base_perm, repeats)
        self._pos  = 0

    def next(self) -> jax.Array:
        """Returns (batch, k_max) int32 indices for the current step."""
        size = self.batch * self.k_max
        idx  = self._perm[self._pos : self._pos + size].reshape(self.batch, self.k_max)
        self._pos = (self._pos + size) % len(self._perm)
        return jnp.array(idx)

    def dummy(self) -> jax.Array:
        """Zero-filled placeholder used in phase 2 (ignored by the model)."""
        return jnp.zeros((self.batch, self.k_max), dtype=jnp.int32)


# ─── Codebook reset ───────────────────────────────────────────────────────────

def reset_dead_vectors(model, rng: np.random.Generator) -> int:
    cfg      = model.config
    ema      = np.array(model.pool.ema_usage.value)
    dead_idx  = np.where(ema < cfg.reset_threshold)[0]
    alive_idx = np.where(ema >= cfg.reset_threshold)[0]
    if len(dead_idx) == 0 or len(alive_idx) == 0:
        return 0
    vectors = np.array(model.pool.vectors.value)
    sources = rng.choice(alive_idx, size=len(dead_idx), replace=True)
    noise   = rng.normal(0, cfg.reset_noise, vectors[dead_idx].shape).astype(np.float32)
    vectors[dead_idx] = vectors[sources] + noise
    model.pool.vectors.value      = jnp.array(vectors)
    new_ema = ema.copy()
    new_ema[dead_idx] = cfg.reset_threshold * 2.0
    model.pool.ema_usage.value = jnp.array(new_ema)
    return len(dead_idx)


# ─── Optimizer ────────────────────────────────────────────────────────────────

def make_optimizer(model: nnx.Module, config) -> nnx.Optimizer:
    def _leaf_label(path, _leaf):
        path_str = "/".join(str(p.key) if hasattr(p, "key") else str(p) for p in path)
        if "pool" in path_str:
            return "pool"
        if "retrieval" in path_str and ("tau" in path_str or "aspect_logits" in path_str):
            return "threshold"
        if "gamma" in path_str:
            return "threshold"
        if "retrieval" in path_str:
            return "retrieval"
        return "parts"

    def label_fn(params):
        return jax.tree_util.tree_map_with_path(_leaf_label, params)

    tx = optax.multi_transform(
        {
            "pool":      optax.adamw(config.lr_pool,          weight_decay=1e-4),
            "parts":     optax.adamw(config.lr_parts,         weight_decay=1e-4),
            "retrieval": optax.adamw(config.lr_retrieval,     weight_decay=1e-4),
            "threshold": optax.adam(config.lr_threshold_gamma),
        },
        param_labels=label_fn,
    )
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


# ─── Loss & train step ────────────────────────────────────────────────────────

def loss_fn(
    model: nnx.Module,
    batch: jax.Array,
    use_sigmoid: bool,
    lambda_sharp: float,
    lambda_entropy_eff: float,
    forced_idx: jax.Array,      # (batch, k_max) — used in phase1, ignored in phase2
):
    x       = batch[:, :-1]
    targets = batch[:, 1:]

    vocab_size = model.config.d_input
    x_onehot   = jax.nn.one_hot(x, vocab_size)

    # Phase 1 uses forced_idx for vector rotation; phase 2 ignores it
    fwd_forced = forced_idx if not use_sigmoid else None
    logits, aux = model(
        x_onehot,
        use_sigmoid=use_sigmoid,
        lambda_sharp=lambda_sharp,
        forced_idx=fwd_forced,
        return_aux=True,
    )

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    task_loss = -jnp.mean(
        jnp.sum(jax.nn.one_hot(targets, vocab_size) * log_probs, axis=-1)
    )

    if use_sigmoid:
        aux_losses = compute_aux_losses(
            model,
            aux["alpha"],
            aux["idx"],
            aux["sims"],
            aux["alpha_raw"],
            lambda_entropy_eff=lambda_entropy_eff,
        )
        total_loss = task_loss + aux_losses["total_aux"]
        metrics    = {"task": task_loss,
                      **{f"aux_{k}": v for k, v in aux_losses.items()}}
    else:
        total_loss = task_loss
        metrics    = {"task": task_loss}

    return total_loss, (metrics, aux)


# use_sigmoid is static → exactly 2 jit compilations over full training
@nnx.jit(static_argnums=(3,))
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    batch: jax.Array,
    use_sigmoid: bool,
    lambda_sharp: float,
    lambda_entropy_eff: float,
    forced_idx: jax.Array,      # always passed; used iff use_sigmoid=False
):
    grad_fn = nnx.value_and_grad(
        loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True
    )
    (total_loss, (metrics, aux)), grads = grad_fn(
        model, batch, use_sigmoid, lambda_sharp, lambda_entropy_eff, forced_idx
    )
    optimizer.update(model, grads)

    # EMA update — aggregate across all blocks for comprehensive pool coverage
    N         = model.pool.N
    alpha_all = aux.get("alpha_all", [aux["alpha"]])
    idx_all   = aux.get("idx_all",   [aux["idx"]])
    n_blocks  = len(alpha_all)
    alpha_sum = jnp.zeros(N)
    for al, ix in zip(alpha_all, idx_all):
        alpha_sum = alpha_sum.at[ix.reshape(-1)].add(
            al.reshape(-1) / (batch.shape[0] * n_blocks)
        )
    model.pool.update_ema(alpha_sum, model.config.beta_ema)

    metrics["loss"] = total_loss
    return metrics


# ─── Training loop ────────────────────────────────────────────────────────────

def train_loop(
    model,
    optimizer,
    data_iter,
    total_steps: int,
    log_every: int = 100,
    seed: int = 0,
):
    cfg      = model.config
    rng      = np.random.default_rng(seed)
    rotator  = Phase1Rotator(cfg.N, cfg.k_max * 4, cfg.k_max, seed=seed)
    # Note: rotator uses batch=k_max*4 as a stand-in; actual batch inferred below
    resets   = 0
    _rotator = None  # lazily init with real batch size

    for step, batch in zip(range(total_steps), data_iter):
        t0 = time.time()

        if cfg.reset_interval > 0 and step > 0 and step % cfg.reset_interval == 0:
            resets += reset_dead_vectors(model, rng)

        use_sigmoid, lambda_sharp, lambda_entropy_eff = get_phase_params(step, cfg)

        # Lazily create rotator with real batch size on first step
        if _rotator is None:
            real_batch = batch.shape[0]
            _rotator = Phase1Rotator(cfg.N, real_batch, cfg.k_max, seed=seed)

        if not use_sigmoid:
            forced_idx = _rotator.next()
        else:
            forced_idx = _rotator.dummy()

        metrics = train_step(
            model, optimizer, batch,
            use_sigmoid, lambda_sharp, lambda_entropy_eff, forced_idx
        )
        metrics = jax.device_get(metrics)

        if step % log_every == 0:
            msg = [f"step={step:05d}"]
            for k, v in sorted(metrics.items()):
                msg.append(f"{k}={float(v):.4f}")
            msg.append(f"lent={lambda_entropy_eff:.4f}  resets={resets}")
            msg.append(f"t={int((time.time()-t0)*1000)}ms")
            print("  ".join(msg))
