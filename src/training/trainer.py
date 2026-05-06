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
    forced_idx: jax.Array,
    soft: bool,
):
    x       = batch[:, :-1]
    targets = batch[:, 1:]

    vocab_size = model.config.d_input
    x_onehot   = jax.nn.one_hot(x, vocab_size)

    fwd_forced = forced_idx if (not use_sigmoid and not soft) else None
    logits, aux = model(
        x_onehot,
        use_sigmoid=use_sigmoid,
        lambda_sharp=lambda_sharp,
        forced_idx=fwd_forced,
        return_aux=True,
        soft=soft,
    )

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    task_loss = -jnp.mean(
        jnp.sum(jax.nn.one_hot(targets, vocab_size) * log_probs, axis=-1)
    )

    # Compute aux losses in phase-2 (hard) OR always in soft mode
    if use_sigmoid or soft:
        aux_losses = compute_aux_losses(
            model,
            aux["alpha"],
            aux["idx"],          # None in soft mode — diversity_loss handles this
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


# use_sigmoid and soft are static → at most 4 jit compilations over full training
@nnx.jit(static_argnums=(3, 4))
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    batch: jax.Array,
    use_sigmoid: bool,
    soft: bool,
    lambda_sharp: float,
    lambda_entropy_eff: float,
    forced_idx: jax.Array,
):
    grad_fn = nnx.value_and_grad(
        loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True
    )
    (total_loss, (metrics, aux)), grads = grad_fn(
        model, batch, use_sigmoid, lambda_sharp, lambda_entropy_eff, forced_idx, soft
    )
    optimizer.update(model, grads)

    # EMA update — soft and hard modes have different alpha shapes.
    N         = model.pool.N
    alpha_all = aux.get("alpha_all", [aux["alpha"]])
    idx_all   = aux.get("idx_all",   [aux["idx"]])
    n_blocks  = len(alpha_all)

    if idx_all[0] is None:
        # Soft mode: alpha is (batch, seq, N) — direct mean, no scatter needed.
        all_alpha = jnp.stack(alpha_all)          # (n_blocks, batch, seq, N)
        alpha_sum = all_alpha.mean(axis=(0, 1, 2)) # (N,) — mean over blocks/batch/seq
    else:
        # Hard mode: scatter alpha into N-dim using idx.
        all_idx   = jnp.stack(idx_all)    # (n_blocks, batch, [seq,] k_max)
        all_alpha = jnp.stack(alpha_all)   # (n_blocks, batch, [seq,] k_max)
        if all_idx.ndim == 3:
            # Sequence-level idx: sum alpha over seq before scattering
            all_alpha = all_alpha.sum(axis=2)
            denom = batch.shape[0] * n_blocks * alpha_all[0].shape[1]
        else:
            denom = batch.shape[0] * n_blocks
        alpha_sum = jnp.zeros(N).at[all_idx.reshape(-1)].add(
            all_alpha.reshape(-1) / denom
        )

    model.pool.update_ema(alpha_sum, model.config.beta_ema)
    metrics["loss"] = total_loss
    return metrics


# ─── Text generation ────────────────────────────────────────────────────────────

@nnx.jit(static_argnums=(4, 5, 6))
def _generate_step(
    model,
    context: jax.Array,        # (1, max_seq_len) int32
    pos: jax.Array,            # scalar int — 1-indexed position to read logit from
    rng_key: jax.Array,        # PRNG key
    temperature: float,        # static
    soft: bool,                # static
    top_k: int,                # static — 0 = disabled
) -> tuple[jax.Array, jax.Array]:
    """Single JIT-compiled generation step. Returns (next_token, new_rng_key)."""
    cfg = model.config
    x_onehot = jax.nn.one_hot(context, cfg.d_input)
    # valid_len=pos: only positions 0..pos-1 are real tokens (rest are padding)
    logits = model(x_onehot, use_sigmoid=True, lambda_sharp=5.0, soft=soft, valid_len=pos)

    # Logit at position (pos-1) predicts token at position pos
    next_logits = logits[:, pos - 1, :] / temperature

    if top_k > 0:
        top_vals = jnp.sort(next_logits, axis=-1)[:, -top_k:]
        threshold = top_vals[:, 0:1]
        next_logits = jnp.where(next_logits >= threshold, next_logits, -1e10)

    rng_key, subkey = jax.random.split(rng_key)
    next_token = jax.random.categorical(subkey, next_logits, axis=-1)
    return next_token, rng_key


def generate(
    model,
    prompt_tokens: jax.Array,   # (1, prompt_len) int32
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = None,
    soft: bool = False,
    seed: int = 42,
) -> jax.Array:
    """
    Autoregressive generation. Returns (1, prompt_len + max_new_tokens) int32.

    JIT-compiled per-step with a fixed-size context window — no shape changes
    between steps so the model compiles exactly once, then runs ~200× faster.
    """
    cfg = model.config
    prompt_len = prompt_tokens.shape[1]
    total_len = prompt_len + max_new_tokens

    token_buffer = jnp.zeros((1, total_len), dtype=jnp.int32)
    token_buffer = token_buffer.at[:, :prompt_len].set(prompt_tokens)

    rng_key = jax.random.PRNGKey(seed)
    top_k_val = top_k if top_k is not None else 0

    for i in range(max_new_tokens):
        pos = prompt_len + i  # absolute position of token to predict

        # Sliding context window of max_seq_len tokens ending at pos
        start = max(0, pos + 1 - cfg.max_seq_len)
        context = token_buffer[:, start:start + cfg.max_seq_len]
        # Pad to exactly max_seq_len if buffer shorter (early steps)
        pad_size = cfg.max_seq_len - context.shape[1]
        if pad_size > 0:
            context = jnp.concatenate(
                [context, jnp.zeros((1, pad_size), dtype=jnp.int32)], axis=1
            )

        # 1-indexed: logit at (pos_in_window - 1) predicts pos_in_window
        pos_in_window = pos - start + 1

        next_token, rng_key = _generate_step(
            model, context,
            jnp.array(pos_in_window),
            rng_key, temperature, soft, top_k_val,
        )
        token_buffer = token_buffer.at[:, pos].set(next_token)

    return token_buffer[:, :total_len]


# ─── Training loop ────────────────────────────────────────────────────────────

def train_loop(
    model,
    optimizer,
    data_iter,
    total_steps: int,
    log_every: int = 100,
    seed: int = 0,
    generate_every: int = 0,
    tokenizer=None,
    generate_prompt: str | None = None,
    generate_max_tokens: int = 200,
    generate_temperature: float = 0.8,
):
    cfg      = model.config
    soft     = getattr(cfg, 'soft_train', False)
    rng      = np.random.default_rng(seed)
    resets   = 0
    _rotator = None  # lazily init with real batch size on first step

    for step, batch in zip(range(total_steps), data_iter):
        # Unwrap (batch_array, tokenizer) tuples from shakespeare_loader
        if isinstance(batch, tuple):
            batch = batch[0]
        t0 = time.time()

        # Codebook resets only in hard mode — soft mode gives every vector
        # gradients each step, so low EMA just means natural sparsity, not death.
        if not soft and cfg.reset_interval > 0 and step > 0 and step % cfg.reset_interval == 0:
            resets += reset_dead_vectors(model, rng)

        use_sigmoid, lambda_sharp, lambda_entropy_eff = get_phase_params(step, cfg)

        # Rotator only needed in hard mode (soft mode doesn't use forced_idx)
        if not soft:
            if _rotator is None:
                _rotator = Phase1Rotator(cfg.N, batch.shape[0], cfg.k_max, seed=seed)
            forced_idx = _rotator.next() if not use_sigmoid else _rotator.dummy()
        else:
            if _rotator is None:
                _rotator = Phase1Rotator(cfg.N, batch.shape[0], cfg.k_max, seed=seed)
            forced_idx = _rotator.dummy()  # ignored by soft_forward

        metrics = train_step(
            model, optimizer, batch,
            use_sigmoid, soft, lambda_sharp, lambda_entropy_eff, forced_idx
        )
        # device_get only at log points — keeps GPU async between steps
        if step % log_every == 0:
            m = jax.device_get(metrics)
            msg = [f"step={step:05d}  soft={soft}"]
            for k, v in sorted(m.items()):
                msg.append(f"{k}={float(v):.4f}")
            msg.append(f"lent={lambda_entropy_eff:.4f}  resets={resets}")
            msg.append(f"t={int((time.time()-t0)*1000)}ms")
            print("  ".join(msg))

        # Periodic text generation
        if generate_every > 0 and step > 0 and step % generate_every == 0 and tokenizer is not None:
            prompt = generate_prompt if generate_prompt else ""
            prompt_tokens = jnp.array(tokenizer.encode(prompt))[None, :]  # (1, seq_len)
            if prompt_tokens.shape[1] == 0:
                prompt_tokens = jnp.zeros((1, 1), dtype=jnp.int32)
            out = generate(model, prompt_tokens, generate_max_tokens,
                           temperature=generate_temperature, soft=soft)
            text = tokenizer.decode(out[0])
            print(f"  [{step}] >> {text}")
