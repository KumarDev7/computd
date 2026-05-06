"""
7B DWA model training on 8-core TPU v5e.

Architecture:  DWAModel — Dynamic Weight Assembly
Mode:          8-way TP sharding + shard_map cross-chip top-k merge
Pool:          N=32768 vectors sharded across 8 chips (4096 per chip)
Memory target: ~5.5 GB params + optimizer per chip on 16 GB HBM

The pallas=True flag routes through pallas_hybrid_forward which uses
shard_map for per-chip similarity computation + all_gather + global top-k.
The Pallas fused kernel (tiled_topk_fused) is available for inference but
not used during training because pallas_call lacks reverse-mode autodiff.

Usage
-----
cd /kaggle/working/computd
python scripts/train_tpu_7b.py

Key flags (edit at bottom of file):
    TOTAL_STEPS   = 20_000
    BATCH_SIZE    = 16      (global; each chip sees batch // 8 = 2)
    LOG_EVERY     = 100
    GEN_EVERY     = 2000
"""

from __future__ import annotations
import os, sys, time, gc, math, functools
import numpy as np

# JAX performance flags — set before importing jax
os.environ.setdefault('JAX_ENABLE_X64',            'False')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'true')
os.environ.setdefault('JAX_DEBUG_NANS',             'True')
os.environ.setdefault('JAX_LOG_COMPILES',           'False')  # verbose during dev: True

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax import nnx

from configs.large_7b import get_7b_config
from src.model.dwa import DWAModel
from src.training.trainer import make_optimizer, get_phase_params, Phase1Rotator
from src.training.trainer import loss_fn, compute_aux_losses
from src.training.sharding import make_mesh, shard_model
from src.data.text_loader import shakespeare_loader


# ---------------------------------------------------------------------------
# Env check
# ---------------------------------------------------------------------------

def env_check():
    devs = jax.devices()
    print(f'[env] platform={devs[0].platform}  devices={len(devs)}  jax={jax.__version__}')
    for lib in ('flax', 'optax'):
        try:
            import importlib; m = importlib.import_module(lib)
            print(f'[env] {lib}={m.__version__}')
        except Exception:
            pass


def count_params(model: nnx.Module) -> int:
    _, state = nnx.split(model)
    params = state.filter(nnx.Param)
    return sum(x.size for x in jax.tree.leaves(params))


# ---------------------------------------------------------------------------
# Sharded train step (runs inside shard_map so each device sees local shard)
# ---------------------------------------------------------------------------

def make_sharded_train_step(mesh: Mesh, cfg, use_sigmoid: bool):
    """
    Returns a sharded train_step that:
      1. Replicates activations across TP
      2. Runs Pallas fused kernel on each chip's local pool shard
      3. Cross-chip all_gather gives global top_k
      4. All-reduces gradients across chips

    use_sigmoid is a compile-time constant (separate JIT for phase 1 vs phase 2).
    """
    import optax
    from src.training.losses import compute_aux_losses

    @nnx.jit
    def train_step(model, optimizer, batch, lambda_sharp, lambda_entropy_eff,
                   forced_idx):
        cfg_local = model.config

        def _loss(model, batch, lambda_sharp, lambda_entropy_eff, forced_idx):
            x       = batch[:, :-1]
            targets = batch[:, 1:]
            vocab   = cfg_local.d_input
            x_oh    = jax.nn.one_hot(x, vocab)

            logits, aux = model(
                x_oh,
                use_sigmoid=use_sigmoid,
                lambda_sharp=lambda_sharp,
                forced_idx=forced_idx if not use_sigmoid else None,
                return_aux=True,
                pallas=True,   # use Pallas fused kernel
                tp_axis='tp',  # cross-chip all_gather inside pallas_hybrid_forward
            )
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            task_loss = -jnp.mean(
                jnp.sum(jax.nn.one_hot(targets, vocab) * log_probs, axis=-1)
            )
            aux_losses = compute_aux_losses(
                model, aux['alpha'], aux['idx'], aux['sims'], aux['alpha_raw'],
                lambda_entropy_eff=lambda_entropy_eff,
            )
            total = task_loss + aux_losses['total_aux']
            return total, ({'task': task_loss, **{f'aux_{k}': v for k, v in aux_losses.items()}}, aux)

        grad_fn = nnx.value_and_grad(_loss, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)
        (total_loss, (metrics, aux)), grads = grad_fn(
            model, batch, lambda_sharp, lambda_entropy_eff, forced_idx
        )
        optimizer.update(model, grads)

        # EMA update (hybrid per-position idx shape)
        N         = model.pool.N
        alpha_all = aux.get('alpha_all', [aux['alpha']])
        idx_all   = aux.get('idx_all',   [aux['idx']])
        all_idx   = jnp.stack(idx_all)
        all_alpha = jnp.stack(alpha_all)
        alpha_sum = jnp.zeros(N).at[all_idx.reshape(-1)].add(
            all_alpha.reshape(-1) / (all_idx.size)
        )
        model.pool.update_ema(alpha_sum, cfg_local.beta_ema)
        metrics['loss'] = total_loss
        return metrics

    return train_step


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_loop_7b(
    model,
    optimizer,
    data_iter,
    mesh: Mesh,
    total_steps: int = 20_000,
    log_every: int = 100,
    generate_every: int = 2000,
    tokenizer=None,
    generate_prompt: str = 'ROMEO:',
):
    from src.training.trainer import generate

    cfg   = model.config
    rng   = np.random.default_rng(0)
    step  = 0
    _rot  = None

    # Two compiled train steps: phase 1 (no sigmoid) and phase 2+ (sigmoid)
    step_phase1 = make_sharded_train_step(mesh, cfg, use_sigmoid=False)
    step_phase2 = make_sharded_train_step(mesh, cfg, use_sigmoid=True)

    for step, batch in zip(range(total_steps), data_iter):
        if isinstance(batch, tuple):
            batch = batch[0]

        t0 = time.time()
        use_sigmoid, lambda_sharp, lambda_entropy_eff = get_phase_params(step, cfg)

        if _rot is None:
            _rot = Phase1Rotator(cfg.N, batch.shape[0], cfg.k_max, seed=0)
        forced_idx = _rot.dummy()   # pallas mode doesn't use phase-1 rotation

        train_fn = step_phase2 if use_sigmoid else step_phase1
        metrics = train_fn(
            model, optimizer, batch,
            lambda_sharp, lambda_entropy_eff, forced_idx
        )

        if step % log_every == 0:
            m   = jax.device_get(metrics)
            msg = [f'step={step:05d}  mode=pallas-tp8']
            for k, v in sorted(m.items()):
                msg.append(f'{k}={float(v):.4f}')
            msg.append(f'lent={lambda_entropy_eff:.4f}')
            msg.append(f't={int((time.time()-t0)*1000)}ms')
            print('  '.join(msg))

        if generate_every > 0 and step > 0 and step % generate_every == 0 and tokenizer:
            prompt_tokens = jnp.array(tokenizer.encode(generate_prompt))[None, :]
            out = generate(model, prompt_tokens, 200, temperature=0.8, hybrid=True)
            print(f'  [{step}] >> {tokenizer.decode(out[0])}')

    print(f'[done] {total_steps} steps completed.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

TOTAL_STEPS  = 20_000
BATCH_SIZE   = 16      # global batch — keep small to fit 16 GB/chip activations
LOG_EVERY    = 100
GEN_EVERY    = 2000
SEQ_LEN_OVERRIDE = 512   # override 7B config's 2048 for initial TPU run

if __name__ == '__main__':
    env_check()

    # ---- Config ----
    cfg = get_7b_config()
    # Override seq_len for first run (2048 needs more HBM for activations)
    cfg.max_seq_len = SEQ_LEN_OVERRIDE

    # ---- Mesh + model ----
    mesh = make_mesh()
    print(f'[mesh] {mesh}')

    rngs  = nnx.Rngs(params=0, dropout=1)
    model = DWAModel(cfg, rngs=rngs, mesh=mesh)

    n_params = count_params(model)
    print(f'[model] trainable params: {n_params:,}  (~{n_params/1e9:.2f}B)')

    # ---- Shard model parameters across 8 chips ----
    print('[sharding] distributing parameters across 8 TPU chips...')
    with mesh:
        shard_model(model, mesh)
    print('[sharding] done')

    # ---- Optimizer ----
    optimizer = make_optimizer(model, cfg)

    # ---- Data ----
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'shakespeare.txt')
    tokenizer = None
    for _, tok in shakespeare_loader(data_path, BATCH_SIZE, cfg.max_seq_len, split='val'):
        tokenizer = tok
        break

    def train_gen():
        while True:
            for batch, _ in shakespeare_loader(data_path, BATCH_SIZE, cfg.max_seq_len, split='train'):
                yield batch

    print(f'[train] steps={TOTAL_STEPS}  batch={BATCH_SIZE}  seq={cfg.max_seq_len}')
    print('─' * 70)

    with mesh:
        train_loop_7b(
            model, optimizer, train_gen(), mesh,
            total_steps=TOTAL_STEPS,
            log_every=LOG_EVERY,
            generate_every=GEN_EVERY,
            tokenizer=tokenizer,
            generate_prompt='ROMEO:',
        )
