"""
1B DWA model training on 8-core TPU v5e with Ultra-FineWeb dataset.

Architecture:  DWAModel — Dynamic Weight Assembly
Config:        ~1B parameters (d_A=d_B=1024, r=8, N=16384, 12 layers)
Dataset:       openbmb/Ultra-FineWeb (streaming)
Tokenizer:     LiquidAI/LFM2.5-1.2B-Thinking (vocab=64400)
Mode:          8-way TP sharding, hybrid_train=True

Usage:
    cd /kaggle/working/computd
    python scripts/train_tpu_1b.py
"""
from __future__ import annotations
import os, sys, time
import numpy as np

os.environ.setdefault('JAX_ENABLE_X64', 'False')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'true')
os.environ.setdefault('JAX_DEBUG_NANS', 'True')
os.environ.setdefault('JAX_LOG_COMPILES', 'False')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from flax import nnx

from configs.large_1b import get_1b_config
from src.model.dwa import DWAModel
from src.training.trainer import make_optimizer, get_phase_params, Phase1Rotator
from src.training.trainer import loss_fn, compute_aux_losses
from src.training.sharding import make_mesh, shard_model, shard_optimizer
from src.data.hf_loader import CachedDataLoader


def env_check():
    devs = jax.devices()
    print(f'[env] platform={devs[0].platform}  devices={len(devs)}  jax={jax.__version__}')
    for lib in ('flax', 'optax', 'transformers', 'datasets'):
        try:
            import importlib
            m = importlib.import_module(lib)
            ver = getattr(m, '__version__', 'unknown')
            print(f'[env] {lib}={ver}')
        except Exception:
            print(f'[env] {lib}=NOT INSTALLED')


def count_params(model: nnx.Module) -> int:
    _, state = nnx.split(model)
    params = state.filter(nnx.Param)
    return sum(x.size for x in jax.tree.leaves(params))


def format_params(n: int) -> str:
    if n >= 1e9:
        return f'{n/1e9:.2f}B'
    return f'{n/1e6:.1f}M'


def make_train_step(cfg, use_sigmoid: bool):
    @nnx.jit
    def train_step(model, optimizer, batch, lambda_sharp, lambda_entropy_eff, forced_idx):
        soft = getattr(cfg, 'soft_train', False)
        hybrid = getattr(cfg, 'hybrid_train', False)

        grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)
        (total_loss, (metrics, aux)), grads = grad_fn(
            model, batch, use_sigmoid, lambda_sharp, lambda_entropy_eff, forced_idx, soft, hybrid
        )
        optimizer.update(model, grads)

        N = model.pool.N
        alpha_all = aux.get('alpha_all', [aux['alpha']])
        idx_all = aux.get('idx_all', [aux['idx']])
        n_blocks = len(alpha_all)

        if idx_all[0] is None:
            all_alpha = jnp.stack(alpha_all)
            alpha_sum = all_alpha.mean(axis=(0, 1, 2))
        elif idx_all[0].ndim == 3 and idx_all[0].shape[1] > 1:
            all_idx = jnp.stack(idx_all)
            all_alpha = jnp.stack(alpha_all)
            alpha_sum = jnp.zeros(N).at[all_idx.reshape(-1)].add(
                all_alpha.reshape(-1) / (all_idx.size)
            )
        else:
            all_idx = jnp.stack(idx_all)
            all_alpha = jnp.stack(alpha_all)
            if all_idx.ndim == 3:
                all_alpha = all_alpha.sum(axis=2)
                denom = batch.shape[0] * n_blocks * alpha_all[0].shape[1]
            else:
                denom = batch.shape[0] * n_blocks
            alpha_sum = jnp.zeros(N).at[all_idx.reshape(-1)].add(
                all_alpha.reshape(-1) / denom
            )

        model.pool.update_ema(alpha_sum, cfg.beta_ema)
        metrics['loss'] = total_loss
        return metrics

    return train_step


def train_loop_1b(
    model, optimizer, data_iter, total_steps, log_every,
    tokenizer=None, generate_every=0, generate_prompt='',
):
    from src.training.trainer import generate

    cfg = model.config
    rng = np.random.default_rng(0)
    _rot = None

    step_phase1 = make_train_step(cfg, use_sigmoid=False)
    step_phase2 = make_train_step(cfg, use_sigmoid=True)

    step_times = []
    print_header = True
    t_start = time.time()

    for step, batch in zip(range(total_steps), data_iter):
        if isinstance(batch, tuple):
            batch = batch[0]

        t0 = time.time()
        use_sigmoid, lambda_sharp, lambda_entropy_eff = get_phase_params(step, cfg)

        if _rot is None:
            _rot = Phase1Rotator(cfg.N, batch.shape[0], cfg.k_max, seed=0)
        forced_idx = _rot.dummy() if (use_sigmoid or cfg.hybrid_train) else (_rot.next() if not use_sigmoid else _rot.dummy())

        train_fn = step_phase2 if use_sigmoid else step_phase1
        metrics = train_fn(
            model, optimizer, batch,
            lambda_sharp, lambda_entropy_eff, forced_idx
        )

        elapsed = time.time() - t0
        step_times.append(elapsed)

        if step % log_every == 0:
            m = jax.device_get(metrics)
            phase = 'phase2+' if use_sigmoid else 'phase1'

            if print_header:
                print(f'\n{"step":>6}  {"phase":<8}  {"loss":>8}  {"task":>8}  {"aux_total":>10}  {"lent":>6}  {"time":>6}  {"tok/s":>8}')
                print_header = False

            task_loss = float(m.get('task', 0))
            aux_total = float(m.get('aux_total', 0))
            total_loss = float(m['loss'])
            lent = lambda_entropy_eff

            recent = step_times[-min(log_every, len(step_times)):]
            avg_step_ms = np.mean(recent) * 1000
            tokens_per_sec = batch.shape[0] * batch.shape[1] / np.mean(recent)
            elapsed_total = time.time() - t_start

            print(f'{step:6d}  {phase:<8}  {total_loss:8.4f}  {task_loss:8.4f}  {aux_total:10.4f}  {lent:6.4f}  {avg_step_ms:5.0f}ms  {tokens_per_sec:8.0f}')

            if step % (log_every * 10) == 0 and step > 0:
                aux_keys = [k for k in sorted(m.keys()) if k.startswith('aux_')]
                aux_str = '  '.join(f'{k}={float(m[k]):.4f}' for k in aux_keys)
                print(f'         aux detail: {aux_str}')

        if generate_every > 0 and step > 0 and step % generate_every == 0 and tokenizer is not None:
            try:
                prompt_tokens = tokenizer.encode(generate_prompt)
                prompt_arr = jnp.array(prompt_tokens)[None, :]
                out = generate(model, prompt_arr, 100, temperature=0.8, hybrid=True)
                text = tokenizer.decode(out[0])
                print(f'\n  [{step}] >> {text[:200]}\n')
            except Exception as e:
                print(f'  [{step}] generation failed: {e}')

    elapsed_total = time.time() - t_start
    print(f'\n[done] {total_steps} steps in {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)')
    print(f'[done] avg step time: {np.mean(step_times)*1000:.0f}ms')


TOTAL_STEPS = 5000
BATCH_SIZE = 4
LOG_EVERY = 50
SEQ_LEN = 1024


if __name__ == '__main__':
    env_check()

    cfg = get_1b_config()
    cfg.max_seq_len = SEQ_LEN

    mesh = make_mesh()
    print(f'[mesh] {mesh}')

    rngs = nnx.Rngs(params=0, dropout=1)
    print('[model] initializing DWAModel (this may take a moment on TPU)...')
    model = DWAModel(cfg, rngs=rngs, mesh=mesh)

    n_params = count_params(model)
    print(f'[model] trainable params: {n_params:,}  (~{format_params(n_params)})')
    print(f'[model] N={cfg.N} (pool)  d_A={cfg.d_A}  d_B={cfg.d_B}')
    print(f'[model] N > d_A: {cfg.N > cfg.d_A}  N > d_B: {cfg.N > cfg.d_B}')

    print('[sharding] distributing parameters across TPU chips...')
    with mesh:
        shard_model(model, mesh)
    print('[sharding] done')

    optimizer = make_optimizer(model, cfg)
    shard_optimizer(optimizer, model, mesh)
    print('[optimizer] sharded and ready')

    cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ultra_fineweb_cache.npy')
    print(f'[data] loading cached dataset: {cache_path}')
    print(f'[data] tokenizer: LiquidAI/LFM2.5-1.2B-Thinking (vocab={cfg.d_input})')
    loader = CachedDataLoader(
        cache_path=cache_path,
        tokenizer_name='LiquidAI/LFM2.5-1.2B-Thinking',
        seq_len=cfg.max_seq_len,
        batch_size=BATCH_SIZE,
        seed=42,
    )
    tokenizer = loader.tokenizer
    first_batch = next(iter(loader))
    print(f'[data] first batch: shape={first_batch.shape} dtype={first_batch.dtype}')

    def data_iter():
        yield first_batch
        for batch in loader:
            yield batch

    print(f'\n[train] steps={TOTAL_STEPS}  batch={BATCH_SIZE}  seq={cfg.max_seq_len}')
    print(f'[train] hybrid={cfg.hybrid_train}  soft={cfg.soft_train}')
    print(f'[train] phase1_end={cfg.phase1_end}  phase2_end={cfg.phase2_end}')
    print('=' * 90)

    with mesh:
        train_loop_1b(
            model, optimizer, data_iter(),
            total_steps=TOTAL_STEPS,
            log_every=LOG_EVERY,
            tokenizer=tokenizer,
            generate_every=1000,
            generate_prompt='The ',
        )

    print('[complete] training finished.')