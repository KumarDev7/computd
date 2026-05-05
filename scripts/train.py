"""
Entry point for DWA training on the synthetic copy task (GPU/CPU).
Small validation config: D=2048, d_A=d_B=64, r=4, N=512, k_max=8, S=2
"""
import os
import sys

# JAX performance + debug flags
os.environ.setdefault("JAX_ENABLE_X64", "False")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("JAX_DEBUG_NANS", "True")
os.environ.setdefault("JAX_LOG_COMPILES", "True")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
from flax import nnx

from configs.small import DWAConfig
from src.model.dwa import DWAModel
from src.training.trainer import make_optimizer, train_loop
from src.data.loader import synthetic_copy_task


def env_check():
    platform = jax.devices()[0].platform
    print(f"[env] platform={platform}  devices={len(jax.devices())}  "
          f"jax={jax.__version__}")
    for lib in ("flax", "optax"):
        try:
            import importlib
            m = importlib.import_module(lib)
            print(f"[env] {lib}={m.__version__}")
        except Exception:
            pass


def count_params(model: nnx.Module) -> int:
    graphdef, state = nnx.split(model)
    params = state.filter(nnx.Param)
    return sum(x.size for x in jax.tree.leaves(params))


def main():
    env_check()

    cfg = DWAConfig(
        d_input=64,
        d_A=64,
        d_B=64,
        D=2048,
        r=4,
        N=512,
        k_max=8,
        S=2,
        d_k=32,
        T=1.0,
        phase1_end=100,   # scaled down for quick validation
        phase2_end=1_000,
    )

    print(f"\n[config] N={cfg.N}  D={cfg.D}  r={cfg.r}  "
          f"k_max={cfg.k_max}  S={cfg.S}")

    rngs = nnx.Rngs(params=0, dropout=1)
    model = DWAModel(cfg, rngs=rngs)

    n_params = count_params(model)
    print(f"[model] trainable params: {n_params:,}  (~{n_params/1e6:.2f}M)")

    optimizer = make_optimizer(model, cfg)

    # Vocab = d_input (one-hot encoding)
    vocab_size = cfg.d_input
    batch_size = 32
    seq_len = 16       # short for GPU memory
    total_steps = 2_000

    data = synthetic_copy_task(vocab_size, batch_size, seq_len, seed=42)

    print(f"\n[train] steps={total_steps}  batch={batch_size}  seq={seq_len}")
    print("─" * 60)

    train_loop(model, optimizer, data, total_steps=total_steps, log_every=50)

    print("─" * 60)
    print("[done] training complete")

    # Quick sanity: run inference
    dummy = jnp.zeros((1, seq_len - 1, vocab_size))
    logits = model(dummy, use_sigmoid=True, lambda_sharp=10.0, return_aux=False)
    print(f"[sanity] output logits shape: {logits.shape}  "
          f"max={float(logits.max()):.3f}")


if __name__ == "__main__":
    main()
