"""
TPU v5e-8 training entry point — 7B DWA model.

Designed for Kaggle TPU v5e-8 (8 cores).

The critical requirement is using `make_sharded_train_step` which routes
assembly through `pool_assemble` — a shard_map that gathers only D/pool_size
columns per chip and psums the tiny (B,T,d_B) h_delta result.
Without this, `vectors[idx]` materialises a 9 GB tensor and crashes.

Usage (Kaggle notebook cell):
    !python scripts/train_tpu_7b.py --steps 20000 --batch 16 --seq 512

Mesh:
    data_axis=2, pool_axis=4 → 8 chips total (v5e-8 default).
    pool axis shards the N=32768 vector pool across 4 chips.
    data axis shards the batch across 2 chips.
"""

import os
import gc
import time
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

# ── Environment flags — set BEFORE any JAX import ────────────────────────
# (These are no-ops if JAX is already initialised, but fine to set here.)
os.environ.setdefault("JAX_ENABLE_X64",                "False")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("JAX_DEBUG_NANS",                "False")  # flip to True to debug NaNs

# ── Startup diagnostics ───────────────────────────────────────────────────
PLATFORM    = jax.devices()[0].platform
NUM_DEVICES = len(jax.devices())
print(f"[env] platform={PLATFORM}  devices={NUM_DEVICES}  jax={jax.__version__}")

import flax, optax
print(f"[env] flax={flax.__version__}")
print(f"[env] optax={optax.__version__}")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps",      type=int, default=20_000)
    p.add_argument("--batch",      type=int, default=16,
                   help="Global batch size (split across data_axis chips)")
    p.add_argument("--seq",        type=int, default=512)
    p.add_argument("--log",        type=int, default=100)
    p.add_argument("--data_axis",  type=int, default=2,
                   help="Number of chips for data parallelism")
    p.add_argument("--pool_axis",  type=int, default=4,
                   help="Number of chips for pool sharding (N axis)")
    p.add_argument("--data",       default="data/shakespeare.txt")
    p.add_argument("--seed",       type=int, default=0)
    return p.parse_args()


def _env_check():
    """Fail-fast on version mismatches likely to cause silent errors."""
    import jaxlib
    print(f"[env] jaxlib={jaxlib.__version__}")
    major, minor = map(int, jax.__version__.split(".")[:2])
    if (major, minor) < (0, 4):
        raise RuntimeError(
            f"JAX >= 0.4.14 required for shard_map.  Got {jax.__version__}"
        )


def main():
    args = _parse_args()
    _env_check()

    required = args.data_axis * args.pool_axis
    if NUM_DEVICES < required:
        raise RuntimeError(
            f"Need {required} devices (data={args.data_axis} × pool={args.pool_axis}) "
            f"but only {NUM_DEVICES} available."
        )

    # ── Config ────────────────────────────────────────────────────────────
    from configs.large_7b import get_7b_config
    cfg = get_7b_config()
    cfg.hybrid_train = True
    cfg.soft_train   = False
    cfg.max_seq_len  = args.seq
    print(
        f"[config] N={cfg.N}  D={cfg.D}  d_A={cfg.d_A}  r={cfg.r}  "
        f"k_max={cfg.k_max}  layers={cfg.n_assembly_layers}"
    )

    # ── 2D Mesh: data × pool ──────────────────────────────────────────────
    # IMPORTANT: This MUST be a 2D mesh named ('data', 'pool').
    # Our sharding code uses these exact axis names.
    # Kaggle TPU v5e-8 exposes 8 chips regardless of how they label the topology.
    from src.training.sharding import (
        make_mesh, MeshContext,
        init_model_cpu_sharded,
        make_sharded_train_step,
        shard_batch,
    )
    mesh = make_mesh(data=args.data_axis, pool=args.pool_axis)
    ctx  = MeshContext(mesh)
    print(f"[mesh] {mesh}  data_size={ctx.data_size}  pool_size={ctx.pool_size}")

    # ── Model: init on CPU RAM, then push shards to TPU ───────────────────
    # init_model_cpu_sharded never puts the full pool on a single 16 GB chip:
    # it slices the N axis in numpy first, then device_puts each slice as bf16.
    print("[model] initialising on CPU RAM...")
    model = init_model_cpu_sharded(cfg, ctx, seed=args.seed)

    n_params = sum(
        x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    )
    print(f"[model] trainable params: {n_params:,}  (~{n_params/1e9:.2f}B)")

    # ── Optimizer: JIT-compiled init so Adam state is sharded ────────────
    from src.training.trainer import make_optimizer
    opt = make_optimizer(model, cfg, mesh=mesh)
    gc.collect()

    # ── Sharded train step ────────────────────────────────────────────────
    # make_sharded_train_step returns a function that:
    #   1. Calls pool_retrieve shard_map → top_idx, alpha  (no big gather)
    #   2. Calls pool_assemble shard_map → gathers D/pool_size cols per chip,
    #      runs partial einsum, psums tiny (B,T,d_B) result
    # This eliminates the 9 GB (B,T,k_max,D) tensor entirely.
    train_fn = make_sharded_train_step(model, opt, cfg, ctx)
    print("[train] using pool-parallel sharded train step (9 GB gather eliminated)")

    # ── Data ──────────────────────────────────────────────────────────────
    from src.data.text_loader import shakespeare_loader
    from src.training.trainer import get_phase_params, Phase1Rotator

    tokenizer = None
    for _, tok in shakespeare_loader(args.data, 1, cfg.max_seq_len, split='val'):
        tokenizer = tok
        break

    def _data_gen():
        while True:
            for batch, _ in shakespeare_loader(
                args.data, args.batch, cfg.max_seq_len, split='train'
            ):
                yield batch

    data_it = _data_gen()
    rotator = Phase1Rotator(cfg.N, args.batch, cfg.k_max, seed=args.seed)

    print(f"[train] steps={args.steps}  batch={args.batch}  seq={cfg.max_seq_len}")
    print("─" * 70)

    t_start = time.time()

    for step, raw_batch in zip(range(args.steps), data_it):
        use_sigmoid, lambda_sharp, lambda_entropy_eff = get_phase_params(step, cfg)

        # Rotator provides forced_idx for phase-1 warmup;
        # in hybrid mode it's ignored by the model but must be passed.
        forced_idx = rotator.dummy()

        t0    = time.time()
        # shard_batch splits batch across the data axis (2 chips get B/2 rows each).
        batch = shard_batch(raw_batch, ctx)

        metrics = train_fn(
            model, opt, batch,
            use_sigmoid,     # static → at most 2 recompilations total
            False,           # soft=False
            lambda_sharp,
            lambda_entropy_eff,
            forced_idx,
            True,            # hybrid=True — static
        )

        if step % args.log == 0:
            m   = jax.device_get(metrics)
            msg = [f"step={step:05d}"]
            for k, v in sorted(m.items()):
                msg.append(f"{k}={float(v):.4f}")
            msg.append(f"t={int((time.time()-t0)*1000)}ms")
            print("  ".join(msg))

    elapsed = time.time() - t_start
    print(
        f"[train] done  steps={args.steps}  wall={elapsed:.1f}s  "
        f"avg={elapsed/max(args.steps,1)*1000:.1f}ms/step"
    )


if __name__ == "__main__":
    main()
