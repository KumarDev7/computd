"""
TPU v5e-8 training entry point for the 1B DWA model.

Architecture (from configs/1b.py):
  d_A = d_B = 768,  r = 8,  D = 13056,  N = 32768
  n_assembly_layers = 12,  n_heads = 12
  12 DWABlocks (attn + query + retrieval + assembly): ~213M
  Pool (32768 × 13056):                                ~428M
  PartA + PartB:                                       ~404M
  Total:                                               ~1.045B

  Pool (428M) > PartA (202M) and Pool > PartB (202M) ✓

Usage:
    # On a TPU v5e-8 (8 cores, 16 GB each):
    python scripts/train_tpu_1b.py --steps 20000 --batch 16 --seq 512

    # Adjust mesh shape for available devices:
    python scripts/train_tpu_1b.py --data_axis 2 --pool_axis 4

    # Single-GPU smoke test (no pool sharding):
    python scripts/train_tpu_1b.py --no_shard --batch 2 --seq 128 --steps 100

Environment flags (set before JAX init):
    XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true"
    JAX_ENABLE_X64=False
    XLA_PYTHON_CLIENT_PREALLOCATE=true
"""

import argparse
import time
import importlib
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

PLATFORM = jax.devices()[0].platform
NUM_DEVICES = len(jax.devices())

print(f"[1b] platform={PLATFORM}  devices={NUM_DEVICES}  jax={jax.__version__}")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps",      type=int, default=20_000)
    p.add_argument("--batch",      type=int, default=16,
                   help="Global batch size (split across data_axis chips)")
    p.add_argument("--seq",        type=int, default=512)
    p.add_argument("--log",        type=int, default=100)
    p.add_argument("--gen",        type=int, default=2000,
                   help="Generate text every N steps (0=off)")
    p.add_argument("--data_axis",  type=int, default=2,
                   help="Number of chips for data parallelism")
    p.add_argument("--pool_axis",  type=int, default=4,
                   help="Number of chips for pool sharding (N axis)")
    p.add_argument("--no_shard",   action="store_true",
                   help="Disable pool sharding (data-parallel only)")
    p.add_argument("--data",       default="data/shakespeare.txt")
    p.add_argument("--seed",       type=int, default=0)
    return p.parse_args()


def main():
    args = _parse_args()

    # ── Config ─────────────────────────────────────────────────────────────
    cfg_mod = importlib.import_module("configs.1b")
    cfg = cfg_mod.get_1b_config()
    cfg.hybrid_train = True
    cfg.soft_train   = False
    cfg.max_seq_len  = args.seq
    print(
        f"[1b] N={cfg.N}  D={cfg.D}  d_A={cfg.d_A}  r={cfg.r}  "
        f"k_max={cfg.k_max}  layers={cfg.n_assembly_layers}  "
        f"vocab={cfg.d_input}"
    )

    # ── Mesh + sharding context ────────────────────────────────────────────
    use_pool_parallel = (
        not args.no_shard
        and NUM_DEVICES >= args.data_axis * args.pool_axis
    )

    if use_pool_parallel:
        from src.training.sharding import make_mesh, MeshContext, shard_initial_state
        mesh = make_mesh(data=args.data_axis, pool=args.pool_axis)
        ctx  = MeshContext(mesh)
        print(f"[1b] mesh={mesh.shape}  pool_parallel=True")
    else:
        print(f"[1b] pool_parallel=False  (single-device or --no_shard)")
        mesh = ctx = None

    # ── Model + optimizer ─────────────────────────────────────────────────
    from src.model.dwa import DWAModel
    from src.training.trainer import make_optimizer

    if use_pool_parallel:
        from src.training.sharding import init_model_cpu_sharded
        model = init_model_cpu_sharded(cfg, ctx, seed=args.seed)
        print("[1b] model init: CPU RAM → sharded TPU push complete")
    else:
        model = DWAModel(cfg, nnx.Rngs(args.seed))

    n_params = sum(
        x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    )
    print(f"[1b] trainable params: {n_params:,}  (~{n_params/1e9:.2f}B)")

    opt = make_optimizer(model, cfg, mesh=mesh if use_pool_parallel else None)

    # ── Data ───────────────────────────────────────────────────────────────
    # 1B model uses 65K vocab (BPE). The text_loader provides char-level
    # tokens suitable for d_input=65 configs. For a real 65K vocab run,
    # replace this with a BPE tokenizer (e.g., sentencepiece, tiktoken).
    # Using random_token_batches for a smoke test with the full vocab size.
    from src.data.loader import random_token_batches

    batch_size = args.batch
    if use_pool_parallel:
        batch_size = args.batch // ctx.data_size

    data_gen = random_token_batches(cfg.d_input, batch_size, cfg.max_seq_len, seed=args.seed)
    print(f"[1b] data: random tokens  vocab={cfg.d_input}  batch={batch_size}")

    # ── Train step selection ───────────────────────────────────────────────
    if use_pool_parallel:
        from src.training.sharding import make_sharded_train_step, shard_batch
        train_step   = make_sharded_train_step(model, opt, cfg, ctx)
        _shard_batch = lambda b: shard_batch(b, ctx)
        print("[1b] using pool-parallel sharded train step")
    else:
        from src.training.trainer import train_step
        _shard_batch = lambda b: b
        print("[1b] using standard nnx.jit train step")

    # ── Training loop ──────────────────────────────────────────────────────
    from src.training.trainer import get_phase_params, Phase1Rotator

    rng      = np.random.default_rng(args.seed)
    rotator  = None
    data_it  = iter(data_gen)

    print(f"[1b] starting training: {args.steps} steps")

    # ── JIT warmup ──────────────────────────────────────────────────────────
    # train_step has static_argnums for (use_sigmoid, soft, hybrid).  When
    # use_sigmoid flips False→True at step phase1_end (~2000) XLA recompiles
    # the full step, causing a multi-minute mid-training stall.
    # Pre-compiling both variants here moves those stalls to startup where
    # they're expected and can be logged clearly.
    print("[1b] JIT warmup: compiling phase-1 (use_sigmoid=False) ...")
    _wup_batch = jnp.zeros((batch_size, cfg.max_seq_len + 1), dtype=jnp.int32)
    _wup_idx   = jnp.zeros((batch_size, cfg.k_max), dtype=jnp.int32)
    _ = train_step(model, opt, _wup_batch,
                   False, False, 1.0, 0.0, _wup_idx, True)
    print("[1b] JIT warmup: compiling phase-2 (use_sigmoid=True) ...")
    _ = train_step(model, opt, _wup_batch,
                   True, False, 1.0, 0.0, _wup_idx, True)
    del _wup_batch, _wup_idx
    print("[1b] JIT warmup done — no recompilation stalls during training")
    print("-" * 70)
    t_start = time.time()

    for step in range(args.steps):
        raw_batch = next(data_it)
        use_sigmoid, lambda_sharp, lambda_entropy_eff = get_phase_params(step, cfg)

        if rotator is None:
            actual_batch = raw_batch.shape[0] if use_pool_parallel else args.batch
            rotator = Phase1Rotator(cfg.N, actual_batch, cfg.k_max, seed=args.seed)
        forced_idx = rotator.dummy()

        t0    = time.time()
        batch = _shard_batch(raw_batch)

        metrics = train_step(
            model, opt, batch,
            use_sigmoid,
            False,
            lambda_sharp,
            lambda_entropy_eff,
            forced_idx,
            True,
        )

        if step % args.log == 0:
            m    = jax.device_get(metrics)
            mode = "pool-parallel" if use_pool_parallel else "hybrid"
            msg  = [f"step={step:05d}  mode={mode}"]
            for k, v in sorted(m.items()):
                msg.append(f"{k}={float(v):.4f}")
            msg.append(f"t={int((time.time()-t0)*1000)}ms")
            print("  ".join(msg))

    elapsed = time.time() - t_start
    print(
        f"[1b] done  steps={args.steps}  wall={elapsed:.1f}s  "
        f"avg={elapsed/max(args.steps,1)*1000:.1f}ms/step"
    )


if __name__ == "__main__":
    main()