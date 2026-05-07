"""
TPU training entry point for DWA — 7B or test config.

Usage:
    # Single device / GPU (test config, hybrid mode)
    python scripts/train_tpu.py --config 500m

    # v5e-8 (8 cores, 7B config, pool-parallel)
    python scripts/train_tpu.py --config 7b --steps 20000

    # v5e-8 (1.5B test, pool-parallel)
    python scripts/train_tpu.py --config 1_5b --steps 10000

Environment flags for TPU:
    XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true"
    JAX_ENABLE_X64=False
    XLA_PYTHON_CLIENT_PREALLOCATE=true
"""

import argparse
import time
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# ── Platform detection ────────────────────────────────────────────────────
PLATFORM = jax.devices()[0].platform
NUM_DEVICES = len(jax.devices())

print(f"[train_tpu] platform={PLATFORM}  devices={NUM_DEVICES}")
print(f"[train_tpu] JAX version: {jax.__version__}")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="500m",
                   choices=["500m", "1_5b", "7b"],
                   help="Model size config")
    p.add_argument("--steps",   type=int, default=20_000)
    p.add_argument("--batch",   type=int, default=4)
    p.add_argument("--log",     type=int, default=100)
    p.add_argument("--gen",     type=int, default=2000,
                   help="Generate text every N steps (0=off)")
    p.add_argument("--data_axis",  type=int, default=2)
    p.add_argument("--pool_axis",  type=int, default=4)
    p.add_argument("--no_shard",   action="store_true",
                   help="Disable pool sharding (data-parallel only)")
    p.add_argument("--data", default="data/shakespeare.txt")
    return p.parse_args()


def main():
    args = _parse_args()

    # ── Config ───────────────────────────────────────────────────────────
    from configs.large_7b import (
        get_500m_test_config, get_1_5b_test_config, get_7b_config,
    )
    cfg_map = {
        "500m":  get_500m_test_config,
        "1_5b":  get_1_5b_test_config,
        "7b":    get_7b_config,
    }
    cfg = cfg_map[args.config]()
    cfg.hybrid_train = True
    cfg.soft_train   = False
    print(f"[train_tpu] config={args.config}  N={cfg.N}  D={cfg.D}  "
          f"d_A={cfg.d_A}  layers={cfg.n_assembly_layers}")

    # ── Mesh + sharding context ───────────────────────────────────────────
    from src.training.sharding import make_mesh, MeshContext, shard_initial_state

    use_pool_parallel = (
        not args.no_shard
        and NUM_DEVICES >= args.data_axis * args.pool_axis
    )

    if use_pool_parallel:
        mesh = make_mesh(data=args.data_axis, pool=args.pool_axis)
        ctx  = MeshContext(mesh)
        print(f"[train_tpu] mesh={mesh.shape}  pool_parallel=True")
    else:
        print(f"[train_tpu] pool_parallel=False  (single-device or --no_shard)")
        mesh = ctx = None

    # ── Model + optimizer ────────────────────────────────────────────────
    model = nnx.Module.__new__(nnx.Module)  # avoids lint; DWAModel below
    from src.model.dwa import DWAModel
    from src.training.trainer import make_optimizer

    if use_pool_parallel:
        from src.training.sharding import init_model_cpu_sharded
        # Init in CPU RAM → split N axis → push each slice to its TPU core.
        # Never puts the full pool on a single device — no OOM at 7B.
        model = init_model_cpu_sharded(cfg, ctx, seed=0)
        print("[train_tpu] model init: CPU RAM → sharded TPU push complete")
    else:
        model = DWAModel(cfg, nnx.Rngs(0))

    opt = make_optimizer(model, cfg)

    # ── Data ─────────────────────────────────────────────────────────────
    from src.data.text_loader import shakespeare_loader

    tokenizer = None
    for _, tok in shakespeare_loader(args.data, args.batch, cfg.max_seq_len, split='val'):
        tokenizer = tok
        break

    def data_gen(split):
        while True:
            for batch, _ in shakespeare_loader(args.data, args.batch, cfg.max_seq_len, split=split):
                yield batch

    # ── Train step selection ──────────────────────────────────────────────
    if use_pool_parallel:
        from src.training.sharding import make_sharded_train_step, shard_batch
        train_step   = make_sharded_train_step(model, opt, cfg, ctx)
        _shard_batch = lambda b: shard_batch(b, ctx)
        print("[train_tpu] using pool-parallel sharded train step")
    else:
        from src.training.trainer import train_step
        _shard_batch = lambda b: b   # no-op
        print("[train_tpu] using standard nnx.jit train step")

    # ── Training loop ─────────────────────────────────────────────────────
    from src.training.trainer import (
        get_phase_params, Phase1Rotator, generate,
    )

    rng      = np.random.default_rng(0)
    rotator  = None
    data_it  = data_gen('train')

    print(f"[train_tpu] starting training: {args.steps} steps")
    t_start = time.time()

    for step, batch in zip(range(args.steps), data_it):
        use_sigmoid, lambda_sharp, lambda_entropy_eff = get_phase_params(step, cfg)

        # Rotator (only used in non-soft, non-hybrid hard warmup — skipped here)
        if rotator is None:
            rotator = Phase1Rotator(cfg.N, batch.shape[0], cfg.k_max, seed=0)
        forced_idx = rotator.dummy()

        t0    = time.time()
        batch = _shard_batch(batch)   # split across data axis, push to TPU cores
        metrics = train_step(
            model, opt, batch,
            use_sigmoid, False, lambda_sharp, lambda_entropy_eff, forced_idx, True,
        )

        if step % args.log == 0:
            m    = jax.device_get(metrics)
            mode = "pool-parallel" if use_pool_parallel else "hybrid"
            msg  = [f"step={step:05d}  mode={mode}"]
            for k, v in sorted(m.items()):
                msg.append(f"{k}={float(v):.4f}")
            msg.append(f"t={int((time.time()-t0)*1000)}ms")
            print("  ".join(msg))

        if args.gen > 0 and step > 0 and step % args.gen == 0 and tokenizer:
            prompt_tok = jnp.array(tokenizer.encode("ROMEO:"))[None, :]
            out  = generate(model, prompt_tok, 200, temperature=0.8, hybrid=True)
            text = tokenizer.decode(out[0])
            print(f"  [{step}] >> {text}")

    elapsed = time.time() - t_start
    print(f"[train_tpu] done  steps={args.steps}  wall={elapsed:.1f}s  "
          f"avg={elapsed/max(args.steps,1)*1000:.1f}ms/step")


if __name__ == "__main__":
    main()
