from configs.small import DWAConfig


def get_shakespeare_v2_config() -> DWAConfig:
    """
    Fixed config addressing all 7 diagnosed bottlenecks:
    1. lr_pool raised 5x (was 2e-5)
    2. W_base=0 + gamma=1.0 forced by assembly.py changes
    3. r=15 → rank-15 W_delta (was r=8, rank-8), D=3968 uses all dims for assembly
    4. n_assembly_layers=2 (was 1) for depth parity with Dense-Small
    5. Aggressive reset (every 500 steps, threshold=0.001)
    6. Phase-1 longer (1000 steps) + per-position rotation in dwa.py
    7. lr_threshold_gamma raised 10x for faster gamma adaptation
    """
    # D must = d_B*r + r*d_A + d_B for zero waste
    # With d_A=d_B=128, r=15: 128*15 + 15*128 + 128 = 1920+1920+128 = 3968
    return DWAConfig(
        d_input=65,
        d_A=128,
        d_B=128,
        D=3968,              # exact assembly dims — no wasted key-only dims
        r=15,                # was 8; W_delta now up to rank 120/128 (was 64/128)

        N=512,
        k_max=8,

        S=4,
        d_k=32,
        T=1.0,

        n_heads=4,
        max_seq_len=256,
        n_assembly_layers=2,  # was 1 (implicit)

        phase1_end=1_000,    # was 500; longer warmup for per-position coverage
        phase2_end=8_000,

        lambda_util=0.01,
        lambda_div=0.005,
        lambda_norm=0.001,
        lambda_sparse=0.001,
        lambda_entropy=0.02,
        beta_ema=0.99,

        reset_interval=500,       # was 3000; kill dead vectors aggressively
        reset_threshold=0.001,    # was 0.0001; 10x higher threshold
        reset_noise=0.01,

        lr_pool=1e-4,             # was 2e-5; 5x faster pool learning
        lr_parts=3e-4,
        lr_retrieval=1e-4,
        lr_threshold_gamma=1e-2,  # was 1e-3; 10x faster gamma adaptation
    )
