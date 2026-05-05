from configs.small import DWAConfig


def get_shakespeare_config() -> DWAConfig:
    """
    Config tuned for character-level Shakespeare training.
    D must satisfy: D >= d_B*r + r*d_A + d_B = 128*8 + 8*128 + 128 = 2176
    """
    return DWAConfig(
        # Dimensions
        d_input=65,         # char vocab size
        d_A=128,
        d_B=128,
        D=4096,             # pool vector dim (>= 2176 required)
        r=8,

        # Pool
        N=512,
        k_max=8,

        # Retrieval
        S=4,
        d_k=32,
        T=1.0,

        # PartA causal attention for cross-position context
        n_heads=4,
        max_seq_len=256,

        # Phase schedule (steps)
        phase1_end=500,
        phase2_end=8_000,

        # Aux losses
        lambda_util=0.01,
        lambda_div=0.005,
        lambda_norm=0.001,
        lambda_sparse=0.001,
        lambda_entropy=0.02,
        beta_ema=0.99,

        # Codebook reset (every 3K steps to keep pool alive)
        reset_interval=3_000,
        reset_threshold=0.0001,
        reset_noise=0.01,

        # Learning rates
        lr_pool=2e-5,
        lr_parts=3e-4,
        lr_retrieval=1e-4,
        lr_threshold_gamma=1e-3,
    )
