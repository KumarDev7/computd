from dataclasses import dataclass
from configs.small import DWAConfig


def get_7b_config() -> DWAConfig:
    """
    ~7B parameter DWA config — requires multi-TPU (v5e-8+ or v5p-8+).

    Memory requirements:
      Params (bf16):          14 GB
      Optimizer (Adam, bf16): 28 GB
      Total static:           42 GB
      With 8-way TP sharding: ~5.3 GB per core (v5e: 16 GB/core)

    Architecture:
      d_A = d_B = 4096
      r = 8  (low rank keeps D manageable)
      D = 69632 = 2 * d_B*r + d_B
      N = 32768 (large pool for 7B knowledge capacity)
      n_assembly_layers = 18

    Hybrid mode is critical at this scale:
      Soft alpha memory: (B, seq, 32768) = 128 MB/layer — blows up HBM
      Hybrid alpha memory: (B, seq, 32) = 0.125 KB/layer — same as hard
    """
    return DWAConfig(
        d_input=65,           # char-level for testing; use 32000+ for BPE
        d_A=4096,
        d_B=4096,
        D=69632,              # 4096*8 + 8*4096 + 4096
        r=8,

        N=32768,
        k_max=32,

        S=16,
        d_k=128,
        T=1.0,

        n_heads=32,
        max_seq_len=2048,
        n_assembly_layers=18,

        phase1_end=2_000,
        phase2_end=16_000,

        lambda_util=0.01,
        lambda_div=0.005,
        lambda_norm=0.0005,
        lambda_sparse=0.001,
        lambda_entropy=0.02,
        beta_ema=0.99,

        reset_interval=0,           # not needed in hybrid/soft mode
        reset_threshold=0.001,
        reset_noise=0.01,

        lr_pool=3e-5,
        lr_parts=1e-4,
        lr_retrieval=5e-5,
        lr_threshold_gamma=1e-3,

        soft_train=False,
        hybrid_train=True,          # critical for 7B — soft would blow up HBM
    )


def get_1_5b_test_config() -> DWAConfig:
    """
    ~1.5B parameter config for testing hybrid mode on a single device.

    Fits in ~24 GB HBM (bf16 + Adam):
      Params (bf16):          3 GB
      Optimizer (Adam, bf16): 6 GB
      Activations:            ~6 GB (batch=4, seq=512)
      Total:                  ~15 GB

    N=8192 demonstrates hybrid's memory advantage:
      Soft alpha: (B, seq, 8192) = 32 MB/layer
      Hybrid alpha: (B, seq, 16) = 0.06 KB/layer — 500x less

    NOTE: The W_K key projection (S x d_k x D) is the memory bottleneck
    during backprop at this D size. For GPU with <24GB, use get_500m_test_config.
    """
    return DWAConfig(
        d_input=65,
        d_A=2048,
        d_B=2048,
        D=67584,              # 2048*16 + 16*2048 + 2048
        r=16,

        N=8192,
        k_max=16,

        S=8,
        d_k=64,
        T=1.0,

        n_heads=16,
        max_seq_len=512,
        n_assembly_layers=16,

        phase1_end=1_000,
        phase2_end=8_000,

        lambda_util=0.01,
        lambda_div=0.005,
        lambda_norm=0.001,
        lambda_sparse=0.001,
        lambda_entropy=0.02,
        beta_ema=0.99,

        reset_interval=0,
        reset_threshold=0.001,
        reset_noise=0.01,

        lr_pool=5e-5,
        lr_parts=2e-4,
        lr_retrieval=1e-4,
        lr_threshold_gamma=5e-3,

        soft_train=False,
        hybrid_train=True,
    )


def get_500m_test_config() -> DWAConfig:
    """
    ~18M parameter config for testing hybrid mode on a single GPU.

    Keeps D=3968 (proven shakespeare_v2 size) but scales N to 4096
    to demonstrate hybrid's memory advantage.

    N=4096 demonstrates hybrid's memory advantage:
      Soft alpha: (B, seq, 4096) = 16 MB/layer
      Hybrid alpha: (B, seq, 16) = 0.06 KB/layer — 256x less

    NOTE: For truly large models (7B), D scales with d_A*r making
    backward pass activations huge. This requires either:
      - Multi-TPU with tensor parallelism (shard pool across cores)
      - Pallas fused GEMM-topk kernel (never materialize full (B,seq,N))
      - Gradient checkpointing (trade compute for memory)
    """
    return DWAConfig(
        d_input=65,
        d_A=128,
        d_B=128,
        D=3968,               # same as shakespeare_v2 — keeps backward pass manageable
        r=15,

        N=4096,
        k_max=16,

        S=8,
        d_k=32,
        T=1.0,

        n_heads=4,
        max_seq_len=256,
        n_assembly_layers=2,

        phase1_end=1_000,
        phase2_end=8_000,

        lambda_util=0.01,
        lambda_div=0.005,
        lambda_norm=0.001,
        lambda_sparse=0.001,
        lambda_entropy=0.02,
        beta_ema=0.99,

        reset_interval=0,
        reset_threshold=0.001,
        reset_noise=0.01,

        lr_pool=1e-4,
        lr_parts=3e-4,
        lr_retrieval=1e-4,
        lr_threshold_gamma=1e-2,

        soft_train=False,
        hybrid_train=True,
    )