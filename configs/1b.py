from configs.small import DWAConfig


def get_1b_config() -> DWAConfig:
    """
    ~1B parameter DWA config — single TPU v5e or multi-GPU.

    Architecture:
      d_A = d_B = 768
      r = 8  (low rank keeps D manageable)
      D = 13056 = 2 * d_B*r + d_B = 768*8 + 8*768 + 768
      N = 32768 (large pool — bigger than both PartA and PartB)
      n_assembly_layers = 12

    Parameter budget per component:
      PartA (768×3072 + 3072×768 + norms):              ~202M
      PartB (768×3072 + 3072×65000 + norms):             ~202M
      12 DWABlocks (attn + query + retrieval + assembly): ~213M
      Pool (32768 × 13056):                               ~428M
      ──────────────────────────────────────────────────────
      Total:                                              ~1.045B

    Pool > PartA (428M > 202M) ✓
    Pool > PartB (428M > 202M) ✓

    Hybrid mode is recommended at this scale:
      Soft alpha memory:  (B, seq, 32768) = 128 MB/layer — HBM pressure
      Hybrid alpha memory: (B, seq, 32)   = 0.125 KB/layer — same as hard
    """
    return DWAConfig(
        d_input=65000,        # BPE vocab size (65K)
        d_A=768,
        d_B=768,
        D=13056,              # 768*8 + 8*768 + 768
        r=8,

        N=32768,
        k_max=32,

        S=8,
        d_k=128,
        T=1.0,

        n_heads=12,
        max_seq_len=2048,
        n_assembly_layers=12,

        phase1_end=2_000,
        phase2_end=16_000,

        lambda_util=0.01,
        lambda_div=0.005,
        lambda_norm=0.0005,
        lambda_sparse=0.001,
        lambda_entropy=0.02,
        beta_ema=0.99,

        reset_interval=0,           # disabled in hybrid mode
        reset_threshold=0.001,
        reset_noise=0.01,

        lr_pool=3e-5,
        lr_parts=1e-4,
        lr_retrieval=5e-5,
        lr_threshold_gamma=1e-3,

        soft_train=False,
        hybrid_train=True,          # recommended for 1B+ — soft alpha HBM
    )