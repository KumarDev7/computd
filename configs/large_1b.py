"""
~1B parameter DWA config for TPU v5e-8 training.

Architecture:
  d_A = d_B = 1280
  r = 8
  D = 21760 (d_B*r + r*d_A + d_B = 10240 + 10240 + 1280)
  N = 16384 (pool size > d_A=1280 and > d_B=1280)
  n_assembly_layers = 12
  vocab = 64400 (LFM2.5 tokenizer)
  use_embedding = True (avoids 64K-dim one_hot, uses Embed lookup instead)

Parameter breakdown:
  Embed+Projection:  ~67M
  Vector Pool:        ~357M  (N * D = 16384 * 21760)
  12 DWA Blocks:     ~613M
  PartB MLP+Head:    ~67M
  ========================================
  Total:              ~1.04B

Memory requirements (bf16 + Adam):
  Params (bf16):          ~2 GB
  Optimizer (Adam, bf16):  ~4 GB
  Activations:             ~4 GB (batch=8, seq=1024)
  Total:                   ~10 GB per chip with 8-way TP sharding
"""
from dataclasses import dataclass
from configs.small import DWAConfig


def get_1b_config() -> DWAConfig:
    return DWAConfig(
        d_input=64400,
        d_A=1280,
        d_B=1280,
        D=21760,
        r=8,

        N=16384,
        k_max=16,

        S=8,
        d_k=64,
        T=1.0,

        n_heads=16,
        max_seq_len=1024,
        n_assembly_layers=12,

        phase1_end=500,
        phase2_end=4000,

        lambda_util=0.01,
        lambda_div=0.005,
        lambda_norm=0.0005,
        lambda_sparse=0.001,
        lambda_entropy=0.02,
        beta_ema=0.99,

        reset_interval=0,
        reset_threshold=0.001,
        reset_noise=0.01,

        lr_pool=3e-5,
        lr_parts=1e-4,
        lr_retrieval=5e-5,
        lr_threshold_gamma=1e-3,

        soft_train=False,
        hybrid_train=True,
        use_embedding=True,
    )