"""
~1B parameter DWA config for TPU v5e-8 training.

Architecture:
  d_A = d_B = 1024
  r = 8
  D = 17408 (d_B*r + r*d_A + d_B = 8192 + 8192 + 1024)
  N = 16384 (pool size > d_A and > d_B)
  n_assembly_layers = 12
  vocab = 64400 (LFM2.5 tokenizer)

Parameter breakdown:
  Embedding (PartA): ~67M
  Head (PartB):       ~67M
  Vector Pool:        ~285M  (N * D = 16384 * 17408)
  12 DWA Blocks:      ~591M
  ========================================
  Total:              ~1.01B

Memory requirements (bf16 + Adam):
  Params (bf16):          ~2 GB
  Optimizer (Adam, bf16):  ~4 GB
  Activations:             ~6 GB (batch=8, seq=2048)
  Total:                   ~12 GB per chip with 8-way TP sharding

Hybrid mode is required at this scale:
  Soft alpha: (B, seq, 16384) = 256 MB/layer — HBM pressure
  Hybrid alpha: (B, seq, 16)  = 0.25 KB/layer — 1000x less
"""
from dataclasses import dataclass
from configs.small import DWAConfig


def get_1b_config() -> DWAConfig:
    return DWAConfig(
        # Vocabulary size from LFM2.5 tokenizer
        d_input=64400,

        # --- Core dimensions ---
        d_A=1024,
        d_B=1024,
        r=8,                                # low-rank factor for W_delta
        D=17408,                            # d_B*r + r*d_A + d_B = exact zero-waste

        # --- Pool ---
        N=16384,                             # pool size: > d_A (1024) and > d_B (1024)
        k_max=16,                            # top-k vectors per position

        # --- Retrieval ---
        S=8,                                 # number of retrieval aspects
        d_k=64,                              # key/query dimension per aspect
        T=1.0,                               # temperature for softmax

        # --- Architecture ---
        n_heads=16,                          # causal self-attention heads (64 dim each)
        max_seq_len=2048,
        n_assembly_layers=12,

        # --- Phase schedule ---
        phase1_end=500,                      # forced rotation warmup
        phase2_end=4000,                     # sigmoid gate ramp-up

        # --- Aux loss weights ---
        lambda_util=0.01,
        lambda_div=0.005,
        lambda_norm=0.0005,
        lambda_sparse=0.001,
        lambda_entropy=0.02,
        beta_ema=0.99,

        # --- Codebook reset (not needed in hybrid mode) ---
        reset_interval=0,
        reset_threshold=0.001,
        reset_noise=0.01,

        # --- Per-group learning rates ---
        lr_pool=3e-5,
        lr_parts=1e-4,
        lr_retrieval=5e-5,
        lr_threshold_gamma=1e-3,

        # --- Training mode ---
        # hybrid_train=True is critical for N=16384:
        # soft would create (B, seq, 16384) tensors per layer → HBM blow-up
        soft_train=False,
        hybrid_train=True,
    )