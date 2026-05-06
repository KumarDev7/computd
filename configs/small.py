from dataclasses import dataclass


@dataclass
class DWAConfig:
    # --- Dimensions ---
    d_input: int = 64
    d_A: int = 64
    d_B: int = 64
    D: int = 2048           # must be >= d_B*r + r*d_A + d_B
    r: int = 4

    # --- Pool ---
    N: int = 512
    k_max: int = 8

    # --- Retrieval ---
    S: int = 2
    d_k: int = 32
    T: float = 1.0

    # --- Phase schedule ---
    phase1_end: int = 1_000
    phase2_end: int = 10_000

    # --- Aux loss weights ---
    lambda_util: float = 0.01
    lambda_div: float = 0.01
    lambda_norm: float = 0.001
    lambda_sparse: float = 0.001
    beta_ema: float = 0.99

    # --- Annealed entropy loss (gradient-connected, prevents premature collapse) ---
    # Annealing: phase1=0, phase2=lambda_entropy→0.3×, phase3=0.1×
    # 0.02 is the sweet spot: -9% ppl vs baseline, no degradation, more alive vectors
    lambda_entropy: float = 0.02

    # --- Codebook reset (optional, use only for longer runs > 10K steps) ---
    reset_interval: int = 0        # 0 = disabled; set 2000+ for long runs
    reset_threshold: float = 0.0001
    reset_noise: float = 0.01

    # --- Per-group learning rates ---
    lr_pool: float = 3e-5
    lr_parts: float = 1e-4
    lr_retrieval: float = 1e-4
    lr_threshold_gamma: float = 1e-3

    # --- PartA causal attention (0 = disabled) ---
    n_heads: int = 0
    max_seq_len: int = 256
    n_assembly_layers: int = 1

    # --- Training mode ---
    # soft_train=True   → soft dense pool (all N vectors, GEMMs only, TPU-optimal)
    # soft_train=False  → hard top-k + gather (current GPU mode)
    # hybrid_train=True → full GEMM compute over all N, keep only top-k for assembly
    #   Best of both worlds: MXU-saturated compute like soft, tiny alpha like hard.
    #   At small scale (N<2K) pure JAX top_k suffices; at 7B+ (N>8K) use Pallas fused kernel.
    # Same checkpoint works for all modes; switch flag between training and inference.
    soft_train: bool = False
    hybrid_train: bool = False
