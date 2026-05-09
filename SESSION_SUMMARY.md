# DWA Project — Full Session Summary

**Last updated:** 2026-05-07  
**Branch:** master (kernel-optimization)  
**All commits pushed.**

---

## Main Goal

Build a **Dynamic Weight Assembly (DWA) language model** that can scale to **7 billion parameters on TPU v5e-8** (8 cores).

The core idea: instead of fixed weight matrices, every token position dynamically assembles its own weight matrix by retrieving vectors from a learned pool and combining them. This gives the model flexible, input-dependent computation with strong gradient signal to every part of the pool.

---

## Architecture in One Diagram

```
x (tokens)
  → one_hot
  → PartA (MLP + optional causal attention)
  → h_A

  → [DWABlock × n_layers]
       │
       ├── query_proj(h)  →  z  (what does this token need?)
       ├── retrieval: z scores against pool of N vectors
       │     → top-k indices + weights α
       ├── assembly: W = W_base + Σ αᵢ (Uᵢ @ Vᵢ)
       │     → apply W to h as residual
       └── h_out

  → PartB (linear → logits)
  → cross-entropy loss
```

**Pool** = N learned vectors, each of dim D.  
Each vector encodes a low-rank factor pair (U ∈ ℝ^(d_B×r), V ∈ ℝ^(r×d_A)) + bias b.  
Assembly builds W_delta = Σ αᵢ Uᵢ Vᵢ — a rank-r residual weight.

---

## Three Forward Modes — Why Each Exists

### Hard Mode (`soft_train=False, hybrid_train=False`)
**What:** Top-k retrieval + explicit gather of k vectors.  
**When:** GPU inference, testing.  
**Why:** Cheapest at inference. Only k=8–32 vectors accessed per step. Pool can stay on disk.  
**Problem at scale:** Top-k on discrete indices → dead vector problem (some vectors never selected → no gradient → atrophy). Required codebook resets.

### Soft Mode (`soft_train=True`)
**What:** All N vectors participate every step via dense GEMMs. No top-k, no gather.  
**When:** TPU training on small models (N ≤ 2K).  
**Why:** Dense GEMMs saturate the TPU MXU perfectly. Every vector gets gradient every step → no dead vectors, no resets needed.  
**Problem at scale:** alpha shape is `(batch, seq, N)`. At 7B (N=32768): **65 MB/layer** in HBM. With 18 layers = 1.2 GB just for alpha tensors. Blows up HBM.

### Hybrid Mode (`hybrid_train=True`) — **main 7B mode**
**What:** Compute all N dot products (MXU-saturated like soft) but only keep top-k for assembly (memory-efficient like hard).  
**Why built:** User asked "what if we combine soft compute density with hard mode memory efficiency?" This is the answer.  
**How:**
1. Full similarity GEMM over all N → `(batch, seq, N)` scores  
2. `jax.lax.top_k` immediately discards all but k_max → only `(batch, seq, k_max)` survives  
3. Assembly uses only the k selected vectors  

**Memory:** `(batch, seq, k_max)` = 0.03 MB/layer vs soft's 65 MB/layer at 7B.  
**Gradient:** All N vectors still get gradient through the full GEMM (like soft) → no dead vectors.  
**Result:** Best of both worlds. This is the core innovation.

---

## What Was Built This Session

### 1. Hybrid Forward Mode (`src/model/retrieval.py`)
Added `hybrid_forward()` to `MultiAspectRetrieval`.  
Wired through `DWABlock`, `DWAModel`, `trainer.train_step`, `generate`, `train_loop`.  
Fixed `diversity_loss` in `losses.py` to handle hybrid's mixed shapes: sims `(batch, N)` but idx `(batch, seq, k_max)`.  
Fixed EMA scatter in `trainer.py` for per-position hybrid idx shape.  

**Why the shape mismatch fix mattered:** The original `diversity_loss` assumed sims and idx shared the same leading dims. Hybrid's sims are sequence-averaged `(batch, N)` for aux losses, but idx is per-position `(batch, seq, k_max)`. Without the fix, the code crashed with incompatible broadcast shapes.

### 2. 7B + Test Configs (`configs/large_7b.py`)
Three configs:
- `get_7b_config()`: real 7B — N=32768, d_A=d_B=4096, r=8, 18 layers. Needs v5e-8.
- `get_1_5b_test_config()`: N=8192, d_A=d_B=2048. Tests hybrid on single large GPU.
- `get_500m_test_config()`: N=4096, D=3968 (same D as shakespeare_v2). Fits single GPU for hybrid testing.

**Why D=3968 for 500m test:** D scales with `d_A * r`. Larger D → W_K shape `(S, d_k, D)` → backward pass activations blow up. Multiple OOM failures at D=67584 and D=33792. Kept D=3968 (proven size) but scaled N to 4096 to demonstrate hybrid's memory advantage. The point is testing the pool-parallel *flow*, not parameter count.

### 3. Pallas Fused GEMM Kernel (`src/kernels/fused_retrieval.py`)
**Problem:** The two-einsum sequence in `hybrid_forward` and `soft_forward`:
```python
sims   = einsum('sbtk,snk->sbtn', queries, keys)  # (S, B, T, N) — 4 GB at 7B!
sims_w = einsum('s,sbtn->btn',    w,       sims)   # same tensor still in HBM
```
At 7B (S=8, B=4, T=2048, N=32768): `(8,4,2048,32768)*2 bytes = 4 GB` just for this one intermediate, then discarded.

**Fix:** `fused_weighted_similarity(queries, keys, w)` fuses both into one call:
- **TPU (Mosaic backend):** Pallas kernel tiles `(block_bt=128, block_n=512)` in VMEM. Accumulates S weighted dot-products in-place. Only `(block_bt, block_n)` lives in VMEM at once. Full `(B,T,N)` written back tile-by-tile. **8× peak HBM reduction** (4 GB → 0.5 GB).
- **GPU/CPU fallback:** Per-aspect einsum loop — still avoids `(S,B,T,N)` peak by accumulating into `(B,T,N)` one aspect at a time. Same 8× benefit, no Pallas needed.

Integrated into both `soft_forward` and `hybrid_forward` in `retrieval.py`.

### 4. Multi-TPU Pool-Parallel Sharding (`src/training/sharding.py`)
**Problem:** 7B pool `(32768, 69632)` in f32 = **8.6 GB**. One v5e core has 16 GB HBM. Pool alone + model params + activations = OOM.

**Mesh:** `data=2 × pool=4 = 8 cores` for v5e-8.
- `data` axis: shards batch (standard data parallelism).
- `pool` axis: shards pool vectors along N → each core holds `(8192, 69632)` = 2.1 GB.

**Pool-parallel retrieval** (`make_pool_parallel_retrieve`, uses `shard_map`):
```
Each core (pool shard j, data shard i):
  1. Local GEMM on pool slice (N/4, D) → local sims (B/2, T, N/4)
  2. lax.all_gather across pool axis → full sims (B/2, T, N)  [ICI comm]
  3. lax.top_k → global top-k indices (B/2, T, k_max)
  4. Masked lax.psum → gather only k_max vectors across shards  [ICI comm]
     (no full pool all-gather — k_max*D = ~4 MB at 7B vs 8.6 GB full pool)
  5. Assembly uses pre-gathered vectors — no second gather from sharded pool
```

**Why masked-psum for gather:** After global top-k, we need the actual pool vectors for assembly. Each shard contributes its local vectors (zeroed for out-of-shard indices), then `lax.psum` merges all contributions. Since each global index belongs to exactly one shard, there's no double-counting. Result: `(B/2, T, k_max, D)` correct vectors on every device. Cost = k_max*D communication, not N*D.

### 5. CPU-Init + Sharded Push (`init_model_cpu_sharded`)
**Problem:** Even with sharding ready, `DWAModel.__init__` still allocates the full pool on one device (default JAX device = first TPU core). 8.6 GB on a 16 GB core → likely OOM during init before sharding even runs.

**Solution — user's own idea:** "Init model in CPU RAM, apply sharding logic in CPU, push split parts to each TPU core."

Implementation:
```python
# 1. Init everything in CPU RAM — zero TPU usage
with jax.default_device(cpu):
    model = DWAModel(cfg, rngs)

# 2. Extract pool as numpy (stays in host DRAM)
pool_np = np.array(model.pool.vectors.value)  # 8.6 GB numpy array

# 3. Split along N axis in numpy (zero-copy slice)
for j in range(pool_size):
    shard = pool_np[j*N_per_shard : (j+1)*N_per_shard]  # 2.1 GB slice
    for i in range(data_size):
        device = mesh.devices[i, j]
        device_arrays.append(jax.device_put(shard, device))  # push to correct TPU

# 4. Assemble into global sharded JAX array
model.pool.vectors.value = jax.make_array_from_single_device_arrays(
    shape=(N, D), sharding=ctx.pool_vecs, arrays=device_arrays
)
```

Host CPU RAM needed: ~11 GB (pool 8.6 GB + other params 2 GB). TPU pod hosts have 200+ GB.

### 6. Gradient Sync Fixes
**Gap 1 (found and fixed):** Batch not sharded before step call. Raw numpy batch passed directly — XLA didn't know to split across data axis.  
Fix: `shard_batch(batch, ctx)` calls `jax.device_put(batch, ctx.batch)` before each step.

**Gap 2 (found and fixed):** No gradient all-reduce across data axis. Each data shard computed loss on its B/2 examples and took an independent gradient step → gradients would diverge across data replicas.  
Fix: `lax.with_sharding_constraint(task_loss, P())` inside `_loss_fn`. Forces the scalar loss to be replicated `P()` across all devices. XLA/GSPMD inserts an `all_reduce` at this point. Autodiff of `all_reduce` = `all_reduce` of upstream gradient → replicated params (W_Q, W_K, assembly, PartA/B) get correctly all-reduced gradients automatically.

**Pool param gradient routing:** Correct by construction via `shard_map` autodiff. The masked-psum in `pool_retrieve` reverses correctly through JAX's autodiff → each pool shard receives gradients only for its N/pool vectors.

### 7. TPU Entry Point (`scripts/train_tpu.py`)
Full training script that stitches everything together:
- Platform detection
- Config selection (500m / 1.5b / 7b)
- `init_model_cpu_sharded` for pool-parallel path
- `shard_batch` every step
- `make_sharded_train_step` for distributed step
- Falls back to standard `train_step` on single device

---

## Full Data/Gradient Flow on v5e-8

```
CPU HOST (200 GB RAM)
  Load shakespeare.txt, tokenize, yield batches (B=4, T=2048)
  shard_batch(batch, ctx) → jax.device_put → P('data', None)
    core [0,0] gets rows 0-1  |  core [1,0] gets rows 0-1  (pool shard 0)
    core [0,1] gets rows 0-1  |  core [1,1] gets rows 0-1  (pool shard 1)
    ...

TPU v5e-8 — mesh(data=2, pool=4):

  ┌─────────────────────────────────────────────────────────────────┐
  │  Each core (i, j): batch shard (B/2, T), pool shard (N/4, D)  │
  │                                                                  │
  │  FORWARD:                                                        │
  │    PartA (replicated weights) → h  (local, no ICI)             │
  │    query_proj (replicated) → z    (local, no ICI)              │
  │    pool_retrieve [shard_map]:                                    │
  │      local GEMM (N/4 keys)   → (B/2, T, N/4) sims  [local]    │
  │      lax.all_gather pool     → (B/2, T, N)   sims  [ICI: ~4MB]│
  │      lax.top_k               → (B/2, T, k_max)      [local]    │
  │      masked lax.psum         → (B/2, T, k_max, D)  [ICI: ~4MB]│
  │    assembly (pre-gathered)   → h_out           [local]          │
  │    PartB (replicated) → logits                  [local]          │
  │                                                                  │
  │  LOSS:                                                           │
  │    cross_entropy(logits, targets) = local_loss                  │
  │    with_sharding_constraint(loss, P())                          │
  │      → all_reduce across data axis               [ICI: scalar]  │
  │      → loss identical on all 8 cores                            │
  │                                                                  │
  │  BACKWARD (autodiff):                                            │
  │    grad(all_reduce) = all_reduce(upstream_grad) [ICI: param sz] │
  │    → W_Q, W_K, assembly, PartA/B: same grad on all cores ✓     │
  │    grad(masked psum) = route to correct pool shard ✓            │
  │    grad(all_gather) = reduce_scatter (auto) ✓                   │
  │                                                                  │
  │  OPTIMIZER:                                                      │
  │    replicated params: same update on all cores (stay in sync)   │
  │    pool shard j: update only its N/4 vectors                    │
  └─────────────────────────────────────────────────────────────────┘
```

---

## ICI Communication Summary per Step

| Operation | Size | Direction |
|---|---|---|
| all_gather (sims) | `(B/2, T, N/4) * pool_size` ≈ 4 MB | pool axis |
| masked psum (vecs) | `k_max * D * 2` ≈ 4 MB | pool axis |
| all_reduce (loss) | scalar × 8 | all |
| all_reduce (grads) | param count × 4 bytes ≈ 2 GB | data axis |

Grad all-reduce is largest but happens once per step, amortized by batch computation.

---

## Current File Map

| File | What it does |
|---|---|
| `src/model/dwa.py` | DWABlock, DWAModel — routes hard/soft/hybrid + pre_gathered_vecs |
| `src/model/retrieval.py` | MultiAspectRetrieval — hard, soft_forward, hybrid_forward (Pallas) |
| `src/model/assembly.py` | WeightAssembler — hard/soft/hybrid + pre_gathered_vecs bypass |
| `src/model/pool.py` | VectorPool with EMA |
| `src/model/parts.py` | PartA, PartB, CausalSelfAttention |
| `src/kernels/fused_retrieval.py` | Pallas fused GEMM (TPU Mosaic) + per-aspect einsum fallback |
| `src/training/sharding.py` | make_mesh, MeshContext, init_model_cpu_sharded, pool_retrieve, sharded_step |
| `src/training/trainer.py` | Phase schedule, train_step (hard/soft/hybrid), generate, train_loop |
| `src/training/losses.py` | All aux losses — handles all idx/alpha shapes |
| `configs/small.py` | DWAConfig dataclass |
| `configs/shakespeare_v2.py` | Small Shakespeare config |
| `configs/large_7b.py` | 7B, 1.5B, 500m configs |
| `scripts/train_tpu.py` | v5e-8 entry point — platform detect, mesh, sharded training loop |
| `data/shakespeare.txt` | Training data |

---

## Known Remaining Gaps

1. **No bfloat16** — all params in float32. TPU v5e MXU is fastest in bf16. Converting would halve pool memory (8.6 GB → 4.3 GB) and double MXU throughput. Need `jax.lax.convert_element_type` at init and careful loss scaling.

2. **No checkpoint save/load** — `train_loop` and `train_tpu.py` don't persist model state. For 7B training runs this is critical. Need `orbax.checkpoint` with mesh-aware sharded save/restore.

3. **No cosine LR decay** — val_ppl oscillates in late training. Cosine schedule in `optax` would stabilize.

4. **W_K not sharded** — `W_K (S, d_k, D)` = 284 MB at 7B, currently replicated on all 8 cores = 2.3 GB total. Could shard D axis across pool axis to reduce per-core footprint. Not critical but would help with activation memory during W_K grad computation.

5. **Pallas top_k not fused** — kernel currently fuses the S-weighted GEMM → `(B,T,N)`. A more aggressive kernel would also fuse the `top_k` → never materialize `(B,T,N)` in HBM at all. This saves another 0.5 GB/layer at 7B. Complex to implement (streaming top-k in VMEM), deferred to later.

6. **Data loader CPU-bound** — `shakespeare_loader` runs Python tokenization. At 7B scale, TPU step ~300 ms but host data prep could become bottleneck. Need async prefetch with `concurrent.futures.ThreadPoolExecutor`.

---

## How to Run

```bash
# Single GPU — hybrid mode test (500m config)
source .venv/bin/activate
python scripts/train_tpu.py --config 500m --steps 5000

# v5e-8 TPU — 7B, pool-parallel
python scripts/train_tpu.py --config 7b --steps 20000 --data_axis 2 --pool_axis 4

# Original small model (GPU, shakespeare)
python -c "
from configs.shakespeare_v2 import get_shakespeare_v2_config
from src.model.dwa import DWAModel
from src.training.trainer import make_optimizer, train_loop
from src.data.text_loader import shakespeare_loader
from flax import nnx

cfg = get_shakespeare_v2_config()
cfg.hybrid_train = True
model = DWAModel(cfg, nnx.Rngs(0))
opt = make_optimizer(model, cfg)
tokenizer = None
for _, tok in shakespeare_loader('data/shakespeare.txt', 32, cfg.max_seq_len, split='val'):
    tokenizer = tok; break
train_loop(model, opt,
    (b for b, _ in shakespeare_loader('data/shakespeare.txt', 32, cfg.max_seq_len, split='train')),
    total_steps=20000, log_every=500, generate_every=2000,
    tokenizer=tokenizer, generate_prompt='ROMEO:')
"
```
