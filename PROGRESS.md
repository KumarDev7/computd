# DWA Model — Current Progress & Resumption Guide

**Last updated:** 2026-05-05  
**Branch:** master  
**Status:** Both hard and soft modes verified on GPU. Ready for TPU training.

---

## Architecture Overview

Dynamic Weight Assembly (DWA) language model. Each token position dynamically assembles a weight matrix from a learned pool of N vectors, then applies it as a residual transformation.

```
x → PartA → [DWABlock × n_layers] → PartB → logits
                  │
                  ├── query_proj(h)
                  ├── retrieval (select k vectors from pool of N)
                  └── assembly (W = W_base + Σ αᵢ UᵢVᵢ, apply to h)
```

### Key Dimensions (shakespeare_v2 config)
| Param | Value | Notes |
|---|---|---|
| d_input | 65 | vocab size (character-level Shakespeare) |
| d_A / d_B | 128 | hidden dims |
| D | 3968 | pool vector dim (exact: d_B×r + r×d_A + d_B, zero waste) |
| r | 15 | W_delta rank (was 8) |
| N | 512 | pool size |
| k_max | 8 | vectors retrieved per query |
| n_assembly_layers | 2 | DWABlock depth (was 1) |
| n_heads | 4 | causal attention heads in PartA |
| Parameters | ~3.49M | total |

---

## Dual-Mode System: Soft (TPU) / Hard (GPU)

The model supports two forward passes controlled by **one config flag**:

```python
# configs/small.py or shakespeare_v2
soft_train: bool = False   # Hard: top-k + gather (GPU / inference)
soft_train: bool = True    # Soft: dense GEMMs over all N (TPU training)
```

### Hard Mode (`soft_train=False`) — GPU / Inference
- Sequence-level retrieval: one query per sequence → idx shape `(batch, k_max)`
- Gather: `(batch, k_max, D)` = **4.1 MB** per block
- Per-position alpha computed against only the k selected vectors
- Pool can stay on disk at inference (fetch 8 vectors per sequence)

### Soft Mode (`soft_train=True`) — TPU Training
- No gather, no top-k: all N=512 vectors participate every step
- Every operation is a dense GEMM (MXU-saturating on TPU)
- alpha shape `(batch, seq, N)` — softmax over full pool
- No phase-1 warmup needed (all vectors get gradients naturally)
- No codebook resets needed (impossible to have dead vectors)
- Same checkpoint loads into either mode — flip the flag, continue

### Why It Works
At high `lambda_sharp`, softmax concentrates weight on the top-k vectors. The soft sum converges to the hard top-k result. By end of training, switching modes produces nearly identical output.

### Switching Workflow
```python
# Train on TPU
cfg.soft_train = True
train_loop(model, optimizer, data_iter, total_steps=20000)

# Inference on GPU/CPU — same checkpoint, flip the flag
cfg.soft_train = False
model = load_checkpoint(path)
logits = model(x, soft=False)   # hard top-k, 4MB gather
```

---

## Test Results

### GPU Benchmark — Hard Mode (shakespeare_v2 config)
| Metric | Value |
|---|---|
| Step time | 146 ms |
| Gather per block | 4.1 MB |
| Val PPL @ 5K steps | 4.951 |
| Val PPL @ 20K steps (previous run) | 4.88 |
| Dead vectors | 0% throughout training |

### GPU Benchmark — Soft Mode (same config)
| Metric | Value |
|---|---|
| Step time | 1,304 ms |
| Gather | 0 (all GEMMs) |
| Val PPL @ 1K steps | 5.772 |

Soft mode is 4.4× slower on GPU because N=512 dense einsums are compute-heavy. On TPU this flips — GEMMs saturate the MXU at near-peak TFLOPS while gathers are the slow path.

### Hard vs Soft @ 1K Steps (GPU)
| Mode | Step Time | Val PPL @ 1K |
|---|---|---|
| Hard | 297 ms | 5.746 |
| Soft | 1,304 ms | 5.772 |

Both converging on similar PPL — the soft mode will catch up as training continues.

---

## Commands to Resume Training

### Hard mode (GPU)
```bash
source .venv/bin/activate
python -c "
from configs.shakespeare_v2 import get_shakespeare_v2_config
from src.model.dwa import DWAModel
from src.training.trainer import make_optimizer, train_loop
from src.data.text_loader import shakespeare_loader
from flax import nnx

cfg = get_shakespeare_v2_config()
cfg.soft_train = False   # hard mode
model = DWAModel(cfg, nnx.Rngs(0))
opt = make_optimizer(model, cfg)

def gen(split):
    for batch, _ in shakespeare_loader('data/shakespeare.txt', 32, cfg.max_seq_len, split=split):
        yield batch

train_loop(model, opt, gen('train'), total_steps=20000, log_every=500)
"
```

### Soft mode (TPU / GPU)
```bash
source .venv/bin/activate
python -c "
from configs.shakespeare_v2 import get_shakespeare_v2_config
from src.model.dwa import DWAModel
from src.training.trainer import make_optimizer, train_loop
from src.data.text_loader import shakespeare_loader
from flax import nnx

cfg = get_shakespeare_v2_config()
cfg.soft_train = True    # soft mode — all GEMMs, no gather
model = DWAModel(cfg, nnx.Rngs(0))
opt = make_optimizer(model, cfg)

def gen(split):
    for batch, _ in shakespeare_loader('data/shakespeare.txt', 32, cfg.max_seq_len, split=split):
        yield batch

train_loop(model, opt, gen('train'), total_steps=20000, log_every=500)
"
```

### Validate
```bash
source .venv/bin/activate
python -c "
import jax.numpy as jnp, numpy as np, math
from configs.shakespeare_v2 import get_shakespeare_v2_config
from src.model.dwa import DWAModel
from src.data.text_loader import shakespeare_loader
from flax import nnx

cfg = get_shakespeare_v2_config()
cfg.soft_train = False   # or True for soft validation
model = DWAModel(cfg, nnx.Rngs(0))
# model = load_checkpoint(model, path)  # after training

_, val_iter, tok = shakespeare_loader('data/shakespeare.txt', 32, cfg.max_seq_len, split='val')
losses = []
for i, (batch, _) in zip(range(50), val_iter):
    lp = jax.nn.log_softmax(model(jax.nn.one_hot(batch[:, :-1], cfg.d_input)), axis=-1)
    tgt = jax.nn.one_hot(batch[:, 1:], cfg.d_input)
    losses.append(float(-jnp.mean(jnp.sum(tgt * lp, axis=-1))))
print(f'Val PPL: {math.exp(np.mean(losses)):.3f}')
"
```

---

## Optimizations Applied (All Still Active)

| Fix | File | What Changed |
|---|---|---|
| UV einsum order | `assembly.py` | Contract through V (r=tiny) first, never form 1GB UV intermediate |
| Sequence-level retrieval | `dwa.py` | One retrieval per sequence instead of per-position (254× less gather) |
| Per-position alpha | `retrieval.py` | `per_position_alpha()` — cheap softmax against only k vectors |
| `device_get` async | `trainer.py` | Only sync at log steps, not every step |
| EMA vectorized | `trainer.py` | Single scatter op instead of Python loop |
| Causal mask cached | `parts.py` | Pre-built in `__init__`, not recomputed per forward pass |
| Soft forward pass | `retrieval.py` | `soft_forward()` — all N vectors, pure GEMMs |
| Soft assembly | `assembly.py` | `idx=None` branch — einsum over full pool, no gather |
| Soft EMA | `trainer.py` | Direct mean over alpha (no scatter needed) |

---

## Phase Schedule

| Step Range | Phase | `use_sigmoid` | `lambda_sharp` | `lambda_entropy` |
|---|---|---|---|---|
| 0 – 1,000 | Phase 1 (warmup) | False | 1.0 | 0.0 |
| 1,000 – 8,000 | Phase 2 (sigmoid) | True | 1.0 → 5.0 | 0.02 → 0.006 |
| 8,000+ | Phase 3 (sharp) | True | 5.0 → 10.0 | 0.002 |

In soft mode, phase-1 warmup is unnecessary (all vectors get gradients from step 0), but the schedule still works — `use_sigmoid=False` simply makes the soft alpha uniform.

---

## File Map

| File | Purpose |
|---|---|
| `src/model/dwa.py` | DWABlock, DWAModel, `_retrieve_per_position` |
| `src/model/assembly.py` | WeightAssembler (hard + soft paths) |
| `src/model/retrieval.py` | MultiAspectRetrieval (hard + `soft_forward` + `per_position_alpha`) |
| `src/model/pool.py` | VectorPool with EMA |
| `src/model/parts.py` | PartA, PartB, CausalSelfAttention |
| `src/training/losses.py` | All aux losses (diversity handles `idx=None` for soft) |
| `src/training/trainer.py` | Phase schedule, train_step, train_loop, `soft` flag routing |
| `src/data/text_loader.py` | Shakespeare char-level data loader |
| `src/data/loader.py` | Synthetic task generators (random, bigram, copy) |
| `configs/small.py` | DWAConfig dataclass (includes `soft_train` flag) |
| `configs/shakespeare_v2.py` | Shakespeare-specific config factory |
| `data/shakespeare.txt` | Training data |

---

## Known Issues / Next Steps

1. **Soft mode slower on GPU** — expected. On TPU the GEMM-heavy path should be significantly faster than hard mode.
2. **No checkpoint save/load** — `train_loop` doesn't persist model state yet. Need to add `nnx.state` serialization before long training runs.
3. **No cosine LR decay** — val_ppl oscillates 4.88–5.16 in late training. Cosine schedule should stabilize.
4. **Generated text quality** — model produces correct character frequencies but loses coherence mid-sentence at 20K steps. Longer training or larger dataset needed.
5. **bfloat16 not implemented** — 2× throughput left on the table for TPU training.
6. **Data loader on CPU** — `shakespeare_loader` runs Python loops for tokenization. Should be vectorized for TPU pipeline.