#!/bin/bash
# TPU v5e-8 verification script for 7B model
# Memory budget per core (16 GB HBM):
#   Model params:   ~5.95 GB (pool + W_K sharded, W_Q + PartA/B replicated)
#   Optimizer (bf16 mu): ~8.93 GB
#   Activations:    ~1 GB (batch=4, seq=2048)
#   Total:          ~15.9 GB (0.1 GB safety buffer)

set -e

export PYTHONPATH=$(pwd)
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true"
export JAX_ENABLE_X64=False
export XLA_PYTHON_CLIENT_PREALLOCATE=true
# Enable NaN detection during verification
export JAX_DEBUG_NANS=True
# Log recompilations (should see exactly 1 per jit function)
export JAX_LOG_COMPILES=True

echo "=== TPU v5e-8: 7B model verification ==="
echo "Testing: W_K D-sharding + fused streaming top-k + bf16 optimizer"
echo ""

# Step 1: Quick 10-step smoke test (catches OOM, shape errors)
echo "── Phase 1: 10-step smoke test ──"
python3 scripts/train_tpu.py \
    --config 7b \
    --steps 10 \
    --batch 4 \
    --log 1 \
    --gen 0 \
    --data_axis 2 \
    --pool_axis 4

echo ""
echo "── Phase 2: 100-step convergence check ──"
# Watch for: loss decreasing, no NaN, stable ms/step after compilation
python3 scripts/train_tpu.py \
    --config 7b \
    --steps 100 \
    --batch 4 \
    --log 10 \
    --gen 0 \
    --data_axis 2 \
    --pool_axis 4

echo ""
echo "=== Verification complete ==="
