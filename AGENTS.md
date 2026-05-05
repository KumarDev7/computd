# AGENTS.md: JAX/NNX/Pallas Engineering Principles

This document defines the constraints and mental models for AI Agents. We prioritize NNX's reference semantics, Pallas hardware-alignment, and cutting-edge model quality over code aesthetics.

## 1. The NNX Philosophy (Mandatory)

* **Reference Semantics & State Tracking:** NNX Modules are Python objects holding their own state. Use `nnx.Param` for trainable weights, `nnx.BatchStat` for non-trainable buffers like running means, and `nnx.Variable` for generic state. This eliminates the need for manual parameter dictionary management and ensures that transformations like `nnx.split` and `nnx.merge` handle the object-functional bridge seamlessly. By maintaining state within the object, NNX allows for PyTorch-like ergonomics while preserving the ability to lower into pure JAX functional representations for compilation.

* **Eager Initialization:** Provide `input_shape` or concrete dimensions in `__init__`. Unlike Linen, NNX does not support lazy shape inference by default; initializing weights immediately allows for instant inspection of parameter counts and memory footprints. This explicit approach prevents "shape-hiding" bugs where mismatches only appear during the first forward pass, enabling faster iteration during the architecture design phase.

* **PRNG Management:** Utilize the `nnx.Rngs` dispenser pattern. Treat the `Rngs` object as a stateful stream—call `rngs.params()` for initialization or `rngs.dropout()` for stochastic ops. Never manually split or manage `jax.random.PRNGKey` logic inside a module method. The `nnx.Rngs` object automatically tracks state updates, ensuring that every call produces a deterministic yet unique key without the boilerplate of manual key threading.

## 2. SOTA Performance & Research-Driven Quality

Model quality and hardware utilization are the ultimate metrics. Prioritize predictive power and training stability using modern architectural research:

* **Dynamic Compute & Token Routing:** Implement techniques like **Mixture-of-Depths (MoD)** or **Early Exit** strategies. By skipping computation for tokens that do not require deep processing (e.g., punctuation or common stop words), you can redirect TFLOPs to harder tokens, effectively decoupling the parameter count from the compute cost per step. This results in models that are significantly faster at inference while maintaining the performance of much larger, static architectures.

* **Advanced Positional Encodings:** Default to **Rotary Positional Embeddings (RoPE)**. When dealing with long-context windows, use **NTK-aware scaling**, **YaRN**, or **RoPE-α** scaling to ensure that the model maintains positional resolution at distances beyond the initial training horizon. These methods prevent the "attention collapse" often seen when models encounter context lengths they weren't explicitly trained on, allowing for extreme context extension with minimal fine-tuning.


* **Numerical Stability & Normalization:** Favor **RMSNorm** for its computational efficiency or **QK-Norm** (Query-Key Normalization) within Attention layers. These techniques prevent "logit drift" and gradient explosions in high-parameter models. QK-Norm, in particular, helps in stabilizing the attention scores, enabling the use of higher learning rates and more aggressive optimizer settings without risking sudden training divergence.

* **Conditional Sparsity:** Implement **Mixture-of-Experts (MoE)** with robust load-balancing strategies such as **Sinkhorn routing** or **Expert-Choice routing**. The goal is to maximize the "parameter-to-compute" ratio, allowing models to scale in knowledge capacity (trillions of parameters) without a linear increase in latency, as only a fraction of experts are activated per token.

## 3. Standardized Folder & Project Structure

To maintain project scalability, the Agent must adhere to this hierarchy:

* **`src/model/`**: Pure NNX module definitions (Architectures).
* **`src/kernels/`**: Low-level Pallas kernels (Custom ops).
* **`src/data/`**: Concurrent data loaders, tokenizers, and sharding utilities.
* **`src/training/`**: On-Device loop logic (`lax.scan`), optimizer setups, and trainers.
* **`scripts/`**: Entry points for `train.py` (TPU Pod) and `bench.py` (Benchmarking).
* **`configs/`**: Static configuration files for hyperparameters and sharding meshes.


## 4. Low-Level Control & Pallas Dominance

* **The 1% Performance Rule:** If a custom Pallas kernel can improve performance by even **1%**, or if it significantly reduces memory pressure by fusing operations (like FlashAttention variants or fused FFNs), implement it immediately. Generic JAX ops often introduce implicit padding or unnecessary materialization of intermediate tensors in HBM. A 1% gain across a 30-day training run on a TPU pod translates to massive savings in time and compute cost.

* **Hardware-Specific Minimalism:** Bypass high-level JAX/Flax abstractions if they introduce overhead or unwanted logging. Use Pallas to gain direct access to the TPU's **VMEM** (Vector Memory) and **SMEM** (Scalar Memory). This direct control ensures that every cycle is spent on computation rather than memory orchestration, allowing you to bypass the overhead of standard XLA buffer management when a custom tiling strategy is more efficient.

* **Manual Memory Pipelining:** Use Pallas to write kernels that overlap memory loads with compute. For example, use the `pallas_call` interface to manually manage double-buffering at the kernel level. By pre-fetching the next tile into VMEM while the current tile is being processed by the Matrix Multiply Unit (MXU), you ensure that the hardware never stalls, reaching near-peak theoretical utilization.

## 5. Parallel Host-to-Device Pipelining & Concurrency

Always design for multi-core TPU pods (e.g., 8+ cores) where the Host (CPU) is the most common bottleneck in the training loop.


* **Concurrent Data Dispatch:** Do not send data to TPU cores sequentially. Use `concurrent.futures.ThreadPoolExecutor` to trigger `jax.device_put` for all 8+ cores in parallel. This parallelization ensures that the time taken to shard and transfer a batch is limited by the single-core transfer speed rather than being a cumulative serial cost that scales linearly with the number of devices.

* **Non-Blocking Host Execution:** Structure the training loop so the CPU is never idle. While the TPU is executing the current training step, the Host should be pre-fetching, tokenizing, augmenting, and sharding the *next* several batches in a background pipeline. This "look-ahead" strategy masks the latency of data preparation, ensuring that the TPU is immediately fed the next batch upon completion of a step.

* **Zero-Copy Sharding:** Leverage `jax.sharding` and `jax.make_array_from_single_device_arrays` to assemble global batches without triggering unnecessary host-side data copies. Avoid CPU-bound reshapes or redundant array cast operations; keep the data in its most compact form (e.g., `uint16` or `int8`) until it reaches the device memory.

## 6. TPU Memory & On-Device Loops

* **Fused On-Device Training (`lax.scan`):** Avoid Python-side training loops for inner iterations. Use `jax.lax.scan` to compile the entire multi-step loop into a single XLA program. This eliminates the "micro-syncs" and Python overhead between Host and Device that usually kill performance in high-speed training scenarios, effectively allowing the TPU to run autonomously for hundreds of steps.


* **Surgical HBM Management:**

  * Call `array.delete()` on data chunks as soon as the `lax.scan` loop completes to signal the XLA allocator for immediate reuse.
  * Follow with `del` and `gc.collect()` to force the JAX runtime to release the memory address space back to the allocator, preventing fragmentation.
  * Always maintain a safety buffer of at least 20% HBM to prevent Out-Of-Memory (OOM) errors during the "activation spike" of the backward pass, where temporary gradients can significantly inflate memory usage.

## 7. Multi-Chip Communication & Interconnect (ICI)

* **Interconnect Bandwidth Maximization:** Modern TPUs have extremely fast chip-to-chip links (ICI). Prioritize collective operations like `jax.lax.psum` (All-Reduce) or `jax.lax.all_gather`. It is often faster to re-distribute data across the entire pod at high bandwidth than to re-calculate it on a single chip or wait for a host-roundtrip.

* **In-JIT Debugging & Telemetry:** During development, use `jax.debug.print` or `jax.debug.callback` inside `@nnx.jit` blocks. This allows you to monitor internal tensor statistics (means, variances, or NaNs) and communication latencies without breaking the XLA execution graph. This visibility is critical for identifying "silent" performance degradations or synchronization bottlenecks that don't trigger errors but slow down training.

## 8. Environment-Aware Development (GPU/TPU)


* **Platform Probing:** Always check `jax.devices()[0].platform` at the entry point of your script. This allows the code to dynamically adjust its behavior based on whether it's running on a development GPU or a production TPU pod.

* **Dynamic Kernel Selection:** Adjust kernel strategies based on the backend. For instance, lower Pallas kernels to **Mosaic** for TPUs and **Triton** for GPUs. Ensure that sharding meshes are dynamically constructed based on the physical device count available in the environment, ensuring the code is "plug-and-play" regardless of the underlying accelerator.

## 9. Dependency and Environment Versioning

Version drift between development and production environments can lead to silent numerical errors or compilation failures.

* **Strict Version Locking:** Use a lockfile (e.g., `requirements.txt` with hashes or `uv.lock`) to ensure that `jax`, `jaxlib`, `flax`, and `optax` versions are identical across the development GPU machine and the training TPU pod.
* **Libtpu Consistency:** Pay special attention to the `libtpu` version on the TPU pod. It must be compatible with the `jaxlib` version used during development to avoid "HLO lowering" errors that only appear at runtime.
* **Environment Validation:** Implement a startup check that logs the versions of all critical libraries. If a version mismatch is detected, the agent should alert the user or fail-fast rather than proceeding with potentially unstable training.

## 10. Critical JAX Environment Flags

The Agent must configure these shell environment variables before initialization to optimize performance and debug precision issues:

* **Maximum Performance:**
  * `XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true"`: Optimizes attention fusions on GPU backends.
  * `JAX_ENABLE_X64=False`: Uses 32-bit/16-bit precision to double throughput on hardware units optimized for those formats (MXU).
  * `XLA_PYTHON_CLIENT_PREALLOCATE=true`: Reserves HBM upfront to avoid allocation fragmentation and potential "hidden" H2D transfers during training.

* **Stability & Inspection:**
  * `JAX_DEBUG_NANS=True`: Essential for catching numerical divergence in complex research architectures (like MoE routing) as soon as they occur.
  * `JAX_LOG_COMPILES=True`: Use this to verify that your JIT functions are "stable." If you see repeated compilation logs, it indicates a "tracer leak" where Python state is accidentally triggering re-compilations.

## 11. The "Karpathy" Workflow for NNX Agents

1. **Thinking Phase:** Identify which parts of the model architecture can be optimized with SOTA research (e.g., replacing standard Attention with Flash-Linear-Attention). Plan the exact memory layout for 8-core sharding and inter-chip communication.
2. **Implementation Phase:** Use `nnx.Optimizer` to wrap the model and Optax state. Group logic into modular `@nnx.jit` functions. Use thread pools to parallelize the "feeder" pipeline, ensuring the TPU never waits for data.
3. **Verification Phase:** Monitor for "Host-to-Device" (H2D) stalls and "Device-to-Host" (D2H) syncs. Prioritize the stability of the loss curve and hardware **Model Flops Utilization (MFU)** over the aesthetics of the implementation code.

## 12. JIT Optimization & Reusability

* **Modular JIT Compilation:** Large monolithic JIT functions lead to long cold-start times and high developer friction. Divide the model into smaller, reusable `@jit` sub-components (e.g., separate `compute_loss`, `apply_gradients`, and `transformer_block`). This enables "instant" incremental compilation where changes to one part of the code don't invalidate the compiled HLO of the rest of the model.

* **Avoid Python Loop Unrolling:** Never use a Python `for` loop inside a `@jit` function for anything involving a high number of iterations. This forces XLA to unroll the loop into a massive, linear sequence of operations, exploding the HLO graph and resulting in minute-long compile times. Always use `jax.lax.scan` or `jax.lax.while_loop`.

* **Compilation Stability:** Design functions to accept `static` arguments (like layer counts or block sizes) carefully. Use `static_argnums` only for values that *must* be known at compile time to define shapes. Ensure that minor code edits outside the core math kernels do not trigger a full model re-compilation, maintaining a fast development loop.
