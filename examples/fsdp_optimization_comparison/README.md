# FSDP Optimization Comparison — 4x GPU Benchmark

**Compare DDP vs FSDP (basic) vs FSDP (optimized) on a ~1.3B GPT model with profiling.**

This example demonstrates the real-world impact of FSDP optimization techniques from the 2025–2026 production playbook, all runnable on a single node with 4 GPUs (e.g., 4x RTX 5090).

## What's Compared

| Config | Parallelism | Precision | Act. Ckpt | Prefetch | Memory Profile |
|---|---|---|---|---|---|
| `01_baseline_ddp.py` | DDP (AllReduce only) | FP32 | No | N/A | Full model on every GPU |
| `02_fsdp_basic.py` | FSDP FULL_SHARD | BF16 | No | No | Sharded states, full activations |
| `03_fsdp_optimized.py` | FSDP FULL_SHARD | BF16 | Yes | Yes | Sharded states + recomputed activations |
| `04_profile_comparison.py` | All of the above | Mixed | Mixed | Mixed | Side-by-side comparison with profiler |

## Key Optimizations Demonstrated

### 1. FSDP FULL_SHARD (ZeRO-3)
- Shards parameters, gradients, and optimizer states across all GPUs
- Each GPU holds only `1/N` of model states (N = number of GPUs)
- Trade-off: requires AllGather before each layer's forward and ReduceScatter after backward

### 2. BF16 Mixed Precision
- Compute in BF16 (2 bytes vs 4 bytes per element)
- Reduces memory by ~50% for activations and communication volume
- Master weights kept in FP32 for numerical stability

### 3. Activation Checkpointing (the biggest memory win)
- Instead of storing all layer activations during forward pass, recompute them during backward
- Reduces activation memory from O(N_layers) to O(1) at cost of ~30% extra FLOPs
- This is what allows you to run 2-4x larger batches or longer sequences

### 4. Forward/Backward Prefetching
- **Forward prefetch**: While computing layer N, start AllGather for layer N+1
- **Backward prefetch (BACKWARD_PRE)**: During backward of layer N, prefetch layer N-1
- Overlaps communication with computation → hides latency on NVLink

### 5. Gradient Accumulation with `no_sync()`
- `03_fsdp_optimized.py` shows proper grad accumulation with FSDP
- Uses `model.no_sync()` to skip ReduceScatter on intermediate micro-steps
- Only synchronizes on the final micro-step → reduces communication by `grad_accum_steps`x

## Quick Start

### Prerequisites
```bash
pip install torch>=2.2.0 tensorboard
```

### Run Individual Scripts
```bash
# Baseline DDP (FP32, no sharding)
torchrun --nproc_per_node=4 01_baseline_ddp.py --seq_len 512 --batch_size 4 --steps 20

# Basic FSDP (FULL_SHARD + BF16)
torchrun --nproc_per_node=4 02_fsdp_basic.py --seq_len 512 --batch_size 4 --steps 20

# Optimized FSDP (all tricks enabled)
torchrun --nproc_per_node=4 03_fsdp_optimized.py --seq_len 1024 --batch_size 8 --steps 20

# Optimized FSDP with gradient accumulation
torchrun --nproc_per_node=4 03_fsdp_optimized.py --seq_len 1024 --batch_size 4 --grad_accum_steps 4 --steps 20
```

### Run the Full Comparison with Profiling
```bash
# This runs all configs back-to-back and prints a comparison table
torchrun --nproc_per_node=4 04_profile_comparison.py --seq_len 512 --batch_size 4 --steps 15

# If DDP OOMs (model too large for FP32 on your GPU), skip it:
torchrun --nproc_per_node=4 04_profile_comparison.py --seq_len 512 --batch_size 4 --skip_ddp
```

### View Profiler Traces
```bash
tensorboard --logdir=./profiles/
# Then open http://localhost:6006 → PyTorch Profiler tab
```

## What to Look for in Profiler Traces

Open the traces in TensorBoard's PyTorch Profiler plugin. Key things to observe:

### DDP Trace (`profiles/ddp/`)
- **One large AllReduce** at the end of backward pass
- Forward and backward are compute-dominated
- Memory timeline shows steady high usage (full model always resident)

### FSDP Basic Trace (`profiles/fsdp_basic/`)
- **AllGather operations before each layer's forward** — params are reconstructed on-the-fly
- **ReduceScatter after each layer's backward** — gradients are sharded immediately
- Memory timeline shows lower baseline but spikes during AllGather (temporarily materializes full layer)
- Communication may not fully overlap with compute (no prefetch)

### FSDP Optimized Trace (`profiles/fsdp_optimized/`)
- **AllGather and compute overlap** — prefetching causes comm to happen during previous layer's compute
- **Activation recomputation visible in backward** — extra forward ops during backward (this is the checkpoint cost)
- Memory timeline is smoother and lower overall
- Better GPU utilization (less idle time between comm and compute)

## Expected Results (4x RTX 5090, 32GB each)

Approximate numbers (will vary by driver, CUDA version, NVLink topology):

| Config | Peak Memory/GPU | Step Time | Tokens/sec | Notes |
|---|---|---|---|---|
| DDP (FP32) | ~25-30 GB | Baseline | Baseline | May OOM with larger models/batches |
| FSDP Basic (BF16) | ~8-12 GB | ~1.0-1.2x DDP | ~0.8-1.0x DDP | Memory savings from sharding + BF16 |
| FSDP Opt (Ckpt+Prefetch) | ~5-8 GB | ~1.1-1.3x DDP | ~0.85-1.0x DDP | Lowest memory, slight recompute cost |
| FSDP Opt (2x batch) | ~10-14 GB | ~1.0-1.1x DDP | ~1.5-2.0x DDP | Memory headroom → bigger batch → more throughput |

**The key insight**: Activation checkpointing trades ~20-30% extra FLOPs for massive memory savings. This lets you run 2x larger batches, which more than compensates for the recompute cost in terms of total throughput (tokens/second).

## Scaling Beyond 4 GPUs

These same techniques scale to multi-node setups. At larger scale:
- **HYBRID_SHARD** (`ShardingStrategy.HYBRID_SHARD`) shards within a node, replicates across nodes — reduces inter-node communication
- Combine with **Tensor Parallelism** (intra-node NVLink) for even larger models
- Add **Pipeline Parallelism** for 100B+ models that don't fit even with FSDP

## File Structure

```
fsdp_optimization_comparison/
├── model.py                  # Shared GPT model + utilities
├── 01_baseline_ddp.py        # DDP baseline (FP32)
├── 02_fsdp_basic.py          # FSDP FULL_SHARD + BF16
├── 03_fsdp_optimized.py      # FSDP + Act.Ckpt + Prefetch + Grad.Accum
├── 04_profile_comparison.py  # Run all configs + comparison table + profiling
├── requirements.txt          # Dependencies
└── README.md                 # This file
```
