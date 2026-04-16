#!/usr/bin/env python3
"""
Basic FSDP Training (ZeRO-3 style sharding)
=============================================
FSDP shards model parameters, gradients, and optimizer states across GPUs.
Each GPU only holds 1/N of the model states (where N = world_size).

Before each layer's forward pass: AllGather to reconstruct full params.
After each layer's backward pass: ReduceScatter to shard gradients.

Memory per GPU ≈ Model/N + Gradients/N + Optimizer/N + FULL activations
For 1.3B on 4 GPUs: ~1.3 GB model + 1.3 GB grads + 2.6 GB optim = ~5.2 GB states
But activations are NOT sharded → still significant.

Compared to DDP baseline:
  - ~4x less memory for model states (sharded across 4 GPUs)
  - Same activation memory
  - Extra communication (AllGather + ReduceScatter per layer vs AllReduce once)

Usage:
    torchrun --nproc_per_node=4 02_fsdp_basic.py [--seq_len 512] [--batch_size 4] [--steps 20]
"""

import os
import argparse
import functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from model import GPTConfig, GPTModel, GPTBlock, get_synthetic_batch, print_memory_summary, ThroughputTimer


def parse_args():
    parser = argparse.ArgumentParser(description="Basic FSDP Training")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro-batch size per GPU")
    parser.add_argument("--steps", type=int, default=20, help="Number of training steps")
    parser.add_argument("--hidden_size", type=int, default=2048, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=24, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--profile", action="store_true", help="Enable torch.profiler")
    parser.add_argument("--profile_dir", type=str, default="./profiles/fsdp_basic", help="Profile output dir")
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Distributed setup ----
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print("=" * 70)
        print("BASIC FSDP: FULL_SHARD (ZeRO-3) + BF16 Mixed Precision")
        print("=" * 70)
        print(f"  World size: {world_size} GPUs")
        print(f"  Sequence length: {args.seq_len}")
        print(f"  Micro-batch per GPU: {args.batch_size}")
        print(f"  Global batch (tokens): {args.batch_size * args.seq_len * world_size:,}")

    # ---- Model (build on CPU first, FSDP will shard and move to GPU) ----
    config = GPTConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_len,
    )
    model = GPTModel(config)

    if rank == 0:
        print(f"  Model params: {model.param_count_str()} ({model.param_count():,})")

    # ---- FSDP wrapping ----
    # Mixed precision: compute in BF16, communicate in BF16, keep master weights in FP32
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Auto-wrap policy: wrap each GPTBlock as an individual FSDP unit
    # This means AllGather/ReduceScatter happens per-block, enabling overlap
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={GPTBlock},
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3: shard params + grads + optim
        mixed_precision=bf16_policy,
        auto_wrap_policy=auto_wrap_policy,
        device_id=local_rank,
        use_orig_params=True,  # Better compatibility with torch.compile
    )

    if rank == 0:
        print_memory_summary("After FSDP wrap", local_rank)

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    if rank == 0:
        print_memory_summary("After optimizer init", local_rank)
        print("-" * 70)

    # ---- Training loop ----
    timer = ThroughputTimer(
        model_params=config.vocab_size * config.hidden_size +  # embedding (approx)
                     config.num_layers * (
                         4 * config.hidden_size * config.hidden_size +  # attn
                         3 * config.hidden_size * config.intermediate_size  # mlp
                     ),
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
    )
    torch.cuda.reset_peak_memory_stats(device)

    # Optional profiling
    profiler = None
    if args.profile:
        os.makedirs(args.profile_dir, exist_ok=True)
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=2, warmup=3, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.profile_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        profiler.start()

    model.train()
    for step in range(args.steps):
        input_ids, labels = get_synthetic_batch(
            args.batch_size, args.seq_len, config.vocab_size, device
        )

        timer.start()

        # Forward (FSDP handles AllGather of params before each layer)
        _, loss = model(input_ids, labels=labels)

        # Backward (FSDP handles ReduceScatter of grads after each layer)
        loss.backward()

        # Optimizer step (operates on sharded params)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        tokens_this_step = args.batch_size * args.seq_len * world_size
        timer.stop(tokens_this_step)

        if profiler:
            profiler.step()

        if rank == 0 and (step % 5 == 0 or step == args.steps - 1):
            print(f"  Step {step:3d} | Loss: {loss.item():.4f} | "
                  f"Step time: {timer.step_times[-1]*1000:.1f} ms")

    if profiler:
        profiler.stop()

    # ---- Summary ----
    if rank == 0:
        print("-" * 70)
        print("RESULTS (Basic FSDP — FULL_SHARD + BF16):")
        print(timer.summary(world_size))
        print_memory_summary("Final memory", local_rank)
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  Peak GPU memory: {peak:.2f} GB")
        if args.profile:
            print(f"  Profile traces saved to: {args.profile_dir}/")
        print("=" * 70)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
