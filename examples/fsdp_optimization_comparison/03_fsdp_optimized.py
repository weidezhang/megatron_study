#!/usr/bin/env python3
"""
Optimized FSDP Training — All the tricks from the 2025–2026 playbook
=====================================================================
Builds on basic FSDP and adds every major optimization:

1. Activation Checkpointing (selective, per-block)
   - Recomputes activations during backward instead of storing them
   - Trades ~20-30% extra FLOPs for massive memory savings
   - Without it: activations for 24 layers × seq_len × batch × hidden dominate memory
   - With it: only need to store boundary activations between checkpointed blocks

2. BF16 Mixed Precision (same as basic)
   - Compute in BF16, reduce in BF16, master weights in FP32

3. Forward Prefetch
   - While computing layer N's forward, prefetch (AllGather) layer N+1's params
   - Overlaps communication with computation → hides AllGather latency

4. Backward Prefetch (BACKWARD_PRE)
   - During backward of layer N, prefetch layer N-1's params
   - Overlaps ReduceScatter/AllGather with backward computation

5. Limit AllGathers
   - Controls how many outstanding AllGather operations can be in flight
   - Prevents memory spikes from too many simultaneously materialized layers

6. Gradient Accumulation
   - Demonstrates how to do gradient accumulation with FSDP
   - Uses no_sync() context to skip ReduceScatter on intermediate micro-steps

Compared to basic FSDP:
  - Much lower peak memory (activation checkpointing)
  - Better compute/comm overlap (prefetching)
  - Can run larger batch sizes or longer sequences in the same memory

Usage:
    torchrun --nproc_per_node=4 03_fsdp_optimized.py [--seq_len 1024] [--batch_size 8] [--steps 20]
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
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from model import GPTConfig, GPTModel, GPTBlock, get_synthetic_batch, print_memory_summary, ThroughputTimer


def parse_args():
    parser = argparse.ArgumentParser(description="Optimized FSDP Training")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length (can go higher!)")
    parser.add_argument("--batch_size", type=int, default=8, help="Micro-batch size per GPU (can go higher!)")
    parser.add_argument("--steps", type=int, default=20, help="Number of training steps")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--hidden_size", type=int, default=2048, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=24, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--no_activation_ckpt", action="store_true", help="Disable activation checkpointing")
    parser.add_argument("--no_prefetch", action="store_true", help="Disable forward/backward prefetch")
    parser.add_argument("--profile", action="store_true", help="Enable torch.profiler")
    parser.add_argument("--profile_dir", type=str, default="./profiles/fsdp_optimized", help="Profile output dir")
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

    use_act_ckpt = not args.no_activation_ckpt
    use_prefetch = not args.no_prefetch

    if rank == 0:
        print("=" * 70)
        print("OPTIMIZED FSDP: FULL_SHARD + BF16 + Act.Ckpt + Prefetch")
        print("=" * 70)
        print(f"  World size: {world_size} GPUs")
        print(f"  Sequence length: {args.seq_len}")
        print(f"  Micro-batch per GPU: {args.batch_size}")
        print(f"  Gradient accumulation: {args.grad_accum_steps}")
        print(f"  Effective batch (tokens): {args.batch_size * args.seq_len * world_size * args.grad_accum_steps:,}")
        print(f"  Activation checkpointing: {'ON' if use_act_ckpt else 'OFF'}")
        print(f"  Forward/Backward prefetch: {'ON' if use_prefetch else 'OFF'}")

    # ---- Model (build on CPU) ----
    config = GPTConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_len,
    )
    model = GPTModel(config)

    if rank == 0:
        print(f"  Model params: {model.param_count_str()} ({model.param_count():,})")

    # ---- FSDP wrapping with all optimizations ----
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={GPTBlock},
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=bf16_policy,
        auto_wrap_policy=auto_wrap_policy,
        device_id=local_rank,
        use_orig_params=True,
        # ---- KEY OPTIMIZATIONS ----
        # Forward prefetch: start AllGather for next layer during current layer's forward
        forward_prefetch=use_prefetch,
        # Backward prefetch: start AllGather for previous layer during current layer's backward
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE if use_prefetch else None,
        # Limit concurrent AllGathers to control memory spikes
        limit_all_gathers=True,
    )

    # ---- Activation Checkpointing ----
    # This is the single biggest memory optimization for FSDP.
    # It wraps each GPTBlock so that activations are recomputed during backward
    # instead of being stored in memory during forward.
    if use_act_ckpt:
        # Apply activation checkpointing to every GPTBlock
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: isinstance(submodule, GPTBlock),
        )

    if rank == 0:
        ckpt_status = "with activation checkpointing" if use_act_ckpt else "without activation checkpointing"
        print(f"  FSDP wrapped {ckpt_status}")
        print_memory_summary("After FSDP wrap + act.ckpt", local_rank)

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    if rank == 0:
        print_memory_summary("After optimizer init", local_rank)
        print("-" * 70)

    # ---- Training loop with gradient accumulation ----
    timer = ThroughputTimer(
        model_params=config.vocab_size * config.hidden_size +
                     config.num_layers * (
                         4 * config.hidden_size * config.hidden_size +
                         3 * config.hidden_size * config.intermediate_size
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
        tokens_this_step = 0
        timer.start()

        for micro_step in range(args.grad_accum_steps):
            input_ids, labels = get_synthetic_batch(
                args.batch_size, args.seq_len, config.vocab_size, device
            )
            tokens_this_step += args.batch_size * args.seq_len * world_size

            # Use no_sync() for all but the last micro-step to skip ReduceScatter
            # This is critical for grad accumulation with FSDP — without it,
            # communication happens on every micro-step, wasting bandwidth.
            is_last_micro = (micro_step == args.grad_accum_steps - 1)

            if not is_last_micro:
                with model.no_sync():
                    _, loss = model(input_ids, labels=labels)
                    loss = loss / args.grad_accum_steps
                    loss.backward()
            else:
                _, loss = model(input_ids, labels=labels)
                loss = loss / args.grad_accum_steps
                loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        timer.stop(tokens_this_step)

        if profiler:
            profiler.step()

        if rank == 0 and (step % 5 == 0 or step == args.steps - 1):
            print(f"  Step {step:3d} | Loss: {loss.item() * args.grad_accum_steps:.4f} | "
                  f"Step time: {timer.step_times[-1]*1000:.1f} ms")

    if profiler:
        profiler.stop()

    # ---- Summary ----
    if rank == 0:
        print("-" * 70)
        optim_label = []
        if use_act_ckpt:
            optim_label.append("Act.Ckpt")
        if use_prefetch:
            optim_label.append("Prefetch")
        label = " + ".join(optim_label) if optim_label else "No optimizations"
        print(f"RESULTS (Optimized FSDP — FULL_SHARD + BF16 + {label}):")
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
