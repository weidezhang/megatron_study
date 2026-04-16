#!/usr/bin/env python3
"""
Baseline: Standard DDP Training (No FSDP)
==========================================
Each GPU holds a FULL copy of the model + optimizer states + gradients.
Only gradients are synchronized (AllReduce) across GPUs.

Memory per GPU ≈ Model (FP32) + Gradients (FP32) + Optimizer (2x FP32 for Adam)
For 1.3B params: ~5.2 GB model + 5.2 GB grads + 10.4 GB optim = ~20.8 GB just for states
Plus activations → easily 25-30 GB per GPU.

Usage:
    torchrun --nproc_per_node=4 01_baseline_ddp.py [--seq_len 512] [--batch_size 4] [--steps 20]
"""

import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPTConfig, GPTModel, get_synthetic_batch, print_memory_summary, ThroughputTimer


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline DDP Training")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro-batch size per GPU")
    parser.add_argument("--steps", type=int, default=20, help="Number of training steps")
    parser.add_argument("--hidden_size", type=int, default=2048, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=24, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--profile", action="store_true", help="Enable torch.profiler")
    parser.add_argument("--profile_dir", type=str, default="./profiles/ddp", help="Profile output dir")
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
        print("BASELINE: DDP (DistributedDataParallel) — FP32")
        print("=" * 70)
        print(f"  World size: {world_size} GPUs")
        print(f"  Sequence length: {args.seq_len}")
        print(f"  Micro-batch per GPU: {args.batch_size}")
        print(f"  Global batch (tokens): {args.batch_size * args.seq_len * world_size:,}")

    # ---- Model ----
    config = GPTConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_len,
    )
    model = GPTModel(config).to(device)

    if rank == 0:
        print(f"  Model params: {model.param_count_str()} ({model.param_count():,})")
        print_memory_summary("After model init", local_rank)

    # ---- Wrap with DDP ----
    # DDP replicates the model on each GPU. Only gradients are AllReduced.
    model = DDP(model, device_ids=[local_rank])

    if rank == 0:
        print_memory_summary("After DDP wrap", local_rank)

    # ---- Optimizer (FP32 Adam — 2 extra states per param) ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    if rank == 0:
        print_memory_summary("After optimizer init", local_rank)
        print("-" * 70)

    # ---- Training loop ----
    timer = ThroughputTimer(
        model_params=model.module.param_count(),
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

        # Forward
        _, loss = model(input_ids, labels=labels)

        # Backward (DDP handles gradient AllReduce automatically)
        loss.backward()

        # Optimizer step
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
        print("RESULTS (Baseline DDP — FP32):")
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
