#!/usr/bin/env python3
"""
Profile & Compare: DDP vs FSDP (basic) vs FSDP (optimized)
============================================================
Runs all three configurations back-to-back in a single process group,
collects memory and throughput metrics, and prints a side-by-side comparison.
Also generates torch.profiler Chrome traces for each configuration.

This script is designed for 4x RTX 5090 (or any 4-GPU single-node setup).

What you'll see in the comparison:
  - DDP:            High memory (full model on each GPU), fast comm (one AllReduce)
  - FSDP basic:     Low memory for states, but activations still large, extra comm per layer
  - FSDP optimized: Lowest memory (act.ckpt), best overlap (prefetch), can run bigger batches

Usage:
    torchrun --nproc_per_node=4 04_profile_comparison.py [--seq_len 512] [--steps 15]

After running, view traces with:
    tensorboard --logdir=./profiles/
"""

import os
import gc
import argparse
import functools
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
from typing import Dict, List

from model import (
    GPTConfig, GPTModel, GPTBlock,
    get_synthetic_batch, get_gpu_memory_gb, ThroughputTimer,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Profile Comparison: DDP vs FSDP vs FSDP+Optimized")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro-batch per GPU")
    parser.add_argument("--steps", type=int, default=15, help="Training steps per config")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup steps (excluded from metrics)")
    parser.add_argument("--hidden_size", type=int, default=2048, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=24, help="Transformer layers")
    parser.add_argument("--num_heads", type=int, default=16, help="Attention heads")
    parser.add_argument("--profile_dir", type=str, default="./profiles", help="Base profile dir")
    parser.add_argument("--skip_ddp", action="store_true", help="Skip DDP baseline (saves time if OOM)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Configuration builders
# ---------------------------------------------------------------------------

def build_ddp_model(config: GPTConfig, local_rank: int) -> torch.nn.Module:
    """Build model wrapped with standard DDP (FP32)."""
    device = torch.device(f"cuda:{local_rank}")
    model = GPTModel(config).to(device)
    return DDP(model, device_ids=[local_rank])


def build_fsdp_basic(config: GPTConfig, local_rank: int) -> torch.nn.Module:
    """Build model wrapped with basic FSDP (FULL_SHARD + BF16)."""
    model = GPTModel(config)

    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={GPTBlock},
    )

    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=bf16_policy,
        auto_wrap_policy=auto_wrap_policy,
        device_id=local_rank,
        use_orig_params=True,
    )


def build_fsdp_optimized(config: GPTConfig, local_rank: int) -> torch.nn.Module:
    """Build model with FSDP + activation checkpointing + prefetching."""
    model = GPTModel(config)

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
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
    )

    # Activation checkpointing on every GPTBlock
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=lambda submodule: isinstance(submodule, GPTBlock),
    )

    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(
    model: torch.nn.Module,
    config: GPTConfig,
    args,
    local_rank: int,
    world_size: int,
    label: str,
    profile_subdir: str,
) -> Dict:
    """Run training loop and collect metrics."""
    device = torch.device(f"cuda:{local_rank}")
    rank = dist.get_rank()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Approximate param count for TFLOPS estimation
    approx_params = (
        config.vocab_size * config.hidden_size +
        config.num_layers * (
            4 * config.hidden_size * config.hidden_size +
            3 * config.hidden_size * config.intermediate_size
        )
    )

    timer = ThroughputTimer(
        model_params=approx_params,
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
    )

    torch.cuda.reset_peak_memory_stats(device)

    # Profiler
    profile_path = os.path.join(args.profile_dir, profile_subdir)
    os.makedirs(profile_path, exist_ok=True)
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=min(args.warmup, 2),
            active=min(args.steps - args.warmup, 8),
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_path),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,  # Disable stacks for smaller trace files
    )
    profiler.start()

    model.train()
    step_times = []

    for step in range(args.steps):
        input_ids, labels = get_synthetic_batch(
            args.batch_size, args.seq_len, config.vocab_size, device
        )

        timer.start()

        _, loss = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        tokens_this_step = args.batch_size * args.seq_len * world_size
        timer.stop(tokens_this_step)

        profiler.step()

        if step >= args.warmup:
            step_times.append(timer.step_times[-1])

        if rank == 0 and (step % 5 == 0 or step == args.steps - 1):
            mem = get_gpu_memory_gb(local_rank)
            print(f"    [{label}] Step {step:3d} | Loss: {loss.item():.4f} | "
                  f"Time: {timer.step_times[-1]*1000:.1f}ms | "
                  f"Mem: {mem['allocated']:.2f}GB")

    profiler.stop()

    # Collect results
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
    final_mem = get_gpu_memory_gb(local_rank)

    # Compute steady-state metrics (excluding warmup)
    if step_times:
        avg_step_ms = (sum(step_times) / len(step_times)) * 1000
        tokens_per_sec = (args.batch_size * args.seq_len * world_size) / (sum(step_times) / len(step_times))
        tflops_per_gpu = (tokens_per_sec * 6 * approx_params) / (1e12 * world_size)
    else:
        avg_step_ms = 0
        tokens_per_sec = 0
        tflops_per_gpu = 0

    return {
        "label": label,
        "peak_memory_gb": peak_mem,
        "allocated_memory_gb": final_mem["allocated"],
        "avg_step_ms": avg_step_ms,
        "tokens_per_sec": tokens_per_sec,
        "tflops_per_gpu": tflops_per_gpu,
        "profile_dir": profile_path,
    }


# ---------------------------------------------------------------------------
# Cleanup between runs
# ---------------------------------------------------------------------------

def cleanup_between_runs(local_rank: int):
    """Aggressively free GPU memory between configurations."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(local_rank)
    # Brief sync to ensure all ranks are ready
    dist.barrier()


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison(results: List[Dict], rank: int, args):
    """Print a formatted comparison table."""
    if rank != 0:
        return

    print("\n")
    print("=" * 90)
    print("  COMPARISON: DDP vs FSDP (basic) vs FSDP (optimized)")
    print(f"  Model: ~1.3B params | Seq len: {args.seq_len} | Batch/GPU: {args.batch_size} | GPUs: {dist.get_world_size()}")
    print("=" * 90)

    # Header
    header = f"{'Config':<30} {'Peak Mem (GB)':>14} {'Step Time (ms)':>15} {'Tokens/sec':>12} {'TFLOPS/GPU':>12}"
    print(header)
    print("-" * 90)

    for r in results:
        if r.get("oom"):
            row = f"{r['label']:<30} {'OOM':>14} {'OOM':>15} {'OOM':>12} {'OOM':>12}"
        else:
            row = (
                f"{r['label']:<30} "
                f"{r['peak_memory_gb']:>14.2f} "
                f"{r['avg_step_ms']:>15.1f} "
                f"{r['tokens_per_sec']:>12,.0f} "
                f"{r['tflops_per_gpu']:>12.1f}"
            )
        print(row)

    print("-" * 90)

    # Compute improvements vs baseline (skip if baseline OOMed)
    valid_results = [r for r in results if not r.get("oom")]
    if len(valid_results) >= 2:
        baseline = valid_results[0]
        print("\nImprovements vs first successful config:")
        for r in valid_results[1:]:
            mem_savings = (1 - r["peak_memory_gb"] / baseline["peak_memory_gb"]) * 100 if baseline["peak_memory_gb"] > 0 else 0
            speed_change = (r["tokens_per_sec"] / baseline["tokens_per_sec"] - 1) * 100 if baseline["tokens_per_sec"] > 0 else 0
            print(f"  {r['label']:<28} Memory: {mem_savings:+.1f}% | Throughput: {speed_change:+.1f}%")

    # Key observations
    print("\nKey observations:")
    print(f"  - DDP stores full model on each GPU → highest memory")
    any_oom = any(r.get("oom") for r in results)
    if any_oom:
        print(f"  - DDP OOMed! This demonstrates why FSDP sharding is essential at scale")
    if len(valid_results) >= 2:
        ratio = valid_results[0]['peak_memory_gb'] / max(valid_results[1]['peak_memory_gb'], 0.01)
        if ratio > 1.0:
            print(f"  - FSDP FULL_SHARD shards params/grads/optim → ~{ratio:.1f}x memory reduction for states")
        print(f"  - But FSDP adds AllGather+ReduceScatter per layer → may affect step time")
    if len(valid_results) >= 3:
        print(f"  - Activation checkpointing further reduces peak memory at cost of recompute FLOPs")
        print(f"  - Forward/backward prefetch overlaps comm with compute → better utilization")

    print(f"\nProfile traces for TensorBoard:")
    for r in results:
        print(f"  {r['label']:<28} → {r['profile_dir']}/")
    print(f"\nVisualize with: tensorboard --logdir={results[0].get('profile_dir', './profiles').rsplit('/', 1)[0]}/")
    print("=" * 90)

    # Save results as JSON
    results_path = os.path.join(os.path.dirname(results[0]["profile_dir"]), "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Distributed setup ----
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    config = GPTConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_len,
    )

    if rank == 0:
        print("=" * 90)
        print("  FSDP Optimization Comparison — Profiling All Configurations")
        print("=" * 90)
        total_mem = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
        print(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"  GPU memory: {total_mem:.1f} GB × {world_size} GPUs")
        print(f"  Model: ~1.3B params ({config.num_layers} layers, hidden={config.hidden_size})")
        print(f"  Seq len: {args.seq_len} | Batch/GPU: {args.batch_size}")
        print(f"  Steps: {args.steps} (warmup: {args.warmup})")
        print()

    results = []

    # ---- Config 1: DDP Baseline ----
    if not args.skip_ddp:
        if rank == 0:
            print(">>> Running Config 1: DDP (FP32 baseline)...")
        try:
            model = build_ddp_model(config, local_rank)
            r = run_training(model, config, args, local_rank, world_size, "DDP (FP32)", "ddp")
            results.append(r)
            del model
            cleanup_between_runs(local_rank)
        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print("    [DDP] OOM! Skipping. This demonstrates why FSDP is needed.")
            results.append({
                "label": "DDP (FP32) — OOM",
                "peak_memory_gb": -1,
                "allocated_memory_gb": 0,
                "avg_step_ms": -1,
                "tokens_per_sec": 0,
                "tflops_per_gpu": 0,
                "profile_dir": os.path.join(args.profile_dir, "ddp"),
                "oom": True,
            })
            cleanup_between_runs(local_rank)
    else:
        if rank == 0:
            print(">>> Skipping DDP baseline (--skip_ddp)")

    # ---- Config 2: FSDP Basic ----
    if rank == 0:
        print(">>> Running Config 2: FSDP Basic (FULL_SHARD + BF16)...")
    model = build_fsdp_basic(config, local_rank)
    r = run_training(model, config, args, local_rank, world_size, "FSDP Basic (BF16)", "fsdp_basic")
    results.append(r)
    del model
    cleanup_between_runs(local_rank)

    # ---- Config 3: FSDP Optimized ----
    if rank == 0:
        print(">>> Running Config 3: FSDP Optimized (Act.Ckpt + Prefetch)...")
    model = build_fsdp_optimized(config, local_rank)
    r = run_training(model, config, args, local_rank, world_size, "FSDP Opt (Ckpt+Prefetch)", "fsdp_optimized")
    results.append(r)
    del model
    cleanup_between_runs(local_rank)

    # ---- Config 4: FSDP Optimized with LARGER batch (demonstrates memory headroom) ----
    # Because activation checkpointing saves so much memory, we can now increase batch size
    big_batch = args.batch_size * 2
    if rank == 0:
        print(f">>> Running Config 4: FSDP Optimized + LARGER batch ({big_batch}/GPU)...")
        print(f"    (This would OOM without activation checkpointing!)")

    try:
        model = build_fsdp_optimized(config, local_rank)
        # Temporarily override batch size
        orig_batch = args.batch_size
        args.batch_size = big_batch
        r = run_training(
            model, config, args, local_rank, world_size,
            f"FSDP Opt (batch={big_batch})", "fsdp_opt_big_batch"
        )
        results.append(r)
        args.batch_size = orig_batch
        del model
        cleanup_between_runs(local_rank)
    except torch.cuda.OutOfMemoryError:
        if rank == 0:
            print(f"    [FSDP Opt batch={big_batch}] OOM — try smaller increase or longer seq_len trade-off")
        cleanup_between_runs(local_rank)

    # ---- Print comparison ----
    print_comparison(results, rank, args)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
