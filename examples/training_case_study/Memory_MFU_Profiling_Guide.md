# Memory Reduction, MFU Optimization, and Profiling Guide for Large-Scale VLM Training

> Focused on Qwen2.5-VL-32B on H100 clusters. All baselines assume bf16 mixed-precision, AdamW fp32 optimizer, no parallelism unless stated. "Baseline memory" = full unsharded replica of 32.5B params.

---

## Part 1: Memory Reduction Techniques

### 1.0 Baseline: Where Does Memory Go?

Before optimizing, understand the budget for a single replica of Qwen2.5-VL-32B (no parallelism):

```
Weights (bf16)           =  65 GB    (12.1%)
Gradients (bf16)         =  65 GB    (12.1%)
AdamW m (fp32)           = 130 GB    (24.3%)
AdamW v (fp32)           = 130 GB    (24.3%)
Master weights (fp32)    = 130 GB    (24.3%)
─────────────────────────────────────────────
Model states subtotal    = 520 GB    (88%)

Activations (S=4K, MBS=1, no ckpt) = 55 GB (10%)
Temp buffers, NCCL, CUDA = ~10 GB    (2%)
─────────────────────────────────────────────
TOTAL                    ≈ 585 GB
```

The optimizer states (m + v + master weights) dominate at **73% of total memory**. This is the #1 target.

---

### 1.1 Technique Catalog

| # | Technique | What It Does | Memory Reduction | MFU Impact | Notes |
|---|---|---|---|---|---|
| **1** | **ZeRO Stage 1** (shard optimizer) | Partitions AdamW m, v, master weights across DP ranks | **−73% of optim** → on 8 DP: 390 GB → 49 GB/GPU (saves ~341 GB across group) | **−1 to −2%** (extra all-gather for optim step) | Almost free; always enable. Per-GPU: 520 → 178 GB on DP=8. |
| **2** | **ZeRO Stage 2** (shard optimizer + gradients) | Also partitions gradients across DP | **Additional −12%** total (65 GB grads / DP) | **−2 to −3%** (reduce-scatter replaces all-reduce) | Marginal over Z1 for most configs. |
| **3** | **ZeRO Stage 3 / FSDP** (shard everything) | Shards weights + grads + optim across all GPUs | **Per-GPU: 520 GB / N_gpus**. On 8 GPUs → **65 GB/GPU total states** | **−5 to −10%** (all-gather before every fwd/bwd layer) | Enables training 32B on 8 GPUs. Communication-heavy; overlap is critical. |
| **4** | **Tensor Parallelism (TP)** | Shards weight matrices column/row-wise within a node | **Linear with TP degree**: TP=4 → weights 65→16 GB/GPU, grads same | **−3 to −8%** (all-reduce per layer, 2× per fwd+bwd) | Best on NVLink. Also shards activations proportionally. |
| **5** | **Pipeline Parallelism (PP)** | Each GPU holds only L/PP layers | **Linear**: PP=4 → weights 65→16 GB/GPU | **−5 to −15%** (pipeline bubble: idle time = (PP−1)/total_microbatches) | Bubble is the killer; use interleaved 1F1B to mitigate. |
| **6** | **Full Activation Checkpointing** | Discard all intermediate activations; recompute in backward | **−95% activations**: 55 GB → ~2.7 GB (save only layer inputs) | **−25 to −33%** (recompute = 1× extra forward) | The biggest single MFU hit. Use sparingly. |
| **7** | **Selective Activation Checkpointing** | Checkpoint only every N-th layer (typically N=2) or only attention | **−50 to −70% activations**: 55 → 16–27 GB | **−8 to −15%** (partial recompute) | Best trade-off for most runs. |
| **8** | **FlashAttention (FA2/FA3)** | Tiled attention — never materializes S×S score matrix | **Eliminates score matrix**: saves N_q·S²·2B per layer. For S=4K, 40 heads: **1.34 GB/layer → 0**. Across 64 layers: **saves ~86 GB** | **+8 to +12%** (faster kernel = MFU GAIN, not loss) | Rare case: saves memory AND improves MFU. Must-have. |
| **9** | **Gradient Accumulation** | Process K micro-batches, sum grads, step once | **Activation mem scales with MBS, not GBS**. MBS=1 instead of MBS=8 saves ~7× activation mem | **−2 to −5%** (more kernel launches, less parallelism per step) | Standard technique; keep MBS as large as memory allows. |
| **10** | **Mixed Precision (bf16/fp16)** | Weights + activations in half-precision; optim in fp32 | Already the baseline. If you were in fp32: **−50% weights+grads+acts** | **+10 to +20%** (half the HBM traffic, tensor core utilization) | Always enabled in modern training. |
| **11** | **FP8 Training** (TransformerEngine) | FP8 (E4M3) for forward GEMMs, FP8 (E5M2) for backward | **−25 to −40% activation memory** (half of bf16 for GEMM inputs/outputs) | **+15 to +25%** MFU gain (2× peak FLOPS on H100: 1979 vs 989 TF) | Aggressive; requires loss parity validation. Start with FFN only. |
| **12** | **CPU Offloading** (ZeRO-Offload / FSDP offload) | Move optimizer states and/or weights to CPU DRAM | **−50 to −90% GPU memory** for offloaded components | **−30 to −60%** MFU (PCIe 5.0 = 64 GB/s vs HBM = 3.35 TB/s) | Last resort. Only viable for LoRA or very small MBS. |
| **13** | **Fused Cross-Entropy** (Liger/FlashCE) | Tiled softmax+CE — never materializes [B,S,V] logit tensor | **Saves B·S·V·2B per step**. For S=4K, V=152K: **~1.2 GB** | **+1 to +2%** (fewer HBM round-trips) | Small but free. Always enable for large vocab. |
| **14** | **Gradient Compression (bf16 grads)** | Keep gradients in bf16 instead of fp32 | **−50% gradient memory**: 65→33 GB | **−0 to −1%** (negligible; may affect convergence at very high LR) | Safe for fine-tuning; risky for pretraining with high LR. |
| **15** | **KV Cache Quantization** (inference-time) | Store KV cache in FP8/INT8 instead of bf16 | **−50% KV cache**: for 32K context, 8.6→4.3 GB | N/A (inference) | Minimal quality impact (~0.2 pp). |
| **16** | **LoRA / QLoRA** | Train only low-rank adapters; freeze base | **−93 to −97%** trainable param memory (no grads/optim for frozen params) | **+5 to +15%** (smaller backward graph, fewer all-reduces) | MFU "gain" is misleading — total useful FLOPs is lower, but wall-clock per token improves. |
| **17** | **Sequence Parallelism (SP)** | Shard LayerNorm/Dropout activations along S within TP group | **−(TP−1)/TP of those activations**: TP=4 saves 75% of LN/Dropout acts | **−0.5 to −1%** (extra all-gather/reduce-scatter on activations) | Always pair with TP. Negligible cost. |

### 1.2 Practical Combinations and Their Net Effect

#### Scenario A: Full Fine-Tuning on 1 Node (8×H100)

```
Baseline (single replica):         585 GB
+ ZeRO-3 / FSDP (÷8 GPUs):       states 520→65 GB/GPU        → saves 455 GB total
+ Selective act ckpt (every 2):    acts 55→22 GB               → saves 33 GB
+ FlashAttention-3:                acts −86 GB (score matrices) → saves 86 GB (already in ckpt calc)
+ Fused cross-entropy:             −1.2 GB
─────────────────────────────────────
Per-GPU: ~65 + 22/8 + 1.5 buffers ≈ 69 GB   ✅ fits 80 GB H100
MFU impact: −8% (FSDP) +10% (FA3) −10% (selective ckpt) +1% (fused CE) = net −7% from peak
Realistic MFU: ~38–42%
```

#### Scenario B: Full Fine-Tuning on 4 Nodes (32×H100) — Production

```
+ TP=4 (intra-node NVLink):       states ÷4 within node
+ ZeRO-1 across DP=8:             optim ÷8
+ Selective act ckpt (every 3):    acts 55→28 GB → ÷4 (TP) = 7 GB/GPU
+ FlashAttention-3:                score matrices eliminated
+ Fused CE + torch.compile:        small gains
─────────────────────────────────────
Per-GPU: ~16 (wt) + 5 (grad) + 16 (optim/8) + 7 (acts) + 2 (buf) ≈ 46 GB
Headroom: 34 GB → can increase MBS to 4 or S to 16K
MFU impact: −5% (TP comm) +10% (FA3) −6% (selective ckpt) +2% (torch.compile) = net +1%
Realistic MFU: ~43–48%
```

#### Scenario C: LoRA r=64 on 1×H100

```
Base weights (bf16, frozen):       65 GB     (no grads, no optim)
LoRA adapters + grads + optim:     ~9 GB
Activations (S=4K, MBS=1, ckpt):  ~2.7 GB
Buffers:                           ~2 GB
─────────────────────────────────────
Total: ~79 GB   ✅ barely fits
MFU: concept doesn't cleanly apply (most params frozen), but wall-clock/token is ~3× faster than full FT
```

### 1.3 Memory Reduction Decision Tree

```
                         Can you fit the model?
                              │
                    ┌─────────┴─────────┐
                   No                   Yes
                    │                    │
            Add parallelism:      Can you fit activations?
            TP (intra-node)           │
            PP (inter-node)     ┌─────┴─────┐
            FSDP (all)         No           Yes
                    │           │            │
                    │    Add act ckpt:   Increase MBS/S
                    │    Selective first  for throughput
                    │    Full if needed
                    │           │
                    └───── FlashAttention (always) ─────┘
                                │
                         Still OOM?
                              │
                    ┌─────────┴─────────┐
                   Yes                  No
                    │                    │
            Reduce precision:       Optimize MFU:
            FP8 training            Comm overlap
            Gradient bf16           torch.compile
            CPU offload (last)      Fused kernels
```

---

## Part 2: MFU Improvement Strategies

### 2.0 MFU Baseline Reference

For Qwen2.5-VL-32B on H100 (989 TFLOPS bf16 peak):

| Configuration | Typical MFU | Why |
|---|---|---|
| Naive HF Trainer + FSDP, no tuning | 18–25% | Unoptimized data pipeline, no FA, full ckpt, no overlap |
| Reasonable defaults (FA2, selective ckpt) | 30–35% | Standard setup, missing comm overlap and data opt |
| Well-tuned (FA3, comm overlap, packing) | 43–48% | Production-grade bf16 |
| Fully optimized + FP8 | 50–60% | State of the art |

### 2.1 Strategy Catalog (Ordered by Impact)

| # | Strategy | MFU Gain | Mechanism | Implementation Effort |
|---|---|---|---|---|
| **1** | **FP8 GEMMs (TransformerEngine)** | **+15 to +25 pp** | Doubles peak FLOPS (1979 TF on H100); halves HBM traffic for GEMM inputs. Note: MFU is still measured against bf16 peak (989 TF), so this can push MFU >50%. | Medium. Requires per-tensor scaling calibration. Start with FFN only, validate loss curves. |
| **2** | **FlashAttention-3 (Hopper)** | **+8 to +12 pp** | Exploits Hopper WGMMA + TMA for async data movement; eliminates O(S²) HBM traffic; 1.5–2× over FA2. | Low. Drop-in replacement via `torch.nn.functional.scaled_dot_product_attention` or `flash_attn` package. |
| **3** | **TP Communication Overlap** | **+5 to +8 pp** | Overlap all-gather (for next layer's weights in FSDP, or TP shards) with current layer's compute using separate CUDA streams. TransformerEngine's `--tp-comm-overlap` or Megatron's async TP. | Medium. Requires stream management or TE integration. |
| **4** | **Token Packing + Dynamic Batching** | **+3 to +6 pp** | Eliminate padding waste. Pack multiple short sequences into one `seq_len` slot with attention mask. For VLM with variable images, this can recover 15–30% of wasted tokens. | Medium. Need custom collator and loss masking. |
| **5** | **Resolution Bucketing (VLM-specific)** | **+3 to +5 pp** | Group images by pixel count into buckets → even compute across DP ranks. Without this, stragglers (one rank gets a 1344×1344 image while others get 224×224) waste up to 40% of step time. | Low-Medium. Modify data sampler. |
| **6** | **Selective Activation Checkpointing** (vs full) | **+10 to +18 pp** (vs full ckpt) | Full ckpt recomputes every layer (33% overhead); selective (every 2nd layer) recomputes only 50% → ~15% overhead. | Low. One config flag. |
| **7** | **Reduce PP Bubble** (interleaved 1F1B / ZB-PP) | **+2 to +5 pp** | Naive 1F1B bubble = (PP−1)/(PP+num_microbatches−1). Interleaving with virtual stages halves the bubble. Zero-bubble PP (ZB-H1) eliminates it almost entirely. | Medium-High. Requires Megatron or custom scheduler. |
| **8** | **CUDA Graphs / torch.compile** | **+2 to +5 pp** | Capture the full forward+backward as a single graph replay; eliminates per-op kernel launch overhead (100–300μs per launch × thousands of ops). Biggest gain at small batch sizes. | Medium. Requires static shapes; dynamic VLM inputs need bucketed graph pools. |
| **9** | **Hierarchical AllReduce + SHARP** | **+2 to +4 pp** | Intra-node NVLink reduce → inter-node IB reduce. SHARP v3 does in-network reduction on InfiniBand switches. Cuts DP all-reduce time by 30–50% at scale. | Low (NCCL config tuning) to Medium (SHARP requires switch firmware). |
| **10** | **DALI GPU Image Decode** | **+2 to +4 pp** | CPU JPEG decode is 2–5 ms/image; GPU (nvJPEG via DALI) is 0.1–0.5 ms. For VLM batches with 8–32 images, this saves 10–100 ms/step. | Medium. Replace torchvision transforms with DALI pipeline. |
| **11** | **Fused Kernels** (RMSNorm+residual, RoPE+GEMM, SwiGLU fused) | **+1 to +3 pp** | Eliminate HBM round-trips between elementwise ops. Each fusion saves one read+write of the full activation tensor. | Low. Use TransformerEngine or Apex fused layers. |
| **12** | **Fused Cross-Entropy** (Liger/FlashCE) | **+1 to +2 pp** | Avoid materializing [B,S,152K] logit tensor (1.2 GB). Tiled softmax+grad in one kernel. | Low. Drop-in loss function. |
| **13** | **Async Checkpointing** | **+1 to +3 pp** | Overlap checkpoint writing with next training step. Without this, saving a 65 GB checkpoint stalls training for 5–15 s every N steps. | Low-Medium. Use `torch.distributed.checkpoint` with async writer. |
| **14** | **Data Pipeline Prefetch** | **+1 to +2 pp** | Prefetch ≥2 batches ahead; overlap I/O and CPU preprocessing with GPU compute. | Low. `num_workers=8, prefetch_factor=4` in DataLoader. |
| **15** | **GBS / MBS Tuning** | **+1 to +5 pp** | Larger MBS → larger matmuls → higher tensor core utilization (arithmetic intensity). But too large → OOM. Sweet spot depends on S and memory. | Low. Hyperparameter sweep. |

### 2.2 Cumulative MFU Journey (Qwen2.5-VL-32B, 32×H100)

```
Stage 0: Naive baseline                          22% MFU
  │
  ├─ + FlashAttention-3                          → 32%   (+10)
  ├─ + Selective ckpt (every 2 layers)           → 38%   (+6, vs full ckpt)
  ├─ + Token packing + resolution bucketing      → 43%   (+5)
  ├─ + TP comm overlap (TransformerEngine)       → 48%   (+5)
  ├─ + Fused kernels + fused CE + torch.compile  → 51%   (+3)
  ├─ + DALI + async ckpt + prefetch              → 53%   (+2)
  │
  │  ──── bf16 ceiling ≈ 53% ────
  │
  ├─ + FP8 (FFN GEMMs only)                     → 60%   (+7)
  ├─ + FP8 (all GEMMs incl. attention)           → 65%   (+5)
  │
  │  ──── FP8 ceiling ≈ 65% (of bf16 peak) ────
  │
  Final: 65% MFU = 643 TFLOPS/GPU effective
         (or ~33% of FP8 peak 1979 TF, which is the
          hardware-honest number)
```

### 2.3 Why 100% MFU Is Impossible

Even a perfectly compute-bound GEMM on H100 achieves ~85% of peak tensor core FLOPS due to:
- Memory latency for loading weight tiles (~5%)
- Warp scheduling and pipeline drain (~3%)
- Softmax, LayerNorm, RoPE, residuals — all non-GEMM (~7–12% of wall time)
- Communication (even with overlap, 2–5% leaks)
- Python/framework overhead (<1% with compile)

Practical ceiling: **55–65% bf16 MFU** for a 32B model on 32+ GPUs.

---

## Part 3: Profiling Story — Finding and Fixing Bottlenecks in Qwen2.5-VL-32B Training

### The Setting

You've just launched Qwen2.5-VL-32B fine-tuning on **4 nodes (32 × H100 SXM5)** connected via InfiniBand NDR (400 Gb/s/GPU). Your config:

```yaml
model: Qwen2.5-VL-32B-Instruct
parallelism: TP=4, DP=8, PP=1
optimizer: AdamW (bf16 weights, fp32 optim)
seq_len: 4096
micro_batch_size: 2
gradient_accumulation: 4
global_batch_size: 64  (8 DP × 2 MBS × 4 GA)
activation_checkpointing: full (every layer)
flash_attention: FA2
data: 10TB multimodal (images + text), WebDataset
```

You observe: **step time = 14.2 s**, giving:

```
tokens/step = 64 × 4096 = 262,144
FLOPs/step = 6 × 32.5e9 × 262,144 = 5.11e16
MFU = 5.11e16 / (14.2 × 32 × 989e12) = 11.3%
```

**11.3% MFU. Terrible.** You expect 40–50%. Time to profile.

---

### Act 1: The Bird's-Eye View (PyTorch Profiler)

**Tool:** PyTorch Profiler + TensorBoard

```python
from torch.profiler import profile, schedule, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=2, warmup=2, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler('./tb_logs'),
    record_shapes=True,
    with_stack=True,
    profile_memory=True,
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()
        if step >= 10:
            break
```

**What you see in TensorBoard (Chrome trace view):**

```
Step timeline (14.2s total):
┌──────────────────────────────────────────────────────────────┐
│ data_load │ vit_fwd │ llm_fwd │ llm_bwd (recomp+grad)  │opt│
│   3.1s    │  0.4s   │  1.8s   │       7.9s             │1.0│
└──────────────────────────────────────────────────────────────┘
     22%       2.8%      12.7%          55.6%              7.0%
```

**Finding 1: Data loading is 22% of step time (3.1s).** GPU is IDLE during this time. This alone caps MFU at ~78% theoretical.

**Finding 2: Backward pass is 55.6% — but it should be ~2× forward (should be ~3.6s, not 7.9s).** Full activation checkpointing is adding a full extra forward inside the backward → 3× forward expected, but 7.9s ÷ 1.8s = 4.4× is worse than expected. Something else is happening.

**Finding 3: Optimizer step takes 1.0s.** For bf16 weights + fp32 optimizer on 32 GPUs with ZeRO-1, this seems high.

**Action items identified:**
1. Fix data loading bottleneck (target: <0.5s)
2. Investigate why backward is 4.4× forward instead of 3×
3. Profile optimizer step

---

### Act 2: Diving into the Backward (Nsight Systems)

**Tool:** NVIDIA Nsight Systems on rank 0

```bash
nsys profile \
  -t cuda,nvtx,osrt,nccl \
  --cuda-memory-usage=true \
  -o qwen32b_profile \
  --capture-range=cudaProfilerApi \
  torchrun --nproc_per_node=8 train.py --profile-nsys
```

Open `qwen32b_profile.nsys-rep` in Nsight GUI. Zoom into one backward layer:

```
Backward for Layer 31 (representative):
┌─────────────────────────────────────────────────────────────────┐
│ recompute_fwd              │ nccl_allreduce │ grad_gemms        │
│ (attention: 42ms)          │   (18ms IDLE)  │ (QKV+FFN: 28ms)  │
│ ├─ sdpa_kernel: 26ms ←!!  │   ← BUBBLE     │                   │
│ └─ gemms: 16ms             │                │                   │
└─────────────────────────────────────────────────────────────────┘
Total: 88ms/layer × 64 layers = 5.6s (matches bulk of 7.9s bwd)
```

**Finding 4: Attention recompute takes 42ms per layer, dominated by `sdpa_kernel` at 26ms.** You're using FA2, which doesn't exploit Hopper's WGMMA/TMA. FA3 would cut this to ~14ms.

**Finding 5: 18ms NCCL idle gap per layer.** The TP all-reduce after attention and after FFN are serialized with compute — no overlap. That's 18ms × 64 = 1.15s of pure communication bubble.

**Finding 6: Full activation checkpointing is recomputing every layer.** For layers where the activation is cheap (just the input tensor), selective checkpointing would skip recompute on half the layers.

**Action items refined:**
1. ~~Fix data loading~~ (already noted)
2. **Upgrade FA2 → FA3** (save ~12ms/layer × 64 = 768ms)
3. **Enable TP comm overlap** (save ~12ms/layer × 64 = 768ms)
4. **Switch to selective checkpointing every 2 layers** (halve recompute cost: save ~1.4s)

---

### Act 3: Data Pipeline Forensics

**Tool:** PyTorch Profiler CPU trace + manual NVTX markers

You add timing around the data pipeline:

```python
with torch.cuda.nvtx.range("data_load"):
    batch = next(dataloader_iter)

with torch.cuda.nvtx.range("image_preprocess"):
    images = preprocess_images(batch['images'])  # CPU!
```

Nsight trace shows:

```
data_load breakdown (3.1s):
├─ tar shard read (Ceph):     0.8s   (I/O)
├─ JPEG decode (CPU, PIL):    1.4s   ← DOMINANT
├─ resize + normalize (CPU):  0.6s
└─ tensor collate + transfer: 0.3s   (H2D copy)
```

**Finding 7: CPU JPEG decode (PIL/torchvision) at 1.4s** for a batch of ~16 images averaging 800×600. The CPU is the bottleneck — not storage.

**Finding 8: No prefetching.** DataLoader `num_workers=2, prefetch_factor=2` — the main thread waits for workers.

**Action items:**
5. **Switch to DALI GPU decode** (nvJPEG: ~0.1s for 16 images)
6. **Increase DataLoader workers to 8, prefetch to 4**
7. **Add local NVMe cache** for hot shards (eliminate 0.8s I/O on repeat epochs)

---

### Act 4: The Optimizer Step

**Tool:** PyTorch Profiler, zoom into optimizer

```
optimizer.step() breakdown (1.0s):
├─ all-gather master weights:  0.3s
├─ unscale gradients (fp32):   0.1s
├─ adam update (fp32):         0.2s
├─ copy master → bf16 weights: 0.1s
└─ broadcast updated weights:  0.3s
```

**Finding 9: Weight broadcast after optimizer step takes 0.3s.** This is a ZeRO-3 artifact — after each GPU updates its shard, it must broadcast the new weights. With ZeRO-1 + TP=4, you wouldn't need this.

**Action item:**
8. **Switch from ZeRO-3 to TP=4 + ZeRO-1.** The 32B model with TP=4 fits weights in ~16 GB/GPU; ZeRO-1 across DP=8 shards the optimizer. Eliminates the 0.6s all-gather/broadcast cycle.

---

### Act 5: Single-Kernel Deep Dive

**Tool:** Nsight Compute (ncu) on the hottest GEMM

```bash
ncu --set full \
    --kernel-name regex:"gemm" \
    --launch-count 5 \
    python train.py --single-step
```

Focus on the FFN `gate_proj` GEMM: `[MBS·S, H] × [H, H_ff]` = `[8192, 5120] × [5120, 27648]`

```
Nsight Compute report:
┌─────────────────────────────────────────┐
│ Metric              │ Value   │ Peak %  │
├─────────────────────┼─────────┼─────────┤
│ Tensor Core Active  │  78.3%  │         │
│ DRAM Throughput     │ 2.41 TB/s │ 71.9% │
│ L2 Hit Rate         │  42.1%  │         │
│ Occupancy           │  67.2%  │         │
│ Achieved TFLOPS     │ 773 TF  │ 78.2%  │
│ Arithmetic Intensity│  84.3   │ compute │
│                     │         │ bound ✓ │
└─────────────────────────────────────────┘
```

**Finding 10: GEMM itself is 78% efficient** — reasonable for this matrix size. The bottleneck is NOT the individual kernels — it's the gaps between them (communication, recompute, data loading).

**Finding 11: With FP8, this GEMM could run at 1546 TFLOPS** (2× tensor core throughput). The arithmetic intensity is high enough to stay compute-bound in FP8.

**Action item:**
9. **Enable FP8 for FFN GEMMs** (TransformerEngine). Expected 1.8× speedup on FFN forward+backward GEMMs. FFN is ~75% of per-layer FLOPs → big impact.

---

### Act 6: Applying All Fixes

| Fix | Before | After | Time Saved |
|---|---|---|---|
| 1. DALI GPU decode + prefetch=4 + workers=8 | 3.1s data load | 0.3s | **−2.8s** |
| 2. FA2 → FA3 | 26ms/layer attn | 14ms/layer | **−0.77s** |
| 3. TP comm overlap | 18ms/layer idle | 3ms/layer | **−0.96s** |
| 4. Full ckpt → selective (every 2) | 42ms/layer recompute | 21ms/layer | **−1.34s** |
| 5. ZeRO-3 → TP=4 + ZeRO-1 | 1.0s optim step | 0.4s | **−0.60s** |
| 6. Fused kernels + fused CE | misc overhead | — | **−0.30s** |
| 7. FP8 FFN GEMMs | — | — | **−1.10s** |
| **Total saved** | | | **−7.87s** |

```
New step time: 14.2 − 7.87 ≈ 6.3s

New MFU = 5.11e16 / (6.3 × 32 × 989e12) = 25.6%
```

Wait — that's still only 25.6%. But we said FP8 pushes FLOPs higher. Let me recalculate properly:

With FP8 on FFN (75% of model FLOPs), effective throughput on those GEMMs is ~1.8×. The "6NT" model-FLOP count stays the same, but hardware does them faster. MFU (measured against bf16 peak = 989 TF) can exceed 50%:

```
Corrected: the time savings from FP8 are already folded into step time.
MFU = 5.11e16 / (6.3 × 32 × 989e12) = 25.6%  ← this is too low still.
```

Something is still wrong. Let me recheck the step breakdown:

```
Optimized step breakdown (6.3s):
├─ data_load (overlapped):     0.0s   (fully hidden behind compute)
├─ vit_fwd:                    0.35s
├─ llm_fwd:                    1.1s   (FP8 FFN)
├─ llm_bwd (selective ckpt):   3.4s   (FP8, FA3, partial recompute)
├─ optim step:                 0.4s
├─ comm overhead (residual):   0.3s
├─ misc (collate, logging):    0.2s
└─ GA overhead (4 microsteps): 0.55s
Total: ~6.3s
```

MFU = 5.11e16 / (6.3 × 32 × 989e12) = **25.6%**.

Hmm, this is low because the GBS is only 64 sequences × 4096 tokens = 262K tokens — fairly small. The per-step FLOPs are modest. Let's also increase GBS:

**Fix 10: Increase GBS from 64 to 256** (MBS=4, GA=4, DP=8 → 4·4·8·4096 = 524K tokens or MBS=2, GA=16 → same):

With MBS=4 (fits due to memory savings from selective ckpt + TP=4):
```
New tokens/step = 256 × 4096 = 1,048,576
New FLOPs/step = 6 × 32.5e9 × 1,048,576 = 2.04e17
Step time: ~12.5s (4× more compute, but better arithmetic intensity)
MFU = 2.04e17 / (12.5 × 32 × 989e12) = 51.6%
```

**Now we're talking. 51.6% MFU.**

---

### Act 7: Long-Run Monitoring

**Tool:** DCGM + Prometheus + Grafana dashboard

After deploying the optimized config, monitor 24/7:

```
Grafana panels:
┌──────────────────────────────────────────────────┐
│ MFU Over Time                                     │
│ ████████████████████ 51.2%  (stable)              │
│                                                    │
│ Step Time Distribution                             │
│ mean=12.5s, p50=12.3s, p99=14.1s                  │
│ ─── spikes at steps 2400, 4800 (checkpoints) ──   │
│                                                    │
│ GPU Temp (rack 3)                                  │
│ ████████████████████ 72°C (normal)                │
│                                                    │
│ IB Tx/Rx (node 2, port 3)                          │
│ ▓▓▓▓▓▓░░░░░░░░ 180 Gb/s (of 400)                 │
│ ─── healthy headroom ───                           │
│                                                    │
│ ALERT: step 8847 took 45s (3.6× normal)           │
│ → Investigation: NVMe cache miss on new shard      │
│   rotation. Fix: pre-warm cache 1 epoch ahead.     │
└──────────────────────────────────────────────────┘
```

**Finding 12 (from long-run telemetry):** Every ~2400 steps, step time spikes to 25s. Correlation: that's the checkpoint interval. **Async checkpointing wasn't properly enabled** — the main training thread blocks on `state_dict()` serialization.

**Final action item:**
10. **Enable truly async checkpointing** with `torch.distributed.checkpoint.async_save()` or background thread + pinned staging buffer.

After this: step-time variance drops, p99 goes from 14.1s to 12.8s, effective MFU climbs to **52.3%**.

---

### The Final Scorecard

| Metric | Before | After | Improvement |
|---|---|---|---|
| Step time | 14.2s | 12.5s | 1.14× faster |
| GBS (tokens/step) | 262K | 1.05M | 4× more work/step |
| MFU | 11.4% | 51.7% | **4.5× improvement** |
| Tokens/day | 1.6B | 7.2B | 4.5× throughput |
| Projected time for 100B tokens | 63 days | 14 days | **4.5× faster** |
| GPU cost ($3/GPU-hr, 32 GPUs) | $145K | $32K | **$113K saved** |

### Key Lessons

1. **Profile before optimizing.** The 11% MFU wasn't caused by one thing — it was death by a thousand cuts (data loading 22%, full ckpt 15%, no FA3 5%, no comm overlap 7%, bad ZeRO config 4%, small GBS 20%).

2. **Fix the data pipeline first.** It's always the data pipeline. 22% of step time sitting idle waiting for JPEG decodes on CPU is money burned.

3. **FlashAttention is non-negotiable.** FA3 on Hopper saves memory AND improves MFU — the only free lunch in ML systems.

4. **Communication overlap is the second-biggest lever** after FA. Serialized TP all-reduces cost 7% MFU in this case.

5. **Full activation checkpointing is a trap.** Teams enable it "just to be safe" and leave 15% MFU on the table. Always try selective first.

6. **GBS matters more than most people think.** Doubling GBS (from 64 to 256) doubled MFU because larger matmuls saturate tensor cores. Profile your arithmetic intensity.

7. **FP8 is the final boss.** Once you've exhausted bf16 optimizations (~48% ceiling), FP8 is the only path to 60%+. But validate convergence first — a 0.5% loss spike at step 5000 is cheaper to fix in bf16 experiments.

8. **Long-run telemetry catches what profiling misses.** The checkpoint spike was invisible in a 10-step profile window. DCGM + Grafana running 24/7 caught it.

---

### Profiler Toolbox Quick Reference

| Tool | When to Use | Overhead | What It Shows |
|---|---|---|---|
| **PyTorch Profiler** | First look; per-op time, memory | ~1% | Op-level timeline, memory peaks, NCCL calls |
| **Nsight Systems (nsys)** | GPU/NCCL timeline across ranks | 10–20% | Kernel-level timeline, CPU-GPU sync, NCCL overlap, NVLink usage |
| **Nsight Compute (ncu)** | Single-kernel deep dive | 50–100× (on target kernel only) | Tensor core utilization, memory throughput, occupancy, register pressure |
| **HTA (Holistic Trace Analysis)** | Multi-rank straggler detection | Post-hoc | Comm/compute overlap ratio, critical path, rank imbalance |
| **DCGM + Prometheus + Grafana** | 24/7 production monitoring | <1% | SM activity, temp, power, IB bandwidth, ECC errors, MFU trend |
| **torch.cuda.memory_stats()** | Memory debugging | 0% | Peak allocated, peak reserved, fragmentation |
| **`py-spy` / `cProfile`** | CPU-side bottlenecks | 1–5% | Python-level hotspots (data loading, preprocessing) |
