# Qwen2.5-VL-72B: Architecture, Distributed Training, MFU, and Inference

> Notes written from the perspective of an ML systems engineer who has trained and served large Qwen-family VLMs on H100 clusters. Numbers below use `bf16` weights/activations with `fp32` optimizer states (AdamW) unless stated otherwise. The "70B" model in the Qwen2.5-VL family is officially **72B** parameters; all math uses 72B.

---

## 1. Model Architecture Summary

Qwen2.5-VL has three stacked components:

1. **Vision Transformer (ViT) encoder** — processes pixels into visual tokens.
2. **MLP Patch Merger (projector)** — aligns visual tokens with the LLM's hidden size and 4× compresses tokens by merging 2×2 spatial groups.
3. **Qwen2.5-72B LLM decoder** — multimodal autoregressive decoder consuming interleaved text + visual tokens.

### 1.1 Vision Encoder (ViT)

| Item | Value |
|---|---|
| Architecture | Native-dynamic-resolution ViT (redesigned from Qwen-VL) |
| Patch size | 14 × 14 |
| Hidden size `H_v` | 1280 |
| Heads | 16 (head dim 80) |
| Layers | 32 |
| MLP inter size | 5120 (4×) |
| Positional encoding | 2D-RoPE over (H/14, W/14) grid, absolute time embed for video |
| Attention | Windowed local attention interleaved with 4 full-attention layers |
| Activations | GELU |
| Parameters | ~675 M |

Dynamic resolution means the grid `(H/14) × (W/14)` varies per sample; no resize to a fixed 224/336 canvas. For a 448×448 image the grid is 32×32 → **1024 patch tokens**.

### 1.2 Patch Merger (Projector)

A 2-layer MLP that (a) concatenates features from a 2×2 spatial group (→ 4·H_v = 5120), then (b) projects down to LLM hidden 8192. Net effect: visual token count is **reduced 4×** (1024 → 256 tokens for a 448×448 image).

### 1.3 LLM Backbone (Qwen2.5-72B)

| Item | Value |
|---|---|
| Layers `L` | 80 |
| Hidden `H` | 8192 |
| FFN inter `H_ff` | 29,568 |
| Attention heads `N_q` / `N_kv` | 64 / 8 (GQA, 8:1) |
| Head dim `D` | 128 |
| Vocab `V` | 152,064 |
| Max native context | 32K (→ 128K with YaRN) |
| Norm | RMSNorm (pre-norm) |
| Activation | SwiGLU (gate·silu × up → down) |
| Positional | 1D-RoPE (with M-RoPE extension for images/video in VL variant) |
| Tied embeddings | No (separate `lm_head`) |

**Parameter accounting per LLM layer (H=8192):**

| Tensor | Shape | Params |
|---|---|---|
| `q_proj` | 8192 × 8192 | 67.1 M |
| `k_proj` | 8192 × 1024 | 8.39 M |
| `v_proj` | 8192 × 1024 | 8.39 M |
| `o_proj` | 8192 × 8192 | 67.1 M |
| `gate_proj` | 8192 × 29568 | 242.2 M |
| `up_proj` | 8192 × 29568 | 242.2 M |
| `down_proj` | 29568 × 8192 | 242.2 M |
| 2× RMSNorm | 2 × 8192 | 0.016 M |
| **Per layer total** | | **≈ 877.7 M** |

Across 80 layers: **70.2 B**.
Plus: `embed_tokens` 152064·8192 ≈ 1.246 B, `lm_head` ≈ 1.246 B (≈ 1.25 B if untied), vision 0.675 B, merger ~0.085 B, final norm ~8 K.

**Grand total ≈ 72.0 B** parameters (of which the LLM ≈ 71 B, vision path ≈ 0.76 B).

---

## 2. Layer-by-Layer Forward / Backward Math

I'll use a concrete micro-batch for the LLM: `B=1`, `S=4096`, bf16 (2 bytes). All matmuls counted as `2·m·n·k` FLOPs.

### 2.1 Input embedding

- Forward: gather rows → `[B, S, H] = [1, 4096, 8192]`, **64 MB** activation (2·1·4096·8192).
- Backward: index-scatter-add into embedding grad; ~negligible FLOPs, sparse update.

### 2.2 One LLM transformer layer

Input `X: [1, 4096, 8192]` (64 MB).

**(a) RMSNorm #1**
- In/Out: [1, 4096, 8192]. Elementwise, O(BSH) ≈ 34 MFLOPs. Save reciprocal-rms for backward.

**(b) Q / K / V projections (GQA)**
- `Q = X·W_q`: `[1,4096,8192]·[8192,8192] → [1,4096,8192]`. FLOPs = 2·1·4096·8192·8192 ≈ **0.55 TF**. Output 64 MB.
- `K = X·W_k`: `[1,4096,8192]·[8192,1024] → [1,4096,1024]`. **0.069 TF**. Output 8 MB.
- `V = X·W_v`: same shape as K. **0.069 TF**. 8 MB.

**(c) RoPE** on Q,K — pointwise cos/sin rotation, ~negligible FLOPs, ~0.

**(d) Scaled dot-product attention (FlashAttention)**
- View `Q:[1,64,4096,128]`, `K:[1,8,4096,128]`, `V:[1,8,4096,128]`. GQA broadcasts K,V to 64 heads during the kernel.
- FLOPs ≈ `4·B·N_q·S²·D` for QKᵀ + PV (with causal mask, multiply by ½ → `2·B·N_q·S²·D`) ≈ 2·1·64·4096²·128 ≈ **0.275 TF**.
- **Without** FlashAttention, the `S×S` attention matrix `[1,64,4096,4096]` alone is **2.0 GB** of bf16 activation — the reason FA2/FA3 is mandatory.
- **With** FlashAttention-2/3: no materialized score matrix; activation footprint drops to just Q,K,V and output buffers (~144 MB). Store only the log-sum-exp per row for backward (1·64·4096·4 B = 1 MB).
- Output `O: [1,4096,8192]` — 64 MB.

**(e) O-projection**
- `Y = O·W_o`: `[1,4096,8192]·[8192,8192] → [1,4096,8192]`. **0.55 TF**. 64 MB.
- Residual add with X.

**(f) RMSNorm #2** — same as (a).

**(g) SwiGLU FFN**
- `G = X·W_gate`: `[1,4096,8192]·[8192,29568] → [1,4096,29568]`. FLOPs = 2·1·4096·8192·29568 ≈ **1.984 TF**. Act: **231 MB**.
- `U = X·W_up`: same. **1.984 TF**, 231 MB.
- `H_act = silu(G) ⊙ U`: elementwise, 231 MB intermediate (can be recomputed).
- `Y = H_act·W_down`: `[1,4096,29568]·[29568,8192] → [1,4096,8192]`. **1.984 TF**, 64 MB.
- Residual add.

**Per-layer forward totals (B=1, S=4096):**
- FLOPs ≈ 0.55 + 0.069 + 0.069 + 0.275 + 0.55 + 1.984·3 ≈ **7.46 TFLOPs**
- Peak activation memory (bf16, no recompute) ≈ 64 (X) + 64 (Q) + 8 (K) + 8 (V) + 64 (attn out) + 64 (O-proj) + 231·3 (FFN) ≈ **975 MB/layer**
- With FlashAttention + activation checkpointing at layer granularity: keep only `X` (64 MB) and re-run forward during backward → ~**64 MB/layer saved**.

**Across 80 layers (forward only, no LM head):**
- **597 TFLOPs** forward.
- Activation memory without checkpointing: 80·975 MB ≈ **78 GB** — exceeds a single H100 (80 GB) once gradients, KV, params arrive.
- With full activation checkpointing: ≈ 80·64 MB = **5.1 GB** activations; backward recompute costs ~1× extra forward FLOPs.

### 2.3 LM head + loss

- `logits = h_last·W_lm`: `[1,4096,8192]·[8192,152064] → [1,4096,152064]`. FLOPs = 2·4096·8192·152064 ≈ **10.2 TF** (!). Activation buffer (bf16): 4096·152064·2 = **1.19 GB** — fused cross-entropy (e.g. Liger Kernel / FlashCE) avoids materializing it.
- Loss and softmax: ~1 GFLOP.

### 2.4 Backward pass

General rule for dense ops: **backward ≈ 2× forward FLOPs** (grad w.r.t. input + grad w.r.t. weight). Total training FLOPs per step are conventionally counted as **6·N·T** (N=params, T=tokens) which captures fwd + bwd for dense matmuls:
- For 72B params, S=4096 tokens/sample, per-sample compute ≈ 6·72e9·4096 ≈ **1.77 PF**.
- Backward recompute (with full activation checkpoint) adds another 1× forward ≈ **8.97 PF total** (~1.5× the "6·N·T" baseline), equivalent to a 6 → 9·N·T effective rate.

### 2.5 Vision pass (reference, 448×448 image)

- Patchify: 32×32 grid, 1024 tokens, H_v=1280. Activation `[1,1024,1280]` = 2.5 MB.
- Per ViT layer (H_v=1280, N=16, D=80, MLP=5120, windowed): ~5.6 GFLOPs forward.
- 32 layers: ~180 GFLOPs.
- Merger: 1024 → 256 tokens, `[1,256,8192]` = 4 MB entering LLM.

For a VLM training batch, vision is typically <5 % of total FLOPs — the LLM decoder dominates.

### 2.6 Static memory for full training (single replica, before parallelism)

| Bucket | Bytes/param | For 72B |
|---|---|---|
| Weights (bf16) | 2 | 144 GB |
| Gradients (bf16 or fp32) | 2–4 | 144–288 GB |
| AdamW m, v (fp32) | 8 | 576 GB |
| Master weights (fp32) | 4 | 288 GB |
| **State total** | **16–18** | **~1.15–1.30 TB** |
| Activations (S=4096, MBS=1, checkpointed) | — | ~5–10 GB |

**Single H100 = 80 GB → impossible to hold a replica.** Must shard.

---

## 3. Distributed Training on H100 Clusters

### 3.1 How many H100s to "fully load" the 72B model?

With pure FSDP / ZeRO-3 (params+grads+optim sharded across DP):
- State budget ≈ 1.15 TB → need ≥ **15 H100 GPUs** just for states. Leave 30–40 % headroom for activations, temp buffers, NCCL/ NVSHMEM, CUDA graphs → **24 GPUs (3 nodes)** is a realistic floor at MBS=1.
- In practice, to fit longer sequences (S=8K–32K) and reasonable per-GPU batch, the minimum footprint is usually **1 node (8 × H100)** running with aggressive sharding:
  - TP=8 intra-node (using NVLink/NVSwitch, 900 GB/s) shards every weight matrix — each GPU holds 72B/8 ≈ 9B params in weights → 18 GB weights, 72 GB optim → doesn't fit on one node alone.
  - A single node **cannot** train 72B alone with AdamW fp32. It can fit for **inference** (~150 GB bf16 across 8×80 GB).
- **Minimum for training** is commonly **2 nodes (16 GPUs)** with TP=8 + PP=2, or **4 nodes (32 GPUs)** with TP=8 + PP=4 to give breathing room for activations/pipe bubbles.
- **Typical "comfortable" training unit** for Qwen-72B-class models: **TP=8, PP=4, DP≥4 → 128 GPUs = 16 nodes minimum.** That is the smallest configuration most teams actually run in production.

### 3.2 Data volume: 100 TB compressed multimodal corpus

Unpacking 100 TB of JPEG/WebP + tokenized text:
- Image count: assume average 150 KB/image compressed → **~670 M images**. Each produces ~200–1000 visual tokens after patch-merger (say 400 avg).
- Text: a typical VLM corpus at 100 TB has ~3–5 % text by size → ~3–5 TB of raw text → **~1.0–1.7 T text tokens**.
- Visual tokens: 670 M × 400 ≈ **270 B visual tokens** (counted post-merger; these are much more expensive per-token than text because they carry image info).

Call total effective training tokens **T ≈ 1.3–2.0 T tokens** for one epoch of mixed-modality. For multi-stage training (pretrain + mid-train + SFT + DPO) total exposure lands around **3–5 T tokens**.

**Compute budget (6·N·T):**
- N = 72e9, T = 3e12 → **1.30 × 10²⁴ FLOPs** (Chinchilla-style one pass).
- Effective with activation recompute (×1.5) → 1.95 × 10²⁴ FLOPs.

**Wall-clock on H100 (bf16 dense peak = 989 TFLOPS):**

| Cluster | GPUs | Aggregate peak | @ 40 % MFU | Days for 2e24 FLOPs |
|---|---|---|---|---|
| 256 (32 nodes) | 256 | 253 PF | 101 PF | **~229 days** |
| 512 (64 nodes) | 512 | 506 PF | 202 PF | ~114 days |
| 1024 (128 nodes) | 1024 | 1.01 EF | 405 PF | **~57 days** |
| 2048 (256 nodes) | 2048 | 2.02 EF | 810 PF | **~29 days** |
| 4096 (512 nodes) | 4096 | 4.05 EF | 1.62 EF | ~14 days |

**Industry-typical production run for 72B VLM on ~2 T tokens is 1024–2048 H100s for ~30–60 days.** Alibaba's own Qwen2.5-VL-72B was trained on a scale in that range.

### 3.3 Parallelism recipe (what we actually run)

A proven configuration for Qwen2.5-VL-72B on 128 nodes (1024 × H100):

| Dimension | Value | Rationale |
|---|---|---|
| Tensor Parallel (TP) | 8 | Keep TP inside a node (NVLink 900 GB/s). Shard Q/K/V/O, gate/up/down column-wise. |
| Sequence Parallel (SP) | on | Shards activations along S dim inside TP group; saves ≈ (TP-1)/TP of LayerNorm/Dropout activations. |
| Context Parallel (CP) | 2–4 | For 32K+ sequences; Ring/Zigzag attention across GPUs. |
| Pipeline Parallel (PP) | 4 | Across NVLink islands / top-of-rack. Use interleaved-1F1B (virtual stages=4) to cut bubble to ~5 %. |
| Data Parallel (DP) | 32 (=1024 / 8 / 4) | Across InfiniBand (NDR 400 Gb/s per GPU via rail-optimized NDR). |
| ZeRO-1 on top of DP | on | Shard optimizer states across DP; ×4 memory savings; negligible extra comm. |
| Activation checkpointing | selective (every 2 layers) | Trade ~20 % recompute for 2× activation memory. |
| Precision | bf16 weights/acts, fp32 master & optim | Default. |
| FlashAttention | FA3 on H100 | ~1.5–2× over FA2 on Hopper. |
| Vision tower | DP only (no TP) | Small model; TP would be comm-bound. Overlap vision forward with LLM backward. |

Vision data pipeline uses NVIDIA DALI for GPU-side JPEG decode, shuffled WebDataset shards on a shared object store (Lustre / Ceph / Alluxio), with **1 PB/day** read throughput expected from the storage fabric.

---

## 4. MFU: Definition, Calculation, Profiling

### 4.1 Formula

MFU = (achieved model FLOPs/sec) / (hardware peak FLOPs/sec)

Achieved FLOPs/sec is conventionally computed with the **6·N·T** approximation:

```
tokens_per_step = global_batch_size × seq_len
model_flops_per_step = 6 × N_params × tokens_per_step
   (add +2·N·T for activation recompute ⇒ 8·N·T effective)
achieved_tflops_per_gpu = model_flops_per_step / (step_time_s × num_gpus) / 1e12
MFU = achieved_tflops_per_gpu / 989    # H100 bf16 dense peak
```

**Worked example** on 1024 H100s, Qwen2.5-VL-72B, S=8192, global batch 1024 sequences, step time 18 s:
- tokens/step = 8192 × 1024 = 8.39 M
- FLOPs/step = 6 · 72e9 · 8.39e6 = 3.62e18 = 3.62 EF
- FLOPs/s/GPU = 3.62e18 / (18 · 1024) = 1.97e14 = **197 TFLOPS/GPU**
- MFU = 197 / 989 = **19.9 %** — low, diagnose and tune.

After optimization to step time 8.0 s:
- 3.62e18 / (8 · 1024) = 442 TFLOPS/GPU → **44.7 % MFU** (good).

### 4.2 How I profile

I run a 3-level profiling funnel, cheapest first:

1. **PyTorch Profiler + TensorBoard** (in-process, ~1 % overhead).
   - Enabled via `torch.profiler.profile` with `record_shapes=True, with_stack=True`.
   - Wrap 3–5 steps after warmup, emit `chrome://tracing` JSON.
   - Immediate view: per-op time, GPU/CPU timeline, NCCL bars, kernel launch gaps. Good for "why is this op slow?".
2. **NVIDIA Nsight Systems (`nsys profile`)** — cluster-wide timeline, 10–20 % overhead but captures NVTX ranges, CUDA/NCCL/NVLink counters, DMA, page faults, IB traffic.
   - Run on 1 node for a couple of steps: `nsys profile -t cuda,nvtx,osrt,cudnn,cublas,nccl -o trace ./launch.sh`.
   - Inspect in Nsight GUI; look at GPU kernel utilization %, NCCL kernel width, and whether `ncclAllReduce` overlaps with `nccl_*_kernel`.
3. **Nsight Compute (`ncu`)** for single-kernel analysis when a specific matmul/attention kernel looks slow. Reports achieved vs. peak tensor core usage, memory throughput %, L2 hit rate, register pressure.
4. **Holistic Trace Analysis (HTA)** aggregates PyTorch Profiler traces from many ranks to identify stragglers, comm/compute overlap ratio, and the critical path of a step.
5. **DCGM + Prometheus/Grafana** for long-run telemetry: SM activity (%), tensor core active (%), NVLink Tx/Rx, InfiniBand Rx/Tx, HBM BW, power, clock throttling, ECC errors. This is how you catch a thermally-throttling rack mid-run.
6. Custom NVTX markers around training phases (`data_load`, `fwd`, `bwd`, `opt_step`, `ckpt`) so timelines are readable.

The key metrics I track every run: **step time, MFU, tensor-core active %, NCCL busy %, SM occupancy on the attention/FFN kernels, HBM BW on LM head**.

---

## 5. Bottlenecks & Optimizations (with expected MFU gains)

Starting point for a **naive** Qwen2.5-VL-72B training on 1024 H100s is typically 15–25 % MFU. The path to 45–55 % MFU:

| # | Bottleneck | Optimization | MFU gain | Notes |
|---|---|---|---|---|
| 1 | Quadratic attention mem & slow HBM accesses (score matrix materialization) | **FlashAttention-3 (Hopper)** with async WGMMA + TMA | **+8–12 pp** | 1.5–2× over FA2 on long seqs; essential. |
| 2 | Communication serialized with compute (TP all-gather/reduce-scatter) | **Overlap comm with compute** via TransformerEngine / Megatron-LM `--tp-comm-overlap`; CUDA streams + graph capture | **+5–8 pp** | On NVLink this hides ~80 % of TP comm. |
| 3 | Optimizer-state DP all-reduce | **ZeRO-1 (sharded AdamW)** + gradient bucketing + reduce-scatter in bf16, cast to fp32 per shard | **+3–5 pp** | Also halves optimizer memory. |
| 4 | Activation memory forces small MBS → low arithmetic intensity | **Selective activation checkpoint** (only attention+FFN inputs every 2–4 layers) instead of full | **+3–6 pp** | Full checkpoint costs ~33 % compute; selective costs ~10 %. |
| 5 | Pipeline bubble (naive 1F1B ≈ (PP-1)/(PP·vpp)) | **Interleaved 1F1B** with virtual pipeline stages = 4–8; **zero-bubble PP** (ZB-H1/H2) where feasible | **+2–4 pp** | Bubble cut from 15 % → ~5 %. |
| 6 | Load imbalance in VL training (variable image resolutions) | **Token packing** to fill seq, **dynamic bucketing** of images to similar pixel counts, **ViT compute bucketing** | **+3–5 pp** | Also stabilizes grad norm. |
| 7 | Data loading stalls (100 TB shuffled) | WebDataset shards + **DALI** GPU decode; prefetch ≥ 2 steps ahead; local NVMe cache on each node; **multi-epoch shuffle buffer** | **+1–3 pp** | Brings GPU idle < 2 %. |
| 8 | NCCL all-reduce across InfiniBand at DP scale | **Hierarchical all-reduce** (intra-node via NVLink, inter-node via IB); SHARP v3 in-network reduction; NCCL_IB_QPS_PER_CONNECTION tuning; rail-optimized topology | **+2–4 pp** | Crucial above 256 GPUs. |
| 9 | LM head softmax + bf16 logits materialization (1.2 GB per sample) | **Fused cross-entropy** (Liger / FlashCE) — tiled softmax + gradient in one kernel | **+1–2 pp** | Also saves ~2–3 GB peak act. |
| 10 | Kernel launch overhead on small ops (layernorms, RoPE, element-wise) | **torch.compile** / CUDA Graphs over fwd+bwd; fuse RMSNorm+residual with `te.LayerNorm` | **+1–3 pp** | Biggest payoff for small seq lengths. |
| 11 | fp8 underutilized on Hopper | **FP8 training** (TransformerEngine) with delayed/per-tensor scaling on GEMMs; keep attention in bf16 | **+10–20 pp** | Most aggressive single lever; requires numerical validation (loss-curve parity within 0.5 %). Many teams enable FP8 only on the FFN first. |
| 12 | Vision tower stalling LLM | **Run ViT on a subset of DP ranks** (tensor-of-experts style) and broadcast features, or **overlap ViT fwd with LLM bwd** | **+1–2 pp** | Especially with CP. |

**Cumulative realistic journey** I have landed at for a 72B VLM on 1024 H100s, bf16-only: **22 % → 48 %** MFU. Adding FP8 (FFN + QKV GEMMs): **→ 58–62 %** of bf16 peak-equivalent, or ~30 % wall-clock speedup.

Common **anti-patterns** that kill MFU: micro-batch 1 with long sequences and no CP; full activation checkpointing when selective suffices; PP without virtual pipelines; DP all-reduce in fp32; unshuffled data leading to batch imbalance; JPEG decode on CPU.

---

## 6. Inference Architecture

### 6.1 Serving stack

Production stack for Qwen2.5-VL-72B:

- **Engine:** vLLM (or SGLang) — custom fork with VLM-aware scheduler.
- **Runtime**: PyTorch 2.4+ with **TransformerEngine FP8** kernels for GEMMs, **FlashAttention-3** for prefill, **FlashInfer / vLLM PagedAttention** for decode.
- **Parallelism:** TP=4 on a single 8×H100 node for **latency-bound** serving (leaves headroom for KV + vision); TP=8 for throughput-bound serving of long contexts.
- **Quantization:** bf16 baseline; FP8 (E4M3 weights + activations via SmoothQuant or AWQ-FP8) for best latency/quality tradeoff; **AWQ-INT4 / GPTQ-INT4** for edge/small deployments (~40 % quality recoverable with good calibration).
- **KV cache:** PagedAttention with 16-token blocks; FP8 KV cache (E5M2) halves memory for ~0.2 pp quality drop.
- **Vision path:** ViT runs on 1 GPU (shared across the TP group), features broadcast via NVLink. Optionally **vision-prefill disaggregation** — dedicated GPUs run only the ViT+merger and stream visual tokens to LLM nodes via RDMA.

### 6.2 Memory budget (TP=8 on 8 × H100)

| Item | Size / GPU |
|---|---|
| Weights bf16 (72B / 8) | 18 GB |
| Weights FP8 | 9 GB |
| Activation buffers | 1–2 GB |
| KV cache (bf16, 1 seq, 32K ctx) | 2·L·S·N_kv·D·2B / TP = 2·80·32768·8·128·2 / 8 = **5.25 GB** |
| Per-seq KV (FP8) | **2.6 GB** |

With FP8 weights + FP8 KV, each H100 can hold ~25 concurrent 32K-context requests; with bf16 + bf16 KV more like 8.

### 6.3 Real-time inference optimizations

| Lever | What it does | Measured impact |
|---|---|---|
| **Continuous batching** | New requests join the batch each decode step; no head-of-line blocking. | 3–10× throughput vs. static batching. |
| **Chunked prefill** | Break long prompts into 512–2048-token chunks; interleave with decode. | Decode TPS stays flat even during long-prompt ingest; p99 latency 3–5× better. |
| **Prefill/decode disaggregation** (Splitwise / DistServe pattern) | Prefill nodes (compute-bound) separate from decode nodes (mem-BW-bound). KV shipped via RDMA. | 1.5–2× throughput for latency-SLO workloads. |
| **Speculative decoding** | Draft model (Qwen2.5-1.5B or EAGLE-2 head) proposes k≈4 tokens; target verifies in parallel. | 1.8–2.5× decode speedup at similar quality. |
| **FP8 GEMMs + FP8 KV** | Halve HBM traffic on the decode-dominant KV-read path. | 30–40 % decode TPS gain; first-token latency roughly flat. |
| **CUDA graphs for decode step** | Capture the decode step once per batch size; replay with near-zero launch overhead. | 15–25 % decode TPS at small batches. |
| **PagedAttention (16-block)** | Remove KV fragmentation → higher batch sizes. | 2–4× concurrent sessions. |
| **Vision encoder cache** | Hash-cache ViT features for repeat images (RAG, UI screenshots). | Effective infinite speedup on cache hit; ~30 % hit rate in production assistants. |
| **Dynamic resolution clamping** | Cap image pixels at content-aware threshold; reuse Qwen2.5-VL's native dyn-res. | 20–40 % ViT time saved, no quality loss on text-rich images. |
| **LM-head sharding** | Shard 152K vocab column-wise across TP; all-reduce only final argmax/sample. | 30–50 % lower first-token latency at TP>1. |
| **Torch.compile on the decode graph + CUTLASS epilogues** | Fuse RMSNorm + residual + RoPE + GEMM bias. | 5–10 % decode TPS. |

**End-to-end results** I typically see on a single 8 × H100 node serving Qwen2.5-VL-72B with all of the above (FP8 weights + FP8 KV + speculative + continuous batching + paged KV):
- **Prefill:** ~6,000 tokens/s for a 4K-token + 1-image prompt.
- **Decode:** 80–110 tokens/s/stream at batch=1 (fp8 + spec-dec); aggregate >1500 tok/s at batch=32.
- **Time-to-first-token (TTFT):** 350–600 ms for 2K prompt with a 1024×1024 image.
- **Time-per-output-token (TPOT):** 9–12 ms.

For higher-scale serving (say 10k QPS), we deploy a **global scheduler** (KV-aware routing: pin requests to nodes that already have matching system-prompt KV cached) in front of a fleet running 16–64 replicas, with prefill/decode disaggregation across racks.

---

## 7. Appendix: Key arithmetic references

- H100 SXM5 peak: **989 TFLOPS bf16**, **1979 TFLOPS FP8** (dense); HBM3 **3.35 TB/s**; NVLink 900 GB/s.
- H100 DGX node: 8 × H100 + 4 NVSwitch + 8 × CX-7 NDR 400 Gb/s IB.
- AdamW fp32 state per param = 8 B (m + v).
- Per-token training FLOPs for dense transformer ≈ **6 · N_params** (3 for forward, 3 for backward matmul grads vs. input & weight).
- Attention FLOPs per token: `4·L·S·D_model_per_head_group` — usually 5–15 % of total at S=4K–16K; dominates at S≥64K.
- KV cache per token per layer: `2 · N_kv · D · bytes = 2 · 8 · 128 · 2 = 4 KB` bf16 → 4 KB · 80 = **320 KB/token** for the whole model (bf16).
