# Qwen2.5-VL-32B: Fine-Tuning Analysis, 32B vs 72B Comparison, and Large-Scale VLM Fine-Tuning Strategy

> Written from the perspective of an ML systems engineer running large-scale Qwen VLM fine-tuning on H100 clusters. All memory/FLOP numbers use bf16 weights+activations with fp32 AdamW unless stated otherwise.

---

## 1. Fundamental Differences: Qwen2.5-VL-32B vs 72B

### 1.1 Architecture Side-by-Side

| Attribute | Qwen2.5-VL-**32B** | Qwen2.5-VL-**72B** | Ratio (72B/32B) |
|---|---|---|---|
| **LLM Layers** | 64 | 80 | 1.25× |
| **Hidden dim (H)** | 5120 | 8192 | 1.6× |
| **FFN intermediate** | 27,648 | 29,568 | 1.07× |
| **Q heads** | 40 | 64 | 1.6× |
| **KV heads** | 8 | 8 | 1.0× |
| **GQA ratio** | 5:1 | 8:1 | — |
| **Head dim (D)** | 128 | 128 | 1.0× |
| **Vocab (V)** | 152,064 | 152,064 | 1.0× |
| **Params per layer** | ~488 M | ~878 M | 1.8× |
| **LLM backbone params** | ~31.2 B | ~70.2 B | 2.25× |
| **Total params** | ~33.4 B | ~72.0 B | 2.16× |
| **FLOPs per layer (S=4K)** | ~4.17 TF | ~7.46 TF | 1.79× |
| **Total fwd FLOPs (S=4K)** | ~267 TF + 6.4 TF (LM head) | ~597 TF + 10.2 TF | 2.22× |
| **Context length** | 32K (128K w/ YaRN) | 32K (128K w/ YaRN) | Same |

### 1.2 Vision Encoder — Identical

Both models share the **exact same** ViT encoder and patch merger:
- ViT: 32 layers, 1280 hidden, 16 heads, native dynamic resolution, 2D-RoPE, windowed + full attention hybrid.
- Merger: 2×2 spatial grouping → 4× token compression, MLP projecting to LLM hidden.
- Only the final projection layer differs: merger output is 5120 (32B) vs 8192 (72B).
- Vision tower parameters: ~675M (ViT) + ~33–85M (merger depending on output dim) — small vs LLM.

### 1.3 Key Structural Differences

**1. Width vs Depth trade-off:**
The 72B is primarily **wider** (8192 vs 5120 hidden → 2.56× more parameters per attention/FFN matmul), while being only **25% deeper** (80 vs 64 layers). This matters for:
- **Parallelism**: wider → better TP utilization (larger matmuls, higher arithmetic intensity); deeper → more natural PP stages.
- **Memory**: wider model has proportionally more KV cache per token (320 KB/tok for 72B vs 256 KB/tok for 32B in bf16 — difference comes from 80 vs 64 layers, since both have 8 KV heads × 128 dim).

**2. GQA ratio:**
32B uses 5:1 (40Q:8KV) vs 72B's 8:1 (64Q:8KV). Both have 8 KV heads (same KV cache size per layer), but the 32B devotes proportionally more compute to key/value projection relative to its total width. In practice: 32B's attention is slightly less "compressed" — may benefit quality on multi-hop reasoning.

**3. FFN width nearly equal:**
27,648 vs 29,568 — only 7% wider in the 72B. The 72B's extra capacity comes overwhelmingly from the hidden dim (which affects attention AND FFN input/output) rather than from a wider FFN.

**4. Memory sweet spot:**
32B in bf16 = ~66 GB (fits on 1×H100 for inference). 72B in bf16 = ~144 GB (needs ≥2 H100s). This means the 32B is the **largest Qwen VL you can serve at TP=1 on a single H100** — huge practical advantage for deployment.

**5. FLOPs-per-token:**
32B ≈ 192B FLOPs/token (6·32e9); 72B ≈ 432B FLOPs/token (6·72e9). The 32B is **2.25× cheaper per token** in both training and inference. For fine-tuning at 50–100 TB data scale, this translates directly to 2.25× less GPU-hours.

**6. Quality gap:**
Based on public benchmarks (MathVista, DocVQA, ChartQA, RealWorldQA, etc.), the 72B leads the 32B by **2–5 percentage points** on most vision-language benchmarks. The gap narrows significantly after domain-specific fine-tuning (often <1 pp on in-domain tasks), making the 32B the better cost/quality trade-off for specialized deployments.

---

## 2. Qwen2.5-VL-32B: Layer-by-Layer Forward and Backward

### 2.1 Architecture Constants

```
H = 5120          # hidden dim
H_ff = 27648      # FFN intermediate (SwiGLU)
N_q = 40          # query heads
N_kv = 8          # KV heads (GQA 5:1)
D = 128           # head dim
L = 64            # transformer layers
V = 152064        # vocabulary
```

Working example: `B=1`, `S=4096`, bf16 (2 bytes/element).

### 2.2 Input Embedding

| Step | Operation | Input → Output tensor | FLOPs | Activation (bf16) |
|---|---|---|---|---|
| Embed | Gather rows from `[V, H]` | token_ids `[1,4096]` → `[1,4096,5120]` | ~0 (lookup) | **40 MB** |

Backward: sparse index-scatter-add into embedding gradient; negligible FLOPs.

### 2.3 One Transformer Layer

Input: `X: [1, 4096, 5120]` — **40 MB**.

#### (a) Pre-Attention RMSNorm

| | Input | Output | FLOPs | Activation |
|---|---|---|---|---|
| RMSNorm | `[1,4096,5120]` | `[1,4096,5120]` | ~21 MF | 40 MB (save reciprocal-rms: 32 KB) |

#### (b) Q / K / V Projections (GQA)

| Projection | Matmul shape | Output tensor | FLOPs | Activation (bf16) |
|---|---|---|---|---|
| **Q = X·W_q** | `[1,4096,5120]·[5120,5120]` | `[1,4096,5120]` | **0.215 TF** | 40 MB |
| **K = X·W_k** | `[1,4096,5120]·[5120,1024]` | `[1,4096,1024]` | **0.043 TF** | 8 MB |
| **V = X·W_v** | `[1,4096,5120]·[5120,1024]` | `[1,4096,1024]` | **0.043 TF** | 8 MB |

#### (c) RoPE (M-RoPE for VL variant)
Pointwise cos/sin rotation on Q and K — ~negligible FLOPs and memory.

#### (d) Scaled Dot-Product Attention (FlashAttention-2/3)

| Step | Details | FLOPs | Memory |
|---|---|---|---|
| Reshape | Q→`[1,40,4096,128]`, K,V→`[1,8,4096,128]` | 0 | 0 (view) |
| QKᵀ (with GQA broadcast) | `[1,40,4096,128]·[1,40,4096,128]ᵀ` | ~0.172 TF | **Without FA: 1.25 GB** (score matrix `[1,40,4096,4096]`) |
| softmax + PV | `[1,40,4096,4096]·[1,40,4096,128]` | ~0.172 TF | **With FA: ~0** (tiled, only log-sum-exp saved: 640 KB) |
| Output | `[1,4096,5120]` | — | 40 MB |
| **Attention total** | | **~0.172 TF** | **40 MB** (with FA) |

#### (e) O-Projection + Residual

| | Matmul | Output | FLOPs | Activation |
|---|---|---|---|---|
| O = Attn·W_o | `[1,4096,5120]·[5120,5120]` | `[1,4096,5120]` | **0.215 TF** | 40 MB |
| Residual add | X + O | `[1,4096,5120]` | ~21 MF | 0 (in-place) |

#### (f) Post-Attention RMSNorm — same as (a), ~21 MF.

#### (g) SwiGLU FFN

| Step | Matmul shape | Output tensor | FLOPs | Activation (bf16) |
|---|---|---|---|---|
| **Gate = X·W_gate** | `[1,4096,5120]·[5120,27648]` | `[1,4096,27648]` | **1.160 TF** | **216 MB** |
| **Up = X·W_up** | `[1,4096,5120]·[5120,27648]` | `[1,4096,27648]` | **1.160 TF** | **216 MB** |
| SiLU(Gate)⊙Up | Elementwise | `[1,4096,27648]` | ~0.23 GF | 216 MB (can recompute) |
| **Down = Act·W_down** | `[1,4096,27648]·[27648,5120]` | `[1,4096,5120]` | **1.160 TF** | 40 MB |
| Residual add | | `[1,4096,5120]` | ~21 MF | 0 |

### 2.4 Per-Layer Summary (B=1, S=4096)

| Category | FLOPs | Activation (bf16, no ckpt) |
|---|---|---|
| RMSNorm ×2 | ~42 MF | ~0 extra |
| Q/K/V projections | 0.301 TF | 96 MB |
| FlashAttention | 0.172 TF | 40 MB (output only) |
| O-projection | 0.215 TF | 40 MB |
| SwiGLU FFN | 3.480 TF | 688 MB |
| **Layer total** | **~4.17 TF fwd** | **~860 MB** |

With activation checkpointing (per-layer): save only input `X` (40 MB); recompute rest during backward.

### 2.5 Full Model Forward (B=1, S=4096)

| Component | FLOPs | Activation (no ckpt) | Activation (full ckpt) |
|---|---|---|---|
| Vision (448² image) | ~0.18 TF | ~50 MB | ~10 MB |
| 64 LLM layers | 266.6 TF | 55.3 GB | 2.7 GB |
| LM head (fused CE) | 6.38 TF | ~0 (fused) | ~0 |
| **Total forward** | **~273 TF** | **~55 GB** | **~2.8 GB** |

### 2.6 Backward Pass

- Backward ≈ 2× forward FLOPs for dense matmuls.
- With full activation checkpointing: +1× forward for recompute.
- **Total per-step (fwd + bwd + recompute): ≈ 4× forward ≈ 1.09 PFLOPs** (or use the 6·N·T convention: 6·32.5e9·4096 ≈ 0.80 PF per sample).

### 2.7 Memory Budget for Fine-Tuning (Full Parameters)

| Component | Bytes/param | For 32.5B | Notes |
|---|---|---|---|
| Weights (bf16) | 2 | **65 GB** | |
| Gradients (bf16) | 2 | **65 GB** | Or fp32 (130 GB) |
| AdamW m (fp32) | 4 | **130 GB** | First moment |
| AdamW v (fp32) | 4 | **130 GB** | Second moment |
| Master weights (fp32) | 4 | **130 GB** | For mixed-precision |
| **Total states** | **16** | **~520 GB** | |
| Activations (S=4K, ckpt) | — | **~2.5–5 GB** | Per micro-batch |
| NCCL / CUDA buffers | — | **~5–10 GB** | Per GPU |

Single H100 = 80 GB → need ≥ **7 GPUs** (1 node) minimum with FSDP/ZeRO-3.

---

## 3. Distributed Fine-Tuning on H100 Clusters

### 3.1 Hardware Configurations for Different Fine-Tuning Regimes

#### Full Fine-Tuning (all parameters trainable)

| Config | GPUs | Parallelism | Per-GPU memory | Fits? |
|---|---|---|---|---|
| **1 node (8×H100)** | 8 | FSDP (ZeRO-3), TP optional | States: 65 GB + acts ~5 GB + buffers ~8 GB ≈ **78 GB** | ✅ Tight but works (S≤4K, MBS=1) |
| **2 nodes (16×H100)** | 16 | FSDP across 16 | ~42 GB/GPU | ✅ Comfortable, S≤8K |
| **4 nodes (32×H100)** | 32 | TP=8 + PP=1 + DP=4 | ~28 GB/GPU | ✅ Large batches, S≤32K |

**Sweet spot for full fine-tuning of 32B: 1–2 nodes (8–16 H100s).** This is a key advantage over 72B which needs ≥16 GPUs comfortably.

#### LoRA Fine-Tuning (r=64, targeting Q/K/V/O + gate/up/down)

| Item | Size | Notes |
|---|---|---|
| Base weights (bf16, frozen) | 65 GB | Loaded but no grads |
| LoRA A+B adapters | ~1.07 B params | ~2.1 GB bf16 |
| LoRA grads + optim | ~1.07B × 16 B/p | ~17 GB |
| Activations (S=4K, ckpt) | ~2.5 GB | |
| **Total per replica** | **~87 GB** | Tight on 1×H100; use gradient ckpt or reduce S |

**LoRA makes 32B fine-tunable on 1–2 H100 GPUs.** At r=64 (half the params) it fits comfortably on a single H100. This is the practical sweet spot for most teams.

#### QLoRA (4-bit base + LoRA)

| Item | Size |
|---|---|
| Base weights (NF4) | ~17 GB |
| LoRA adapters + optim (r=64) | ~9 GB |
| Activations | ~2.5 GB |
| **Total** | **~29 GB** |

Fits on a single A100-40GB or L40S. On H100, you can run S=16K+.

### 3.2 Parallelism Recipe for Full Fine-Tuning at Scale

For 32 × H100 (4 nodes), fine-tuning on large multimodal data:

```
TP = 4          # intra-node NVLink (could also use 8)
PP = 1          # 32B is shallow enough to skip PP
DP = 8          # 32/4 = 8-way data parallel
ZeRO stage = 1  # shard optimizer across DP (sufficient with TP=4)
SP = on          # sequence parallel within TP group
CP = 2           # for sequences > 16K
```

Why PP=1 is fine for 32B: 64 layers at 5120 hidden — each TP=4 shard holds ~16 GB weights + ~16 GB optim states per GPU with ZeRO-1. That's ~32 GB for states, leaving ~48 GB for activations, KV, and buffers. Plenty.

### 3.3 Software Stack

For fine-tuning Qwen2.5-VL specifically, the practical stack:

| Framework | Use case | Notes |
|---|---|---|
| **LLaMA-Factory** | Full / LoRA / QLoRA SFT on Qwen2.5-VL | Best out-of-box Qwen VL support; handles image preprocessing, dynamic resolution, chat templates |
| **Axolotl** | Full / LoRA SFT | Good multi-modal support, config-driven |
| **Megatron-LM + Energon** | Full fine-tuning at 100+ GPU scale | Maximum perf but requires Megatron model conversion |
| **DeepSpeed (ZeRO-2/3)** | Full fine-tuning | Works with HuggingFace Trainer; moderate effort |
| **FSDP2 (torch.distributed)** | Full fine-tuning | Native PyTorch, good for custom loops |
| **Unsloth** | QLoRA on consumer GPUs | 2× speedup via triton kernels; great for prototyping |
| **ms-swift** | Alibaba's own fine-tuning framework | Native Qwen2.5-VL support; recommended starting point |

---

## 4. Practical Fine-Tuning Methods for 50 TB – 100 TB VLM Dataset

### 4.1 Let's size this dataset

50–100 TB of compressed multimodal data (JPEG/WebP images + tokenized text):

| Metric | 50 TB | 100 TB |
|---|---|---|
| Images (avg 150 KB) | ~335 M | ~670 M |
| Visual tokens (avg 400/image post-merger) | ~134 B | ~268 B |
| Text tokens (assume 3–5% by size) | ~0.5–0.8 T | ~1.0–1.7 T |
| **Total effective tokens** | **~0.7–1.0 T** | **~1.3–2.0 T** |

**This is NOT traditional SFT.** At 0.7–2.0 T tokens, this is **continued pretraining / domain adaptation** scale — comparable to what was used for the original Qwen2.5-VL pretraining. The fine-tuning strategy must be fundamentally different from typical 10K–1M sample instruction tuning.

### 4.2 The Three-Stage Fine-Tuning Pipeline

For 50–100 TB VLM data, I recommend a **three-stage pipeline** that progressively narrows the learning:

---

#### Stage 1: Continued Pretraining (CPT) — "Domain Soak"
**Goal:** Absorb domain knowledge into the model weights.

| Setting | Value | Rationale |
|---|---|---|
| Method | **Full parameter training** | At this data scale, LoRA is insufficient — you need full-rank weight updates to absorb domain knowledge |
| Data | 80–90% of total corpus, mixed: raw domain text + image-text pairs + interleaved docs | Broad domain absorption |
| Learning rate | 1e-5 → 5e-6 (cosine decay) | 10× lower than pretraining (1e-4) to avoid catastrophic forgetting |
| Warmup | 1–2% of steps | Short warmup; model is already pretrained |
| Batch size | 4M–16M tokens/step | Standard for CPT; scale with cluster |
| Sequence length | 4K → 8K → 32K (staged) | Progressive context extension |
| Epochs | **1 epoch** (single pass) | At T-scale, 1 epoch is sufficient; >1 risks overfitting |
| Vision tower | **Frozen for first 20%, then unfreeze** | Prevents early vision degradation; ViT is already well-trained |
| Precision | bf16 (FP8 for FFN GEMMs if validated) | Standard |
| Duration | ~15–30 days on 256 H100s | See compute table below |
| Regularization | Weight decay 0.1, dropout 0 | Standard |

**Compute budget (Stage 1):**
- Tokens: 0.7–1.5 T (80% of corpus)
- FLOPs: 6 · 32.5e9 · 1e12 = 1.95e23 FLOPs (for 1T tokens)
- At 45% MFU on 256 H100s: effective 114 PFLOPS → **~20 days**

---

#### Stage 2: Supervised Fine-Tuning (SFT) — "Task Alignment"
**Goal:** Align the model to follow instructions and produce structured outputs.

| Setting | Value | Rationale |
|---|---|---|
| Method | **Full parameter** or **LoRA r=128** (if GPU-constrained) | Full is better if you can afford it after CPT |
| Data | 10–15% of corpus: curated instruction-response pairs, VQA, chart/doc understanding, grounded responses | Quality over quantity |
| Data volume | 50B–200B tokens | 5–10% of CPT; carefully curated |
| Learning rate | 5e-6 → 1e-6 (cosine) | Lower than CPT |
| Batch size | 512K–2M tokens/step | Smaller than CPT for stability |
| Epochs | 2–3 on instruction data | Instruction data benefits from repetition |
| Vision tower | **Unfrozen** (lr = 0.1× LLM lr) | Lower LR for vision to prevent drift |
| Chat template | Qwen2.5-VL chat format (`<|im_start|>...`) | Must match inference format |
| Packing | **Sequence packing** with loss masking on padding | Critical for throughput with variable-length conversations |
| Duration | 2–5 days on 64 H100s | Much smaller than CPT |

**LoRA configuration (if GPU-constrained):**
```python
lora_config = {
    "r": 128,                    # high rank for domain-rich SFT
    "alpha": 256,                # alpha = 2r
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "lora_dropout": 0.05,
    "modules_to_save": [         # full fine-tune these
        "visual",                # vision tower (if unfreezing)
        "embed_tokens",          # new domain tokens
        "lm_head"
    ]
}
# Trainable params: ~1.07B (3.2% of model) with r=128 on all linear
```

---

#### Stage 3: Preference Optimization (DPO/RLHF) — "Quality Polish"
**Goal:** Align outputs with human preferences; reduce hallucination.

| Setting | Value |
|---|---|
| Method | **DPO** (simpler than RLHF, no reward model needed) |
| Data | 1M–10M preference pairs (chosen/rejected per query) |
| Method variant | Vision-DPO with image-grounded preferences |
| Learning rate | 1e-6 → 5e-7 |
| β (DPO) | 0.1 |
| Epochs | 1–2 |
| LoRA | r=64 is sufficient (small update) |
| Duration | 1–2 days on 16 H100s |

### 4.3 Method Comparison for 50–100 TB Scale

| Method | Trainable params | GPU-hours (256 H100, 1T tokens) | Quality | When to use |
|---|---|---|---|---|
| **Full CPT** | 32.5 B (100%) | ~120K | ★★★★★ | Default for T-scale domain adaptation |
| **LoRA r=128** | ~1.07 B (3.2%) | ~30K | ★★★★ | Budget-constrained; 70–90% of full FT quality |
| **LoRA r=16** | ~0.13 B (0.4%) | ~25K | ★★★ | Quick experiments, light adaptation |
| **QLoRA r=64** | ~0.53 B (1.6%) | ~20K | ★★★ | Prototyping on few GPUs; NOT for production T-scale |
| **Freeze LLM, train vision** | ~0.7 B (2%) | ~8K | ★★ | Only when adding new visual capabilities to existing text model |
| **Full FT + DPO** | 32.5B + 0.66B | ~130K | ★★★★★+ | Best quality; the pipeline I recommend |

### 4.4 Why Full Fine-Tuning Wins at 50–100 TB

At this data scale, **LoRA is fundamentally limited** because:

1. **Rank bottleneck**: LoRA constrains weight updates to a low-rank subspace. At 1T+ tokens, the model needs to absorb enough information to justify full-rank updates. Empirically, full FT outperforms LoRA r=256 by 3–8 pp on domain benchmarks at this scale.

2. **Learning capacity**: LoRA r=128 adds ~1.3B trainable params. With 1T tokens, you have ~770 tokens per parameter — extreme data-to-parameter ratio means the adapter bottleneck wastes data.

3. **Vision tower**: LoRA on the ViT is awkward (attention patterns differ from LLM); full fine-tuning of the ViT works better for new visual domains.

4. **Catastrophic forgetting trade-off**: At small data scales, LoRA's implicit regularization (frozen base) prevents forgetting. At T-scale, you WANT the model to shift substantially — full FT with proper LR and warmup is better.

**Exception**: If you need to maintain multiple domain-specific variants of the same base model (e.g., medical + legal + finance), LoRA makes sense for storage efficiency (each adapter is ~2.5 GB vs. 65 GB per full checkpoint).

### 4.5 Critical Implementation Details for T-Scale VLM Fine-Tuning

#### Data pipeline architecture
```
Object Store (S3/Ceph/Lustre, 100TB)
    │
    ├─ WebDataset .tar shards (1GB each, ~100K shards)
    │  ├─ Shuffled at shard level (buffer = 10K shards)
    │  └─ Shuffled within shard (buffer = 1K samples)
    │
    ├─ DALI GPU JPEG decode (bypass CPU bottleneck)
    │  ├─ Dynamic resolution: resize to nearest 14×14 grid
    │  └─ Aspect ratio preserving, pad to bucket
    │
    ├─ Token packing engine
    │  ├─ Pack multiple image-text pairs to fill seq_len
    │  ├─ Loss mask on padding tokens
    │  └─ Respect conversation boundaries
    │
    └─ Multi-worker prefetch (≥2 steps ahead)
```

#### Image resolution bucketing strategy
```python
RESOLUTION_BUCKETS = [
    (224, 224),   # thumbnails
    (448, 448),   # standard photos
    (448, 896),   # tall docs
    (896, 448),   # wide panoramas
    (896, 896),   # high-res detailed
    (1344, 448),  # ultra-wide
    (448, 1344),  # ultra-tall (full-page docs)
]
# Each image snapped to nearest bucket → consistent token counts
# per bucket → even compute across DP ranks
```

#### Checkpointing strategy for long runs
- **Async checkpointing** (torch.distributed.checkpoint) to avoid stalling training.
- Checkpoint every 500–1000 steps (~2–4 hours of wall time).
- Keep last 5 + every 5000th checkpoint.
- **Validate every 1000 steps** on a held-out set (1000 samples across domains) to detect divergence early.
- Track **loss-per-domain** separately — if one domain spikes, rebalance sampling weights.

---

## 5. MFU and Optimization for Fine-Tuning 32B

### 5.1 MFU Calculation for Fine-Tuning

Same formula as pretraining:
```
tokens_per_step = global_batch_size × seq_len
model_flops_per_step = 6 × N × tokens_per_step
MFU = (model_flops_per_step / step_time / num_gpus) / 989e12
```

**Worked example**: 32B on 64 H100s, S=4096, GBS=256 sequences, step time = 4.2 s:
```
tokens/step = 256 × 4096 = 1.05M
FLOPs/step  = 6 × 32.5e9 × 1.05e6 = 2.05e17
FLOPs/s/GPU = 2.05e17 / (4.2 × 64) = 7.62e14 = 762 TFLOPS
                                        → wait, that's > peak (989)
```
Something's off — this means the step time is unrealistically low. Let me correct:
```
Realistic step time for above config: ~7.5s
FLOPs/s/GPU = 2.05e17 / (7.5 × 64) = 4.27e14 = 427 TFLOPS
MFU = 427 / 989 = 43.2%  ← realistic for fine-tuning
```

**Fine-tuning MFU is typically 3–8% LOWER than pretraining** because:
- Frequent evaluation/validation checkpoints cause pipeline stalls.
- Data loading is more complex (image decode + dynamic resolution).
- Gradient accumulation steps may have comm overhead.
- Smaller global batch sizes (common in SFT) reduce arithmetic intensity.

### 5.2 Fine-Tuning-Specific Bottlenecks & Optimizations

| # | Bottleneck | Optimization | MFU gain |
|---|---|---|---|
| 1 | **Vision preprocessing stalls** (JPEG decode, resize, normalize on CPU) | **DALI GPU decode** + torchvision.transforms.v2 on GPU | +3–5 pp |
| 2 | **Variable image sizes → uneven DP loads** | **Resolution bucketing + token packing** (pack multiple short samples to seq_len) | +3–6 pp |
| 3 | **Gradient accumulation gaps** (idle between micro-steps at small MBS) | **Increase MBS** (enabled by ZeRO-3/FSDP act ckpt); or **eliminate grad accum** by using enough DP | +2–4 pp |
| 4 | **Checkpoint I/O blocking training** | **Async distributed checkpointing** (torch.distributed.checkpoint with S3/Lustre backend) | +1–3 pp |
| 5 | **FlashAttention not using Hopper features** | **FA3** with WGMMA + TMA (vs FA2) | +3–5 pp |
| 6 | **TP comm not overlapped** | **TransformerEngine comm overlap** | +3–5 pp |
| 7 | **Full activation ckpt overhead** | **Selective ckpt** (every 2 layers) | +2–4 pp |
| 8 | **LoRA introduces non-fusible small matmuls** | **Fused LoRA kernels** (Unsloth/PEFT optimized); or full FT at scale | +2–3 pp (LoRA only) |
| 9 | **LM head logits materialization** | **Fused cross-entropy** (Liger kernel) | +1–2 pp |
| 10 | **Data loading from remote storage** | **Local NVMe cache tier** + pre-sharded WebDataset | +1–3 pp |

**Realistic MFU journey for 32B fine-tuning on 64 H100s:**
- Naive (HF Trainer + basic FSDP): **25–30%**
- + FA3 + selective ckpt + fused CE: **33–38%**
- + DALI + token packing + resolution bucketing: **38–43%**
- + TP comm overlap + async ckpt + NVMe cache: **43–48%**
- + FP8 FFN GEMMs: **50–55%**

### 5.3 Wall-Clock Estimates for Full Pipeline

For the three-stage pipeline on 50 TB data (≈0.7T tokens for CPT):

| Stage | Tokens | GPUs | MFU | Days |
|---|---|---|---|---|
| Stage 1: CPT | 700B | 256 × H100 | 45% | **~14 days** |
| Stage 2: SFT | 100B | 64 × H100 | 42% | **~8.5 days** |
| Stage 3: DPO | 2B | 16 × H100 | 38% | **~1 day** |
| **Total** | | | | **~23.5 days** |

For 100 TB data (≈1.5T tokens for CPT):

| Stage | Tokens | GPUs | MFU | Days |
|---|---|---|---|---|
| Stage 1: CPT | 1.3T | 512 × H100 | 45% | **~13 days** |
| Stage 2: SFT | 200B | 128 × H100 | 42% | **~8.5 days** |
| Stage 3: DPO | 5B | 32 × H100 | 38% | **~1 day** |
| **Total** | | | | **~22.5 days** |

---

## 6. Inference After Fine-Tuning

### 6.1 Serving the Fine-Tuned 32B

The 32B's **single-H100 inference** capability is its killer advantage:

| Config | GPUs | Use case | Throughput |
|---|---|---|---|
| TP=1, 1×H100 (bf16) | 1 | Low-latency, single-stream | 30–50 tok/s decode |
| TP=1, 1×H100 (FP8) | 1 | **Recommended production** | 50–80 tok/s decode |
| TP=1, 1×H100 (AWQ-INT4) | 1 | Max throughput, slight quality drop | 80–120 tok/s decode |
| TP=2, 2×H100 (FP8) | 2 | Low TTFT for long contexts | 80–110 tok/s, TTFT 200ms |
| TP=4, 4×H100 (FP8) | 4 | High concurrency (batch=64+) | 2000+ tok/s aggregate |

### 6.2 LoRA Serving Options

If you fine-tuned with LoRA:

1. **Merge and serve**: `model.merge_and_unload()` — zero runtime overhead; one checkpoint per domain.
2. **Multi-LoRA serving** (vLLM native): load base model once + multiple LoRA adapters hot-swappable per request. Adapter overhead: ~2.5 GB per LoRA r=128. Serve 20+ domain variants on one H100.
3. **S-LoRA** (vLLM integration): paged LoRA adapter memory, unified KV + adapter paging. Scales to 100+ concurrent adapters.

### 6.3 Key Memory Budget (32B, TP=1, FP8)

| Component | Size |
|---|---|
| Weights (FP8) | **~33 GB** |
| Activation buffers | 1–2 GB |
| KV cache per 32K-ctx session (bf16) | 2·64·32768·8·128·2 / 1 = **~8.6 GB** |
| KV cache (FP8) | **~4.3 GB** |
| Available for batching (from 80GB) | **~43 GB** |
| Max concurrent 8K sessions (FP8 KV) | **~40** |

### 6.4 Real-Time Optimizations (same principles as 72B, tuned for 32B)

All optimizations from the 72B analysis apply. The 32B-specific advantages:
- **TP=1 viable** → zero inter-GPU comm overhead for decode → 15–20% higher per-GPU MBU (model bandwidth utilization) than 72B at TP=2.
- **Smaller KV cache** → 2× more concurrent sessions → better batching → higher aggregate throughput.
- **Speculative decoding**: use Qwen2.5-VL-3B as draft model (same tokenizer/architecture family) → acceptance rate ~70% → 1.8–2.2× decode speedup.
- **Quantization headroom**: 32B at INT4 (AWQ) = ~17 GB → fits on L40S/A100-40GB → **massive deployment cost reduction**.

---

## 7. Summary Decision Matrix

| Question | Answer |
|---|---|
| When to choose 32B over 72B? | When you need single-GPU inference, 2× lower training cost, or multi-adapter serving. Quality gap is 2–5 pp pre-finetune, <1 pp after domain-specific CPT. |
| Min GPUs for full FT (32B)? | 8 × H100 (1 node) with FSDP ZeRO-3. |
| Min GPUs for LoRA FT (32B)? | **1 × H100** (r≤128) or 1 × A100-80GB. |
| Best method for 50–100 TB? | Three-stage: CPT (full FT) → SFT (full or LoRA r=128) → DPO (LoRA r=64). |
| Why not just LoRA for 50–100 TB? | Low-rank bottleneck wastes data; full FT absorbs 3–8 pp more domain knowledge at T-scale. |
| Cluster size for 100 TB? | CPT: 256–512 H100s for 12–20 days. SFT: 64–128. DPO: 16–32. |
| Target MFU (fine-tuning)? | 43–48% bf16 (50–55% with FP8). |
| Inference sweet spot? | 1×H100 FP8 (50–80 tok/s) or INT4 on L40S for cost-optimized. |

---

## 8. Appendix: 32B Arithmetic References

- Params per LLM layer: **487.7 M**
- FLOPs per layer (fwd, S=4K): **4.17 TF**
- Total fwd FLOPs (S=4K): **273 TF** (LLM+head+vision)
- Training FLOPs per token: **6 · 32.5e9 = 195 GF** (cf. 72B: 432 GF)
- KV cache per token: **2 · 8 · 128 · 2 bytes · 64 layers = 256 KB** bf16 (cf. 72B: 320 KB)
- Full model states: **~520 GB** (cf. 72B: ~1.15 TB)
- Weights bf16: **65 GB** (cf. 72B: 144 GB)
- Single-GPU inference viable: **Yes (bf16 with ~15 GB headroom)** (cf. 72B: No)
