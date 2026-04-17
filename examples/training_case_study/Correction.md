Great catch — the **"~28 GB/GPU"** in my original doc was a sloppy shorthand that only counted weights + optimizer. Here's the full picture:

### The Corrected Breakdown (TP=8, DP=4, ZeRO-1, MBS=2, S=4096)

The real total is **~52 GB/GPU** with **28 GB headroom** on an 80 GB H100:

| Component | How it's calculated | Per GPU |
|---|---|---|
| **Weights** | 32.5B params ÷ TP=8 ≈ 4.86B params (includes replicated ViT 760M), × 2 bytes bf16 | **9.7 GB** |
| **Gradients** | Same shape as weights, × 2 bytes bf16 | **9.7 GB** |
| **Optimizer** | 4.86B params × 12 bytes (master_w + m + v in fp32) ÷ DP=4 (ZeRO-1 shards across DP) | **14.6 GB** |
| **Activations** | Selective ckpt every 2 layers: 32 layers save only input (84 MB), 32 layers save everything (363 MB) — activations are TP-sharded too (FFN intermediates ÷8) | **14.3 GB** |
| **Buffers** | NCCL, CUDA context, allreduce staging, fragmentation | **3.5 GB** |
| **Total** | | **51.8 GB** |
| **Headroom** | 80 − 51.8 | **28.2 GB** |

The key mechanics behind each sharding:

**TP=8 on weights:** Every Q/K/V/O and gate/up/down matrix is sliced into 8 pieces along the head or FFN dimension. For example, `gate_proj [5120, 27648]` becomes `[5120, 3456]` per GPU. The ViT is an exception — it's too small to benefit from TP, so it's replicated (760M params on every GPU).

**ZeRO-1 on optimizer:** Each of the 4 DP ranks holds only ¼ of the AdamW states (master weights + first moment + second moment). Without ZeRO-1, the optimizer alone would be 58.3 GB/GPU — with it, just 14.6 GB.

**Why "S≤32K":** With full activation checkpointing (save only layer inputs), activations scale linearly with S. At S=32768 you hit 80.4 GB — just over the limit. At S=16384 you're at 59 GB with room to spare.