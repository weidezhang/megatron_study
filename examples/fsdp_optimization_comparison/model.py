#!/usr/bin/env python3
"""
Shared GPT-style transformer model for FSDP optimization comparison.

Model: ~1.3B parameters (GPT-2 XL scale)
  - hidden_size=2048, num_layers=24, num_heads=16, vocab_size=50257
  - Sized to be meaningful on 4x RTX 5090 (32GB each)

This module provides:
  - GPTBlock: single transformer layer
  - GPTModel: full model with embedding + transformer layers + LM head
  - Synthetic data generation for benchmarking
  - Memory/throughput measurement utilities
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List


@dataclass
class GPTConfig:
    """Model configuration with sensible defaults for ~1.3B params."""
    vocab_size: int = 50257
    max_seq_len: int = 1024
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    intermediate_size: int = 8192  # 4x hidden
    dropout: float = 0.0
    bias: bool = False


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.hidden_size % config.num_heads == 0
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Use PyTorch's scaled_dot_product_attention (FlashAttention-2 backend when available)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.attn_dropout.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))


class MLP(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class GPTBlock(nn.Module):
    """Single transformer block: RMSNorm -> Attention -> RMSNorm -> MLP."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTModel(nn.Module):
    """
    GPT-style language model.

    Architecture:
      - Token + Position embeddings
      - N transformer blocks
      - RMSNorm + LM head (tied weights)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.num_layers)])
        self.ln_f = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        # Init weights
        self.apply(self._init_weights)
        # Scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('down_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.size()
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)

        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_count_str(self) -> str:
        n = self.param_count()
        if n >= 1e9:
            return f"{n/1e9:.2f}B"
        return f"{n/1e6:.0f}M"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_synthetic_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a random batch of token IDs and shifted labels."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return input_ids, labels


def get_gpu_memory_gb(device: Optional[int] = None) -> Dict[str, float]:
    """Return current GPU memory stats in GB."""
    if device is None:
        device = torch.cuda.current_device()
    return {
        "allocated": torch.cuda.memory_allocated(device) / 1e9,
        "reserved": torch.cuda.memory_reserved(device) / 1e9,
        "max_allocated": torch.cuda.max_memory_allocated(device) / 1e9,
        "max_reserved": torch.cuda.max_memory_reserved(device) / 1e9,
    }


def print_memory_summary(label: str, device: Optional[int] = None):
    """Print a one-line memory summary."""
    m = get_gpu_memory_gb(device)
    print(f"  [{label}] Alloc: {m['allocated']:.2f} GB | "
          f"Reserved: {m['reserved']:.2f} GB | "
          f"Peak Alloc: {m['max_allocated']:.2f} GB | "
          f"Peak Reserved: {m['max_reserved']:.2f} GB")


class ThroughputTimer:
    """Measure tokens/second and MFU."""

    def __init__(self, model_params: int, num_layers: int, hidden_size: int):
        self.model_params = model_params
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.start_time = None
        self.total_tokens = 0
        self.step_times: List[float] = []

    def start(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def stop(self, tokens_in_step: int):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start_time
        self.step_times.append(elapsed)
        self.total_tokens += tokens_in_step

    def tokens_per_second(self) -> float:
        if not self.step_times:
            return 0.0
        total_time = sum(self.step_times)
        return self.total_tokens / total_time

    def avg_step_time_ms(self) -> float:
        if not self.step_times:
            return 0.0
        return (sum(self.step_times) / len(self.step_times)) * 1000

    def estimate_flops_per_token(self) -> float:
        """Rough estimate: 6 * N (params) for fwd+bwd per token."""
        return 6 * self.model_params

    def estimate_tflops(self, num_gpus: int = 1) -> float:
        """Estimate achieved TFLOPS across all GPUs."""
        tps = self.tokens_per_second()
        flops_per_token = self.estimate_flops_per_token()
        total_tflops = (tps * flops_per_token) / 1e12
        return total_tflops / num_gpus  # per-GPU TFLOPS

    def summary(self, num_gpus: int = 1) -> str:
        tps = self.tokens_per_second()
        tflops = self.estimate_tflops(num_gpus)
        avg_ms = self.avg_step_time_ms()
        # Skip first step (warmup) for cleaner avg
        if len(self.step_times) > 1:
            steady_times = self.step_times[1:]
            steady_avg_ms = (sum(steady_times) / len(steady_times)) * 1000
        else:
            steady_avg_ms = avg_ms
        return (
            f"  Tokens/sec: {tps:,.0f} | "
            f"Avg step: {avg_ms:.1f} ms | "
            f"Steady-state step: {steady_avg_ms:.1f} ms | "
            f"Est. TFLOPS/GPU: {tflops:.1f}"
        )
