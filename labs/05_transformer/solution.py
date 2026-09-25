"""Lab 05 -- a modern decoder-only transformer from scratch.

RMSNorm, rotary position embeddings (RoPE), grouped-query attention (GQA),
SwiGLU MLP, optional QK-norm, pre-norm residual blocks, tied embeddings and
GPT-2-style depth-scaled init. You may use nn.Linear / nn.Embedding and
F.softmax, but not nn.MultiheadAttention or F.scaled_dot_product_attention.

Handout: labs/05_transformer/README.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 256  # maximum context length
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 4  # < n_head => grouped-query attention; 1 => multi-query
    d_model: int = 128
    d_ff: int = 0  # 0 => SwiGLU default: ~8/3 * d_model rounded up to a multiple of 64
    rope_theta: float = 10000.0
    qk_norm: bool = False
    tie_embeddings: bool = True

    def __post_init__(self):
        assert self.d_model % self.n_head == 0 and self.n_head % self.n_kv_head == 0
        if self.d_ff == 0:
            self.d_ff = 64 * math.ceil(8 * self.d_model / 3 / 64)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# -------------------------------------------------------------------- RoPE
def rope_cache(head_dim: int, max_seq: int, theta: float = 10000.0) -> tuple[Tensor, Tensor]:
    """cos and sin tables of shape (max_seq, head_dim // 2): angle[m, i] = m * theta^(-2i/head_dim)."""
    # BEGIN SOLUTION
    inv_freq = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    angles = torch.outer(torch.arange(max_seq, dtype=torch.float32), inv_freq)
    return angles.cos(), angles.sin()
    # END SOLUTION


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate each pair (x[..., 2i], x[..., 2i+1]) of x (B, H, T, D) by angle[t, i].

    ``cos``/``sin`` have shape (T, D/2) for the positions of x. (This is the interleaved
    convention of the RoPE paper and Meta's reference code; Hugging Face rotates halves
    (i, i + D/2) instead. Porting weights between conventions needs a permutation.)
    """
    # BEGIN SOLUTION
    # HINT: x1 = x[..., 0::2], x2 = x[..., 1::2]; (x1, x2) -> (x1 cos - x2 sin, x1 sin + x2 cos)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2).type_as(x)
    # END SOLUTION


# --------------------------------------------------------------- attention
def causal_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """softmax(q k^T / sqrt(d) + causal mask) v for q, k, v of shape (B, H, T, D)."""
    # BEGIN SOLUTION
    T = q.shape[-2]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    mask = torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1)
    scores = scores.masked_fill(mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ v
    # END SOLUTION


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """(B, H_kv, T, D) -> (B, H_kv * n_rep, T, D): each KV head serves n_rep consecutive query heads."""
    # BEGIN SOLUTION
    return x if n_rep == 1 else x.repeat_interleave(n_rep, dim=1)
    # END SOLUTION


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        hd = cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_head * hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_head * hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_head * hd, bias=False)
        self.wo = nn.Linear(cfg.n_head * hd, cfg.d_model, bias=False)
        if cfg.qk_norm:
            self.q_norm = RMSNorm(hd)
            self.k_norm = RMSNorm(hd)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        cfg = self.cfg
        B, T, _ = x.shape
        # BEGIN SOLUTION
        # HINT: project, view as (B, T, heads, hd), transpose to (B, heads, T, hd),
        # HINT: [qk-norm], RoPE on q and k, repeat k/v for GQA, attend, merge heads, project out.
        q = self.wq(x).view(B, T, cfg.n_head, cfg.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, cfg.n_kv_head, cfg.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, cfg.n_kv_head, cfg.head_dim).transpose(1, 2)
        if cfg.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        n_rep = cfg.n_head // cfg.n_kv_head
        y = causal_attention(q, repeat_kv(k, n_rep), repeat_kv(v, n_rep))
        return self.wo(y.transpose(1, 2).reshape(B, T, cfg.d_model))
        # END SOLUTION


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN SOLUTION
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        # END SOLUTION


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        # BEGIN SOLUTION
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))
        # END SOLUTION


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        cos, sin = rope_cache(cfg.head_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        self._init_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_residual_projections(self) -> None:
        """Scale the two projections that write into the residual stream by 1/sqrt(2 * n_layer)."""
        # BEGIN SOLUTION
        std = 0.02 / math.sqrt(2 * self.cfg.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.wo.weight, mean=0.0, std=std)
            nn.init.normal_(block.mlp.w_down.weight, mean=0.0, std=std)
        # END SOLUTION

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied weights are counted once
        return n - self.tok_emb.weight.numel() if non_embedding else n

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """idx (B, T) token ids -> logits (B, T, V) and, if targets are given, mean cross-entropy."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        # BEGIN SOLUTION
        x = self.tok_emb(idx)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
        return logits, loss
        # END SOLUTION

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None) -> Tensor:
        """Autoregressive sampling without a KV cache (lab 07 adds one)."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size :])
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
