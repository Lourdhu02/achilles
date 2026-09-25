# Exercise stub generated from solution.py by tools/make_exercises.py.
# Replace every NotImplementedError, then run: pytest labs/07_kv_cache_sampling
"""Lab 07 -- inference mechanics: a KV cache (with chunked prefill) and the sampling stack.

The model below is a compact Llama-style decoder (it mirrors lab 05) whose attention can
read from and append to a KV cache. You implement the cache, the cache-aware attention
mask, generation, and temperature / top-k / top-p / min-p sampling.

Handout: labs/07_kv_cache_sampling/README.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


@dataclass
class Config:
    vocab_size: int = 64
    max_seq: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_kv_head: int = 2
    d_model: int = 64

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


# ------------------------------------------------------------------- KV cache
class KVCache:
    """Preallocated keys/values: (n_layer, batch, n_kv_head, max_seq, head_dim) each.

    ``pos`` counts how many positions are filled. A forward pass over T new tokens writes
    positions [pos, pos + T) in every layer and then calls ``advance(T)``.
    """

    def __init__(self, cfg: Config, batch: int, dtype=torch.float32, device=None):
        shape = (cfg.n_layer, batch, cfg.n_kv_head, cfg.max_seq, cfg.head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.pos = 0

    def update(self, layer: int, k_new: Tensor, v_new: Tensor) -> tuple[Tensor, Tensor]:
        """Write k_new/v_new (B, H_kv, T, hd) at [pos, pos+T) and return all keys/values so far."""
        raise NotImplementedError("07_kv_cache_sampling: implement update")

    def advance(self, t: int) -> None:
        self.pos += t

    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes


def cache_attention_mask(t_new: int, pos: int, device=None) -> Tensor:
    """Boolean (t_new, pos + t_new) mask, True where attention is NOT allowed.

    Query i sits at absolute position pos + i and may attend to keys 0 .. pos + i.
    """
    raise NotImplementedError("07_kv_cache_sampling: implement cache_attention_mask")


# ---------------------------------------------------------------------- model
def _rope(x: Tensor, positions: Tensor, theta: float = 10000.0) -> Tensor:
    hd = x.shape[-1]
    inv_freq = theta ** (-torch.arange(0, hd, 2, dtype=torch.float32, device=x.device) / hd)
    ang = positions.float()[:, None] * inv_freq[None, :]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2).type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        hd = cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_head * hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_head * hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_head * hd, bias=False)
        self.wo = nn.Linear(cfg.n_head * hd, cfg.d_model, bias=False)

    def forward(self, x: Tensor, layer: int, pos: int, cache: KVCache | None) -> Tensor:
        cfg = self.cfg
        B, T, _ = x.shape
        positions = torch.arange(pos, pos + T, device=x.device)
        q = _rope(self.wq(x).view(B, T, cfg.n_head, cfg.head_dim).transpose(1, 2), positions)
        k = _rope(self.wk(x).view(B, T, cfg.n_kv_head, cfg.head_dim).transpose(1, 2), positions)
        v = self.wv(x).view(B, T, cfg.n_kv_head, cfg.head_dim).transpose(1, 2)
        if cache is not None:
            k, v = cache.update(layer, k, v)
        rep = cfg.n_head // cfg.n_kv_head
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(cfg.head_dim)
        scores = scores.masked_fill(cache_attention_mask(T, pos, x.device), float("-inf"))
        y = torch.softmax(scores, dim=-1) @ v
        return self.wo(y.transpose(1, 2).reshape(B, T, cfg.d_model))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n1, self.attn, self.n2 = RMSNorm(cfg.d_model), Attention(cfg), RMSNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False), nn.GELU(), nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False))

    def forward(self, x, layer, pos, cache):
        x = x + self.attn(self.n1(x), layer, pos, cache)
        return x + self.mlp(self.n2(x))


class TinyLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx: Tensor, cache: KVCache | None = None) -> Tensor:
        """Logits for the new tokens ``idx``. With a cache, idx continues the cached sequence."""
        pos = cache.pos if cache is not None else 0
        x = self.emb(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, i, pos, cache)
        if cache is not None:
            cache.advance(idx.shape[1])
        return self.head(self.norm(x))


# ------------------------------------------------------------------- sampling
def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Keep the k largest logits per row (ties at the boundary kept), set the rest to -inf."""
    raise NotImplementedError("07_kv_cache_sampling: implement top_k_filter")


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Nucleus: keep the smallest set of highest-probability tokens whose total mass is >= p.

    The token that crosses the threshold is kept; the top token is always kept.
    """
    # HINT: sort descending, cumulative-sum the probs, drop tokens whose *preceding* mass already >= p.
    raise NotImplementedError("07_kv_cache_sampling: implement top_p_filter")


def min_p_filter(logits: Tensor, min_p: float) -> Tensor:
    """Keep tokens with probability >= min_p * (probability of the most likely token)."""
    raise NotImplementedError("07_kv_cache_sampling: implement min_p_filter")


def sample_next(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one token id per row of (B, V) logits. temperature == 0 means greedy.

    Order: temperature, then top-k, top-p, min-p filters, then sample from the softmax.
    """
    raise NotImplementedError("07_kv_cache_sampling: implement sample_next")


# ----------------------------------------------------------------- generation
@torch.no_grad()
def generate(model: TinyLM, prompt: Tensor, max_new_tokens: int, use_cache: bool = True, prefill_chunk: int | None = None, **sampling) -> Tensor:
    """Autoregressive generation. With ``use_cache``, prefill the prompt (optionally in chunks of
    ``prefill_chunk`` tokens) and then decode one token per step reading the cache. Without it,
    recompute the full sequence every step (O(T^2) extra work -- the baseline to beat)."""
    raise NotImplementedError("07_kv_cache_sampling: implement generate")
