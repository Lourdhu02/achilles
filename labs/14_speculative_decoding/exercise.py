# Exercise stub generated from solution.py by tools/make_exercises.py.
# Replace every NotImplementedError, then run: pytest labs/14_speculative_decoding
"""Lab 14 -- speculative decoding with exact rejection sampling.

Models are functions prefix(tuple[int]) -> next-token probability vector (NumPy).
Handout: labs/14_speculative_decoding/README.md
"""

from __future__ import annotations

from typing import Callable

import numpy as np

import torch
import torch.nn as nn

Model = Callable[[tuple], np.ndarray]

class Cache(list):
    pass

class CausalModel(nn.Module):
    """A minimal causal transformer layer with KV cache for exact verifier tests."""
    def __init__(self, vocab_size: int = 5, d_model: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(64, d_model)
        self.out = nn.Linear(d_model, vocab_size)
        self.w = nn.Linear(d_model, d_model * 3)

    def forward(self, x: torch.Tensor, cache: Cache) -> torch.Tensor:
        seq_len = x.size(0)
        start = len(cache)
        h = self.embed(x) + self.pos(torch.arange(start, start + seq_len, device=x.device))
        q, k, v = self.w(h).chunk(3, dim=-1)
        cache.extend(list(zip(k, v)))
        all_k = torch.stack([item[0] for item in cache])
        all_v = torch.stack([item[1] for item in cache])
        attn = q @ all_k.T / (q.size(-1) ** 0.5)
        mask = torch.ones(seq_len, len(cache), dtype=torch.bool, device=x.device)
        for i in range(seq_len):
            mask[i, start + i + 1:] = False
        attn.masked_fill_(~mask, float('-inf'))
        return self.out(torch.softmax(attn, -1) @ all_v)

def rollback(cache: Cache, n_keep: int) -> Cache:
    """Truncates the cache to the accepted prefix."""
    raise NotImplementedError("14_speculative_decoding: implement rollback")

def verify_block(target: CausalModel, draft: CausalModel, x: torch.Tensor,
                 cache_t: Cache, cache_d: Cache, k: int, rng: torch.Generator) -> tuple[torch.Tensor, Cache, Cache]:
    """Draft k tokens autogressively from draft model, then verify in one pass with target model.
    Returns (accepted_tokens, new_cache_t, new_cache_d).
    x contains the new tokens that have not been processed by the models yet."""
    raise NotImplementedError("14_speculative_decoding: implement verify_block")


def speculative_step(p_fn: Model, q_fn: Model, prefix: tuple, k: int, rng: np.random.Generator) -> list[int]:
    """One round: draft k tokens from q, verify with p. Returns the accepted tokens (1..k+1 of them).

    Accept draft x_i with prob min(1, p(x_i)/q(x_i)). On the first rejection, sample from
    normalize(max(0, p - q)) and stop. If all k are accepted, sample one bonus token from p.
    The output is distributed exactly as sampling from p.
    """
    raise NotImplementedError("14_speculative_decoding: implement speculative_step")


def acceptance_rate(p: np.ndarray, q: np.ndarray) -> float:
    """alpha = sum_x min(p(x), q(x)) = 1 - TV(p, q)."""
    raise NotImplementedError("14_speculative_decoding: implement acceptance_rate")


def expected_tokens(alpha: float, k: int) -> float:
    """Expected tokens produced per target forward pass: (1 - alpha^(k+1)) / (1 - alpha)."""
    raise NotImplementedError("14_speculative_decoding: implement expected_tokens")


def expected_speedup(alpha: float, k: int, cost_ratio: float) -> float:
    """Speedup over plain decoding when a draft call costs `cost_ratio` of a target call."""
    raise NotImplementedError("14_speculative_decoding: implement expected_speedup")
