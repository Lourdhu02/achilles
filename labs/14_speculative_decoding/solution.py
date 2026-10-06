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
    # BEGIN SOLUTION
    return Cache(cache[:n_keep])
    # END SOLUTION

def verify_block(target: CausalModel, draft: CausalModel, x: torch.Tensor,
                 cache_t: Cache, cache_d: Cache, k: int, rng: torch.Generator) -> tuple[torch.Tensor, Cache, Cache]:
    """Draft k tokens autogressively from draft model, then verify in one pass with target model.
    Returns (accepted_tokens, new_cache_t, new_cache_d).
    x contains the new tokens that have not been processed by the models yet."""
    # BEGIN SOLUTION
    drafts = []
    qs = []
    curr = x
    for _ in range(k):
        logits = draft(curr, cache_d)[-1]
        probs = torch.softmax(logits, -1)
        curr = torch.multinomial(probs, 1, generator=rng)
        drafts.append(curr)
        qs.append(probs)
    
    if k > 0:
        draft(drafts[-1], cache_d)
    else:
        draft(x, cache_d)
        
    draft_tokens = torch.cat(drafts) if k > 0 else torch.empty(0, dtype=torch.long, device=x.device)
    p_logits = target(torch.cat([x, draft_tokens]), cache_t)
    p_probs = torch.softmax(p_logits, -1)
    
    start = len(x) - 1
    out = []
    accepted = 0
    for i in range(k):
        draft_tok = draft_tokens[i].item()
        q_prob = qs[i][draft_tok].item()
        p_prob = p_probs[start + i, draft_tok].item()
        if torch.rand(1, generator=rng).item() < min(1.0, p_prob / q_prob):
            out.append(draft_tok)
            accepted += 1
        else:
            resid = torch.clamp(p_probs[start + i] - qs[i], min=0.0)
            resid /= resid.sum()
            out.append(torch.multinomial(resid, 1, generator=rng).item())
            break
    else:
        out.append(torch.multinomial(p_probs[-1], 1, generator=rng).item())
    
    cache_t = rollback(cache_t, len(cache_t) - k + accepted)
    cache_d = rollback(cache_d, len(cache_d) - k + accepted)
    
    return torch.tensor(out, device=x.device), cache_t, cache_d
    # END SOLUTION


def speculative_step(p_fn: Model, q_fn: Model, prefix: tuple, k: int, rng: np.random.Generator) -> list[int]:
    """One round: draft k tokens from q, verify with p. Returns the accepted tokens (1..k+1 of them).

    Accept draft x_i with prob min(1, p(x_i)/q(x_i)). On the first rejection, sample from
    normalize(max(0, p - q)) and stop. If all k are accepted, sample one bonus token from p.
    The output is distributed exactly as sampling from p.
    """
    # BEGIN SOLUTION
    drafts, qs = [], []
    for _ in range(k):
        q = q_fn(prefix + tuple(drafts))
        drafts.append(int(rng.choice(len(q), p=q)))
        qs.append(q)
    out: list[int] = []
    for i, x in enumerate(drafts):
        p = p_fn(prefix + tuple(drafts[:i]))
        if rng.random() < min(1.0, p[x] / qs[i][x]):
            out.append(x)
            continue
        resid = np.maximum(p - qs[i], 0.0)
        out.append(int(rng.choice(len(p), p=resid / resid.sum())))
        return out
    p = p_fn(prefix + tuple(drafts))
    out.append(int(rng.choice(len(p), p=p)))
    return out
    # END SOLUTION


def acceptance_rate(p: np.ndarray, q: np.ndarray) -> float:
    """alpha = sum_x min(p(x), q(x)) = 1 - TV(p, q)."""
    # BEGIN SOLUTION
    return float(np.minimum(p, q).sum())
    # END SOLUTION


def expected_tokens(alpha: float, k: int) -> float:
    """Expected tokens produced per target forward pass: (1 - alpha^(k+1)) / (1 - alpha)."""
    # BEGIN SOLUTION
    return float(k + 1) if alpha == 1 else (1 - alpha ** (k + 1)) / (1 - alpha)
    # END SOLUTION


def expected_speedup(alpha: float, k: int, cost_ratio: float) -> float:
    """Speedup over plain decoding when a draft call costs `cost_ratio` of a target call."""
    # BEGIN SOLUTION
    return expected_tokens(alpha, k) / (k * cost_ratio + 1)
    # END SOLUTION
