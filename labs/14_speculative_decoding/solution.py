"""Lab 14 -- speculative decoding with exact rejection sampling.

Models are functions prefix(tuple[int]) -> next-token probability vector (NumPy).
Handout: labs/14_speculative_decoding/README.md
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Model = Callable[[tuple], np.ndarray]


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
