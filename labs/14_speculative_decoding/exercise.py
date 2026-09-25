# Exercise stub generated from solution.py by tools/make_exercises.py.
# Replace every NotImplementedError, then run: pytest labs/14_speculative_decoding
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
