# Exercise stub generated from solution.py by tools/make_exercises.py.
# Replace every NotImplementedError, then run: pytest labs/15_eval_stats
"""Lab 15 -- evaluation statistics: CIs, paired tests, clustered SEs, pass@k, power, judges, Bradley-Terry.

Handout: labs/15_eval_stats/README.md
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


def mean_ci(scores, conf: float = 0.95) -> tuple[float, float, float]:
    """(mean, low, high) by the CLT with the sample std (ddof=1)."""
    raise NotImplementedError("15_eval_stats: implement mean_ci")


def bootstrap_ci(scores, n_boot: int = 10000, conf: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    raise NotImplementedError("15_eval_stats: implement bootstrap_ci")


def paired_permutation_test(a, b, n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sided p-value for mean(a - b) == 0 by random sign flips of the per-item differences."""
    raise NotImplementedError("15_eval_stats: implement paired_permutation_test")


def mcnemar_exact(a_correct, b_correct) -> float:
    """Exact two-sided McNemar p-value from the discordant pairs."""
    raise NotImplementedError("15_eval_stats: implement mcnemar_exact")


def clustered_se(scores, clusters) -> float:
    """Cluster-robust standard error of the mean: sqrt(sum_c (sum_{i in c} (x_i - mean))^2) / n."""
    raise NotImplementedError("15_eval_stats: implement clustered_se")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct: 1 - C(n-c, k) / C(n, k), computed stably."""
    raise NotImplementedError("15_eval_stats: implement pass_at_k")


def required_n(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Items per model to detect accuracy p1 vs p2 with an unpaired two-proportion z-test."""
    raise NotImplementedError("15_eval_stats: implement required_n")


def holm_bonferroni(pvals, alpha: float = 0.05) -> list[bool]:
    """Reject flags controlling the family-wise error rate (step-down Holm)."""
    raise NotImplementedError("15_eval_stats: implement holm_bonferroni")


def cohens_kappa(a, b) -> float:
    """Agreement beyond chance between two raters (e.g. an LLM judge vs a human)."""
    raise NotImplementedError("15_eval_stats: implement cohens_kappa")


def bradley_terry(wins: np.ndarray, iters: int = 500) -> np.ndarray:
    """Strengths s (sum 1) from a win matrix wins[i, j] = times i beat j, via the MM algorithm:
    s_i <- W_i / sum_j n_ij / (s_i + s_j)."""
    raise NotImplementedError("15_eval_stats: implement bradley_terry")
