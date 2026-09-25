"""Lab 15 -- evaluation statistics: CIs, paired tests, clustered SEs, pass@k, power, judges, Bradley-Terry.

Handout: labs/15_eval_stats/README.md
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


def mean_ci(scores, conf: float = 0.95) -> tuple[float, float, float]:
    """(mean, low, high) by the CLT with the sample std (ddof=1)."""
    # BEGIN SOLUTION
    x = np.asarray(scores, float)
    z = NormalDist().inv_cdf(0.5 + conf / 2)
    se = x.std(ddof=1) / math.sqrt(len(x))
    return float(x.mean()), float(x.mean() - z * se), float(x.mean() + z * se)
    # END SOLUTION


def bootstrap_ci(scores, n_boot: int = 10000, conf: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    # BEGIN SOLUTION
    x = np.asarray(scores, float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(np.quantile(means, (1 - conf) / 2)), float(np.quantile(means, (1 + conf) / 2))
    # END SOLUTION


def paired_permutation_test(a, b, n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sided p-value for mean(a - b) == 0 by random sign flips of the per-item differences."""
    # BEGIN SOLUTION
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs((flips * d).mean(1))
    return float((np.sum(null >= obs - 1e-12) + 1) / (n_perm + 1))
    # END SOLUTION


def mcnemar_exact(a_correct, b_correct) -> float:
    """Exact two-sided McNemar p-value from the discordant pairs."""
    # BEGIN SOLUTION
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = n01 + n10
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2**n
    return min(1.0, 2 * tail)
    # END SOLUTION


def clustered_se(scores, clusters) -> float:
    """Cluster-robust standard error of the mean: sqrt(sum_c (sum_{i in c} (x_i - mean))^2) / n."""
    # BEGIN SOLUTION
    x, c = np.asarray(scores, float), np.asarray(clusters)
    resid = x - x.mean()
    return float(math.sqrt(sum(resid[c == k].sum() ** 2 for k in np.unique(c))) / len(x))
    # END SOLUTION


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct: 1 - C(n-c, k) / C(n, k), computed stably."""
    # BEGIN SOLUTION
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))
    # END SOLUTION


def required_n(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Items per model to detect accuracy p1 vs p2 with an unpaired two-proportion z-test."""
    # BEGIN SOLUTION
    za, zb = NormalDist().inv_cdf(1 - alpha / 2), NormalDist().inv_cdf(power)
    pbar = (p1 + p2) / 2
    num = za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil(num**2 / (p1 - p2) ** 2)
    # END SOLUTION


def holm_bonferroni(pvals, alpha: float = 0.05) -> list[bool]:
    """Reject flags controlling the family-wise error rate (step-down Holm)."""
    # BEGIN SOLUTION
    order = np.argsort(pvals)
    reject = [False] * len(pvals)
    for rank, i in enumerate(order):
        if pvals[i] > alpha / (len(pvals) - rank):
            break
        reject[i] = True
    return reject
    # END SOLUTION


def cohens_kappa(a, b) -> float:
    """Agreement beyond chance between two raters (e.g. an LLM judge vs a human)."""
    # BEGIN SOLUTION
    a, b = np.asarray(a), np.asarray(b)
    labels = np.unique(np.concatenate([a, b]))
    po = np.mean(a == b)
    pe = sum(np.mean(a == k) * np.mean(b == k) for k in labels)
    return float((po - pe) / (1 - pe))
    # END SOLUTION


def bradley_terry(wins: np.ndarray, iters: int = 500) -> np.ndarray:
    """Strengths s (sum 1) from a win matrix wins[i, j] = times i beat j, via the MM algorithm:
    s_i <- W_i / sum_j n_ij / (s_i + s_j)."""
    # BEGIN SOLUTION
    w = np.asarray(wins, float)
    n = w + w.T
    s = np.ones(len(w)) / len(w)
    for _ in range(iters):
        denom = (n / (s[:, None] + s[None, :])).sum(1)
        s = w.sum(1) / denom
        s /= s.sum()
    return s
    # END SOLUTION
