"""Lab 08 -- scaling laws: fit power laws, find compute-optimal and inference-aware model sizes.

Handout: labs/08_scaling_laws/README.md
"""

from __future__ import annotations

import numpy as np


def fit_power_law(N, L, n_grid: int = 4000) -> tuple[float, float, float]:
    """Fit L(N) = E + A * N**(-alpha). Returns (E, A, alpha).

    For fixed E the model is linear in log space: log(L - E) = log A - alpha log N.
    Grid-search E in [0, min(L)) and keep the least-squares best.
    """
    N, L = np.asarray(N, float), np.asarray(L, float)
    # BEGIN SOLUTION
    X = np.stack([np.ones_like(N), np.log(N)], axis=1)
    best = None
    for E in np.linspace(0.0, L.min() * (1 - 1e-6), n_grid):
        y = np.log(L - E)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = float(np.sum((X @ coef - y) ** 2))
        if best is None or resid < best[0]:
            best = (resid, E, float(np.exp(coef[0])), float(-coef[1]))
    return best[1], best[2], best[3]
    # END SOLUTION


def chinchilla_loss(N, D, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    """Parametric loss L(N, D) = E + A/N^alpha + B/D^beta (defaults: Hoffmann et al. 2022 fit)."""
    # BEGIN SOLUTION
    return E + A / np.power(N, alpha) + B / np.power(D, beta)
    # END SOLUTION


def compute_optimal(C: float, A=406.4, B=410.7, alpha=0.34, beta=0.28) -> tuple[float, float]:
    """Minimize chinchilla_loss subject to C = 6 N D. Returns (N_opt, D_opt).

    N* = G (C/6)^(beta/(alpha+beta)),  G = (alpha A / (beta B))^(1/(alpha+beta)),  D* = C / (6 N*).
    """
    # BEGIN SOLUTION
    G = (alpha * A / (beta * B)) ** (1.0 / (alpha + beta))
    n = G * (C / 6.0) ** (beta / (alpha + beta))
    return n, C / (6.0 * n)
    # END SOLUTION


def inference_aware_optimum(target_loss: float, inference_tokens: float, n_grid=None, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28) -> tuple[float, float]:
    """Smallest total FLOPs (training 6ND + inference 2N per served token) reaching target_loss.

    For each N on the grid, solve for the D that hits the target (skip N that cannot).
    Returns (N, D) of the cheapest option (Sardana et al., "Beyond Chinchilla-Optimal").
    """
    n_grid = np.logspace(7, 12, 2000) if n_grid is None else np.asarray(n_grid, float)
    # BEGIN SOLUTION
    best = None
    for n in n_grid:
        gap = target_loss - E - A / n**alpha
        if gap <= 0:
            continue
        d = (B / gap) ** (1.0 / beta)
        total = 6 * n * d + 2 * n * inference_tokens
        if best is None or total < best[0]:
            best = (total, n, d)
    if best is None:
        raise ValueError("target loss unreachable on this grid")
    return best[1], best[2]
    # END SOLUTION
