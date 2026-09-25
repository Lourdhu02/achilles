# Exercise stub generated from solution.py by tools/make_exercises.py.
# Replace every NotImplementedError, then run: pytest labs/08_scaling_laws
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
    raise NotImplementedError("08_scaling_laws: implement fit_power_law")


def chinchilla_loss(N, D, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    """Parametric loss L(N, D) = E + A/N^alpha + B/D^beta (defaults: Hoffmann et al. 2022 fit)."""
    raise NotImplementedError("08_scaling_laws: implement chinchilla_loss")


def compute_optimal(C: float, A=406.4, B=410.7, alpha=0.34, beta=0.28) -> tuple[float, float]:
    """Minimize chinchilla_loss subject to C = 6 N D. Returns (N_opt, D_opt).

    N* = G (C/6)^(beta/(alpha+beta)),  G = (alpha A / (beta B))^(1/(alpha+beta)),  D* = C / (6 N*).
    """
    raise NotImplementedError("08_scaling_laws: implement compute_optimal")


def inference_aware_optimum(target_loss: float, inference_tokens: float, n_grid=None, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28) -> tuple[float, float]:
    """Smallest total FLOPs (training 6ND + inference 2N per served token) reaching target_loss.

    For each N on the grid, solve for the D that hits the target (skip N that cannot).
    Returns (N, D) of the cheapest option (Sardana et al., "Beyond Chinchilla-Optimal").
    """
    n_grid = np.logspace(7, 12, 2000) if n_grid is None else np.asarray(n_grid, float)
    raise NotImplementedError("08_scaling_laws: implement inference_aware_optimum")
