import numpy as np
import pytest

from labs._impl import load

sl = load(__file__)


def test_fit_recovers_known_law():
    rng = np.random.default_rng(0)
    N = np.logspace(6, 10, 12)
    L = (1.8 + 400 * N**-0.34) * (1 + 0.002 * rng.normal(size=N.size))
    E, A, alpha = sl.fit_power_law(N, L)
    assert E == pytest.approx(1.8, abs=0.1)
    assert alpha == pytest.approx(0.34, abs=0.04)


def test_loss_decreases_in_both_axes():
    assert sl.chinchilla_loss(1e9, 2e10) < sl.chinchilla_loss(1e8, 2e10)
    assert sl.chinchilla_loss(1e9, 2e11) < sl.chinchilla_loss(1e9, 2e10)


@pytest.mark.parametrize("C", [1e20, 5.76e23])
def test_closed_form_matches_grid_search(C):
    n_opt, d_opt = sl.compute_optimal(C)
    assert 6 * n_opt * d_opt == pytest.approx(C)
    grid = np.logspace(6, 13, 20000)
    losses = sl.chinchilla_loss(grid, C / (6 * grid))
    assert n_opt == pytest.approx(grid[np.argmin(losses)], rel=0.01)


def test_more_inference_means_smaller_models_trained_longer():
    n0, d0 = sl.inference_aware_optimum(2.2, inference_tokens=0)
    n1, d1 = sl.inference_aware_optimum(2.2, inference_tokens=1e13)
    assert n1 < n0 and d1 > d0
    assert sl.chinchilla_loss(n1, d1) == pytest.approx(2.2, abs=1e-6)

def test_inference_aware_matches_grid_search():
    target = 2.2
    inf = 1e13
    n, d = sl.inference_aware_optimum(target, inf)
    grid = np.logspace(7, 12, 1000)
    gap = target - 1.69 - 406.4 / grid**0.34
    valid = gap > 0
    grid = grid[valid]
    gap = gap[valid]
    d_grid = (410.7 / gap) ** (1 / 0.28)
    cost = 6 * grid * d_grid + 2 * grid * inf
    assert n == pytest.approx(grid[np.argmin(cost)], rel=0.01)
