import numpy as np
import pytest

from labs._impl import load

par = load(__file__)
RNG = np.random.default_rng(0)


@pytest.mark.parametrize("n", [2, 4, 8])
def test_ring_all_reduce_sums_and_moves_2_n_minus_1_over_n(n):
    arrays = [RNG.normal(size=64) for _ in range(n)]
    results, sent = par.ring_all_reduce(arrays)
    for r in results:
        np.testing.assert_allclose(r, np.sum(arrays, axis=0))
    assert all(s == pytest.approx(2 * (n - 1) / n * 64 * 8) for s in sent)


def test_tensor_parallel_linear_layers():
    x, W = RNG.normal(size=(5, 12)), RNG.normal(size=(12, 8))
    np.testing.assert_allclose(par.column_parallel(x, np.array_split(W, 4, axis=1)), x @ W)
    np.testing.assert_allclose(par.row_parallel(np.array_split(x, 3, axis=1), np.array_split(W, 3, axis=0)), x @ W)


def test_megatron_mlp_equals_unsharded():
    x, A, B = RNG.normal(size=(4, 16)), RNG.normal(size=(16, 64)), RNG.normal(size=(64, 16))
    np.testing.assert_allclose(par.megatron_mlp(x, A, B, n=4), par.gelu(x @ A) @ B)


def test_zero_paper_numbers():
    gb = [par.zero_memory_bytes(7.5e9, 64, s) / 1e9 for s in range(4)]
    assert gb == pytest.approx([120, 31.4, 16.6, 1.875], rel=3e-3)


def test_pipeline_bubble_and_data_parallel():
    assert par.pipeline_bubble(4, 1) == pytest.approx(0.75)
    assert par.pipeline_bubble(4, 29) == pytest.approx(3 / 32)
    X, y, w = RNG.normal(size=(64, 5)), RNG.normal(size=64), RNG.normal(size=5)
    np.testing.assert_allclose(par.data_parallel_grad(X, y, w, 8), X.T @ (X @ w - y) / 64)
