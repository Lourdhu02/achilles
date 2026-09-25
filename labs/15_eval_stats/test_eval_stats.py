import math

import numpy as np
import pytest

from labs._impl import load

es = load(__file__)


def test_ci_width_for_a_1000_item_benchmark():
    scores = np.array([1] * 700 + [0] * 300)
    mean, lo, hi = es.mean_ci(scores)
    assert mean == 0.7 and (hi - lo) / 2 == pytest.approx(0.0284, abs=5e-4)
    blo, bhi = es.bootstrap_ci(scores)
    assert blo == pytest.approx(lo, abs=0.004) and bhi == pytest.approx(hi, abs=0.004)


def test_paired_tests_detect_small_consistent_gains():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 500)
    b = a.copy()
    flip = rng.choice(500, 30, replace=False)
    b[flip] = 1  # b fixes up to 30 of a's items and never breaks one
    assert es.paired_permutation_test(b, a) < 0.01
    assert es.mcnemar_exact(b, a) < 0.01
    assert es.mcnemar_exact(a, a) == 1.0


def test_clustered_se_exceeds_naive_when_clusters_correlate():
    scores = np.repeat([1, 0, 1, 1, 0, 0, 1, 0], 10)  # 8 passages, 10 identical-outcome questions each
    clusters = np.repeat(np.arange(8), 10)
    naive = scores.std() / math.sqrt(len(scores))
    assert es.clustered_se(scores, clusters) > 2.5 * naive


def test_pass_at_k():
    assert es.pass_at_k(10, 0, 1) == 0.0
    assert es.pass_at_k(10, 3, 1) == pytest.approx(0.3)
    assert es.pass_at_k(10, 3, 2) == pytest.approx(1 - math.comb(7, 2) / math.comb(10, 2))
    assert es.pass_at_k(5, 4, 2) == 1.0


def test_power_and_multiple_comparisons():
    assert es.required_n(0.5, 0.6) == 388
    assert es.holm_bonferroni([0.01, 0.04, 0.03, 0.005]) == [True, False, False, True]


def test_kappa_and_bradley_terry():
    assert es.cohens_kappa([1, 0, 1, 1], [1, 0, 1, 1]) == 1.0
    assert es.cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0
    true = np.array([4.0, 2.0, 1.0])
    true /= true.sum()
    wins = 1000 * true[:, None] / (true[:, None] + true[None, :])
    np.fill_diagonal(wins, 0)
    np.testing.assert_allclose(es.bradley_terry(wins), true, rtol=1e-4)
