import numpy as np
import pytest

from labs._impl import load

sd = load(__file__)

P_TABLE = {0: [0.6, 0.3, 0.1], 1: [0.1, 0.2, 0.7], 2: [0.3, 0.3, 0.4]}
Q_TABLE = {0: [0.2, 0.5, 0.3], 1: [0.4, 0.4, 0.2], 2: [0.1, 0.8, 0.1]}


def markov(table):
    return lambda prefix: np.array(table[prefix[-1]])


def test_output_distribution_is_exactly_the_target():
    rng = np.random.default_rng(0)
    p, q = markov(P_TABLE), markov(Q_TABLE)
    counts = np.zeros((3, 3))
    n = 30000
    for _ in range(n):
        seq = []
        while len(seq) < 2:
            seq += sd.speculative_step(p, q, (0,) + tuple(seq), k=2, rng=rng)
        counts[seq[0], seq[1]] += 1
    exact = np.array([[P_TABLE[0][a] * P_TABLE[a][b] for b in range(3)] for a in range(3)])
    assert np.abs(counts / n - exact).sum() / 2 < 0.015  # total-variation distance


def test_tokens_per_step_matches_theory():
    rng = np.random.default_rng(1)
    pv, qv = np.array([0.5, 0.3, 0.2]), np.array([0.3, 0.3, 0.4])
    alpha = sd.acceptance_rate(pv, qv)
    assert alpha == pytest.approx(0.8)
    lengths = [len(sd.speculative_step(lambda _: pv, lambda _: qv, (0,), 4, rng)) for _ in range(20000)]
    assert np.mean(lengths) == pytest.approx(sd.expected_tokens(alpha, 4), rel=0.02)


def test_formulas():
    assert sd.expected_tokens(0.0, 5) == 1.0
    assert sd.expected_tokens(1.0, 5) == 6.0
    assert sd.expected_speedup(0.8, 4, 0.05) == pytest.approx((1 - 0.8**5) / 0.2 / 1.2)
