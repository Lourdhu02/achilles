"""Lab 09 -- parallelism, simulated with NumPy: ring all-reduce, tensor parallelism, ZeRO, pipelines.

Handout: labs/09_parallelism/README.md
"""

from __future__ import annotations

import numpy as np


def ring_all_reduce(arrays: list[np.ndarray]) -> tuple[list[np.ndarray], list[int]]:
    """Simulate a ring all-reduce (sum) over len(arrays) ranks.

    Each 1-D array is split into n chunks. Reduce-scatter: n-1 steps, rank r sends one chunk to
    rank r+1, which adds it. All-gather: n-1 steps forwarding finished chunks. All sends in a
    step happen simultaneously. Returns (per-rank results, bytes sent per rank).
    """
    n = len(arrays)
    chunks = [np.array_split(a.astype(np.float64).copy(), n) for a in arrays]
    sent = [0] * n
    # BEGIN SOLUTION
    for step in range(n - 1):  # reduce-scatter
        msgs = [(r, (r - step) % n, chunks[r][(r - step) % n].copy()) for r in range(n)]
        for r, c, data in msgs:
            chunks[(r + 1) % n][c] += data
            sent[r] += data.nbytes
    for step in range(n - 1):  # all-gather: rank r owns reduced chunk (r + 1) % n
        msgs = [(r, (r + 1 - step) % n, chunks[r][(r + 1 - step) % n].copy()) for r in range(n)]
        for r, c, data in msgs:
            chunks[(r + 1) % n][c] = data
            sent[r] += data.nbytes
    # END SOLUTION
    return [np.concatenate(c) for c in chunks], sent


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def column_parallel(x: np.ndarray, w_shards: list[np.ndarray]) -> np.ndarray:
    """Each rank holds a column block W_i (in, out/n); outputs are concatenated (all-gather)."""
    # BEGIN SOLUTION
    return np.concatenate([x @ w for w in w_shards], axis=-1)
    # END SOLUTION


def row_parallel(x_shards: list[np.ndarray], w_shards: list[np.ndarray]) -> np.ndarray:
    """Rank i holds input slice x_i and row block W_i; partial products are summed (all-reduce)."""
    # BEGIN SOLUTION
    return sum(x @ w for x, w in zip(x_shards, w_shards))
    # END SOLUTION


def megatron_mlp(x: np.ndarray, A: np.ndarray, B: np.ndarray, n: int) -> np.ndarray:
    """gelu(x A) B with A split by columns and B by rows over n ranks: one all-reduce, no gather."""
    # BEGIN SOLUTION
    a_shards = np.array_split(A, n, axis=1)
    b_shards = np.array_split(B, n, axis=0)
    return row_parallel([gelu(x @ a) for a in a_shards], b_shards)
    # END SOLUTION


def zero_memory_bytes(n_params: float, n_gpus: int, stage: int, k: int = 12) -> float:
    """Per-GPU bytes for mixed-precision Adam (2 + 2 + K bytes/param) under ZeRO stage 0-3."""
    # BEGIN SOLUTION
    p, g, o = 2 * n_params, 2 * n_params, k * n_params
    if stage >= 1:
        o /= n_gpus
    if stage >= 2:
        g /= n_gpus
    if stage >= 3:
        p /= n_gpus
    return p + g + o
    # END SOLUTION


def pipeline_bubble(stages: int, micro_batches: int) -> float:
    """Idle fraction of a GPipe/1F1B schedule."""
    # BEGIN SOLUTION
    return (stages - 1) / (micro_batches + stages - 1)
    # END SOLUTION


def data_parallel_grad(X: np.ndarray, y: np.ndarray, w: np.ndarray, n: int) -> np.ndarray:
    """Gradient of mean squared error 0.5*mean((Xw-y)^2) computed on n equal shards and averaged."""
    # BEGIN SOLUTION
    grads = [xs.T @ (xs @ w - ys) / len(ys) for xs, ys in zip(np.array_split(X, n), np.array_split(y, n))]
    return np.mean(grads, axis=0)
    # END SOLUTION
