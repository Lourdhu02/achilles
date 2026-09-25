# Exercise stub generated from solution.py by tools/make_exercises.py.
# Replace every NotImplementedError, then run: pytest labs/09_parallelism
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
    raise NotImplementedError("09_parallelism: implement ring_all_reduce")
    return [np.concatenate(c) for c in chunks], sent


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def column_parallel(x: np.ndarray, w_shards: list[np.ndarray]) -> np.ndarray:
    """Each rank holds a column block W_i (in, out/n); outputs are concatenated (all-gather)."""
    raise NotImplementedError("09_parallelism: implement column_parallel")


def row_parallel(x_shards: list[np.ndarray], w_shards: list[np.ndarray]) -> np.ndarray:
    """Rank i holds input slice x_i and row block W_i; partial products are summed (all-reduce)."""
    raise NotImplementedError("09_parallelism: implement row_parallel")


def megatron_mlp(x: np.ndarray, A: np.ndarray, B: np.ndarray, n: int) -> np.ndarray:
    """gelu(x A) B with A split by columns and B by rows over n ranks: one all-reduce, no gather."""
    raise NotImplementedError("09_parallelism: implement megatron_mlp")


def zero_memory_bytes(n_params: float, n_gpus: int, stage: int, k: int = 12) -> float:
    """Per-GPU bytes for mixed-precision Adam (2 + 2 + K bytes/param) under ZeRO stage 0-3."""
    raise NotImplementedError("09_parallelism: implement zero_memory_bytes")


def pipeline_bubble(stages: int, micro_batches: int) -> float:
    """Idle fraction of a GPipe/1F1B schedule."""
    raise NotImplementedError("09_parallelism: implement pipeline_bubble")


def data_parallel_grad(X: np.ndarray, y: np.ndarray, w: np.ndarray, n: int) -> np.ndarray:
    """Gradient of mean squared error 0.5*mean((Xw-y)^2) computed on n equal shards and averaged."""
    raise NotImplementedError("09_parallelism: implement data_parallel_grad")
