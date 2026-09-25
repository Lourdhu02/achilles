# Exercise stub generated from solution.py by tools/make_exercises.py.
# Replace every NotImplementedError, then run: pytest labs/03_napkin_math
"""Lab 03 -- napkin math for large models.

Parameters, FLOPs, memory, bandwidth, time and money: the arithmetic behind every
"can we train / serve this?" decision and every frontier-lab systems interview.
Pure Python, no frameworks.

Handout: labs/03_napkin_math/README.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    glu: bool = True  # SwiGLU-style MLP: 3 matrices (gate, up, down) instead of 2
    tied_embeddings: bool = False  # LM head shares the input embedding matrix
    bias: bool = False  # biases in every linear layer (GPT-2: yes; Llama: no)
    norm: str = "rms"  # "rms" = weight only; "layer" = weight + bias
    learned_pos: int = 0  # learned absolute position embeddings (GPT-2: 1024)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


# ------------------------------------------------------------------ parameters
def attention_weights(cfg: TransformerConfig) -> int:
    """Weight entries in one attention block (Q, K, V, O projections), excluding biases."""
    raise NotImplementedError("03_napkin_math: implement attention_weights")


def mlp_weights(cfg: TransformerConfig) -> int:
    """Weight entries in one MLP block, excluding biases."""
    raise NotImplementedError("03_napkin_math: implement mlp_weights")


def count_params(cfg: TransformerConfig) -> int:
    """Every parameter: embeddings, all blocks (weights, biases, norms), final norm, LM head."""
    raise NotImplementedError("03_napkin_math: implement count_params")


def matmul_params(cfg: TransformerConfig) -> int:
    """Weights that take part in a matmul for every token: all block weights + the LM head.

    The embedding lookup is a gather (no FLOPs); the LM head is a d x V matmul even when tied.
    """
    raise NotImplementedError("03_napkin_math: implement matmul_params")


# ----------------------------------------------------------------------- FLOPs
def flops_per_token(cfg: TransformerConfig, seq_len: int, training: bool = True, causal: bool = True) -> float:
    """FLOPs per token (1 multiply-accumulate = 2 FLOPs).

    forward  = 2 * matmul_params + attention, where attention per layer is QK^T plus AV:
               2 * (2 * T_eff * d_model) with T_eff = seq_len / 2 if causal else seq_len.
    training = 3 * forward (the backward pass costs ~2x the forward).
    """
    raise NotImplementedError("03_napkin_math: implement flops_per_token")


def train_flops(n_params: float, n_tokens: float) -> float:
    """The 6ND rule: total training compute for N parameters on D tokens."""
    raise NotImplementedError("03_napkin_math: implement train_flops")


def chinchilla_optimal(compute: float, tokens_per_param: float = 20.0) -> tuple[float, float]:
    """Compute-optimal (N, D) with C = 6ND and D = tokens_per_param * N."""
    raise NotImplementedError("03_napkin_math: implement chinchilla_optimal")


def training_days(total_flops: float, n_gpus: int, peak_flops: float, mfu: float) -> float:
    """Wall-clock days given per-GPU peak FLOP/s and model FLOPs utilization."""
    raise NotImplementedError("03_napkin_math: implement training_days")


def mfu(tokens_per_second: float, flops_per_token_: float, peak_flops: float) -> float:
    """Model FLOPs utilization: achieved model FLOP/s over hardware peak."""
    raise NotImplementedError("03_napkin_math: implement mfu")


# ---------------------------------------------------------------------- memory
def training_memory_bytes(
    n_params: float,
    n_gpus: int = 1,
    zero_stage: int = 0,
    weight_bytes: int = 2,
    grad_bytes: int = 2,
    optimizer_bytes: int = 12,
) -> dict[str, float]:
    """Per-GPU bytes for weights, grads and optimizer state under ZeRO stage 0-3.

    Mixed-precision Adam: bf16 weights (2) + bf16 grads (2) + fp32 master, m, v (12) = 16 B/param.
    Stage 1 shards the optimizer state, 2 also the grads, 3 also the weights.
    Activations are excluded (see activation_bytes_per_layer).
    """
    raise NotImplementedError("03_napkin_math: implement training_memory_bytes")


def activation_bytes_per_layer(seq_len: int, batch: int, d_model: int, n_heads: int, flash: bool = False) -> float:
    """Activation memory of one transformer layer in 16-bit, no recomputation, no parallelism.

    Korthikanti et al. (2022): s*b*h*(34 + 5*a*s/h). FlashAttention never stores the s x s
    score/softmax/dropout tensors, leaving 34*s*b*h.
    """
    raise NotImplementedError("03_napkin_math: implement activation_bytes_per_layer")


def kv_cache_bytes(n_layers: int, n_kv_heads: int, head_dim: int, seq_len: int, batch: int = 1, bytes_per_elem: float = 2) -> float:
    """K and V for every layer, KV head, position and sequence."""
    raise NotImplementedError("03_napkin_math: implement kv_cache_bytes")


# ------------------------------------------------------------ roofline & speed
def arithmetic_intensity_matmul(m: int, k: int, n: int, bytes_per_elem: float = 2) -> float:
    """FLOPs per byte for (m x k) @ (k x n): read A and B once, write C once."""
    raise NotImplementedError("03_napkin_math: implement arithmetic_intensity_matmul")


def ridge_point(peak_flops: float, peak_bandwidth: float) -> float:
    """Arithmetic intensity (FLOP/byte) above which a kernel is compute-bound."""
    raise NotImplementedError("03_napkin_math: implement ridge_point")


def roofline_seconds(flops: float, bytes_moved: float, peak_flops: float, peak_bandwidth: float) -> float:
    """Lower bound on runtime: whichever of compute and memory traffic takes longer."""
    raise NotImplementedError("03_napkin_math: implement roofline_seconds")


def decode_tokens_per_second(weight_bytes: float, kv_bytes_per_seq: float, batch: int, bandwidth: float) -> float:
    """Upper bound on aggregate decode throughput for a memory-bound decoder.

    Each step streams all weights once (shared by the batch) plus every sequence's KV cache,
    and produces one token per sequence.
    """
    raise NotImplementedError("03_napkin_math: implement decode_tokens_per_second")


def ring_allreduce_seconds(n_bytes: float, n_devices: int, bandwidth: float, latency: float = 0.0) -> float:
    """Ring all-reduce: each device sends/receives 2(n-1)/n of the buffer in 2(n-1) steps."""
    raise NotImplementedError("03_napkin_math: implement ring_allreduce_seconds")


def cost_per_million_tokens(gpu_dollars_per_hour: float, tokens_per_second: float) -> float:
    """Serving cost in $ per 1M generated tokens for one fully used GPU."""
    raise NotImplementedError("03_napkin_math: implement cost_per_million_tokens")


# ------------------------------------------------------------ reference models
GPT2_SMALL = TransformerConfig(
    vocab_size=50257, d_model=768, n_layers=12, n_heads=12, n_kv_heads=12, d_ff=3072,
    glu=False, tied_embeddings=True, bias=True, norm="layer", learned_pos=1024,
)
LLAMA2_7B = TransformerConfig(vocab_size=32000, d_model=4096, n_layers=32, n_heads=32, n_kv_heads=32, d_ff=11008)
LLAMA3_8B = TransformerConfig(vocab_size=128256, d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8, d_ff=14336)
MISTRAL_7B = TransformerConfig(vocab_size=32000, d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8, d_ff=14336)
LLAMA2_70B = TransformerConfig(vocab_size=32000, d_model=8192, n_layers=80, n_heads=64, n_kv_heads=8, d_ff=28672)
