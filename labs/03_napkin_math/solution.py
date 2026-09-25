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
    # BEGIN SOLUTION
    kv_dim = cfg.n_kv_heads * cfg.head_dim
    return 2 * cfg.d_model * cfg.d_model + 2 * cfg.d_model * kv_dim
    # END SOLUTION


def mlp_weights(cfg: TransformerConfig) -> int:
    """Weight entries in one MLP block, excluding biases."""
    # BEGIN SOLUTION
    return (3 if cfg.glu else 2) * cfg.d_model * cfg.d_ff
    # END SOLUTION


def count_params(cfg: TransformerConfig) -> int:
    """Every parameter: embeddings, all blocks (weights, biases, norms), final norm, LM head."""
    # BEGIN SOLUTION
    d = cfg.d_model
    kv_dim = cfg.n_kv_heads * cfg.head_dim
    norm = d * (2 if cfg.norm == "layer" else 1)
    per_layer = attention_weights(cfg) + mlp_weights(cfg) + 2 * norm
    if cfg.bias:
        per_layer += (d + 2 * kv_dim + d) + ((2 * cfg.d_ff if cfg.glu else cfg.d_ff) + d)
    embeddings = cfg.vocab_size * d + cfg.learned_pos * d
    head = 0 if cfg.tied_embeddings else cfg.vocab_size * d
    return embeddings + cfg.n_layers * per_layer + norm + head
    # END SOLUTION


def matmul_params(cfg: TransformerConfig) -> int:
    """Weights that take part in a matmul for every token: all block weights + the LM head.

    The embedding lookup is a gather (no FLOPs); the LM head is a d x V matmul even when tied.
    """
    # BEGIN SOLUTION
    return cfg.n_layers * (attention_weights(cfg) + mlp_weights(cfg)) + cfg.vocab_size * cfg.d_model
    # END SOLUTION


# ----------------------------------------------------------------------- FLOPs
def flops_per_token(cfg: TransformerConfig, seq_len: int, training: bool = True, causal: bool = True) -> float:
    """FLOPs per token (1 multiply-accumulate = 2 FLOPs).

    forward  = 2 * matmul_params + attention, where attention per layer is QK^T plus AV:
               2 * (2 * T_eff * d_model) with T_eff = seq_len / 2 if causal else seq_len.
    training = 3 * forward (the backward pass costs ~2x the forward).
    """
    # BEGIN SOLUTION
    t_eff = seq_len / 2 if causal else seq_len
    attention = cfg.n_layers * 2 * (2 * t_eff * cfg.d_model)
    forward = 2 * matmul_params(cfg) + attention
    return 3 * forward if training else forward
    # END SOLUTION


def train_flops(n_params: float, n_tokens: float) -> float:
    """The 6ND rule: total training compute for N parameters on D tokens."""
    # BEGIN SOLUTION
    return 6.0 * n_params * n_tokens
    # END SOLUTION


def chinchilla_optimal(compute: float, tokens_per_param: float = 20.0) -> tuple[float, float]:
    """Compute-optimal (N, D) with C = 6ND and D = tokens_per_param * N."""
    # BEGIN SOLUTION
    n = math.sqrt(compute / (6.0 * tokens_per_param))
    return n, tokens_per_param * n
    # END SOLUTION


def training_days(total_flops: float, n_gpus: int, peak_flops: float, mfu: float) -> float:
    """Wall-clock days given per-GPU peak FLOP/s and model FLOPs utilization."""
    # BEGIN SOLUTION
    return total_flops / (n_gpus * peak_flops * mfu) / 86400.0
    # END SOLUTION


def mfu(tokens_per_second: float, flops_per_token_: float, peak_flops: float) -> float:
    """Model FLOPs utilization: achieved model FLOP/s over hardware peak."""
    # BEGIN SOLUTION
    return tokens_per_second * flops_per_token_ / peak_flops
    # END SOLUTION


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
    # BEGIN SOLUTION
    shard = lambda b, on: b / n_gpus if on else b  # noqa: E731
    out = {
        "weights": shard(n_params * weight_bytes, zero_stage >= 3),
        "grads": shard(n_params * grad_bytes, zero_stage >= 2),
        "optimizer": shard(n_params * optimizer_bytes, zero_stage >= 1),
    }
    out["total"] = out["weights"] + out["grads"] + out["optimizer"]
    return out
    # END SOLUTION


def activation_bytes_per_layer(seq_len: int, batch: int, d_model: int, n_heads: int, flash: bool = False) -> float:
    """Activation memory of one transformer layer in 16-bit, no recomputation, no parallelism.

    Korthikanti et al. (2022): s*b*h*(34 + 5*a*s/h). FlashAttention never stores the s x s
    score/softmax/dropout tensors, leaving 34*s*b*h.
    """
    # BEGIN SOLUTION
    s, b, h, a = seq_len, batch, d_model, n_heads
    return s * b * h * 34 + (0 if flash else 5 * a * s * s * b)
    # END SOLUTION


def kv_cache_bytes(n_layers: int, n_kv_heads: int, head_dim: int, seq_len: int, batch: int = 1, bytes_per_elem: float = 2) -> float:
    """K and V for every layer, KV head, position and sequence."""
    # BEGIN SOLUTION
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_elem
    # END SOLUTION


# ------------------------------------------------------------ roofline & speed
def arithmetic_intensity_matmul(m: int, k: int, n: int, bytes_per_elem: float = 2) -> float:
    """FLOPs per byte for (m x k) @ (k x n): read A and B once, write C once."""
    # BEGIN SOLUTION
    return 2 * m * k * n / (bytes_per_elem * (m * k + k * n + m * n))
    # END SOLUTION


def ridge_point(peak_flops: float, peak_bandwidth: float) -> float:
    """Arithmetic intensity (FLOP/byte) above which a kernel is compute-bound."""
    # BEGIN SOLUTION
    return peak_flops / peak_bandwidth
    # END SOLUTION


def roofline_seconds(flops: float, bytes_moved: float, peak_flops: float, peak_bandwidth: float) -> float:
    """Lower bound on runtime: whichever of compute and memory traffic takes longer."""
    # BEGIN SOLUTION
    return max(flops / peak_flops, bytes_moved / peak_bandwidth)
    # END SOLUTION


def decode_tokens_per_second(weight_bytes: float, kv_bytes_per_seq: float, batch: int, bandwidth: float) -> float:
    """Upper bound on aggregate decode throughput for a memory-bound decoder.

    Each step streams all weights once (shared by the batch) plus every sequence's KV cache,
    and produces one token per sequence.
    """
    # BEGIN SOLUTION
    step_seconds = (weight_bytes + batch * kv_bytes_per_seq) / bandwidth
    return batch / step_seconds
    # END SOLUTION


def ring_allreduce_seconds(n_bytes: float, n_devices: int, bandwidth: float, latency: float = 0.0) -> float:
    """Ring all-reduce: each device sends/receives 2(n-1)/n of the buffer in 2(n-1) steps."""
    # BEGIN SOLUTION
    n = n_devices
    return 2 * (n - 1) / n * n_bytes / bandwidth + 2 * (n - 1) * latency
    # END SOLUTION


def cost_per_million_tokens(gpu_dollars_per_hour: float, tokens_per_second: float) -> float:
    """Serving cost in $ per 1M generated tokens for one fully used GPU."""
    # BEGIN SOLUTION
    return gpu_dollars_per_hour / (tokens_per_second * 3600.0) * 1e6
    # END SOLUTION


# ------------------------------------------------------------ reference models
GPT2_SMALL = TransformerConfig(
    vocab_size=50257, d_model=768, n_layers=12, n_heads=12, n_kv_heads=12, d_ff=3072,
    glu=False, tied_embeddings=True, bias=True, norm="layer", learned_pos=1024,
)
LLAMA2_7B = TransformerConfig(vocab_size=32000, d_model=4096, n_layers=32, n_heads=32, n_kv_heads=32, d_ff=11008)
LLAMA3_8B = TransformerConfig(vocab_size=128256, d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8, d_ff=14336)
MISTRAL_7B = TransformerConfig(vocab_size=32000, d_model=4096, n_layers=32, n_heads=32, n_kv_heads=8, d_ff=14336)
LLAMA2_70B = TransformerConfig(vocab_size=32000, d_model=8192, n_layers=80, n_heads=64, n_kv_heads=8, d_ff=28672)
