# Exercise stub generated from solution.py by tools/make_exercises.py.
# Replace every NotImplementedError, then run: pytest labs/13_quantization
"""Lab 13 -- quantization: INT8/INT4 (per-tensor, per-channel, group-wise), int4 packing,
NF4, SmoothQuant and GPTQ.

Handout: labs/13_quantization/README.md
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def quantize_symmetric(w: Tensor, bits: int = 8, dim: int | None = None) -> tuple[Tensor, Tensor]:
    """q = round(w / s) clamped to [-(2^(b-1)-1), 2^(b-1)-1] with s = absmax / qmax.
    dim=None: one scale for the tensor; dim=d: one scale per slice along d (keepdim)."""
    raise NotImplementedError("13_quantization: implement quantize_symmetric")


def dequantize(q: Tensor, s: Tensor) -> Tensor:
    return q.float() * s


def quantize_groupwise(w: Tensor, bits: int = 4, group: int = 64) -> tuple[Tensor, Tensor]:
    """Symmetric quantization with one scale per contiguous group of `group` values in each row."""
    raise NotImplementedError("13_quantization: implement quantize_groupwise")


def dequantize_groupwise(q: Tensor, s: Tensor, group: int = 64) -> Tensor:
    rows, cols = q.shape
    return (q.float().reshape(rows, cols // group, group) * s).reshape(rows, cols)


def pack_int4(q: Tensor) -> Tensor:
    """Pack signed int4 values in [-8, 7] (even count) two per uint8: low nibble first."""
    raise NotImplementedError("13_quantization: implement pack_int4")


def unpack_int4(packed: Tensor, shape) -> Tensor:
    raise NotImplementedError("13_quantization: implement unpack_int4")


def nf4_codebook() -> Tensor:
    """The 16 NormalFloat4 levels (QLoRA): quantiles of N(0,1), asymmetric so that 0 is exact.
    8 positive levels from linspace(offset, 0.5, 9)[:-1], 7 negative from linspace(offset, 0.5, 8)[:-1],
    plus 0, normalized to [-1, 1]; offset = 0.9677083."""
    raise NotImplementedError("13_quantization: implement nf4_codebook")


def quantize_nf4(w: Tensor, block: int = 64) -> tuple[Tensor, Tensor]:
    """Per-block absmax scaling to [-1, 1], then the index of the nearest NF4 level."""
    raise NotImplementedError("13_quantization: implement quantize_nf4")


def dequantize_nf4(idx: Tensor, s: Tensor, block: int = 64) -> Tensor:
    return (nf4_codebook()[idx.long()].reshape(-1, block) * s).reshape(idx.shape)


def smoothquant(w: Tensor, act_absmax: Tensor, alpha: float = 0.5) -> tuple[Tensor, Tensor]:
    """Migrate activation outliers into weights. w (out, in); act_absmax (in,).
    s_j = act_absmax_j^alpha / max_i|w_ij|^(1-alpha). Returns (w * s, s); run activations as x / s."""
    raise NotImplementedError("13_quantization: implement smoothquant")


def gptq_quantize(w: Tensor, x: Tensor, bits: int = 4, damp: float = 0.01) -> Tensor:
    """GPTQ for one linear layer (w: out x in, calibration inputs x: n x in).

    Quantize columns left to right; after each column, spread its error over the remaining columns
    using the upper Cholesky factor U of H^-1 (H = 2 X^T X / n + damping):
        err = (w_j - q_j) / U_jj;   W[:, j+1:] -= err ⊗ U_{j, j+1:}
    Uses per-output-row symmetric scales computed from the original weights. Returns dequantized W.
    """
    raise NotImplementedError("13_quantization: implement gptq_quantize")
