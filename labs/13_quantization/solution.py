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
    # BEGIN SOLUTION
    qmax = 2 ** (bits - 1) - 1
    absmax = w.abs().amax() if dim is None else w.abs().amax(dim=dim, keepdim=True)
    s = torch.clamp(absmax, min=1e-12) / qmax
    return torch.clamp(torch.round(w / s), -qmax, qmax).to(torch.int8), s
    # END SOLUTION


def dequantize(q: Tensor, s: Tensor) -> Tensor:
    return q.float() * s


def quantize_groupwise(w: Tensor, bits: int = 4, group: int = 64) -> tuple[Tensor, Tensor]:
    """Symmetric quantization with one scale per contiguous group of `group` values in each row."""
    # BEGIN SOLUTION
    rows, cols = w.shape
    q, s = quantize_symmetric(w.reshape(rows, cols // group, group), bits, dim=-1)
    return q.reshape(rows, cols), s
    # END SOLUTION


def dequantize_groupwise(q: Tensor, s: Tensor, group: int = 64) -> Tensor:
    rows, cols = q.shape
    return (q.float().reshape(rows, cols // group, group) * s).reshape(rows, cols)


def pack_int4(q: Tensor) -> Tensor:
    """Pack signed int4 values in [-8, 7] (even count) two per uint8: low nibble first."""
    # BEGIN SOLUTION
    u = (q.to(torch.int16) & 0xF).to(torch.uint8).flatten()
    return u[0::2] | (u[1::2] << 4)
    # END SOLUTION


def unpack_int4(packed: Tensor, shape) -> Tensor:
    # BEGIN SOLUTION
    lo = (packed & 0xF).to(torch.int16)
    hi = (packed >> 4).to(torch.int16)
    u = torch.stack([lo, hi], 1).flatten()
    return torch.where(u > 7, u - 16, u).to(torch.int8).reshape(shape)
    # END SOLUTION


def nf4_codebook() -> Tensor:
    """The 16 NormalFloat4 levels (QLoRA): quantiles of N(0,1), asymmetric so that 0 is exact.
    8 positive levels from linspace(offset, 0.5, 9)[:-1], 7 negative from linspace(offset, 0.5, 8)[:-1],
    plus 0, normalized to [-1, 1]; offset = 0.9677083."""
    # BEGIN SOLUTION
    offset = 0.9677083
    pos = torch.special.ndtri(torch.linspace(offset, 0.5, 9, dtype=torch.float64)[:-1])
    neg = -torch.special.ndtri(torch.linspace(offset, 0.5, 8, dtype=torch.float64)[:-1])
    v = torch.cat([pos, torch.zeros(1, dtype=torch.float64), neg]).sort().values
    return (v / v.abs().max()).float()
    # END SOLUTION


def quantize_nf4(w: Tensor, block: int = 64) -> tuple[Tensor, Tensor]:
    """Per-block absmax scaling to [-1, 1], then the index of the nearest NF4 level."""
    # BEGIN SOLUTION
    code = nf4_codebook()
    x = w.reshape(-1, block)
    s = x.abs().amax(1, keepdim=True).clamp(min=1e-12)
    idx = ((x / s)[..., None] - code).abs().argmin(-1)
    return idx.to(torch.uint8).reshape(w.shape), s
    # END SOLUTION


def dequantize_nf4(idx: Tensor, s: Tensor, block: int = 64) -> Tensor:
    return (nf4_codebook()[idx.long()].reshape(-1, block) * s).reshape(idx.shape)


def smoothquant(w: Tensor, act_absmax: Tensor, alpha: float = 0.5) -> tuple[Tensor, Tensor]:
    """Migrate activation outliers into weights. w (out, in); act_absmax (in,).
    s_j = act_absmax_j^alpha / max_i|w_ij|^(1-alpha). Returns (w * s, s); run activations as x / s."""
    # BEGIN SOLUTION
    s = act_absmax.clamp(min=1e-5) ** alpha / w.abs().amax(0).clamp(min=1e-5) ** (1 - alpha)
    return w * s, s
    # END SOLUTION


def gptq_quantize(w: Tensor, x: Tensor, bits: int = 4, damp: float = 0.01) -> Tensor:
    """GPTQ for one linear layer (w: out x in, calibration inputs x: n x in).

    Quantize columns left to right; after each column, spread its error over the remaining columns
    using the upper Cholesky factor U of H^-1 (H = 2 X^T X / n + damping):
        err = (w_j - q_j) / U_jj;   W[:, j+1:] -= err ⊗ U_{j, j+1:}
    Uses per-output-row symmetric scales computed from the original weights. Returns dequantized W.
    """
    # BEGIN SOLUTION
    W = w.clone().double()
    n = x.shape[0]
    H = 2 * x.double().T @ x.double() / n
    H += damp * H.diag().mean() * torch.eye(H.shape[0], dtype=H.dtype)
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    qmax = 2 ** (bits - 1) - 1
    scale = W.abs().amax(1).clamp(min=1e-12) / qmax
    Q = torch.zeros_like(W)
    for j in range(W.shape[1]):
        q = torch.clamp(torch.round(W[:, j] / scale), -qmax, qmax) * scale
        Q[:, j] = q
        err = (W[:, j] - q) / U[j, j]
        W[:, j + 1 :] -= err[:, None] * U[j, j + 1 :][None, :]
    return Q.to(w.dtype)
    # END SOLUTION
