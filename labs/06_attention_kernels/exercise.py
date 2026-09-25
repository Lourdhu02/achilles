# Exercise stub generated from solution.py by tools/make_exercises.py.
# Replace every NotImplementedError, then run: pytest labs/06_attention_kernels
"""Lab 06 -- attention kernels: online softmax -> FlashAttention (forward + backward) -> Triton.

Part A (PyTorch, CPU-testable): the FlashAttention algorithm expressed with tiles and loops.
It is slow in Python -- the point is the algorithm: never materialize the T x T matrix,
keep running softmax statistics, recompute in the backward pass.

Part B (Triton): the same forward pass as a real GPU kernel, plus a fused row softmax.
Runs on your RTX 5060, or on CPU with TRITON_INTERPRET=1 (the tests do this automatically).

Handout: labs/06_attention_kernels/README.md
"""

from __future__ import annotations

import math

import torch

Tensor = torch.Tensor

try:  # Triton ships with Linux CUDA wheels of PyTorch; on Windows use WSL2 or `triton-windows`.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None


# ============================================================== Part A: PyTorch
def naive_attention(q: Tensor, k: Tensor, v: Tensor, causal: bool = False) -> Tensor:
    """Reference: materializes the full (T_q x T_k) score matrix."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if causal:
        tq, tk = scores.shape[-2:]
        scores = scores.masked_fill(torch.ones(tq, tk, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def online_softmax_stats(blocks: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Stream over 1-D blocks of one row and return (m, l): the running max and
    sum(exp(x - m)) over everything seen, so that logsumexp(row) = m + log(l).

    Never hold the whole row. When the max grows from m to m', rescale l by exp(m - m').
    """
    raise NotImplementedError("06_attention_kernels: implement online_softmax_stats")


def flash_attention_forward(
    q: Tensor, k: Tensor, v: Tensor, causal: bool = False, block_q: int = 16, block_k: int = 16
) -> tuple[Tensor, Tensor]:
    """Tiled attention with online softmax. q, k, v: (B, H, T, D) with equal T when causal.

    Returns O (B, H, T, D) and LSE (B, H, T) = logsumexp of each row of scaled scores,
    which the backward pass uses to recompute probabilities without storing them.
    """
    B, H, Tq, D = q.shape
    Tk = k.shape[-2]
    scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    lse = torch.empty(B, H, Tq, dtype=q.dtype, device=q.device)
    # HINT: for each query tile keep m (running max), l (running sum), acc (unnormalized output);
    # HINT: for each key tile: s = q k^T * scale, mask, m_new = max(m, rowmax(s)), p = exp(s - m_new),
    # HINT: rescale l and acc by exp(m - m_new), add p's contribution. Finally O = acc / l, LSE = m + log l.
    raise NotImplementedError("06_attention_kernels: implement flash_attention_forward")
    return out, lse


def flash_attention_backward(
    q: Tensor, k: Tensor, v: Tensor, out: Tensor, lse: Tensor, dout: Tensor,
    causal: bool = False, block_q: int = 16, block_k: int = 16,
) -> tuple[Tensor, Tensor, Tensor]:
    """Gradients (dq, dk, dv) by recomputation, tile by tile, never storing P.

    With P = softmax(S), S = q k^T * scale:
        dV = P^T dO          dP = dO V^T
        dS = P * (dP - D)    where D_i = rowsum(dO_i * O_i)   (= rowsum(P_i * dP_i))
        dQ = dS K * scale    dK = dS^T Q * scale
    P for a tile is recomputed as exp(S - LSE).
    """
    B, H, Tq, D = q.shape
    Tk = k.shape[-2]
    scale = 1.0 / math.sqrt(D)
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    # HINT: precompute Drow = (dout * out).sum(-1); loop over key tiles (outer) and query tiles (inner).
    raise NotImplementedError("06_attention_kernels: implement flash_attention_backward")
    return dq, dk, dv


class FlashAttention(torch.autograd.Function):
    """Autograd wrapper: saves only q, k, v, O and LSE -- O(T) extra memory, not O(T^2)."""

    @staticmethod
    def forward(ctx, q, k, v, causal=False):
        out, lse = flash_attention_forward(q, k, v, causal)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = flash_attention_backward(q, k, v, out, lse, dout, ctx.causal)
        return dq, dk, dv, None


def flash_attention(q: Tensor, k: Tensor, v: Tensor, causal: bool = False) -> Tensor:
    return FlashAttention.apply(q, k, v, causal)


# =============================================================== Part B: Triton
if triton is not None:

    @triton.jit
    def _softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols, BLOCK: tl.constexpr):
        """One program per row: load the row once, softmax in registers, store once."""
        # HINT: offs = tl.arange(0, BLOCK); load with mask and other=-inf; subtract tl.max; exp; divide by tl.sum.
        raise NotImplementedError("06_attention_kernels: implement _softmax_kernel")

    @triton.jit
    def _flash_fwd_kernel(
        Q, K, V, O, L, stride_bh, stride_t, T, scale,
        CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    ):
        """Program (m, bh) computes BLOCK_M query rows of one (batch, head) with online softmax."""
        raise NotImplementedError("06_attention_kernels: implement _flash_fwd_kernel")


def triton_softmax(x: Tensor) -> Tensor:
    """Row-wise softmax of a 2-D tensor with one Triton program per row."""
    assert triton is not None, "Triton is not installed"
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    _softmax_kernel[(n_rows,)](out, x, x.stride(0), out.stride(0), n_cols, BLOCK=triton.next_power_of_2(n_cols))
    return out


def triton_flash_attention(q: Tensor, k: Tensor, v: Tensor, causal: bool = False, block_m: int = 16, block_n: int = 16) -> tuple[Tensor, Tensor]:
    """FlashAttention forward in Triton. q, k, v: (B, H, T, D), D a power of two >= 16."""
    assert triton is not None, "Triton is not installed"
    B, H, T, D = q.shape
    q, k, v = (t.contiguous() for t in (q, k, v))
    out = torch.empty_like(q)
    lse = torch.empty(B * H, T, device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(T, block_m), B * H)
    _flash_fwd_kernel[grid](
        q, k, v, out, lse, q.stride(1), q.stride(2), T, 1.0 / math.sqrt(D),
        CAUSAL=causal, BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=D,
    )
    return out, lse.view(B, H, T)
