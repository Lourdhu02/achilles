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
    # BEGIN SOLUTION
    m = torch.tensor(float("-inf"))
    l = torch.tensor(0.0)
    for block in blocks:
        m_new = torch.maximum(m, block.max())
        l = l * torch.exp(m - m_new) + torch.exp(block - m_new).sum()
        m = m_new
    return m, l
    # END SOLUTION


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
    # BEGIN SOLUTION
    # HINT: for each query tile keep m (running max), l (running sum), acc (unnormalized output);
    # HINT: for each key tile: s = q k^T * scale, mask, m_new = max(m, rowmax(s)), p = exp(s - m_new),
    # HINT: rescale l and acc by exp(m - m_new), add p's contribution. Finally O = acc / l, LSE = m + log l.
    for i0 in range(0, Tq, block_q):
        i1 = min(i0 + block_q, Tq)
        qi = q[..., i0:i1, :]
        m = torch.full((B, H, i1 - i0), float("-inf"), dtype=q.dtype, device=q.device)
        l = torch.zeros((B, H, i1 - i0), dtype=q.dtype, device=q.device)
        acc = torch.zeros((B, H, i1 - i0, D), dtype=q.dtype, device=q.device)
        for j0 in range(0, Tk, block_k):
            if causal and j0 > i1 - 1:
                break  # every key in this tile is in the future of every query
            j1 = min(j0 + block_k, Tk)
            s = (qi @ k[..., j0:j1, :].transpose(-2, -1)) * scale
            if causal:
                rows = torch.arange(i0, i1, device=q.device)[:, None]
                cols = torch.arange(j0, j1, device=q.device)[None, :]
                s = s.masked_fill(cols > rows, float("-inf"))
            m_new = torch.maximum(m, s.amax(dim=-1))
            p = torch.exp(s - m_new[..., None])
            correction = torch.exp(m - m_new)
            l = l * correction + p.sum(dim=-1)
            acc = acc * correction[..., None] + p @ v[..., j0:j1, :]
            m = m_new
        out[..., i0:i1, :] = acc / l[..., None]
        lse[..., i0:i1] = m + torch.log(l)
    # END SOLUTION
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
    # BEGIN SOLUTION
    # HINT: precompute Drow = (dout * out).sum(-1); loop over key tiles (outer) and query tiles (inner).
    d_row = (dout * out).sum(dim=-1)
    for j0 in range(0, Tk, block_k):
        j1 = min(j0 + block_k, Tk)
        kj, vj = k[..., j0:j1, :], v[..., j0:j1, :]
        for i0 in range(0, Tq, block_q):
            i1 = min(i0 + block_q, Tq)
            if causal and j0 > i1 - 1:
                continue
            qi, doi = q[..., i0:i1, :], dout[..., i0:i1, :]
            s = (qi @ kj.transpose(-2, -1)) * scale
            if causal:
                rows = torch.arange(i0, i1, device=q.device)[:, None]
                cols = torch.arange(j0, j1, device=q.device)[None, :]
                s = s.masked_fill(cols > rows, float("-inf"))
            p = torch.exp(s - lse[..., i0:i1, None])
            dv[..., j0:j1, :] += p.transpose(-2, -1) @ doi
            dp = doi @ vj.transpose(-2, -1)
            ds = p * (dp - d_row[..., i0:i1, None])
            dq[..., i0:i1, :] += ds @ kj * scale
            dk[..., j0:j1, :] += ds.transpose(-2, -1) @ qi * scale
    # END SOLUTION
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
        # BEGIN SOLUTION
        # HINT: offs = tl.arange(0, BLOCK); load with mask and other=-inf; subtract tl.max; exp; divide by tl.sum.
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(in_ptr + row * in_row_stride + offs, mask=mask, other=float("-inf"))
        x = x - tl.max(x, axis=0)
        num = tl.exp(x)
        tl.store(out_ptr + row * out_row_stride + offs, num / tl.sum(num, axis=0), mask=mask)
        # END SOLUTION

    @triton.jit
    def _flash_fwd_kernel(
        Q, K, V, O, L, stride_bh, stride_t, T, scale,
        CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    ):
        """Program (m, bh) computes BLOCK_M query rows of one (batch, head) with online softmax."""
        # BEGIN SOLUTION
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        base = pid_bh * stride_bh
        q = tl.load(Q + base + offs_m[:, None] * stride_t + offs_d[None, :], mask=offs_m[:, None] < T, other=0.0)
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
        hi = T
        if CAUSAL:
            hi = tl.minimum((pid_m + 1) * BLOCK_M, T)  # skip tiles entirely in the future
        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_mask = offs_n[:, None] < T
            k = tl.load(K + base + offs_n[:, None] * stride_t + offs_d[None, :], mask=kv_mask, other=0.0)
            v = tl.load(V + base + offs_n[:, None] * stride_t + offs_d[None, :], mask=kv_mask, other=0.0)
            s = tl.dot(q, tl.trans(k)) * scale
            valid = offs_n[None, :] < T
            if CAUSAL:
                valid = valid & (offs_m[:, None] >= offs_n[None, :])
            s = tl.where(valid, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        acc = acc / l_i[:, None]
        row_mask = offs_m < T
        tl.store(O + base + offs_m[:, None] * stride_t + offs_d[None, :], acc.to(O.dtype.element_ty), mask=row_mask[:, None])
        tl.store(L + pid_bh * T + offs_m, m_i + tl.log(l_i), mask=row_mask)
        # END SOLUTION


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
