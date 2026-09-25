import math
import os

import pytest
import torch

# Without a GPU, run Triton kernels in its CPU interpreter. Must be set before triton is imported.
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

from labs._impl import load  # noqa: E402

ak = load(__file__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
has_triton = getattr(ak, "triton", None) is not None


def qkv(B=2, H=3, T=37, D=16, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, H, T, D, generator=g, dtype=dtype) for _ in range(3)]


# --------------------------------------------------------------- online softmax
def test_online_softmax_matches_logsumexp():
    torch.manual_seed(0)
    x = torch.randn(103, dtype=torch.float64) * 5
    m, l = ak.online_softmax_stats(list(x.split(16)))
    torch.testing.assert_close(m + torch.log(l), torch.logsumexp(x, 0))
    assert m == x.max()


def test_online_softmax_handles_growing_max():
    blocks = [torch.tensor([-50.0, -40.0]), torch.tensor([100.0]), torch.tensor([99.0, 1000.0])]
    m, l = ak.online_softmax_stats(blocks)
    torch.testing.assert_close(m + torch.log(l), torch.logsumexp(torch.cat(blocks), 0))


# ------------------------------------------------------------ flash (PyTorch)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("T, bq, bk", [(37, 16, 16), (64, 16, 32), (5, 8, 8), (33, 7, 11)])
def test_flash_forward_matches_naive(causal, T, bq, bk):
    q, k, v = qkv(T=T)
    out, lse = ak.flash_attention_forward(q, k, v, causal=causal, block_q=bq, block_k=bk)
    torch.testing.assert_close(out, ak.naive_attention(q, k, v, causal))
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if causal:
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
    torch.testing.assert_close(lse, torch.logsumexp(s, dim=-1))


def test_flash_forward_cross_attention_lengths():
    q = qkv(T=9)[0]
    _, k, v = qkv(T=40, seed=1)
    out, _ = ak.flash_attention_forward(q, k, v, block_q=4, block_k=16)
    torch.testing.assert_close(out, ak.naive_attention(q, k, v))


@pytest.mark.parametrize("causal", [False, True])
def test_flash_backward_matches_autograd(causal):
    q, k, v = (t.requires_grad_() for t in qkv(T=29))
    ref = ak.naive_attention(q, k, v, causal)
    dout = torch.randn_like(ref)
    gq, gk, gv = torch.autograd.grad(ref, (q, k, v), dout)
    with torch.no_grad():
        out, lse = ak.flash_attention_forward(q, k, v, causal)
        dq, dk, dv = ak.flash_attention_backward(q, k, v, out, lse, dout, causal)
    torch.testing.assert_close(dq, gq)
    torch.testing.assert_close(dk, gk)
    torch.testing.assert_close(dv, gv)


def test_autograd_function_end_to_end():
    q, k, v = (t.requires_grad_() for t in qkv(T=20))
    ak.flash_attention(q, k, v, causal=True).square().sum().backward()
    flash_grads = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None
    ak.naive_attention(q, k, v, causal=True).square().sum().backward()
    for got, want in zip(flash_grads, (q.grad, k.grad, v.grad)):
        torch.testing.assert_close(got, want)


# ------------------------------------------------------------------- Triton
@pytest.mark.skipif(not has_triton, reason="triton not installed")
@pytest.mark.parametrize("shape", [(4, 37), (3, 128), (1, 1000)])
def test_triton_softmax(shape):
    x = torch.randn(*shape, device=DEVICE) * 3
    torch.testing.assert_close(ak.triton_softmax(x), torch.softmax(x, dim=-1))


@pytest.mark.skipif(not has_triton, reason="triton not installed")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("T", [16, 37])
def test_triton_flash_attention(causal, T):
    q, k, v = (t.float().to(DEVICE) for t in qkv(T=T, D=32))
    out, lse = ak.triton_flash_attention(q, k, v, causal=causal)
    ref = ak.naive_attention(q, k, v, causal)
    tol = 1e-5 if DEVICE == "cpu" else 2e-3  # tf32 dot products on GPU
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)
    assert lse.shape == (2, 3, T)
