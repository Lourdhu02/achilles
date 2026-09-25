# Exercise stub generated from solution.py by tools/make_exercises.py.
# Replace every NotImplementedError, then run: pytest labs/02_training_core
"""Lab 02 -- the training toolkit from scratch: init, norms, losses, AdamW, Muon,
LR schedules, gradient clipping and correct gradient accumulation.

Everything here also exists in torch.nn / torch.optim. You re-implement it using
only tensor ops and autograd; the tests compare against PyTorch.

Handout: labs/02_training_core/README.md
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

import torch
import torch.nn as nn

Tensor = torch.Tensor


# ------------------------------------------------------------ initialization
def kaiming_normal_(w: Tensor, fan_in: int, gain: float = math.sqrt(2.0)) -> Tensor:
    """In place: w ~ N(0, gain^2 / fan_in). gain = sqrt(2) preserves variance through ReLU."""
    raise NotImplementedError("02_training_core: implement kaiming_normal_")


def xavier_uniform_(w: Tensor, fan_in: int, fan_out: int) -> Tensor:
    """In place: w ~ U(-a, a) with Var = 2 / (fan_in + fan_out)."""
    # HINT: Var(U(-a, a)) = a^2 / 3
    raise NotImplementedError("02_training_core: implement xavier_uniform_")


def activation_stds(depth: int, width: int, init: str, batch: int = 512, seed: int = 0) -> list[float]:
    """Std of each layer's output in a bias-free ReLU MLP at initialization.

    init: "kaiming" (N(0, 2/width)), "standard" (N(0, 1)) or "small" (N(0, 0.01^2)).
    """
    torch.manual_seed(seed)
    x = torch.randn(batch, width)
    stds = []
    for _ in range(depth):
        w = torch.empty(width, width)
        if init == "kaiming":
            kaiming_normal_(w, fan_in=width)
        elif init == "standard":
            w.normal_(0.0, 1.0)
        elif init == "small":
            w.normal_(0.0, 0.01)
        else:
            raise ValueError(init)
        x = torch.relu(x @ w)
        stds.append(x.std().item())
    return stds


# --------------------------------------------------------------- normalization
class LayerNorm(nn.Module):
    """y = (x - mean) / sqrt(var + eps) * weight + bias, statistics over the last dim."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        # HINT: use the biased variance (divide by N, not N-1), like PyTorch.
        raise NotImplementedError("02_training_core: implement forward")


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight. No centering, no bias (Llama, Qwen, Mistral)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("02_training_core: implement forward")


# ---------------------------------------------------------------------- losses
def cross_entropy(
    logits: Tensor,
    targets: Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> Tensor:
    """Cross-entropy over the last dim of ``logits`` (..., C) for integer ``targets`` (...).

    Positions where targets == ignore_index contribute nothing, and "mean" divides by the
    number of non-ignored positions. This is exactly how SFT masks prompt tokens.
    Matches torch.nn.functional.cross_entropy for reduction in {"mean", "sum", "none"}.
    """
    # HINT: logp = logits - logsumexp(logits); gather the target column; mask ignored positions.
    raise NotImplementedError("02_training_core: implement cross_entropy")


# ------------------------------------------------------------------ optimizers
class AdamW(torch.optim.Optimizer):
    """Adam with decoupled weight decay (Loshchilov & Hutter). Must match torch.optim.AdamW."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                # HINT: decay the weights first (p *= 1 - lr*wd), then the bias-corrected Adam step.
                raise NotImplementedError("02_training_core: implement step")
        return loss


def newton_schulz(G: Tensor, steps: int = 5, coefficients=(3.4445, -4.7750, 2.0315), eps: float = 1e-7) -> Tensor:
    """Approximate the orthogonal polar factor U V^T of G (G = U S V^T) without an SVD.

    Iterates X <- a X + (b A + c A^2) X with A = X X^T, after scaling G so its spectral
    norm is <= 1. The default (quintic) coefficients are Muon's: after 5 steps the
    singular values land roughly in [0.6, 1.2] -- fast, not exact. The cubic choice
    (1.5, -0.5, 0.0) converges to U V^T exactly, just more slowly.
    """
    # HINT: normalize by the Frobenius norm; transpose tall matrices so A = X X^T is small.
    raise NotImplementedError("02_training_core: implement newton_schulz")


class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton-Schulz, for 2-D weight matrices only.

    update = NS(momentum) * sqrt(max(1, rows / cols)); p <- p(1 - lr*wd) - lr * update.
    Use AdamW for embeddings, the output head, norms and biases.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Muon is for 2-D weights; use AdamW for the rest")
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(p)
                raise NotImplementedError("02_training_core: implement step")
        return loss


# ------------------------------------------------------------------- schedules
def lr_cosine(step: int, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup over steps [0, warmup) reaching max_lr at step warmup-1, then cosine
    decay from max_lr (at step = warmup) to min_lr (at step = total_steps), constant after."""
    raise NotImplementedError("02_training_core: implement lr_cosine")


def lr_wsd(step: int, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int, decay_frac: float = 0.2) -> float:
    """Warmup-Stable-Decay: linear warmup, flat at max_lr, then a linear decay to min_lr over
    the final ``decay_frac`` of training. The stable phase can be extended indefinitely."""
    raise NotImplementedError("02_training_core: implement lr_wsd")


# --------------------------------------------------------------------- clipping
def clip_grad_norm_(params: Iterable[Tensor], max_norm: float) -> Tensor:
    """Scale all grads in place so their global L2 norm is at most max_norm.

    Returns the total norm *before* clipping. Matches torch.nn.utils.clip_grad_norm_,
    which scales by max_norm / (total_norm + 1e-6) when that is < 1.
    """
    raise NotImplementedError("02_training_core: implement clip_grad_norm_")


# --------------------------------------------------------- gradient accumulation
def accumulate_gradients(
    model: nn.Module,
    micro_batches: list[tuple[Tensor, Tensor]],
    ignore_index: int = -100,
) -> Tensor:
    """Accumulate grads over micro-batches so they EQUAL the full-batch gradient of the
    mean token loss, even when micro-batches hold different numbers of valid tokens.

    ``model(x)`` returns logits (..., C); targets use ``ignore_index`` for padding.
    Returns the full-batch mean loss (detached). The classic bug is averaging the
    per-micro-batch *means*, which over-weights micro-batches with few tokens.
    """
    # HINT: count valid tokens across ALL micro-batches first; backprop sum-loss / total.
    raise NotImplementedError("02_training_core: implement accumulate_gradients")
