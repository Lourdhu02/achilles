"""Lab 10 -- LoRA: low-rank adapters on frozen linear layers.

Handout: labs/10_lora/README.md
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """y = base(x) + (alpha / r) * x A^T B^T with base frozen, A (r, in) Kaiming, B (out, r) zeros."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BEGIN SOLUTION
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
        # END SOLUTION

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """A plain nn.Linear with W + scale * B A (zero inference overhead)."""
        # BEGIN SOLUTION
        merged = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        merged.weight.copy_(self.base.weight + self.scale * self.B @ self.A)
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged
        # END SOLUTION


def apply_lora(model: nn.Module, targets: tuple[str, ...], r: int = 8, alpha: float = 16.0) -> nn.Module:
    """Replace every nn.Linear whose attribute name is in ``targets`` with a LoRALinear, in place.
    All other parameters are frozen."""
    # BEGIN SOLUTION
    for p in model.parameters():
        p.requires_grad_(False)
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(module, name, LoRALinear(child, r, alpha))
    return model
    # END SOLUTION


def trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fit_low_rank_delta(r: int, true_rank: int = 2, dim: int = 16, steps: int = 600, seed: int = 0) -> float:
    """Frozen base W0; the target is W0 + a rank-`true_rank` delta. Train LoRA(r) and return the final MSE."""
    g = torch.Generator().manual_seed(seed)
    base = nn.Linear(dim, dim, bias=False)
    with torch.no_grad():
        base.weight.copy_(torch.randn(dim, dim, generator=g) / math.sqrt(dim))
    delta = torch.randn(dim, true_rank, generator=g) @ torch.randn(true_rank, dim, generator=g) / math.sqrt(dim)
    target_w = base.weight.detach() + delta
    layer = LoRALinear(base, r=r, alpha=r)
    opt = torch.optim.Adam([layer.A, layer.B], lr=2e-2)
    x = torch.randn(512, dim, generator=g)
    y = x @ target_w.T
    for _ in range(steps):
        loss = ((layer(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()
