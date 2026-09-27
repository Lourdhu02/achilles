# Exercise stub generated from solution.py by tools/make_exercises.py.
# Replace every NotImplementedError, then run: pytest labs/18_moe
"""Lab 18 -- mixture of experts: top-k routing, load balancing, capacity and token dropping.

Handout: labs/18_moe/README.md
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# ------------------------------------------------------------------ routing
def route(logits: Tensor, k: int, renormalize: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    """logits (T, E) -> (weights (T, k), experts (T, k), probs (T, E)).

    probs is the softmax over all experts (the load-balancing loss needs it). experts holds each
    token's k highest-scoring experts, best first. weights are their probs, renormalized to sum
    to 1 per token when `renormalize` (Mixtral), or left as they are (Switch Transformer, k = 1).
    """
    raise NotImplementedError("18_moe: implement route")


def load_balancing_loss(probs: Tensor, experts: Tensor, n_experts: int) -> Tensor:
    """Switch Transformer auxiliary loss E * sum_i f_i * P_i.

    f_i: fraction of the T * k (token, slot) assignments that went to expert i (no gradient).
    P_i: mean router probability of expert i over the T tokens (carries the gradient).
    Equals 1 when routing is uniform and grows to E when every token picks the same expert (k = 1).
    """
    raise NotImplementedError("18_moe: implement load_balancing_loss")


def router_z_loss(logits: Tensor) -> Tensor:
    """ST-MoE router z-loss: mean over tokens of logsumexp(logits)^2. Keeps router logits small."""
    raise NotImplementedError("18_moe: implement router_z_loss")


# ----------------------------------------------------------------- capacity
def expert_capacity(n_tokens: int, n_experts: int, k: int, capacity_factor: float) -> int:
    """Slots per expert: ceil(capacity_factor * n_tokens * k / n_experts)."""
    return math.ceil(capacity_factor * n_tokens * k / n_experts)


def capacity_mask(experts: Tensor, n_experts: int, capacity: int) -> Tensor:
    """experts (T, k) -> keep (T, k) bool. Each expert accepts at most `capacity` assignments.

    Priority: every token's first choice is placed before any token's second choice (GShard), and
    within one choice rank, earlier tokens go first. Assignments beyond capacity are dropped.
    """
    # HINT: flatten rank-major with experts.T.reshape(-1), one-hot it, cumsum down the rows to get
    # HINT: each assignment's position inside its expert, keep position < capacity, reshape back.
    raise NotImplementedError("18_moe: implement capacity_mask")


# ------------------------------------------------------------------ the layer
class MoE(nn.Module):
    """Token-choice top-k MoE feed-forward layer: y = sum over kept (expert, weight) of weight * expert(x).

    Experts are two-layer GELU MLPs stored as stacked tensors w1 (E, d, d_ff) and w2 (E, d_ff, d).
    With a capacity factor, assignments beyond an expert's capacity are dropped; a token whose
    assignments are all dropped gets zero output here and passes through the residual connection.
    """

    def __init__(self, d: int, d_ff: int, n_experts: int, k: int = 2, capacity_factor: float | None = None):
        super().__init__()
        self.n_experts, self.k, self.capacity_factor = n_experts, k, capacity_factor
        self.router = nn.Linear(d, n_experts, bias=False)
        self.w1 = nn.Parameter(torch.randn(n_experts, d, d_ff) / math.sqrt(d))
        self.w2 = nn.Parameter(torch.randn(n_experts, d_ff, d) / math.sqrt(d_ff))

    def expert(self, e: int, x: Tensor) -> Tensor:
        return F.gelu(x @ self.w1[e]) @ self.w2[e]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """x (T, d) -> (y (T, d), load-balancing loss)."""
        # HINT: route, build the keep mask if capacity_factor is set, then loop over experts: gather
        # HINT: the (token, slot) pairs routed to e, run the expert once on those tokens, and
        # HINT: index_add_ weight * output back into y. Return y and load_balancing_loss.
        raise NotImplementedError("18_moe: implement forward")


def moe_param_counts(d: int, d_ff: int, n_experts: int, k: int, n_shared: int = 0) -> tuple[int, int]:
    """(total, active-per-token) parameters of one MoE feed-forward layer, router included.

    Each expert (routed or shared) is a two-matrix MLP with 2 * d * d_ff weights; the router is d * n_experts.
    Shared experts (DeepSeekMoE) process every token, so they count as active.
    """
    raise NotImplementedError("18_moe: implement moe_param_counts")


def train_router_for_balance(steps: int = 300, n_experts: int = 8, d: int = 16, seed: int = 0) -> tuple[float, float]:
    """Start a router that sends most tokens to expert 0, train it on the load-balancing loss alone,
    and return the largest expert's share of tokens (top-1) before and after."""
    torch.manual_seed(seed)
    x = torch.randn(1024, d)
    router = nn.Linear(d, n_experts)
    with torch.no_grad():
        router.bias.zero_()
        router.bias[0] = 4.0                                   # skewed start: expert 0 wins most tokens
    opt = torch.optim.Adam(router.parameters(), lr=0.05)

    def max_share() -> float:
        with torch.no_grad():
            top1 = router(x).argmax(dim=-1)
        return torch.bincount(top1, minlength=n_experts).max().item() / x.shape[0]

    before = max_share()
    for _ in range(steps):
        _, experts, probs = route(router(x), k=1)
        loss = load_balancing_loss(probs, experts, n_experts)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return before, max_share()
