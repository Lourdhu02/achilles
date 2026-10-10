# Exercise stub generated from solution.py by tools/make_exercises.py.
# Replace every NotImplementedError, then run: pytest labs/11_dpo
"""Lab 11 -- DPO and friends: sequence log-probs, DPO/IPO/SimPO losses, and a numerical check of the theory.

Handout: labs/11_dpo/README.md
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def sequence_logprobs(logits: Tensor, labels: Tensor, mask: Tensor, average: bool = False) -> Tensor:
    """Sum (or mean) of log p(labels[t] | < t) over positions where mask[t] == 1.

    logits (B, T, V) predict the NEXT token, so logits[:, t] scores labels[:, t+1].
    """
    raise NotImplementedError("11_dpo: implement sequence_logprobs")


def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta: float = 0.1, label_smoothing=0.0):
    """Returns (mean loss, chosen implicit rewards, rejected implicit rewards).

    h = beta * [(pi_c - ref_c) - (pi_r - ref_r)];  loss = -(1-eps) log sig(h) - eps log sig(-h).
    Implicit reward = beta * (log pi - log ref).
    """
    raise NotImplementedError("11_dpo: implement dpo_loss")


def ipo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta: float = 0.1) -> Tensor:
    """IPO: regress the log-ratio margin to 1/(2 beta)."""
    raise NotImplementedError("11_dpo: implement ipo_loss")


def simpo_loss(avg_chosen, avg_rejected, beta: float = 2.0, gamma: float = 0.5) -> Tensor:
    """SimPO: reference-free, length-normalized log-probs with a target margin gamma."""
    raise NotImplementedError("11_dpo: implement simpo_loss")


def tabular_dpo(rewards: Tensor, ref_logp: Tensor, beta: float, steps: int = 3000, lr: float = 0.05) -> Tensor:
    """Train a tabular policy over K responses with DPO on ALL ordered pairs, using Bradley-Terry soft
    labels (label_smoothing = 1 - sigmoid(r_i - r_j)). Returns the learned log-probs.
    Theory: the optimum is log pi* = ref_logp + r / beta - log Z."""
    K = rewards.numel()
    theta = ref_logp.clone().requires_grad_()
    i, j = torch.meshgrid(torch.arange(K), torch.arange(K), indexing="ij")
    keep = i != j
    i, j = i[keep], j[keep]
    eps = 1 - torch.sigmoid(rewards[i] - rewards[j])
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        logp = torch.log_softmax(theta, 0)
        loss, _, _ = dpo_loss(logp[i], logp[j], ref_logp[i], ref_logp[j], beta, eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.log_softmax(theta.detach(), 0)
