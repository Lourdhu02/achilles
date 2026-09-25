# Exercise stub generated from solution.py by tools/make_exercises.py.
# Replace every NotImplementedError, then run: pytest labs/12_grpo
"""Lab 12 -- from REINFORCE to GRPO: unbiased policy gradients, group-relative advantages,
the clipped GRPO objective with a k3 KL penalty, and RL on a toy verifiable task.

Handout: labs/12_grpo/README.md
"""

from __future__ import annotations

import torch
import torch.nn as nn

Tensor = torch.Tensor


# ------------------------------------------------------------ bandit warm-up
def exact_policy_gradient(theta: Tensor, rewards: Tensor) -> Tensor:
    """d/dtheta E_{a ~ softmax(theta)}[r(a)] = pi * (r - E_pi[r])."""
    raise NotImplementedError("12_grpo: implement exact_policy_gradient")


def reinforce_gradient(theta: Tensor, rewards: Tensor, n: int, baseline: float = 0.0, generator=None) -> Tensor:
    """Monte-Carlo estimate: mean over samples of (r(a) - baseline) * grad log pi(a),
    where grad_theta log softmax(theta)[a] = onehot(a) - pi."""
    raise NotImplementedError("12_grpo: implement reinforce_gradient")


# --------------------------------------------------------------------- GRPO
def group_advantages(rewards: Tensor, std_norm: bool = True, eps: float = 1e-6) -> Tensor:
    """rewards (B, G) for G samples per prompt -> advantages (B, G): r - mean_group,
    divided by the group std (unbiased=False) + eps when std_norm (GRPO); not divided (Dr. GRPO)."""
    raise NotImplementedError("12_grpo: implement group_advantages")


def grpo_loss(logp_new: Tensor, logp_old: Tensor, logp_ref: Tensor, adv: Tensor, mask: Tensor,
              clip_eps: float = 0.2, beta: float = 0.04, agg: str = "token") -> Tensor:
    """Per-token log-probs (N, T), per-sequence advantages (N,), response mask (N, T).

    ratio = exp(new - old); surrogate = min(ratio A, clip(ratio, 1-eps, 1+eps) A)
    kl (k3) = exp(ref - new) - (ref - new) - 1;  per-token loss = -(surrogate - beta kl)
    agg="token": mean over all valid tokens (DAPO); agg="seq": mean over each sequence, then over sequences (GRPO).
    """
    raise NotImplementedError("12_grpo: implement grpo_loss")


# --------------------------------------------------------- toy verifiable RL
class CounterPolicy(nn.Module):
    """Autoregressive policy: next-token logits from (previous token, position)."""

    def __init__(self, vocab: int = 8, length: int = 5, d: int = 32):
        super().__init__()
        self.vocab, self.length = vocab, length
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(length, d)
        self.out = nn.Sequential(nn.Tanh(), nn.Linear(d, vocab))

    def logits(self, prev: Tensor, t: int) -> Tensor:
        return self.out(self.tok(prev) + self.pos.weight[t])

    def sample(self, prompts: Tensor, generator=None) -> Tensor:
        seqs, prev = [], prompts
        for t in range(self.length):
            probs = torch.softmax(self.logits(prev, t), -1)
            prev = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            seqs.append(prev)
        return torch.stack(seqs, 1)

    def token_logprobs(self, prompts: Tensor, seqs: Tensor) -> Tensor:
        prevs = torch.cat([prompts[:, None], seqs[:, :-1]], 1)
        out = [torch.log_softmax(self.logits(prevs[:, t], t), -1).gather(1, seqs[:, t : t + 1]) for t in range(self.length)]
        return torch.cat(out, 1)


def counting_reward(prompts: Tensor, seqs: Tensor) -> Tensor:
    """Verifiable reward: fraction of positions where seq[i] == (prompt + i + 1) mod V."""
    vocab = 8
    target = (prompts[:, None] + torch.arange(1, seqs.shape[1] + 1)) % vocab
    return (seqs == target).float().mean(1)


def train_counting(steps: int = 80, prompts_per_step: int = 16, group: int = 8, lr: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """GRPO on the counting task. Returns (mean reward at the start, mean reward at the end)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    policy = CounterPolicy()
    ref = CounterPolicy()
    ref.load_state_dict(policy.state_dict())
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []
    for _ in range(steps):
        prompts = torch.randint(0, 8, (prompts_per_step,), generator=g).repeat_interleave(group)
        with torch.no_grad():
            seqs = policy.sample(prompts, generator=g)
            old = policy.token_logprobs(prompts, seqs)
            ref_lp = ref.token_logprobs(prompts, seqs)
        rewards = counting_reward(prompts, seqs)
        history.append(rewards.mean().item())
        adv = group_advantages(rewards.view(prompts_per_step, group)).view(-1)
        for _ in range(2):  # a couple of inner epochs on the same rollouts (off-policy by one step)
            loss = grpo_loss(policy.token_logprobs(prompts, seqs), old, ref_lp, adv, torch.ones_like(old), beta=0.01)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return sum(history[:5]) / 5, sum(history[-5:]) / 5
