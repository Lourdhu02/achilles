import torch

from labs._impl import load

rl = load(__file__)


def test_reinforce_is_unbiased_and_baseline_reduces_variance():
    theta = torch.tensor([0.5, -1.0, 0.2, 1.0])
    r = torch.tensor([1.0, 0.0, 3.0, 0.5])
    exact = rl.exact_policy_gradient(theta, r)
    g = torch.Generator().manual_seed(0)
    est = rl.reinforce_gradient(theta, r, n=200_000, generator=g)
    torch.testing.assert_close(est, exact, atol=0.01, rtol=0)
    mean_r = (torch.softmax(theta, 0) * r).sum().item()
    no_base = torch.stack([rl.reinforce_gradient(theta, r, 32, 0.0, g) for _ in range(300)])
    with_base = torch.stack([rl.reinforce_gradient(theta, r, 32, mean_r, g) for _ in range(300)])
    assert with_base.var(0).sum() < 0.5 * no_base.var(0).sum()


def test_group_advantages():
    r = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    adv = rl.group_advantages(r)
    torch.testing.assert_close(adv[0], torch.tensor([1.0, -1.0, -1.0, 1.0]), atol=1e-4, rtol=0)
    assert torch.all(adv[1] == 0)  # all-correct groups carry no signal
    torch.testing.assert_close(rl.group_advantages(r, std_norm=False)[0], torch.tensor([0.5, -0.5, -0.5, 0.5]))


def test_grpo_loss_clipping_and_kl():
    lp = torch.zeros(1, 3)
    mask = torch.ones(1, 3)
    # identical policies, A = 1: loss = -1
    torch.testing.assert_close(rl.grpo_loss(lp, lp, lp, torch.tensor([1.0]), mask), torch.tensor(-1.0))
    # ratio already above 1 + eps with A > 0: the clipped objective gives zero gradient
    new = torch.full((1, 3), 0.5, requires_grad=True)
    rl.grpo_loss(new, lp, new.detach(), torch.tensor([1.0]), mask).backward()
    assert torch.all(new.grad == 0)
    # k3 KL is >= 0 and zero only at equality
    kl_loss = rl.grpo_loss(lp, lp, torch.full((1, 3), 0.7), torch.tensor([0.0]), mask, beta=1.0)
    assert kl_loss > 0


def test_aggregation_modes_differ_with_uneven_lengths():
    lp = torch.zeros(2, 4)
    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]])
    adv = torch.tensor([1.0, -1.0])
    token = rl.grpo_loss(lp, lp, lp, adv, mask, agg="token")
    seq = rl.grpo_loss(lp, lp, lp, adv, mask, agg="seq")
    torch.testing.assert_close(token, torch.tensor(-3 / 5))
    torch.testing.assert_close(seq, torch.tensor(0.0))


def test_grpo_learns_the_counting_task():
    start, end = rl.train_counting()
    assert start < 0.3
    assert end > 0.7, f"reward only reached {end:.2f}"
