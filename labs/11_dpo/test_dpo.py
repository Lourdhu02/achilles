import math

import torch

from labs._impl import load

dpo = load(__file__)


def test_sequence_logprobs_shift_and_mask():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    labels = torch.randint(0, 7, (2, 5))
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])
    logp = torch.log_softmax(logits, -1)
    expected = torch.stack([
        sum(logp[b, t - 1, labels[b, t]] for t in range(1, 5) if mask[b, t]) for b in range(2)
    ])
    torch.testing.assert_close(dpo.sequence_logprobs(logits, labels, mask), expected)
    torch.testing.assert_close(dpo.sequence_logprobs(logits, labels, mask, average=True), expected / torch.tensor([3.0, 2.0]))


def test_dpo_loss_values():
    t = lambda *v: torch.tensor(v)  # noqa: E731
    loss, rc, rr = dpo.dpo_loss(t(-1.0), t(-3.0), t(-2.0), t(-2.0), beta=0.5)
    torch.testing.assert_close(loss, torch.tensor(-math.log(1 / (1 + math.exp(-1.0)))))
    torch.testing.assert_close(rc, t(0.5))
    torch.testing.assert_close(rr, t(-0.5))
    at_ref, _, _ = dpo.dpo_loss(t(-1.0), t(-1.0), t(-1.0), t(-1.0))
    torch.testing.assert_close(at_ref, torch.tensor(math.log(2)))


def test_gradient_pushes_chosen_up_and_rejected_down():
    c, r = torch.tensor([-2.0], requires_grad=True), torch.tensor([-2.0], requires_grad=True)
    loss, _, _ = dpo.dpo_loss(c, r, torch.tensor([-2.0]), torch.tensor([-2.0]))
    loss.backward()
    assert c.grad < 0 and r.grad > 0


def test_implicit_rewards_are_detached():
    c = torch.tensor([-1.0], requires_grad=True)
    r = torch.tensor([-2.0], requires_grad=True)
    _, rc, rr = dpo.dpo_loss(c, r, torch.tensor([-1.0]), torch.tensor([-2.0]))
    assert not rc.requires_grad
    assert not rr.requires_grad


def test_ipo_and_simpo():
    t = lambda *v: torch.tensor(v)  # noqa: E731
    torch.testing.assert_close(dpo.ipo_loss(t(0.0), t(0.0), t(0.0), t(0.0), beta=0.5), t(1.0)[0])
    torch.testing.assert_close(dpo.simpo_loss(t(-1.0), t(-1.25), beta=2.0, gamma=0.5), torch.tensor(math.log(2)))


def test_dpo_converges_to_the_closed_form_optimum():
    torch.manual_seed(0)
    rewards = torch.randn(6)
    ref = torch.log_softmax(torch.randn(6), 0)
    beta = 0.5
    learned = dpo.tabular_dpo(rewards, ref, beta)
    theory = torch.log_softmax(ref + rewards / beta, 0)
    torch.testing.assert_close(learned, theory, atol=2e-2, rtol=0)
