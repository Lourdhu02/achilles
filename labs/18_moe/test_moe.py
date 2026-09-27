import math

import pytest
import torch

from labs._impl import load

moe = load(__file__)


def test_routing_picks_top_k_and_weights_sum_to_one():
    torch.manual_seed(0)
    logits = torch.randn(6, 8)
    weights, experts, probs = moe.route(logits, k=2)
    torch.testing.assert_close(probs, logits.softmax(-1))
    assert torch.equal(experts, logits.topk(2, dim=-1).indices)
    torch.testing.assert_close(weights.sum(-1), torch.ones(6))


def test_renormalized_top_k_equals_softmax_over_the_selected_logits():
    torch.manual_seed(1)
    logits = torch.randn(5, 8)
    weights, experts, _ = moe.route(logits, k=2)
    torch.testing.assert_close(weights, logits.gather(-1, experts).softmax(-1))


def test_top1_gate_needs_the_raw_probability_to_train_the_router():
    torch.manual_seed(2)
    logits = torch.randn(4, 8, requires_grad=True)
    weights, _, _ = moe.route(logits, k=1, renormalize=True)
    assert torch.allclose(weights, torch.ones_like(weights))      # renormalized top-1: always exactly 1
    weights, _, _ = moe.route(logits, k=1, renormalize=False)
    weights.sum().backward()
    assert logits.grad.abs().sum() > 0                             # the raw probability carries a gradient


def test_load_balancing_loss_is_one_when_uniform_and_E_when_collapsed():
    E, T = 8, 64
    uniform_probs = torch.full((T, E), 1 / E)
    round_robin = torch.arange(T).remainder(E)[:, None]
    assert moe.load_balancing_loss(uniform_probs, round_robin, E).item() == pytest.approx(1.0)
    collapsed_probs = torch.zeros(T, E)
    collapsed_probs[:, 0] = 1.0
    all_to_zero = torch.zeros(T, 1, dtype=torch.long)
    assert moe.load_balancing_loss(collapsed_probs, all_to_zero, E).item() == pytest.approx(E)


def test_load_balancing_loss_trains_a_skewed_router_toward_balance():
    before, after = moe.train_router_for_balance()
    assert before > 0.9 and after < 0.25, (before, after)          # perfect balance is 1/8


def test_router_z_loss_matches_definition():
    torch.manual_seed(3)
    logits = torch.randn(7, 5)
    expected = torch.logsumexp(logits, -1).pow(2).mean()
    torch.testing.assert_close(moe.router_z_loss(logits), expected)


def test_capacity_gives_first_choices_priority_over_second_choices():
    assert moe.expert_capacity(64, 8, 2, 1.25) == 20
    experts = torch.tensor([[1, 0], [0, 1], [0, 1]])               # (T=3, k=2)
    keep = moe.capacity_mask(experts, n_experts=2, capacity=2)
    # expert 0 takes the first choices of tokens 1 and 2 before token 0's second choice
    assert keep.tolist() == [[True, False], [True, True], [True, False]]


def _brute_force(layer, x):
    weights, experts, _ = moe.route(layer.router(x), layer.k, renormalize=layer.k > 1)
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for s in range(layer.k):
            out[t] += weights[t, s] * layer.expert(int(experts[t, s]), x[t : t + 1])[0]
    return out


def test_moe_forward_matches_a_per_token_reference():
    torch.manual_seed(4)
    layer = moe.MoE(d=8, d_ff=16, n_experts=4, k=2)
    x = torch.randn(10, 8)
    y, aux = layer(x)
    torch.testing.assert_close(y, _brute_force(layer, x), atol=1e-5, rtol=1e-5)
    assert aux.requires_grad                                       # the gradient flows through P, not f


def test_dropped_assignments_contribute_nothing():
    torch.manual_seed(5)
    layer = moe.MoE(d=8, d_ff=16, n_experts=4, k=1, capacity_factor=0.25)   # capacity = 1 slot per expert
    x = torch.randn(12, 8)
    y, _ = layer(x)
    kept_tokens = (y.abs().sum(-1) > 0).sum().item()
    assert kept_tokens <= 4                                         # at most one token per expert survives


def test_param_counts_total_vs_active():
    total, active = moe.moe_param_counts(d=4096, d_ff=14336, n_experts=8, k=2)
    per_expert = 2 * 4096 * 14336
    assert total == 8 * per_expert + 4096 * 8
    assert active == 2 * per_expert + 4096 * 8
    total, active = moe.moe_param_counts(d=1024, d_ff=512, n_experts=64, k=6, n_shared=2)
    assert active == (6 + 2) * 2 * 1024 * 512 + 1024 * 64
    assert math.isclose(total / active, (66 * 2 * 1024 * 512 + 65536) / (8 * 2 * 1024 * 512 + 65536))
