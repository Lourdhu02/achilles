import torch
import torch.nn as nn

from labs._impl import load

lo = load(__file__)


def test_identity_at_init_and_only_adapters_train():
    torch.manual_seed(0)
    base = nn.Linear(12, 7)
    layer = lo.LoRALinear(base, r=4, alpha=8)
    x = torch.randn(3, 12)
    torch.testing.assert_close(layer(x), base(x))
    assert [n for n, p in layer.named_parameters() if p.requires_grad] == ["A", "B"]


def test_merge_matches_adapter():
    torch.manual_seed(0)
    layer = lo.LoRALinear(nn.Linear(12, 7), r=4, alpha=8)
    with torch.no_grad():
        layer.B.normal_()
    x = torch.randn(5, 12)
    torch.testing.assert_close(layer.merge()(x), layer(x), atol=1e-5, rtol=1e-5)


def test_apply_lora_and_param_count():
    model = nn.Sequential()
    model.q = nn.Linear(32, 32)
    model.v = nn.Linear(32, 32)
    model.mlp = nn.Linear(32, 64)
    lo.apply_lora(model, targets=("q", "v"), r=4)
    assert isinstance(model.q, lo.LoRALinear) and isinstance(model.mlp, nn.Linear)
    assert lo.trainable_params(model) == 2 * 4 * (32 + 32)


def test_low_rank_hypothesis():
    exact = lo.fit_low_rank_delta(r=2)
    too_small = lo.fit_low_rank_delta(r=1)
    assert exact < 1e-3
    assert too_small > 20 * exact
