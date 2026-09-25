import math

import pytest
import torch
import torch.nn.functional as F

from labs._impl import load

tc = load(__file__)


# ---------------------------------------------------------------- init
def test_kaiming_and_xavier_statistics():
    w = tc.kaiming_normal_(torch.empty(512, 1024), fan_in=1024)
    assert abs(w.std().item() - math.sqrt(2 / 1024)) < 0.002
    w = tc.xavier_uniform_(torch.empty(512, 1024), fan_in=1024, fan_out=512)
    a = math.sqrt(6 / 1536)
    assert w.abs().max().item() <= a
    assert abs(w.var().item() - 2 / 1536) < 1e-4


def test_init_scale_decides_signal_propagation():
    kaiming = tc.activation_stds(depth=30, width=256, init="kaiming")
    standard = tc.activation_stds(depth=30, width=256, init="standard")
    small = tc.activation_stds(depth=30, width=256, init="small")
    assert 0.3 < kaiming[-1] < 3.0, "Kaiming init should keep activations O(1) through 30 ReLU layers"
    assert standard[-1] > 1e10, "N(0,1) weights should explode"
    assert small[-1] < 1e-10, "N(0,0.01) weights should vanish"


# ---------------------------------------------------------------- norms
def test_layernorm_matches_torch():
    torch.manual_seed(0)
    x = torch.randn(4, 7, 32) * 3 + 1
    ln = tc.LayerNorm(32)
    with torch.no_grad():
        ln.weight.uniform_(0.5, 1.5)
        ln.bias.uniform_(-1, 1)
    ref = F.layer_norm(x, (32,), ln.weight, ln.bias, eps=1e-5)
    torch.testing.assert_close(ln(x), ref)


def test_rmsnorm_matches_reference_and_is_scale_invariant():
    torch.manual_seed(0)
    x = torch.randn(4, 7, 32)
    rms = tc.RMSNorm(32, eps=1e-6)
    with torch.no_grad():
        rms.weight.uniform_(0.5, 1.5)
    ref = x / torch.sqrt((x**2).mean(-1, keepdim=True) + 1e-6) * rms.weight
    torch.testing.assert_close(rms(x), ref)
    torch.testing.assert_close(rms(10 * x), rms(x), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------- losses
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("smoothing", [0.0, 0.1])
def test_cross_entropy_matches_torch(reduction, smoothing):
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 11, requires_grad=True)
    targets = torch.randint(0, 11, (3, 5))
    targets[0, :2] = -100  # masked prompt tokens
    ours = tc.cross_entropy(logits, targets, label_smoothing=smoothing, reduction=reduction)
    ref = F.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1), label_smoothing=smoothing, reduction=reduction)
    torch.testing.assert_close(ours, ref)
    g1 = torch.autograd.grad(ours.sum(), logits)[0]
    g2 = torch.autograd.grad(ref.sum(), logits)[0]
    torch.testing.assert_close(g1, g2)


def test_cross_entropy_is_stable_for_huge_logits():
    logits = torch.tensor([[1e4, 0.0, -1e4]])
    assert torch.isfinite(tc.cross_entropy(logits, torch.tensor([2])))


# ---------------------------------------------------------------- AdamW
def _trajectory(opt_cls, steps=25, **kw):
    torch.manual_seed(0)
    w = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    x = torch.randn(32, 6, dtype=torch.float64)
    y = torch.randn(32, 4, dtype=torch.float64)
    opt = opt_cls([w], **kw)
    for _ in range(steps):
        opt.zero_grad()
        ((x @ w - y) ** 2).mean().backward()
        opt.step()
    return w.detach()


def test_adamw_matches_torch_exactly():
    kw = dict(lr=3e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    torch.testing.assert_close(_trajectory(tc.AdamW, **kw), _trajectory(torch.optim.AdamW, **kw), rtol=1e-10, atol=1e-12)


def test_weight_decay_is_decoupled():
    p = torch.nn.Parameter(torch.full((3,), 2.0))
    opt = tc.AdamW([p], lr=0.1, weight_decay=0.5)
    p.grad = torch.zeros(3)  # zero loss gradient: only weight decay acts
    opt.step()
    torch.testing.assert_close(p.detach(), torch.full((3,), 2.0 * (1 - 0.1 * 0.5)))


def test_first_adam_step_has_magnitude_lr():
    # Bias correction makes the first update exactly lr * sign(g) (up to eps).
    p = torch.nn.Parameter(torch.zeros(4))
    opt = tc.AdamW([p], lr=0.01, weight_decay=0.0)
    p.grad = torch.tensor([1e-5, -3.0, 50.0, -0.2])
    opt.step()
    torch.testing.assert_close(p.detach(), -0.01 * torch.sign(p.grad), atol=1e-4, rtol=0)


# ---------------------------------------------------------------- Muon
@pytest.mark.parametrize("shape", [(32, 64), (64, 32), (48, 48)])
def test_newton_schulz_cubic_converges_to_polar_factor(shape):
    torch.manual_seed(0)
    G = torch.randn(*shape, dtype=torch.float64)
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    X = tc.newton_schulz(G, steps=40, coefficients=(1.5, -0.5, 0.0))
    torch.testing.assert_close(X, U @ Vh, atol=1e-6, rtol=0)


def test_newton_schulz_quintic_is_approximately_orthogonal():
    torch.manual_seed(0)
    X = tc.newton_schulz(torch.randn(64, 128, dtype=torch.float64), steps=5)
    sv = torch.linalg.svdvals(X)
    assert sv.min() > 0.5 and sv.max() < 1.25


def test_muon_reduces_loss():
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.randn(16, 32) * 0.1)
    target = torch.randn(16, 32)
    opt = tc.Muon([W], lr=0.1)
    start = ((W - target) ** 2).mean().item()
    for _ in range(50):
        opt.zero_grad()
        ((W - target) ** 2).mean().backward()
        opt.step()
    assert ((W - target) ** 2).mean().item() < 0.2 * start


# ---------------------------------------------------------------- schedules
def test_cosine_schedule_shape():
    kw = dict(max_lr=1.0, min_lr=0.1, warmup_steps=10, total_steps=110)
    assert tc.lr_cosine(0, **kw) == pytest.approx(0.1)
    assert tc.lr_cosine(9, **kw) == pytest.approx(1.0)
    assert tc.lr_cosine(10, **kw) == pytest.approx(1.0)
    assert tc.lr_cosine(60, **kw) == pytest.approx(0.55)
    assert tc.lr_cosine(110, **kw) == pytest.approx(0.1)
    assert tc.lr_cosine(500, **kw) == pytest.approx(0.1)
    lrs = [tc.lr_cosine(s, **kw) for s in range(10, 111)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:])), "decay must be monotone"


def test_wsd_schedule_shape():
    kw = dict(max_lr=1.0, min_lr=0.0, warmup_steps=10, total_steps=100, decay_frac=0.2)
    assert tc.lr_wsd(5, **kw) == pytest.approx(0.6)
    assert tc.lr_wsd(50, **kw) == pytest.approx(1.0)
    assert tc.lr_wsd(79, **kw) == pytest.approx(1.0)
    assert tc.lr_wsd(90, **kw) == pytest.approx(0.5)
    assert tc.lr_wsd(100, **kw) == pytest.approx(0.0)


# ---------------------------------------------------------------- clipping
def test_clip_grad_norm_matches_torch():
    torch.manual_seed(0)
    a = [torch.nn.Parameter(torch.randn(5, 3)), torch.nn.Parameter(torch.randn(7))]
    b = [torch.nn.Parameter(t.detach().clone()) for t in a]
    for pa, pb in zip(a, b):
        pa.grad = torch.randn_like(pa) * 10
        pb.grad = pa.grad.clone()
    n1 = tc.clip_grad_norm_(a, 1.0)
    n2 = torch.nn.utils.clip_grad_norm_(b, 1.0)
    torch.testing.assert_close(n1, n2)
    for pa, pb in zip(a, b):
        torch.testing.assert_close(pa.grad, pb.grad)


def test_clip_is_noop_below_threshold():
    p = torch.nn.Parameter(torch.zeros(3))
    p.grad = torch.tensor([0.1, 0.2, 0.2])
    tc.clip_grad_norm_([p], 5.0)
    torch.testing.assert_close(p.grad, torch.tensor([0.1, 0.2, 0.2]))


# ---------------------------------------------------- gradient accumulation
def test_accumulation_equals_full_batch_with_uneven_masks():
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 5)
    xs = [torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)]
    ys = [torch.randint(0, 5, (4,)) for _ in xs]
    ys[0][:3] = -100  # micro-batch 0 has 1 valid token, the others have 4
    tc.accumulate_gradients(model, list(zip(xs, ys)))
    accumulated = [p.grad.clone() for p in model.parameters()]

    model.zero_grad()
    F.cross_entropy(model(torch.cat(xs)), torch.cat(ys)).backward()
    for acc, full in zip(accumulated, [p.grad for p in model.parameters()]):
        torch.testing.assert_close(acc, full)
