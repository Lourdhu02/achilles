import numpy as np
import pytest

from labs._impl import load

ag = load(__file__)
Tensor = ag.Tensor
RNG = np.random.default_rng(0)


def finite_diff(f, x, eps=1e-6):
    """Reference central differences (independent of your numerical_gradient)."""
    x = x.astype(np.float64).copy()
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + eps
        fp = f(x)
        x[i] = old - eps
        fm = f(x)
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


def check_vjp(fn, *arrays, rtol=1e-5, atol=1e-6):
    """Backprop a random upstream gradient G through fn and compare every input's
    gradient with finite differences of sum(fn(inputs) * G)."""
    tensors = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    out = fn(*tensors)
    G = RNG.normal(size=out.shape)
    out.backward(G)
    for i, a in enumerate(arrays):

        def scalar(x, i=i):
            args = [Tensor(x if j == i else arrays[j]) for j in range(len(arrays))]
            return float(np.sum(fn(*args).data * G))

        assert tensors[i].grad is not None, f"input {i} received no gradient"
        np.testing.assert_allclose(tensors[i].grad, finite_diff(scalar, a), rtol=rtol, atol=atol)


def randn(*shape):
    return RNG.normal(size=shape)


def positive(*shape):
    return RNG.uniform(0.5, 2.0, size=shape)


# ---------------------------------------------------------------- unbroadcast
@pytest.mark.parametrize(
    "grad_shape, shape",
    [((3, 4), (3, 4)), ((3, 4), (4,)), ((3, 4), (1, 4)), ((3, 4), (3, 1)), ((2, 3, 4), (4,)), ((2, 3, 4), (3, 1))],
)
def test_unbroadcast_matches_sum(grad_shape, shape):
    g = randn(*grad_shape)
    out = ag.unbroadcast(g, shape)
    assert out.shape == shape
    # Summing the reduced gradient must equal summing the original one.
    np.testing.assert_allclose(out.sum(), g.sum())


# -------------------------------------------------------------------- forward
def test_forward_matches_numpy():
    a, b = randn(3, 4), randn(4, 5)
    p = positive(3, 4)
    np.testing.assert_allclose((Tensor(a) + Tensor(a)).data, a + a)
    np.testing.assert_allclose((Tensor(a) * 3.0).data, a * 3)
    np.testing.assert_allclose((Tensor(a) @ Tensor(b)).data, a @ b)
    np.testing.assert_allclose((Tensor(a) / Tensor(p)).data, a / p)
    np.testing.assert_allclose((Tensor(p) ** 1.5).data, p**1.5)
    np.testing.assert_allclose(Tensor(a).exp().data, np.exp(a))
    np.testing.assert_allclose(Tensor(p).log().data, np.log(p))
    np.testing.assert_allclose(Tensor(a).relu().data, np.maximum(a, 0))
    np.testing.assert_allclose(Tensor(a).tanh().data, np.tanh(a))
    np.testing.assert_allclose(Tensor(a).sum(axis=0).data, a.sum(0))
    np.testing.assert_allclose(Tensor(a).mean().data, a.mean())
    np.testing.assert_allclose(Tensor(a).reshape(4, 3).data, a.reshape(4, 3))
    np.testing.assert_allclose(Tensor(a).T.data, a.T)
    np.testing.assert_allclose(Tensor(a)[1:, [0, 2]].data, a[1:, [0, 2]])


# ----------------------------------------------------------------------- VJPs
@pytest.mark.parametrize("sa, sb", [((3, 4), (3, 4)), ((3, 4), (4,)), ((3, 1), (1, 4)), ((2, 3, 4), (3, 4))])
def test_add_mul_div_broadcasting(sa, sb):
    check_vjp(lambda a, b: a + b, randn(*sa), randn(*sb))
    check_vjp(lambda a, b: a * b, randn(*sa), randn(*sb))
    check_vjp(lambda a, b: a / b, randn(*sa), positive(*sb))
    check_vjp(lambda a, b: a - b, randn(*sa), randn(*sb))


@pytest.mark.parametrize("sa, sb", [((3, 4), (4, 5)), ((2, 3, 4), (4, 5)), ((2, 3, 4), (2, 4, 5)), ((2, 1, 3, 4), (5, 4, 2))])
def test_matmul(sa, sb):
    check_vjp(lambda a, b: a @ b, randn(*sa), randn(*sb))


def test_unary_ops():
    check_vjp(lambda a: a.exp(), randn(3, 4))
    check_vjp(lambda a: a.log(), positive(3, 4))
    check_vjp(lambda a: a.tanh(), randn(3, 4))
    check_vjp(lambda a: a.relu(), randn(3, 4) + 0.1)  # avoid the kink at exactly 0
    check_vjp(lambda a: a**3, randn(3, 4))
    check_vjp(lambda a: a**0.5, positive(3, 4))


@pytest.mark.parametrize("axis, keepdims", [(None, False), (0, False), (1, True), (-1, False), ((0, 2), False)])
def test_sum_and_mean(axis, keepdims):
    check_vjp(lambda a: a.sum(axis=axis, keepdims=keepdims), randn(2, 3, 4))
    check_vjp(lambda a: a.mean(axis=axis, keepdims=keepdims), randn(2, 3, 4))


def test_shape_ops():
    check_vjp(lambda a: a.reshape(6, 4), randn(2, 3, 4))
    check_vjp(lambda a: a.transpose(2, 0, 1), randn(2, 3, 4))
    check_vjp(lambda a: a.T, randn(3, 4))


def test_getitem_scatter_adds_repeated_indices():
    x = Tensor(np.arange(4.0), requires_grad=True)
    x[np.array([0, 0, 2])].sum().backward()
    np.testing.assert_allclose(x.grad, [2.0, 0.0, 1.0, 0.0])
    check_vjp(lambda a: a[np.arange(3), np.array([1, 0, 1])], randn(3, 2))


@pytest.mark.parametrize("axis, keepdims", [(-1, False), (0, True), (1, False)])
def test_logsumexp(axis, keepdims):
    check_vjp(lambda a: a.logsumexp(axis=axis, keepdims=keepdims), randn(3, 4))


def test_logsumexp_is_stable():
    x = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -1000.0, -1000.0]])
    out = Tensor(x).logsumexp(axis=-1).data
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, [1002.0 + np.log(1 + np.exp(-1) + np.exp(-2)), -1000.0 + np.log(3)])


# --------------------------------------------------------------- graph logic
def test_node_used_twice_accumulates():
    x = Tensor(np.array([1.5, -2.0]), requires_grad=True)
    y = x * x + x  # dy/dx = 2x + 1
    y.sum().backward()
    np.testing.assert_allclose(x.grad, 2 * x.data + 1)


def test_diamond_graph():
    check_vjp(lambda a, b: (a * b) + (a * b) * (a * b), randn(3), randn(3))


def test_constants_get_no_grad():
    x = Tensor(randn(3), requires_grad=True)
    c = Tensor(randn(3))
    (x * c).sum().backward()
    assert c.grad is None and x.grad is not None


def test_backward_on_nonscalar_needs_grad():
    with pytest.raises(RuntimeError) as err:
        Tensor(randn(3), requires_grad=True).backward()
    assert not isinstance(err.value, NotImplementedError)


def test_deep_graph_does_not_hit_recursion_limit():
    x = Tensor(np.array([1.0]), requires_grad=True)
    y = x
    for _ in range(5000):
        y = y * 1.0
    y.sum().backward()
    np.testing.assert_allclose(x.grad, [1.0])


# ------------------------------------------------------------------- losses
def test_cross_entropy_value_and_grad():
    logits, y = randn(5, 4), np.array([0, 3, 1, 1, 2])
    t = Tensor(logits, requires_grad=True)
    loss = ag.cross_entropy(t, y)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    np.testing.assert_allclose(loss.data, -np.log(p[np.arange(5), y]).mean())
    loss.backward()
    onehot = np.eye(4)[y]
    np.testing.assert_allclose(t.grad, (p - onehot) / 5, atol=1e-12)


def test_log_softmax_rows_normalize():
    out = ag.log_softmax(Tensor(randn(3, 5)), axis=-1).data
    np.testing.assert_allclose(np.exp(out).sum(-1), np.ones(3))


def test_numerical_gradient_helper():
    x = randn(3, 2)
    g = ag.numerical_gradient(lambda v: float(np.sum(np.sin(v))), x.copy())
    np.testing.assert_allclose(g, np.cos(x), atol=1e-6)


# ------------------------------------------------------------------ training
def test_sgd_step_updates_and_clears():
    p = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    p.grad = np.array([0.5, -1.0])
    ag.sgd_step([p], lr=0.1)
    np.testing.assert_allclose(p.data, [0.95, 2.1])
    assert p.grad is None


def test_mlp_learns_spirals():
    loss, acc = ag.train_spirals()
    assert acc >= 0.97, f"accuracy {acc:.3f}: your backprop or SGD is off"
    assert loss < 0.1
