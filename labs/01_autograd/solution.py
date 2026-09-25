"""Lab 01 -- reverse-mode automatic differentiation from scratch (NumPy only).

A Tensor records every operation applied to it. ``backward()`` walks that
graph in reverse topological order, applying each op's vector-Jacobian
product (VJP) and accumulating gradients. This is the core of PyTorch, JAX
and every other deep learning framework.

Handout: labs/01_autograd/README.md
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

Array = np.ndarray


def unbroadcast(grad: Array, shape: tuple[int, ...]) -> Array:
    """Sum ``grad`` down to ``shape``, undoing NumPy broadcasting.

    Broadcasting copies a value along an axis. The gradient of a copied value
    is the sum of the gradients of all its copies.
    """
    # BEGIN SOLUTION
    # HINT: sum away the leading axes broadcasting prepended,
    # HINT: then sum (keepdims=True) over axes where `shape` has size 1.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad
    # END SOLUTION


def _topological_order(root: "Tensor") -> list["Tensor"]:
    """Every node reachable from ``root``, parents before children (iterative DFS)."""
    order: list[Tensor] = []
    visited: set[int] = set()
    stack: list[tuple[Tensor, bool]] = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        if id(node) in visited:
            continue
        visited.add(id(node))
        stack.append((node, True))
        stack.extend((p, False) for p in node._parents if id(p) not in visited)
    return order


def _as_index(idx):
    """Convert Tensor components of an index into integer arrays."""
    if isinstance(idx, tuple):
        return tuple(_as_index(i) for i in idx)
    if isinstance(idx, Tensor):
        return idx.data.astype(np.int64)
    return idx


class Tensor:
    """An n-dimensional array that remembers how it was computed."""

    def __init__(self, data, requires_grad: bool = False, _parents: tuple = (), _op: str = ""):
        self.data: Array = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad: Array | None = None
        self._parents: tuple[Tensor, ...] = _parents
        # Maps the gradient w.r.t. this tensor to a tuple of gradients w.r.t. _parents.
        self._vjp: Callable[[Array], tuple[Array | None, ...]] | None = None
        self._op = _op

    # ---------------------------------------------------------------- plumbing
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __repr__(self) -> str:
        return f"Tensor({self.data!r}, requires_grad={self.requires_grad}, op={self._op!r})"

    @staticmethod
    def lift(x) -> "Tensor":
        return x if isinstance(x, Tensor) else Tensor(x)

    def _make(self, data: Array, parents: tuple, vjp: Callable, op: str) -> "Tensor":
        """Create an output node. Only record the graph if some parent needs grads."""
        needs = any(p.requires_grad for p in parents)
        out = Tensor(data, requires_grad=needs, _parents=parents if needs else (), _op=op)
        if needs:
            out._vjp = vjp
        return out

    # ------------------------------------------------------- elementwise binary
    def __add__(self, other) -> "Tensor":
        other = Tensor.lift(other)
        # BEGIN SOLUTION
        a, b = self, other
        return self._make(
            a.data + b.data,
            (a, b),
            lambda g: (unbroadcast(g, a.shape), unbroadcast(g, b.shape)),
            "add",
        )
        # END SOLUTION

    def __mul__(self, other) -> "Tensor":
        other = Tensor.lift(other)
        # BEGIN SOLUTION
        a, b = self, other
        return self._make(
            a.data * b.data,
            (a, b),
            lambda g: (unbroadcast(g * b.data, a.shape), unbroadcast(g * a.data, b.shape)),
            "mul",
        )
        # END SOLUTION

    def __truediv__(self, other) -> "Tensor":
        other = Tensor.lift(other)
        # BEGIN SOLUTION
        a, b = self, other

        def vjp(g):
            return (
                unbroadcast(g / b.data, a.shape),
                unbroadcast(-g * a.data / b.data**2, b.shape),
            )

        return self._make(a.data / b.data, (a, b), vjp, "div")
        # END SOLUTION

    def __pow__(self, p: float) -> "Tensor":
        """Power with a constant (non-Tensor) exponent."""
        # BEGIN SOLUTION
        x = self.data
        return self._make(x**p, (self,), lambda g: (g * p * x ** (p - 1),), f"pow{p}")
        # END SOLUTION

    def __matmul__(self, other) -> "Tensor":
        """Matrix product of tensors with ndim >= 2 (leading dims broadcast)."""
        other = Tensor.lift(other)
        # BEGIN SOLUTION
        # HINT: for C = A @ B:  dA = dC @ B^T  and  dB = A^T @ dC  (then unbroadcast)
        a, b = self, other

        def vjp(g):
            ga = g @ np.swapaxes(b.data, -1, -2)
            gb = np.swapaxes(a.data, -1, -2) @ g
            return unbroadcast(ga, a.shape), unbroadcast(gb, b.shape)

        return self._make(a.data @ b.data, (a, b), vjp, "matmul")
        # END SOLUTION

    # Derived operators: no new gradients needed.
    def __radd__(self, other) -> "Tensor":
        return self + other

    def __rmul__(self, other) -> "Tensor":
        return self * other

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other) -> "Tensor":
        return self + (-Tensor.lift(other))

    def __rsub__(self, other) -> "Tensor":
        return Tensor.lift(other) + (-self)

    def __rtruediv__(self, other) -> "Tensor":
        return Tensor.lift(other) / self

    # -------------------------------------------------------- elementwise unary
    def exp(self) -> "Tensor":
        # BEGIN SOLUTION
        out = np.exp(self.data)
        return self._make(out, (self,), lambda g: (g * out,), "exp")
        # END SOLUTION

    def log(self) -> "Tensor":
        # BEGIN SOLUTION
        x = self.data
        return self._make(np.log(x), (self,), lambda g: (g / x,), "log")
        # END SOLUTION

    def relu(self) -> "Tensor":
        # BEGIN SOLUTION
        x = self.data
        return self._make(np.maximum(x, 0.0), (self,), lambda g: (g * (x > 0),), "relu")
        # END SOLUTION

    def tanh(self) -> "Tensor":
        # BEGIN SOLUTION
        out = np.tanh(self.data)
        return self._make(out, (self,), lambda g: (g * (1.0 - out**2),), "tanh")
        # END SOLUTION

    # --------------------------------------------------------------- reductions
    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        # BEGIN SOLUTION
        # HINT: the VJP of a sum is a broadcast; restore reduced axes before broadcasting.
        shape = self.shape

        def vjp(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return (np.broadcast_to(g, shape).copy(),)

        return self._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), vjp, "sum")
        # END SOLUTION

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        total = self.sum(axis=axis, keepdims=keepdims)
        return total * (total.data.size / self.data.size)

    def logsumexp(self, axis: int = -1, keepdims: bool = False) -> "Tensor":
        """log(sum(exp(x))) along ``axis``, stable for large inputs."""
        # BEGIN SOLUTION
        # HINT: subtract the max before exp; the gradient of logsumexp is softmax.
        x = self.data
        m = x.max(axis=axis, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)
        lse = np.log(np.exp(x - m).sum(axis=axis, keepdims=True)) + m
        softmax = np.exp(x - lse)

        def vjp(g):
            if not keepdims:
                g = np.expand_dims(g, axis)
            return (g * softmax,)

        out = lse if keepdims else np.squeeze(lse, axis=axis)
        return self._make(out, (self,), vjp, "logsumexp")
        # END SOLUTION

    # ------------------------------------------------------------------- shapes
    def reshape(self, *shape) -> "Tensor":
        # BEGIN SOLUTION
        old = self.shape
        return self._make(self.data.reshape(*shape), (self,), lambda g: (g.reshape(old),), "reshape")
        # END SOLUTION

    def transpose(self, *axes) -> "Tensor":
        axes = axes or tuple(reversed(range(self.ndim)))
        # BEGIN SOLUTION
        inverse = tuple(np.argsort(axes))
        return self._make(self.data.transpose(axes), (self,), lambda g: (g.transpose(inverse),), "transpose")
        # END SOLUTION

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def __getitem__(self, idx) -> "Tensor":
        idx = _as_index(idx)
        # BEGIN SOLUTION
        # HINT: gather forward => scatter-ADD backward (np.add.at handles repeated indices).
        shape = self.shape

        def vjp(g):
            full = np.zeros(shape)
            np.add.at(full, idx, g)
            return (full,)

        return self._make(self.data[idx], (self,), vjp, "getitem")
        # END SOLUTION

    # ----------------------------------------------------------------- backward
    def backward(self, grad: Array | None = None) -> None:
        """Accumulate d(self)/d(node) into ``node.grad`` for every node that requires grad.

        ``grad`` is the upstream gradient; it may be omitted only for scalars.
        """
        # BEGIN SOLUTION
        # HINT: visit nodes in reverse topological order (use _topological_order),
        # HINT: summing contributions when a node feeds several children.
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("backward() on a non-scalar needs an explicit upstream grad")
            grad = np.ones_like(self.data)
        pending: dict[int, Array] = {id(self): np.asarray(grad, dtype=np.float64)}
        for node in reversed(_topological_order(self)):
            g = pending.pop(id(node), None)
            if g is None:
                continue
            if node.requires_grad:
                node.grad = g.copy() if node.grad is None else node.grad + g
            if node._vjp is None:
                continue
            for parent, pg in zip(node._parents, node._vjp(g)):
                if pg is None or not parent.requires_grad:
                    continue
                key = id(parent)
                pending[key] = pending[key] + pg if key in pending else pg
        # END SOLUTION


# ------------------------------------------------------------------ functions
def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    # BEGIN SOLUTION
    return x - x.logsumexp(axis=axis, keepdims=True)
    # END SOLUTION


def cross_entropy(logits: Tensor, targets: Array) -> Tensor:
    """Mean negative log-likelihood of integer ``targets`` (N,) under softmax(logits) (N, C)."""
    # BEGIN SOLUTION
    n = logits.shape[0]
    return -log_softmax(logits, axis=-1)[np.arange(n), np.asarray(targets)].mean()
    # END SOLUTION


def numerical_gradient(f: Callable[[Array], float], x: Array, eps: float = 1e-6) -> Array:
    """Central-difference estimate of df/dx; ``x`` must be float64 and is restored."""
    # BEGIN SOLUTION
    grad = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + eps
        f_plus = f(x)
        x[i] = old - eps
        f_minus = f(x)
        x[i] = old
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad
    # END SOLUTION


# ------------------------------------------------------------------ networks
class Linear:
    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator):
        bound = np.sqrt(6.0 / n_in)  # Kaiming-uniform: Var(W) = 2 / n_in for ReLU nets
        self.W = Tensor(rng.uniform(-bound, bound, size=(n_in, n_out)), requires_grad=True)
        self.b = Tensor(np.zeros(n_out), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b

    def parameters(self) -> list[Tensor]:
        return [self.W, self.b]


class MLP:
    def __init__(self, sizes: list[int], rng: np.random.Generator):
        self.layers = [Linear(a, b, rng) for a, b in zip(sizes[:-1], sizes[1:])]

    def __call__(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x.relu()
        return x

    def parameters(self) -> list[Tensor]:
        return [p for layer in self.layers for p in layer.parameters()]


def sgd_step(params: Iterable[Tensor], lr: float) -> None:
    """Vanilla SGD update, then clear gradients."""
    # BEGIN SOLUTION
    for p in params:
        if p.grad is not None:
            p.data -= lr * p.grad
        p.grad = None
    # END SOLUTION


def make_spirals(n_per_class: int = 100, n_classes: int = 3, noise: float = 0.2, seed: int = 0):
    """The CS231n spiral dataset: not linearly separable, easy for a small MLP."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(n_classes):
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(4.0 * c, 4.0 * (c + 1), n_per_class) + rng.normal(0.0, noise, n_per_class)
        xs.append(np.stack([r * np.sin(t), r * np.cos(t)], axis=1))
        ys.append(np.full(n_per_class, c))
    return np.concatenate(xs), np.concatenate(ys)


def train_spirals(steps: int = 600, lr: float = 0.5, seed: int = 0) -> tuple[float, float]:
    """Train an MLP on the spirals with full-batch SGD. Returns (final loss, accuracy)."""
    X, y = make_spirals(seed=seed)
    rng = np.random.default_rng(seed)
    model = MLP([2, 64, 64, 3], rng)
    loss = None
    for _ in range(steps):
        loss = cross_entropy(model(Tensor(X)), y)
        loss.backward()
        sgd_step(model.parameters(), lr)
    accuracy = float((model(Tensor(X)).data.argmax(axis=1) == y).mean())
    return float(loss.data), accuracy
