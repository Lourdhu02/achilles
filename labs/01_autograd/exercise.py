# Exercise stub generated from solution.py by tools/make_exercises.py.
# Replace every NotImplementedError, then run: pytest labs/01_autograd
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
    # HINT: sum away the leading axes broadcasting prepended,
    # HINT: then sum (keepdims=True) over axes where `shape` has size 1.
    raise NotImplementedError("01_autograd: implement unbroadcast")


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
        raise NotImplementedError("01_autograd: implement __add__")

    def __mul__(self, other) -> "Tensor":
        other = Tensor.lift(other)
        raise NotImplementedError("01_autograd: implement __mul__")

    def __truediv__(self, other) -> "Tensor":
        other = Tensor.lift(other)
        raise NotImplementedError("01_autograd: implement __truediv__")

    def __pow__(self, p: float) -> "Tensor":
        """Power with a constant (non-Tensor) exponent."""
        raise NotImplementedError("01_autograd: implement __pow__")

    def __matmul__(self, other) -> "Tensor":
        """Matrix product of tensors with ndim >= 2 (leading dims broadcast)."""
        other = Tensor.lift(other)
        # HINT: for C = A @ B:  dA = dC @ B^T  and  dB = A^T @ dC  (then unbroadcast)
        raise NotImplementedError("01_autograd: implement __matmul__")

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
        raise NotImplementedError("01_autograd: implement exp")

    def log(self) -> "Tensor":
        raise NotImplementedError("01_autograd: implement log")

    def relu(self) -> "Tensor":
        raise NotImplementedError("01_autograd: implement relu")

    def tanh(self) -> "Tensor":
        raise NotImplementedError("01_autograd: implement tanh")

    # --------------------------------------------------------------- reductions
    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        # HINT: the VJP of a sum is a broadcast; restore reduced axes before broadcasting.
        raise NotImplementedError("01_autograd: implement sum")

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        total = self.sum(axis=axis, keepdims=keepdims)
        return total * (total.data.size / self.data.size)

    def logsumexp(self, axis: int = -1, keepdims: bool = False) -> "Tensor":
        """log(sum(exp(x))) along ``axis``, stable for large inputs."""
        # HINT: subtract the max before exp; the gradient of logsumexp is softmax.
        raise NotImplementedError("01_autograd: implement logsumexp")

    # ------------------------------------------------------------------- shapes
    def reshape(self, *shape) -> "Tensor":
        raise NotImplementedError("01_autograd: implement reshape")

    def transpose(self, *axes) -> "Tensor":
        axes = axes or tuple(reversed(range(self.ndim)))
        raise NotImplementedError("01_autograd: implement transpose")

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def __getitem__(self, idx) -> "Tensor":
        idx = _as_index(idx)
        # HINT: gather forward => scatter-ADD backward (np.add.at handles repeated indices).
        raise NotImplementedError("01_autograd: implement __getitem__")

    # ----------------------------------------------------------------- backward
    def backward(self, grad: Array | None = None) -> None:
        """Accumulate d(self)/d(node) into ``node.grad`` for every node that requires grad.

        ``grad`` is the upstream gradient; it may be omitted only for scalars.
        """
        # HINT: visit nodes in reverse topological order (use _topological_order),
        # HINT: summing contributions when a node feeds several children.
        raise NotImplementedError("01_autograd: implement backward")


# ------------------------------------------------------------------ functions
def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    raise NotImplementedError("01_autograd: implement log_softmax")


def cross_entropy(logits: Tensor, targets: Array) -> Tensor:
    """Mean negative log-likelihood of integer ``targets`` (N,) under softmax(logits) (N, C)."""
    raise NotImplementedError("01_autograd: implement cross_entropy")


def numerical_gradient(f: Callable[[Array], float], x: Array, eps: float = 1e-6) -> Array:
    """Central-difference estimate of df/dx; ``x`` must be float64 and is restored."""
    raise NotImplementedError("01_autograd: implement numerical_gradient")


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
    raise NotImplementedError("01_autograd: implement sgd_step")


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
