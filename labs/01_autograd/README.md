# Lab 01 — Autograd from scratch

**Build:** a NumPy `Tensor` with reverse-mode automatic differentiation, then train an MLP with it.<br>
**Time:** 8–12 h · **Reads first:** [math §2](../../curriculum/01-math.md#2-matrix-calculus-the-only-calculus-you-need), [deep learning §1](../../curriculum/03-deep-learning.md#1-autodiff-what-backward-actually-does)<br>
**Run:** `pytest labs/01_autograd` (your code) · `pytest labs/01_autograd --impl=solution` (reference)

After this lab you should be able to derive, on a whiteboard, the backward pass of any layer you meet,
and to explain why frameworks are built the way they are. Every later lab rests on this one.

---

## 1. The ideas

### A computation is a DAG
`loss = cross_entropy(relu(x @ W1 + b1) @ W2 + b2, y)` is a directed acyclic graph: leaves are
parameters and data, internal nodes are ops, the root is a scalar. Each `Tensor` you create keeps
`_parents` and a `_vjp` closure. That record *is* the graph.

### Vector–Jacobian products (VJPs), not Jacobians
For an op `y = f(x)` with upstream gradient `ḡ = ∂L/∂y`, the backward pass needs
`∂L/∂x = Jᵀ ḡ`. You never build `J` (for a 4096×4096 matmul it would have 2.8·10¹⁴ entries).
Every op supplies a closed-form VJP:

| op | forward | VJP (given ḡ = ∂L/∂out) |
|---|---|---|
| add | `a + b` | `ḡ`, `ḡ` (then unbroadcast) |
| mul | `a * b` | `ḡ·b`, `ḡ·a` |
| div | `a / b` | `ḡ/b`, `−ḡ·a/b²` |
| matmul | `C = A @ B` | `Ā = C̄ Bᵀ`, `B̄ = Aᵀ C̄` |
| exp / log | `eˣ` / `ln x` | `ḡ·eˣ` / `ḡ/x` |
| tanh / relu | | `ḡ·(1 − tanh²x)` / `ḡ·𝟙[x>0]` |
| sum | `Σ` over axes | broadcast `ḡ` back to the input shape |
| logsumexp | `log Σ eˣ` | `ḡ · softmax(x)` |
| gather `x[idx]` | pick entries | **scatter-add** `ḡ` into zeros (`np.add.at`) |

Derive the matmul rule once, by hand: `dL = tr(C̄ᵀ dC)` and `dC = dA·B + A·dB`, so
`dL = tr((C̄Bᵀ)ᵀ dA) + tr((AᵀC̄)ᵀ dB)`. Then check yourself: the shapes must match
`A` and `B`, and there is exactly one arrangement of `C̄, A, B` that does.

### Why reverse mode
For `f: ℝⁿ → ℝ` (a loss over *n* parameters), forward mode costs *n* passes, one per input
direction. Reverse mode costs **one** backward pass, about 2–3× a forward pass, whatever *n* is.
The price is memory: the backward pass needs the forward pass's intermediates. That trade-off
leads to activation checkpointing ([lab 02](../02_training_core/README.md), [curriculum 05](../../curriculum/05-pretraining.md)).

### Topological order and accumulation
A node can feed several children (`x*x + x`). Its gradient is the **sum** over all paths, so you
can only call its VJP once every child has contributed. Process nodes in reverse topological order
and accumulate into a pending dict. Use an iterative DFS: a 5,000-op chain overflows Python's
recursion limit, and the test for exactly that is on purpose.

### Broadcasting: copies forward, sums backward
`x (3,4) + b (4,)` copies `b` onto each of 3 rows. The gradient of a copied value is the sum of
the copies' gradients, so `unbroadcast` sums away prepended axes and axes where the original had
size 1. Half of all "my gradients are wrong" bugs in real code are this.

### Gather forward, scatter-add backward
`x[[0, 0, 2]]` reads element 0 twice, so element 0 must receive **two** gradient contributions.
`full[idx] += g` silently drops the duplicate; `np.add.at` does not. The same duality explains
embedding-layer gradients (sparse rows) and why `index_add_` exists in PyTorch.

### Numerically stable logsumexp
`log Σ exp(x)` overflows for `x = 1000`. Use `m + log Σ exp(x − m)` with `m = max(x)`. The
gradient is `softmax(x)`, which you already have from the forward pass. Cross-entropy then
reduces to `logsumexp(z) − z_y`, and its gradient is `(softmax(z) − onehot(y)) / N`. That identity
is the most-asked "show me you can do calculus" interview question.

### Gradient checking
Central differences `(f(x+ε) − f(x−ε)) / 2ε` have error O(ε²), versus O(ε) for one-sided ones.
Use float64 and ε ≈ 1e-6: much smaller and floating-point cancellation dominates. The tests
backprop a *random* upstream gradient `G` and compare against finite differences of
`Σ f(x)·G`, which checks the full VJP rather than one row of it.

---

## 2. What to implement (in this order)

`_topological_order` (iterative DFS), the derived operators (`-`, `__neg__`, `__radd__`, `mean`, `.T`, ...), `Linear`, `MLP` and the spirals data are given. You write the rest of `exercise.py`, in this order:

| step | implement | tests that turn green |
|---|---|---|
| 1 | `unbroadcast` | `test_unbroadcast_matches_sum` (6 shape pairs) |
| 2 | `__add__`, `__mul__`, `backward` | `test_node_used_twice_accumulates` (your first real milestone), `test_diamond_graph`, `test_constants_get_no_grad`, `test_backward_on_nonscalar_needs_grad`, `test_deep_graph_does_not_hit_recursion_limit` |
| 3 | `__truediv__`, `__pow__`, `exp`, `log`, `relu`, `tanh` | `test_add_mul_div_broadcasting` (4), `test_unary_ops` |
| 4 | `__matmul__` | `test_matmul` (4, including batched and broadcast batch dims) |
| 5 | `sum`, `reshape`, `transpose`, `__getitem__`, `logsumexp` | `test_sum_and_mean` (5), `test_shape_ops`, `test_getitem_scatter_adds_repeated_indices`, `test_logsumexp` (3), `test_logsumexp_is_stable`, and now `test_forward_matches_numpy`, which uses every forward op |
| 6 | `log_softmax`, `cross_entropy`, `numerical_gradient`, `sgd_step` | `test_log_softmax_rows_normalize`, `test_cross_entropy_value_and_grad`, `test_numerical_gradient_helper`, `test_sgd_step_updates_and_clears`, `test_mlp_learns_spirals` (accuracy ≥ 97% and loss < 0.1) |

37 tests in total. `pytest labs/01_autograd -x` stops at the first failure; work top to bottom.

Two contracts the tests check that are easy to miss:
- `backward()` on a non-scalar without an explicit upstream gradient must raise `RuntimeError` (not `NotImplementedError`).
- A tensor created with `requires_grad=False` must end with `.grad is None`, even when it took part in the computation.

> [!TIP]
> While iterating, skip the slow end-to-end test: `pytest labs/01_autograd -k "not spirals"`. The spirals test trains a 2-64-64-3 MLP for 600 full-batch steps in pure NumPy; the reference solution reaches 99.3% accuracy and takes about a minute on a slow CPU.

> [!TIP]
> When a VJP test fails, reproduce it by hand on the smallest input that fails (a 2×3 array), print your gradient next to `finite_diff` from the test file, and look at the *pattern* of the difference: off by a constant factor (a missing `p` in `pow`, a missing `/N` in the mean), transposed (wrong matmul order), or summed over the wrong axis (`unbroadcast`).

## 3. Check yourself (answer without looking)

1. Why does `backward()` need a topological order? Construct a graph where a naive DFS order gives a wrong answer.
2. What is the memory cost of the backward pass for an L-layer MLP with batch B and width d? What would you store and what could you recompute?
3. Your loss is a scalar. What changes if you want the full Jacobian of a vector output? What does `torch.func.jacrev` do under the hood?
4. Why is `softmax` followed by `log` numerically worse than `log_softmax`? Give a concrete input where it fails in float32.
5. `relu` has no derivative at 0. Why does it not matter in practice, and when could it?
6. The tests backprop a *random* upstream gradient `G` instead of `ones`. What bug would `ones` miss?
7. Your `sum` VJP returns `np.broadcast_to(g, shape)` without `.copy()`. What can go wrong later?

<details><summary>Answers (after you try)</summary>

1. A node's VJP must see the *total* gradient from all its children. If you visit `x` before one of its children has pushed its contribution, `x` propagates an incomplete gradient and the missing part never arrives. Example: `y = x*x + x`, visiting `x` right after the `+` node.
2. It stores each layer's input activations (B·d per layer, so O(L·B·d)) and the ReLU masks. With checkpointing you store every k-th activation and recompute the rest in the backward pass, trading about one extra forward pass for √L-scale memory.
3. You need one VJP per output coordinate, that is, one backward pass per row of the Jacobian. `jacrev` vectorizes those VJPs with `vmap`. For wide outputs and narrow inputs, forward mode (`jacfwd`) is cheaper.
4. `softmax` can underflow to exactly 0 for very negative logits, and then `log(0) = −inf`. In float32, `exp(−110)` is already 0, so logits `[0, −200]` produce `log p₂ = −inf`, while `log_softmax` returns −200 correctly.
5. The kink is a measure-zero set: activations are almost never exactly 0 under continuous noise. It matters for quantized or saturated activations, or when a bug initializes everything to exactly 0.
6. With `G = ones`, any VJP that routes gradient entries to the wrong positions still returns all ones: for example, a `transpose` VJP that applies the forward permutation instead of its inverse on a 3×3×3 input, or a `reshape` VJP that reorders elements. A random `G` gives every output entry a different weight, so every row of the Jacobian is tested.
7. `broadcast_to` returns a read-only view whose entries share memory. Any later in-place update of that gradient (`grad += ...` in your accumulation code, or an optimizer that modifies `.grad` in place) raises `ValueError: output array is read-only`. Copy before handing gradients to code that may modify them.
</details>

## 4. Stretch goals

- Add `max(axis)` with a correct tie-breaking VJP, then build `conv1d` using only existing ops and check its gradient.
- Add a `no_grad()` context manager and measure the memory saved (`tracemalloc`).
- **Double backward.** Make each VJP build `Tensor` operations instead of NumPy ones, so `backward` itself is differentiable. Then compute a Hessian-vector product and check it against finite differences of your gradient. This is what `create_graph=True` does in PyTorch.
- Time your MLP step against the same MLP in PyTorch. Where does your time go? (Profile with `cProfile`: the answer is Python overhead per op, which is why frameworks fuse kernels.)
- Your `Tensor` keeps `.grad` on every node that requires grad, including intermediates; PyTorch keeps it only on leaves. Change that and measure the memory difference on the spirals MLP.
- Read [micrograd](https://github.com/karpathy/micrograd) and [tinygrad's `Tensor`](https://github.com/tinygrad/tinygrad). What does tinygrad do that you don't?

## 5. Common bugs

| symptom | likely cause |
|---|---|
| gradient right for `(3,4)+(3,4)` but wrong for `(3,4)+(4,)` or `(3,1)+(1,4)` | missing or incorrect `unbroadcast`: sum prepended axes first, then axes where the target has size 1, with `keepdims=True` |
| `x*x` gives `x.grad == x` instead of `2x` | you overwrite rather than accumulate into `pending` / `.grad` |
| `test_node_used_twice_accumulates` passes but `test_diamond_graph` fails | a node propagates before all its consumers contributed: process in reverse topological order, one visit per node |
| gradient doubles every time you call `backward` in a loop | `.grad` never cleared between steps (`sgd_step` must set `p.grad = None`) |
| `RecursionError` on long chains | recursive traversal; use the provided iterative `_topological_order` |
| `test_getitem_scatter_adds_repeated_indices` gives `[1, 0, 1, 0]` instead of `[2, 0, 1, 0]` | `full[idx] += g` drops repeated indices; use `np.add.at(full, idx, g)` |
| `sum` VJP raises a broadcast error for `axis=1, keepdims=False` | restore the reduced axis with `np.expand_dims(g, axis)` before `np.broadcast_to` |
| `transpose` passes for `.T` but fails for `transpose(2, 0, 1)` | the VJP needs the *inverse* permutation (`np.argsort(axes)`); a 2-D transpose is its own inverse, so 2-D tests hide this |
| batched `test_matmul` cases fail with shape errors | `.T` reverses *all* axes of a 3-D array; swap only the last two (`np.swapaxes(b, -1, -2)`), then `unbroadcast` to each input's shape |
| NaN or inf in cross-entropy, or `test_logsumexp_is_stable` fails | softmax or logsumexp computed without subtracting the row max |
| spirals accuracy stuck near 33% or loss explodes | `cross_entropy` sums instead of averaging (gradients N× too big at lr = 0.5), or gradients never reach the weights (constants lifted with `requires_grad=False` in the wrong place) |
| `test_numerical_gradient_helper` off in later entries | you forgot to restore `x[i]` after perturbing it |

## 6. CPU and GPU notes

Everything here is NumPy in float64 on the CPU, deliberately: finite-difference checks need float64 (with ε = 1e-6 the central difference is accurate to ~1e-10; in float32 the best you can do is ~1e-5, too coarse to separate a subtle bug from rounding; see [01 §2.6](../../curriculum/01-math.md#26-gradient-checking-and-choosing-the-step-size)). Nothing in this lab needs a GPU. The profiling stretch goal below shows why real frameworks move the same design onto GPU kernels.
