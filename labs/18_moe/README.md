# Lab 18 — Mixture of experts

**Build:** top-k routing, the Switch load-balancing loss and the router z-loss, expert capacity with token dropping, a token-choice MoE layer checked against a per-token reference, and the arithmetic of total versus active parameters.<br>
**Time:** 3–4 h for the tests, 4–8 h for the scale-up · **Reads first:** [transformers §6](../../curriculum/04-transformers.md#6-mixture-of-experts)<br>
**Run:** `pytest labs/18_moe` (your code) · `pytest labs/18_moe --impl=solution` (reference). The ten tests run on CPU in about 5 seconds.

A mixture-of-experts (MoE) layer decouples how many parameters a model has from how much compute each token uses. Mixtral 8x7B has 47B parameters but uses 13B per token ([arXiv 2401.04088](https://arxiv.org/abs/2401.04088)); DeepSeek-V3 has 671B and uses 37B ([arXiv 2412.19437](https://arxiv.org/abs/2412.19437)). Many of the largest open models, including DeepSeek-V3, the biggest Qwen3 variants and Llama 4, are MoE. Interviews ask why the router needs a balancing loss, what happens to a token when its expert is full, where the memory goes, and why expert parallelism needs all-to-all communication. This lab makes each answer something you have implemented and measured.

---

## 1. The layer

A dense transformer block has one feed-forward network (FFN). An MoE block has $E$ expert FFNs and a **router**, a linear layer that scores every expert for every token:

$$z = x W_r \in \mathbb{R}^E, \qquad p = \text{softmax}(z), \qquad y = \sum_{i \in \text{top-}k(p)} g_i \, \text{FFN}_i(x).$$

Each token runs only $k$ experts, so its FLOPs are those of $k$ FFNs while the model stores $E$ of them. In this lab an expert is a two-matrix GELU MLP ($2 d \, d_\text{ff}$ weights); Mixtral's experts are three-matrix SwiGLU FFNs ($3 d \, d_\text{ff}$).

**Gate values.** Two conventions, and the difference matters:

- **Renormalized** (Mixtral, $k = 2$): $g_i = p_i / \sum_{j \in \text{top-}k} p_j$. Dividing the full softmax by the selected mass is the same as a softmax over only the selected logits, because the shared denominator cancels: $\frac{e^{z_i}/S}{\sum_{j \in K} e^{z_j}/S} = \frac{e^{z_i}}{\sum_{j \in K} e^{z_j}}$. `test_renormalized_top_k_equals_softmax_over_the_selected_logits` checks it.
- **Raw** (Switch Transformer, $k = 1$, [arXiv 2101.03961](https://arxiv.org/abs/2101.03961)): $g_i = p_i$. With one expert, a renormalized gate is always exactly 1, so the task loss sends no gradient to the router at all. The raw probability keeps the router learning. `test_top1_gate_needs_the_raw_probability_to_train_the_router` shows both halves; the lab's `MoE` renormalizes only when $k > 1$.

DeepSeek-V3 uses sigmoid scores instead of a softmax and normalizes them over the selected experts; the ideas below carry over.

**Parameter arithmetic** (`moe_param_counts`, router included, one layer):

| layer | total params | active per token |
|---|---|---|
| dense GELU FFN, $d = 4096$, $d_\text{ff} = 14336$ | 117.4 M | 117.4 M |
| 8 experts, top-2, same expert size | 939.6 M | 234.9 M |
| 64 small experts ($d_\text{ff} = 512$) top-6 plus 2 shared, $d = 1024$ | 69.3 M | 8.5 M |

The last row is the DeepSeekMoE pattern ([arXiv 2401.06066](https://arxiv.org/abs/2401.06066)): many fine-grained experts give the router more combinations, and **shared experts** that see every token hold common knowledge so the routed experts can specialize.

## 2. Load balancing

Routing collapses without help. An expert that receives more tokens gets more gradient, gets better, and wins even more tokens; a few experts end up doing all the work while the rest are dead weight.

**Switch auxiliary loss.** With $f_i$ the fraction of the $T k$ (token, slot) assignments sent to expert $i$ and $P_i$ the mean router probability of expert $i$ over the batch,

$$\mathcal{L}_\text{aux} = \alpha \, E \sum_{i=1}^{E} f_i P_i, \qquad \alpha = 10^{-2} \text{ in Switch.}$$

$f_i$ comes from counts and has no gradient; the gradient flows through $P_i$ and pushes probability away from overloaded experts. Under perfectly uniform routing $f_i = P_i = 1/E$ and the loss is exactly 1; if every token picks one expert ($k = 1$) it is $E$. It is not a strict lower bound: when routing and probabilities disagree, the value can dip below 1 (with $E = 2$, routing 90% of tokens to a 0.5/0.5 tie and 10% to a certain expert 1 gives 0.92), so read "about 1" as balanced. In `train_router_for_balance`, a router that starts by sending 100% of 1,024 tokens to expert 0 reaches exactly 12.5% per expert after 300 Adam steps on this loss alone.

**Router z-loss** (ST-MoE, [arXiv 2202.08906](https://arxiv.org/abs/2202.08906)): $\mathcal{L}_z = \frac{1}{T}\sum_t \big(\log \sum_i e^{z_{t,i}}\big)^2$, with coefficient $10^{-3}$ there. It keeps router logits small, which matters because large logits in bf16 round badly and destabilize training.

**Auxiliary-loss-free balancing** (DeepSeek-V3): add a per-expert bias $b_i$ to the scores used to *select* the top-$k$, but not to the gate values used to *combine* outputs. After each step, lower $b_i$ for overloaded experts and raise it for underloaded ones by a fixed small amount. Balance then comes from selection alone, with no extra gradient term competing with the language-modeling loss. The report keeps a very small sequence-wise balance loss as a safeguard.

## 3. Capacity and dropped tokens

Accelerators want static shapes, so each expert gets a fixed number of slots per batch:

$$C = \left\lceil \text{capacity factor} \times \frac{T k}{E} \right\rceil.$$

With $T = 4096$ tokens, $E = 8$, $k = 2$ and a capacity factor of 1.25, each expert has 1,280 slots against 1,024 at perfect balance. Assignments beyond capacity are **dropped**: that slot contributes nothing, and a token whose every assignment was dropped reaches the next layer through the residual connection unchanged.

Who gets dropped depends on priority. `capacity_mask` follows GShard ([arXiv 2006.16668](https://arxiv.org/abs/2006.16668)): every token's first choice is placed before any token's second choice, then token order within a rank. `test_capacity_gives_first_choices_priority_over_second_choices` pins down a case where token 0 loses its second choice to tokens 1 and 2. Alternatives: V-MoE's batch-prioritized routing ranks assignments by router score ([arXiv 2106.05974](https://arxiv.org/abs/2106.05974)); dropless MoE with block-sparse kernels (MegaBlocks, [arXiv 2211.15841](https://arxiv.org/abs/2211.15841)) removes the capacity limit. At inference, capacity is usually set high enough that nothing is dropped, because a drop changes the output.

## 4. Napkin math: memory, compute, communication

- **Memory follows total parameters; FLOPs follow active ones.** Mixtral's 47B parameters are 94 GB in bf16 and must all be resident; a forward pass costs about $2 \times 13\text{B}$ FLOPs per token.
- **Decode reads more than the active experts once the batch grows.** At batch 1, a layer reads only the two chosen experts. With uniform top-2-of-8 routing, a batch of 4 tokens touches 5.5 of 8 experts on average and a batch of 16 touches 7.9, so weight traffic per step approaches the full model while FLOPs per token stay at the active count. The upside: at batch 1, Mixtral reads about 26 GB of weights per token, not 94.
- **Expert parallelism needs all-to-all.** With experts sharded across GPUs, each token's hidden state travels to the GPUs that hold its experts and back. In bf16 with $d = 4096$ and $k = 2$, that is 16 KiB out and 16 KiB back per token per layer. For 8,192 tokens per GPU, that is 256 MiB per layer; with 8-way expert parallelism, 7/8 of it leaves the device. This is why DeepSeek-V3 limits each token to experts on at most 4 nodes, and why MoE training benefits so much from fast interconnects.

## What to implement

| # | function | tests | what the tests pin down |
|---|---|---|---|
| 1 | `route(logits, k, renormalize=True)` → `(weights, experts, probs)` | first three tests | `probs` is the full softmax; `experts` are the top-$k$, best first; renormalized weights sum to 1 and equal a softmax over the selected logits; with $k = 1$ and `renormalize=False` the weight carries a gradient |
| 2 | `load_balancing_loss(probs, experts, n_experts)` | `test_load_balancing_loss_is_one_when_uniform_and_E_when_collapsed`, `test_load_balancing_loss_trains_a_skewed_router_toward_balance` | 1.0 for uniform routing, $E$ for total collapse; training on it alone takes the largest share from over 90% to under 25% |
| 3 | `router_z_loss(logits)` | `test_router_z_loss_matches_definition` | mean squared logsumexp |
| 4 | `capacity_mask(experts, n_experts, capacity)` | `test_capacity_gives_first_choices_priority_over_second_choices` | rank-major priority, then token order; `expert_capacity` is given |
| 5 | `MoE.forward(x)` → `(y, aux)` | `test_moe_forward_matches_a_per_token_reference`, `test_dropped_assignments_contribute_nothing` | equals the per-token sum of gated expert outputs; with one slot per expert at most four tokens get output |
| 6 | `moe_param_counts(d, d_ff, n_experts, k, n_shared=0)` | `test_param_counts_total_vs_active` | total and active parameters, router and shared experts included |

Given: `expert_capacity`, the `MoE` constructor and its `expert` method, and `train_router_for_balance`.

## Tips

> [!TIP]
> Loop over experts, never over tokens. For expert $e$, `torch.nonzero((experts == e) & keep, as_tuple=True)` returns the matching token and slot indices; run the expert once on `x[tok]` and `index_add` the gated result into the output. Production kernels do the same thing as one grouped matrix multiply over all experts.

- `torch.bincount(experts.reshape(-1), minlength=E)` gives the counts for $f$ in one call, including zeros for idle experts.
- For `capacity_mask`, flatten rank-major with `experts.T.reshape(-1)`, one-hot it, and take a cumulative sum down the rows: the entry for each assignment's own expert, minus one, is its position in that expert's queue.

## Common bugs

- **Renormalizing a top-1 gate:** the weight is always 1 and the router never learns from the task. The third test catches it.
- **Computing $P$ from the renormalized top-$k$ weights** instead of the full softmax: the loss then ignores probability mass on unselected experts and stops discouraging collapse.
- **Token-major capacity order:** flattening `experts.reshape(-1)` lets early tokens' second choices take slots from later tokens' first choices.
- **Adding expert outputs without their gate weights,** or with weights taken from the wrong slot after indexing.
- **An auxiliary coefficient that is too large:** balance improves but the language-model loss worsens. Tune $\alpha$ on a short run and watch both.

## CPU experiments (no GPU needed)

1. **Watch collapse happen.** Train `MoE(d=32, d_ff=64, n_experts=8, k=2)` on a toy regression task with and without $0.01 \times$ the auxiliary loss. Log each expert's share of assignments every 20 steps and plot them.
2. **Capacity sweep.** For a random router and for one trained in experiment 1, measure the fraction of dropped assignments at capacity factors 0.5, 1.0, 1.25 and 2.0. Balanced routing drops almost nothing at 1.25; a skewed router drops a lot even at 2.0.
3. **The top-1 trap.** Train with the task loss only, $k = 1$ and `renormalize=True`; confirm that the router weights never change. Switch to raw gates and watch them move.
4. **Loss-free balancing.** Implement DeepSeek-V3's bias update (a step of 0.001, adding the bias for selection only) and compare load balance and task loss against the auxiliary loss.

## GPU scale-up (8 GB): an MoE version of your lab 05 model

Replace the MLP in each block of your [lab 05](../05_transformer/README.md) GPT with this layer, flattening $(B, T, d)$ to $(BT, d)$ and back, and add $0.01 \times$ the auxiliary loss (plus $0.001 \times$ the z-loss) to the language-modeling loss. A free Colab or Kaggle T4 works too.

```python
h = self.norm2(x).view(-1, d)
y, aux = self.moe(h)                  # MoE(d, d_ff, n_experts=8, k=2)
x = x + y.view(B, T, d)
self.aux_loss = aux                   # sum over blocks, then loss = lm_loss + 0.01 * aux_total
```

- Compare three models trained for the same tokens on TinyStories: the dense baseline, an MoE with the same **active** parameters (8 experts of half the dense width, top-2), and an MoE with the same **total** parameters. Report validation loss with seeds, tokens per second, and peak memory.
- Log per-expert load per layer every few hundred steps: early layers often balance differently from late ones.
- Measure what the Python loop over experts costs in tokens per second, then replace it with one padded batched matmul (`torch.bmm` over experts) and measure again.

## Check yourself

1. Why does top-1 routing with a renormalized gate stop the router from learning, and how does Switch avoid it?
2. $f_i$ has no gradient. How does the load-balancing loss still change the router?
3. Mixtral has 47B parameters and 13B active. How much memory do its bf16 weights take, and how many bytes does batch-1 decoding read per token?
4. With $T = 4096$, $E = 8$, $k = 2$ and a capacity factor of 1.25, how many slots does each expert get, and what happens to the 1,281st assignment?
5. Why does expert parallelism need all-to-all communication, and how many bytes move per token per layer for $d = 4096$, $k = 2$ in bf16?
6. How can DeepSeek-V3 balance load without a balancing term in its loss?

<details><summary>Answers</summary>

1. With one selected expert, $g = p_i / p_i = 1$ whatever the logits are, so the task loss has no path to the router weights. Switch uses the raw probability $p_i$ as the gate, so a better expert choice raises $p_i$ and the output scale, and the gradient reaches the router.
2. Through $P_i$, the mean softmax probability, which is differentiable. The gradient of $E \sum_i f_i P_i$ with respect to the logits lowers probability most for experts with large $f_i$.
3. $47\text{B} \times 2$ bytes = 94 GB resident; at batch 1 each token reads roughly the active 13B parameters, about 26 GB. That is why MoE decoding is cheap per token but expensive in memory.
4. $\lceil 1.25 \times 4096 \times 2 / 8 \rceil = 1280$. The 1,281st assignment in priority order is dropped: that slot contributes zero and the token's residual stream passes through.
5. Tokens live on the GPU that owns their batch shard, but their experts live on other GPUs, so every token's hidden state goes to its experts' devices and the results come back. That is $k \times d \times 2$ bytes = 16 KiB each way, 32 KiB per token per layer.
6. It adds a per-expert bias to the scores used only for top-$k$ selection and adjusts it after each step according to each expert's load. The gate values that scale the outputs are unchanged, so no gradient term fights the language-modeling objective.
</details>

## Stretch

- **Expert-choice routing** ([arXiv 2202.09368](https://arxiv.org/abs/2202.09368)): each expert picks its top tokens instead of each token picking experts. Balance is perfect by construction; work out why it leaks information across a causal sequence during decoding.
- **Noisy top-k gating** from the original sparsely-gated MoE paper ([arXiv 1701.06538](https://arxiv.org/abs/1701.06538)): add learned noise to the logits before the top-$k$ and compare balance without an auxiliary loss.
- **Shared experts:** add one always-on expert to the lab's layer and compare specialization (the entropy of each expert's token distribution) with and without it.
- **Grouped GEMM:** sort assignments by expert, pad each group, and run one `torch.bmm`; benchmark it against the loop on CPU and GPU.
