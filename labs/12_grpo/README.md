# Lab 12 — Policy gradients to GRPO

**Build:** the exact and Monte Carlo (REINFORCE) policy gradients for a softmax bandit, with a baseline; group-relative advantages (GRPO and Dr. GRPO); the clipped GRPO objective with a k3 KL penalty and two aggregation modes (GRPO's sequence mean, DAPO's token mean); then RL on a toy verifiable task that learns from reward ~0.13 to > 0.7 on CPU. Scale-up: GRPO with LoRA on Qwen2.5-0.5B-Instruct on 8 GB.
**Time:** 4–5 h for the tests, 8–12 h for the scale-up · **Reads first:** [post-training §3 and §5](../../curriculum/06-post-training.md#5-rl-with-verifiable-rewards-grpo-and-friends)
**Run:** `pytest labs/12_grpo` (your code) · `pytest labs/12_grpo --impl=solution` (reference). All five tests run on CPU; the counting task takes a few seconds.

RL for reasoning (RLVR) is where post-training moved in 2025, and GRPO and its fixes are the standard interview topic. This lab builds the objective from the policy-gradient theorem up, so every term in the GRPO loss is something you derived and tested.

---

## 1. The bandit warm-up: REINFORCE is unbiased, baselines are free

A softmax policy over actions, $\pi = \mathrm{softmax}(\theta)$, with fixed rewards $r(a)$. The objective is $J(\theta) = \sum_a \pi(a) r(a)$.

- **Exact gradient** (`exact_policy_gradient`): $\nabla_\theta J = \pi \odot (r - \mathbb{E}_\pi[r])$. Derive it from $\partial \pi_a / \partial \theta_b = \pi_a(\mathbb{1}[a=b] - \pi_b)$.
- **Monte Carlo** (`reinforce_gradient`): sample $n$ actions, average $(r(a) - b)\,\nabla_\theta\log\pi(a)$, where $\nabla_\theta\log\pi(a) = \mathrm{onehot}(a) - \pi$.

`test_reinforce_is_unbiased_and_baseline_reduces_variance` checks both halves of the theory: with 200,000 samples the estimate matches the exact gradient within 0.01, and with a baseline equal to the mean reward the variance of 32-sample estimates falls by more than half. The mean is unchanged because $\mathbb{E}[b\,\nabla\log\pi] = b\,\nabla\sum_a\pi(a) = 0$.

## 2. Group-relative advantages

For LLMs, each prompt gets $G$ samples and the baseline is their mean reward. `group_advantages(rewards)` takes rewards of shape `(B, G)`:

$$A_{i} = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r) + \epsilon} \quad (\text{GRPO}), \qquad A_i = r_i - \mathrm{mean}(r) \quad (\text{Dr. GRPO, } \texttt{std\_norm=False})$$

using the population std (`unbiased=False`). Worked: rewards `[1, 0, 0, 1]` have mean 0.5 and std 0.5, so GRPO advantages are `[1, −1, −1, 1]` and Dr. GRPO's are `[0.5, −0.5, −0.5, 0.5]`. A group of all-correct answers gets advantages of exactly zero: no signal. That is why DAPO's dynamic sampling drops such groups.

## 3. The GRPO loss

`grpo_loss(logp_new, logp_old, logp_ref, adv, mask, clip_eps, beta, agg)` takes per-token log-probs of shape `(N, T)`, one advantage per sequence `(N,)` and a response mask `(N, T)`:

```
ratio     = exp(logp_new − logp_old)
surrogate = min(ratio · A, clip(ratio, 1 − ε, 1 + ε) · A)          (A broadcast over tokens)
kl (k3)   = exp(logp_ref − logp_new) − (logp_ref − logp_new) − 1    (≥ 0, zero iff equal)
per_token = −(surrogate − β · kl)
agg="token": Σ(per_token · mask) / Σ mask                  (DAPO: every token weighs the same)
agg="seq":   mean over sequences of [Σ_t per_token · mask / Σ_t mask]   (GRPO: every sequence weighs the same)
```

Why the aggregation matters: in `"seq"` mode a 1,000-token answer and a 10-token answer contribute equally, so each token of the long answer gets 1/100 of the weight. For a wrong answer (A < 0) that means a smaller penalty per token, so long wrong answers are punished less and length drifts up. `test_aggregation_modes_differ_with_uneven_lengths` makes it concrete: a 4-token sequence with A = +1 and a 1-token sequence with A = −1 give `−3/5` in token mode and exactly `0` in sequence mode.

The clipping case that the tests check: with A > 0 and a ratio already above 1 + ε, the min selects the clipped constant and the gradient is zero. Work through all four cases (A ≷ 0, ratio inside or outside the band) with the table in the curriculum before coding.

## 4. The toy verifiable task

`CounterPolicy` (given) is a tiny autoregressive policy: next-token logits from the previous token's embedding plus a position embedding, vocabulary 8, 5 tokens per completion. The prompt is a start token `p`; the reward (`counting_reward`, given) is the fraction of positions where the sequence continues the count, `seq[i] == (p + i + 1) mod 8`. A random policy scores about 1/8.

`train_counting` (given) runs 80 GRPO steps: 16 prompts × a group of 8, rewards → your `group_advantages` → two inner epochs of your `grpo_loss` on the same rollouts (the second epoch is off-policy by one update, so the ratio and clipping matter) with β = 0.01. `test_grpo_learns_the_counting_task` requires the mean reward to start below 0.3 and end above 0.7.

## What to implement

| # | function | test | what the test pins down |
|---|---|---|---|
| 1 | `exact_policy_gradient(theta, rewards)` | `test_reinforce_is_unbiased_and_baseline_reduces_variance` | the closed form $\pi \odot (r - \mathbb{E}_\pi r)$ |
| 2 | `reinforce_gradient(theta, rewards, n, baseline, generator)` | same test | unbiased within 0.01 at n = 200k; baseline halves the variance; uses the passed `generator` |
| 3 | `group_advantages(rewards, std_norm, eps)` | `test_group_advantages` | ±1 for `[1,0,0,1]`; zeros for an all-correct group; ±0.5 without std normalization |
| 4 | `grpo_loss(...)` | `test_grpo_loss_clipping_and_kl`, `test_aggregation_modes_differ_with_uneven_lengths` | loss −1 at identical policies with A = 1; zero gradient when clipped; k3 KL > 0; both aggregation modes |
| 5 | (given) `train_counting` | `test_grpo_learns_the_counting_task` | your pieces learn the task end to end |

## Tips

> [!TIP]
> For `reinforce_gradient`, sample with `torch.multinomial(pi, n, replacement=True, generator=generator)` and build the score function as `one_hot(a) − pi` for all samples at once. The test calls the function 600 times with the same generator; a Python loop over samples is slow and an ignored generator makes the variance comparison flaky.

- In `grpo_loss`, broadcast the per-sequence advantage with `adv[:, None]`, and apply the mask **after** computing per-token values (masked positions may hold arbitrary log-probs).
- Keep the KL term's sign straight: the loss is the negative of (surrogate − β·KL), so a larger KL increases the loss.
- `torch.minimum`, not `torch.min`, for the elementwise minimum of two tensors.

## Common bugs

- **Baseline that depends on the sampled action** (for example, subtracting that sample's own reward): biased, and the unbiasedness assertion fails.
- **`std(unbiased=True)`**: the `[1, 0, 0, 1]` group then gets ±0.866, not ±1.
- **Clipping the ratio without the min**, i.e. `clip(ratio) · A` alone: the clipped region still has zero gradient, but for A < 0 and a large ratio you lose the pessimistic unclipped term.
- **k1 instead of k3** for the KL (`logp_new − logp_ref`): can be negative, and the "KL > 0" test fails for some inputs.
- **Aggregation mixed up**: token mode divides the masked sum by the total number of valid tokens in the batch, not by N·T.
- **Gradients through `logp_old` or `logp_ref`**: both come from `torch.no_grad()` in real training; in the tests they are constants.

## GPU scale-up (S5): GRPO with LoRA on Qwen2.5-0.5B-Instruct

Start from your lab 10 SFT model (or the instruct model directly). Use a task with a programmatic checker. Two good choices:

- **Synthetic arithmetic:** generate problems like `What is 347 * 29? Put the final answer in \boxed{}.` on the fly. Infinite fresh data, a trivial checker, no contamination, and you control difficulty so the pass rate stays between 0 and 1.
- **GSM8K:** extract the number after the final `\boxed{}` (or the last number) and compare with the `#### <number>` answer.

One training step:

```
1. Sample B prompts (8–16); generate G = 8 completions each      torch.no_grad(), KV cache, temperature 1.0, ≤ 256 new tokens
2. Reward each completion with the checker (1 correct, 0 otherwise; 0 if unparseable)
3. adv = group_advantages(rewards.view(B, G), std_norm=...).view(-1)
4. logp_old = per-token log-probs of the completions under the current model   (no_grad, the training forward, not generate's scores)
5. logp_ref = the same with LoRA scale set to 0 (only if β > 0)
6. for 1–2 inner epochs, in micro-batches:
       logp_new = per-token log-probs with grad
       loss = grpo_loss(logp_new, logp_old, logp_ref, adv, mask, clip_eps=0.2, beta=β, agg="token")
       loss.backward()   (accumulate), then optimizer step
```

The response mask covers completion tokens up to and including the first end-of-turn token, and nothing after it. Compute `logp_old` with the same forward code as `logp_new`: sampler and trainer log-probs differ slightly (different kernels and batch shapes), and mixing them makes the ratio noisy.

**Memory plan:** B·G = 64 sequences of ~384 tokens. The fp32 logits alone for 8 of them are 8 × 384 × 151,936 × 4 bytes ≈ 1.9 GB, so run the training forward in micro-batches of 2–8 sequences and accumulate. Generation dominates wall-clock time: measure your sampling tokens/s first and size B, G and the completion length from it.

**Starting hyperparameters:** LoRA r = 16 on all linear layers; learning rate around 1e-5 to 5e-5 for LoRA (LoRA needs a higher LR than full fine-tuning); ε = 0.2; β ∈ {0, 0.04}; token-mean aggregation.

**Log every step:** mean reward; fraction of groups with zero variance; mean length of correct and of incorrect completions; mean token entropy of the policy on its completions (from the training forward's logits); clip fraction (share of tokens where the ratio left [1 − ε, 1 + ε]); KL to the reference.

**Reproduce one failure mode and one fix:**

- **Length bias:** run with `agg="seq"` and `std_norm=True` (original GRPO) and watch the length of incorrect completions; rerun with `agg="token"` and `std_norm=False`.
- **Entropy collapse:** with a symmetric ε = 0.2, entropy often falls quickly; add a separate upper bound (`clip_eps_high = 0.28`, DAPO's clip-higher) to your copy of `grpo_loss` and compare entropy and final accuracy.
- **Zero-signal groups:** make the task too easy (2-digit additions) and watch the zero-variance fraction approach 1 and learning stop; then filter those groups and resample (dynamic sampling) or raise the difficulty.

Evaluate the final policy against the starting model on a fixed held-out set with a paired bootstrap CI ([lab 15](../15_eval_stats/README.md)), and read 20 completions from each: check for checker exploits (several boxed answers, answers without reasoning, truncated outputs).

## Check yourself

1. Derive the exact gradient of $J(\theta) = \sum_a \mathrm{softmax}(\theta)_a\, r_a$.
2. Why does subtracting a baseline not bias the REINFORCE gradient, and why does it reduce variance?
3. A group has rewards `[1, 1, 1, 1, 1, 1, 1, 0]`. Compute the GRPO and Dr. GRPO advantages. Which prompt type does std normalization up-weight?
4. With A = −1 and ratio = 0.7 (ε = 0.2), what does the clipped objective pick and what is the gradient?
5. Why is k3 preferred over k1 as a per-token KL estimate?
6. Why is token-mean aggregation less biased toward long wrong answers than sequence-mean?
7. Why does the counting task use two inner epochs, and what would happen to clipping with only one?

<details><summary>Answers</summary>

1. With $\partial\pi_a/\partial\theta_b = \pi_a(\mathbb{1}[a=b] - \pi_b)$: $\partial J/\partial\theta_b = \sum_a r_a\pi_a(\mathbb{1}[a=b] - \pi_b) = \pi_b r_b - \pi_b\sum_a\pi_a r_a = \pi_b(r_b - \mathbb{E}_\pi r)$.
2. $\mathbb{E}[b\,\nabla\log\pi(a)] = b\sum_a\nabla\pi(a) = b\,\nabla 1 = 0$. Variance falls because $(r - b)$ is smaller in magnitude than $r$ when $b$ is near the mean, so each sample's score function is scaled by a smaller, centered weight.
3. Mean 0.875, population std ≈ 0.331. GRPO: ≈ +0.378 for each correct sample, ≈ −2.65 for the wrong one. Dr. GRPO: +0.125 and −0.875. Dividing by a small std inflates the advantages of prompts that are almost always right (or almost always wrong).
4. Ratio 0.7 < 0.8 with A < 0: the candidates are 0.7·(−1) = −0.7 and 0.8·(−1) = −0.8; the min picks −0.8, the clipped constant, so the gradient is zero. The probability has already fallen enough this update.
5. k1 ($-\log r$) is unbiased but can be negative per sample and has high variance; k3 ($r - 1 - \log r$) is also unbiased, always ≥ 0, and has lower variance when the policies are close.
6. In token mode every token has weight 1/(total tokens), so a long wrong answer is penalized in proportion to its length. In sequence mode its total weight is fixed, so the penalty per token shrinks as it gets longer.
7. In the first inner epoch `logp_new == logp_old`, so every ratio is 1 and clipping never triggers. The second epoch reuses the rollouts after one update, which is what clipping exists to control. With one epoch, GRPO reduces to REINFORCE with a group baseline (plus the KL term).
</details>

## Stretch

- **RLOO:** use the leave-one-out mean of the other G − 1 rewards as the baseline and compare learning curves with GRPO on the counting task.
- **Clip-higher and dynamic sampling** in the toy task: make rewards sparse (reward 1 only for the full correct sequence) so that most groups have zero variance, and show that filtering them speeds learning.
- **GSPO:** replace per-token ratios with a sequence-level, length-normalized ratio and compare stability on the S5 task.
- **PPO with a value head:** add a value head to `CounterPolicy`, compute GAE advantages per token, and compare sample efficiency with GRPO.
