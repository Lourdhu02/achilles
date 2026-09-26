# Lab 14 — Speculative decoding

**Build:** one round of speculative sampling with exact rejection sampling, the acceptance rate α = Σ min(p, q), and the formulas for tokens per target pass and end-to-end speedup, each checked against simulation. Then a real draft–target pair on your 8 GB GPU.
**Time:** 2–3 h for the tests, 4–6 h for the scale-up · **Reads first:** [inference §3](../../curriculum/07-inference.md#3-speculative-decoding-done-correctly)
**Run:** `pytest labs/14_speculative_decoding` (your code) · `pytest labs/14_speculative_decoding --impl=solution` (reference). The three tests run on CPU in a few seconds.

Every major serving engine ships speculative decoding, and it is the cleanest example of a systems speedup that provably changes nothing about the output distribution. Interviewers ask for the acceptance rule, the proof that it is exact, the expected speedup, and when it stops paying. This lab has you prove the first, simulate the second and derive the rest.

---

## 1. Why verification is nearly free

Batch-1 decode is memory-bound: each step streams every weight once ([lab 03](../03_napkin_math/README.md), rule 9). A target forward pass over γ + 1 positions reads the same weights as a pass over one, so it costs about the same time. A cheap drafter q proposes γ tokens one at a time, and one target pass computes p at all γ + 1 positions:

```
prefix ──► draft q: x1 x2 ... xγ      (γ cheap steps, keep q(·) at each)
       ──► target p: one pass over prefix + x1..xγ  ->  p(·) at γ+1 positions
       ──► accept x1, x2, ... left to right; stop at the first rejection
```

## 2. The accept rule, and why the output is exactly p

For draft token x at some position, with target p and drafter q there:

- accept with probability $\min\big(1, p(x)/q(x)\big)$;
- on rejection, emit a sample from the residual $r = \mathrm{norm}\big(\max(0,\ p - q)\big)$ and end the round;
- if all γ drafts are accepted, emit one bonus token from p at position γ + 1.

**Proof for one position.** Let $\alpha = \sum_x \min(p(x), q(x))$. The rejection probability is $\sum_x q(x)\big(1 - \min(1, p(x)/q(x))\big) = \sum_x \big(q(x) - \min(p(x), q(x))\big) = 1 - \alpha$, and the residual's normalizer is $\sum_x \max(0, p(x) - q(x)) = 1 - \alpha$ as well. So

$$P(\text{emit } x) = \underbrace{\min\big(p(x), q(x)\big)}_{\text{drafted and accepted}} + (1-\alpha)\,\frac{\max\big(0,\ p(x) - q(x)\big)}{1-\alpha} = p(x).$$

Accepted tokens extend the prefix exactly as target sampling would, so the argument repeats at the next position. After a rejection the remaining drafts were conditioned on a token that was not emitted, so they must be discarded. Induction over positions gives the whole sequence. Note that $\alpha = 1 - \mathrm{TV}(p, q)$: the acceptance rate is one minus the total-variation distance between the two models. In the test's toy chain, α is 0.6 after token 0 and 0.5 after tokens 1 and 2.

## 3. Tokens per target pass and speedup

Assume each draft is accepted independently with probability α. At least i drafts are accepted with probability αⁱ, and a round emits (accepted drafts + 1) tokens, since a rejection or the bonus always adds one:

$$\mathbb{E}[\text{tokens per target pass}] = \sum_{i=0}^{\gamma} \alpha^i = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}.$$

A round costs γ draft steps and one target pass. With c = draft step time / target step time:

$$\text{speedup} = \frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(\gamma c + 1)}.$$

With c = 0.05:

| γ | α = 0.6: tokens per pass | speedup | α = 0.8: tokens per pass | speedup |
|---|---|---|---|---|
| 1 | 1.60 | 1.52× | 1.80 | 1.71× |
| 2 | 1.96 | 1.78× | 2.44 | 2.22× |
| 3 | 2.18 | 1.89× | 2.95 | 2.57× |
| 4 | 2.31 | **1.92×** | 3.36 | 2.80× |
| 5 | 2.38 | 1.91× | 3.69 | 2.95× |
| 6 | 2.43 | 1.87× | 3.95 | 3.04× |
| 7 | 2.46 | 1.82× | 4.16 | 3.08× |
| 8 | 2.47 | 1.77× | 4.33 | **3.09×** |
| 9 | 2.48 | 1.71× | 4.46 | 3.08× |
| 10 | 2.49 | 1.66× | 4.57 | 3.05× |

Tokens per pass saturate at 1/(1 − α) while the draft cost keeps growing, so each α has a best γ. Speculation pays only when c < (E[tokens] − 1)/γ: at γ = 4 that is c < 0.59 for α = 0.8 and c < 0.33 for α = 0.6. For a draft costing c = 0.32 of the target (the parameter ratio of the pair in the scale-up), the best is γ = 3 at 1.51× for α = 0.8 and γ = 1 at 1.21× for α = 0.6. The formula assumes i.i.d. acceptance (real acceptance varies with content and position) and free verification of γ + 1 tokens, which fails at large batch, where decode approaches the compute roof.

## 4. Greedy versus sampled verification

At temperature 0 the target distribution is one-hot at its argmax. Plug that into the rule: a draft is accepted exactly when it equals the argmax, and the residual is the argmax itself. So "accept while the draft matches the target's argmax" is the exact rule for greedy decoding, and only for greedy decoding. With temperature > 0 it returns the target's greedy text instead of a sample.

Two details matter on real models. Apply the same temperature, top-k and top-p processing to p and q before computing ratios; the guarantee holds for the processed target distribution. And q must be the distribution the draft was actually sampled from: a drafter that picks its argmax is exact if you set q to that one-hot vector (then acceptance is p(x) and the residual is p without x), and wrong if you plug in its softmax.

## 5. The statistical test

`test_output_distribution_is_exactly_the_target` runs speculative rounds (γ = 2) from prefix (0,) until two tokens are out, 30,000 times, and compares the empirical distribution of the 9 possible token pairs with the exact target probabilities p(a | 0) p(b | a) by total-variation distance, requiring TV < 0.015. Two tokens, not one, because many bugs only act after the first rejection.

An exact sampler has TV noise from finite samples: over 2,000 simulated runs of 30,000, the mean was 0.0056, the 99th percentile 0.010 and the maximum 0.013. The threshold therefore almost never fails a correct implementation, and it catches every common bug by a wide margin:

| implementation (30,000 runs, same seed) | TV |
|---|---|
| correct | 0.009 |
| no bonus token (still exact, but fewer tokens) | 0.005 |
| on rejection, resample from p instead of the residual | 0.17 |
| keep verifying drafts after a rejection | 0.18 |
| ratio inverted, min(1, q/p) | 0.52 |
| greedy argmax check while sampling | 0.64 |
| no verification (emit the drafts; exact TV) | 0.52 |

`test_tokens_per_step_matches_theory` uses p and q that ignore the prefix, so acceptance really is i.i.d. with α = 0.8. Over 20,000 rounds with γ = 4 the mean round length must be within 2% of 3.3616. The round length has a standard deviation of 1.60, so the standard error is 0.011 and 2% is about 6 standard errors. Dropping the bonus token lowers the mean to 2.95 and fails.

## 6. Drafting without a separate model

**Medusa** (Cai et al., [2401.10774](https://arxiv.org/abs/2401.10774)) adds extra decoding heads to the target's last hidden state; head k predicts the token k + 1 positions ahead. The top candidates from each head form a tree, and one target pass with a tree-shaped attention mask verifies every branch and keeps the longest accepted prefix. There is no second model or KV cache, but the heads must be trained, and each head guesses without seeing the earlier guesses, which caps acceptance. The paper also proposes a "typical acceptance" rule that is faster but does not preserve the target distribution.

**EAGLE** (Li et al., [2401.15077](https://arxiv.org/abs/2401.15077)) drafts with a small autoregressive head, about one transformer layer, that runs on the target's own last hidden states plus the embedding of the token sampled one step ahead, and reuses the target's LM head. Drafting in feature space with the sampled tokens as input gives high acceptance, and verification uses the standard rule, so it is exact. EAGLE-2 ([2406.16858](https://arxiv.org/abs/2406.16858)) grows the draft tree from the draft's confidence; EAGLE-3 ([2503.01840](https://arxiv.org/abs/2503.01840)) fuses features from several layers.

**Lookahead decoding** (Fu et al., [2402.02057](https://arxiv.org/abs/2402.02057)) needs no draft model and no training. It runs Jacobi-style parallel guessing of future tokens with the target itself, collects the n-grams that appear along those trajectories, and verifies promising n-grams in the same forward pass; under greedy decoding the output is exact. **Prompt lookup** is the simplest relative: copy the continuation of the last few tokens from the prompt. It costs nothing and does well on editing, RAG and code.

## What to implement

| # | function | test | what the test pins down |
|---|---|---|---|
| 1 | `speculative_step(p_fn, q_fn, prefix, k, rng)` → list of 1 to k + 1 tokens | `test_output_distribution_is_exactly_the_target`, `test_tokens_per_step_matches_theory` | the output distribution equals the target's (TV < 0.015 over 30,000 runs); mean length matches theory within 2% |
| 2 | `acceptance_rate(p, q)` | `test_tokens_per_step_matches_theory` | Σ min(p, q) = 0.8 for p = [0.5, 0.3, 0.2], q = [0.3, 0.3, 0.4] |
| 3 | `expected_tokens(alpha, k)` | `test_formulas`, `test_tokens_per_step_matches_theory` | exactly 1.0 at α = 0 and 6.0 at α = 1, k = 5 (handle the 0/0); 3.3616 at α = 0.8, k = 4 |
| 4 | `expected_speedup(alpha, k, cost_ratio)` | `test_formulas` | (1 − 0.8⁵)/0.2/1.2 = 2.80 at α = 0.8, k = 4, c = 0.05 |

Models are plain functions from a prefix tuple to a NumPy probability vector. `k` in the code is γ in this handout.

## Tips

> [!TIP]
> Keep every q vector you sampled a draft from. The ratio must use exactly the distribution the draft came from; recomputing it later, with a different temperature or random state, silently breaks exactness.

- For draft i, the target distribution is `p_fn(prefix + tuple(drafts[:i]))`: conditioned on the drafts before i, not including i. In a real model all of these come from one forward pass, where the logits at position t predict token t + 1.
- Draw all randomness from the `rng` you are given (`rng.choice(len(q), p=q)`, `rng.random()`), so runs are reproducible.
- Return as soon as a draft is rejected; sample the bonus token only when all k are accepted.

## Common bugs

- **Off-by-one in the target position**, using `drafts[:i + 1]`: TV 0.61 in the same simulation, far above 0.015.
- **Resampling from p on rejection**, or continuing after a rejection: TV 0.17 and 0.18 in the table above.
- **Forgetting the bonus token:** still exact, but 12% fewer tokens per round, and `test_tokens_per_step_matches_theory` fails.
- **`expected_tokens(1.0, k)` divides by zero:** return k + 1.
- **On real models:** not truncating both KV caches to the accepted length after a round; applying temperature to one model only; or pairing models whose tokenizers differ.

## CPU experiments (no GPU needed)

1. **Noise versus bias:** plot TV against the number of runs (10³ to 10⁵) for the correct step and for one buggy variant. The correct one falls like 1/√n; the bug levels off at its bias.
2. **Non-i.i.d. acceptance:** in the test's Markov chain α depends on the last token. Measure mean tokens per round for γ = 1…6 and compare with the formula at the average α.
3. **Temperature:** build p and q as softmax(logits / T) from two correlated random logit vectors and plot α against T. Explain both ends.
4. **Greedy drafter, sampled verification:** make the drafter emit its argmax, set q to the one-hot vector, and confirm with the TV test that the output is still exactly p.

## GPU scale-up (8 GB)

Qwen2.5-0.5B-Instruct (0.49 B parameters) drafts for Qwen2.5-1.5B-Instruct (1.54 B); both use the same tokenizer, and in bf16 they take 4.1 GB together. A free Colab or Kaggle T4 works too; it has no bf16 tensor cores, so load both in fp16 there. Check that both configs have the same `vocab_size`; if the padded vocabularies differ, compare only the first `len(tok)` logits.

Predict α before building anything, by teacher forcing both models over text the target generated:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")   # older transformers versions: torch_dtype= below, not dtype=
target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.bfloat16).to("cuda")
draft = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.bfloat16).to("cuda")

@torch.no_grad()
def alpha_per_position(ids, n_prompt, temperature=1.0):
    """sum_x min(p_t(x), q_t(x)) at each generated position of ids (shape 1 x L)."""
    V = len(tok)
    p = torch.softmax(target(ids).logits[0, n_prompt - 1:-1, :V].float() / temperature, -1)
    q = torch.softmax(draft(ids).logits[0, n_prompt - 1:-1, :V].float() / temperature, -1)
    return torch.minimum(p, q).sum(-1)
```

1. **Predict.** Generate 20 answers from the target for code prompts (for example "add type hints to this function" with the function in the prompt) and 20 for open-ended chat. Average α per workload, measure c by timing 100 cached single-token decode steps of each model, and predict tokens per pass and speedup with your lab functions.
2. **Build.** Draft γ tokens with the 0.5B and its KV cache, run one target forward over the γ + 1 new positions, apply your accept rule to the real probabilities, and crop both caches to the accepted length (`DynamicCache` has a `crop` method). At greedy, check that your output matches target-only greedy decoding; batched verification uses different kernels, so a rare near-tie can differ.
3. **Measure.** Report α, mean tokens per pass, tokens/s and speedup against the prediction, for both workloads and for greedy and T = 0.7. Then compare with the library's version, `target.generate(**inputs, assistant_model=draft, ...)`.

> [!WARNING]
> At batch 1 in eager PyTorch, small models are often limited by kernel-launch overhead rather than bandwidth, so the measured c can be far above the 0.32 parameter ratio. If c approaches the break-even in §3, speculation cannot help until you cut the overhead (a static cache with `torch.compile`, or CUDA graphs).

## Check yourself

1. p = [0.5, 0.3, 0.2] and q = [0.3, 0.3, 0.4]. What is α? If the drafter proposes token 2, what is the acceptance probability, and what does the residual sample from?
2. α = 0.8, γ = 4, c = 0.05: tokens per target pass and speedup?
3. Show in two lines that an emitted token has distribution p.
4. Why must a round end at the first rejection, even when later drafts would have been accepted?
5. Your drafter has c = 0.32 and α = 0.6. Which γ, and what speedup do you expect? What would you change?
6. A colleague verifies with "accept while the draft equals the target's argmax" at temperature 0.8. What goes wrong?
7. Why does speculative decoding help less at batch 64 than at batch 1?

<details><summary>Answers</summary>

1. α = 0.3 + 0.3 + 0.2 = 0.8. Token 2 is accepted with probability min(1, 0.2/0.4) = 0.5. The residual is max(0, p − q) = [0.2, 0, 0], normalized to [1, 0, 0]: a rejection always emits token 0.
2. (1 − 0.8⁵)/0.2 = 3.36 tokens per pass; 3.36 / (4 · 0.05 + 1) = 2.80×.
3. P(x) = q(x)·min(1, p(x)/q(x)) + (1 − α)·max(0, p(x) − q(x))/(1 − α) = min(p(x), q(x)) + max(0, p(x) − q(x)) = p(x).
4. The later drafts were sampled conditioned on the rejected token, which is not in the output. Their p and q belong to a prefix that no longer exists, so accepting them would sample from the wrong conditional distribution (TV 0.18 in the lab's toy).
5. γ = 1 gives 1.21×, the best available; longer drafts lose. Raise α (a better-matched drafter, EAGLE-style heads, prompt lookup for copy-heavy tasks) or lower c.
6. Every emitted token is the target's argmax, so the output is greedy text, not a sample at temperature 0.8. In the lab's toy chain that is TV 0.64 from the correct distribution.
7. At large batch, decode approaches the compute roof, so verifying γ + 1 tokens per sequence costs close to γ + 1 times a normal step, and rejected drafts become wasted compute. Engines shorten γ or switch speculation off under load.
</details>

## Stretch

- **Tree drafts:** draft the top 2 tokens at each position, verify them with a tree attention mask, and work out an acceptance rule for several candidates that keeps the output exact; test it with the TV test.
- **Prompt lookup:** draft by copying from the prompt on a code-editing workload and compare α and speedup with the 0.5B drafter.
- **Adaptive γ:** set γ each round from a running estimate of α and c, and compare with the best fixed γ.
- **A Medusa-style head:** train one linear head on the frozen 1.5B's last hidden state to predict token t + 2, and measure its acceptance as a one-token drafter.
