# Lab 12 — Policy gradients to GRPO

**Run:** `pytest labs/12_grpo` (your code) · `pytest labs/12_grpo --impl=solution` (reference)

**Reads first:** [post-training §5](../../curriculum/06-post-training.md#5-rl-with-verifiable-rewards-grpo-and-friends)

## What you implement (the tests check each property)
- `reinforce_gradient(...)` for a softmax bandit. Test: the Monte Carlo estimate matches the exact gradient `Σ π(a)(r(a) − E r)∇logπ(a)`, so it is unbiased; a baseline reduces its variance.
- `group_advantages(rewards[B, G], std_norm=True)`, with the Dr. GRPO variant (no std normalization).
- `grpo_loss(logp_new, logp_old, logp_ref, adv, mask, ε, β, agg)`: clipped ratio, k3 KL `e^{Δ} − Δ − 1`, and aggregation modes (sequence-mean vs token-mean). Test the clipping gradient: zero when A > 0 and ratio > 1 + ε.
- A toy verifiable task: a small autoregressive policy (previous-token and position embeddings) must emit a counting sequence from a prompt token. The reward must rise from ~0.13 to > 0.7 on CPU.

## GPU scale-up / stretch
**S5:** GRPO with LoRA on Qwen2.5-0.5B-Instruct for a verifiable task (arithmetic or countdown) on 8 GB: G = 8, short completions, β = 0 to 0.04. Plot reward, response length and entropy. Reproduce one failure mode (length hacking or entropy collapse) and one fix (DAPO clip-higher or token-mean).
