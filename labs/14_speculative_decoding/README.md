# Lab 14 — Speculative decoding

**Run:** `pytest labs/14_speculative_decoding` (your code) · `pytest labs/14_speculative_decoding --impl=solution` (reference)

**Reads first:** [inference §3](../../curriculum/07-inference.md#3-speculative-decoding-done-correctly)

## What you implement (the tests check each property)
- `speculative_step(p_fn, q_fn, prefix, k)`: the draft proposes k tokens; accept each with probability `min(1, p/q)`; on the first rejection resample from `norm(max(0, p − q))`; if all k are accepted, sample a bonus token from p.
- **Correctness test:** over 100k runs with toy Markov models, the output distribution equals the target's exactly (small TV distance). The greedy "argmax matches" variant is lossless only under greedy decoding. Show that too.
- `expected_tokens(α, k) = (1 − α^{k+1})/(1 − α)` with `α = Σ min(p, q)`. Test against simulation.
- `expected_speedup(α, k, c)` where c = draft cost / target cost.

## GPU scale-up / stretch
Use Qwen2.5-0.5B as the draft for Qwen2.5-1.5B (both fit in 8 GB). Measure α and the real speedup on code and on chat prompts, and explain the difference.
