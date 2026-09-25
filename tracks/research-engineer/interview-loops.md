# Interview loops (the durable parts)

Company-specific formats change every year; check [career/market-intel.md](../../career/market-intel.md) and ask the recruiter. The round *types* are stable:

| round | what it looks like | how to prepare |
|---|---|---|
| **Practical coding** | 60–90 min, multi-part, growing requirements (build a class, extend it, handle edge cases), in a real IDE; judged on correctness, clarity and speed | Timed labs rewritten from memory; practical builds like an LRU cache with TTL, a rate limiter, a tokenizer, an inference batcher |
| **ML coding** | implement attention, a sampler, beam search, a training loop, the DPO loss, top-p, BPE merges | [coding-interviews.md](coding-interviews.md) drills: every lab's core in 15–30 min without references |
| **ML debugging** | a broken training script or loss curve; find the bugs | [module 03 §7](../../curriculum/03-deep-learning.md#7-debugging-a-training-run-the-playbook); plant bugs in your own labs for a friend and swap |
| **Napkin math / systems** | "how long to train X on Y?", "why is decode slow?", "how many GPUs to serve Z?" | [lab 03](../../labs/03_napkin_math/README.md) drills until each takes < 2 min |
| **ML system design** | design an LLM serving platform, RAG, an RL training pipeline, an eval platform | [ml-system-design.md](ml-system-design.md) |
| **Research / project deep dive** | defend every number on your résumé; "what would you try next?", "what was surprising?" | Journal write-ups; rehearse two projects at three depths (1, 5 and 20 minutes) |
| **Values / mission** | safety views, trade-offs, what you'd refuse to build | [career/stories.md](../../career/stories.md#values-and-mission-prep), [module 09](../../curriculum/09-interpretability-and-safety.md) |
| **Behavioral** | STAR stories about ownership, conflict, failure | [career/stories.md](../../career/stories.md) |
| **DSA (Big Tech)** | 2 LeetCode mediums in 45 min | [dsa-patterns.md](dsa-patterns.md); 3–4 h/week maintenance |

## Mock protocol
From month 6: one mock every two weeks, alternating ML coding, system design and deep dive. Record yourself, then write a debrief (what was asked, where you stalled, what to drill). Trade mocks with peers from GPU MODE or EleutherAI, or use a paid service before the final rounds.

## The two failure modes to avoid
1. **Breadth without depth:** naming ten techniques and deriving none. Interviewers go three "why"s deep on one.
2. **Unverifiable claims:** every number on your résumé will be probed. If you can't reproduce the reasoning behind it on a whiteboard, remove it.
