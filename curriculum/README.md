# Curriculum — first principles to frontier

Eleven modules. Each follows the same anatomy: **why it matters → mechanisms with derivations → napkin math → traps → lab → check yourself → papers.**
Read a module, then do its lab. Understanding shows up in what you build and measure, not in what you have read.

```
00 learning OS ─▶ 01 math ─▶ 02 compute & hardware ─▶ 03 deep learning ─▶ 04 transformers
                                                                              │
          ┌──────────────────────┬──────────────────────┬─────────────────────┤
          ▼                      ▼                      ▼                     ▼
   05 pretraining ─▶ 06 post-training          07 inference        10 applied LLM systems
          │                      │                      │                     │
          └──────────▶ 08 evaluation & research ◀───────┘                     │
                                 │                                            │
                     09 interpretability & safety              tracks/founder ◀┘
```

| # | module | labs | mastery target |
|---|---|---|---|
| 00 | [How to learn at this level](00-learning-os.md) | — | a weekly loop that produces artifacts |
| 01 | [Math](01-math.md) | 01 | derive any layer's backward pass; PPO/DPO from their objectives |
| 02 | [Compute & hardware](02-compute-and-hardware.md) | 03 | predict runtime/memory within 2× before measuring |
| 03 | [Deep learning](03-deep-learning.md) | 01, 02 | debug a diverging run from its curves |
| 04 | [Transformers](04-transformers.md) | 04, 05 | implement a modern LLM from a blank file |
| 05 | [Pretraining](05-pretraining.md) | 08, 09 | plan a 7B run end to end: data, compute, parallelism |
| 06 | [Post-training](06-post-training.md) | 10–12 | implement SFT/DPO/GRPO and name their failure modes |
| 07 | [Inference](07-inference.md) | 06, 07, 13, 14 | size and optimize a serving system |
| 08 | [Evaluation & research](08-evaluation-and-research.md) | 15 | run an ablation that survives review |
| 09 | [Interpretability & safety](09-interpretability-and-safety.md) | 16 | find a circuit; argue a safety position |
| 10 | [Applied LLM systems](10-applied-llm-systems.md) | 17 | ship an eval-driven RAG or agent product |

**Mastery levels** (rate yourself monthly in the journal): 0 heard of it · 1 can explain · 2 can derive · 3 implemented and tested it · 4 predicted the numbers before measuring · 5 extended it or found where it breaks.
Core-AI interviews probe levels 3–4. Research roles probe level 5.

Also: [papers.md](papers.md) (reading list by module) · [resources.md](resources.md) (courses, books, communities).
