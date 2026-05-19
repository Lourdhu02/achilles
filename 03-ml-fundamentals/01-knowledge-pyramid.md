# The ML Knowledge Pyramid

A visual + textual mental model for "what do I really need to know vs. just recognize?"

---

## The pyramid

```
                    ╱╲
                   ╱  ╲          TIER 5 — Pioneer
                  ╱    ╲         (publish papers; not your league for 2027)
                 ╱──────╲
                ╱        ╲       TIER 4 — Expert
               ╱          ╲      (explain at workshop level; build novel things)
              ╱────────────╲     [YOUR GenAI TIER 4 by Month 8]
             ╱              ╲    TIER 3 — Practitioner
            ╱                ╲   (use confidently in production; debate tradeoffs)
           ╱──────────────────╲  [Most ML topics by Month 8]
          ╱                    ╲ TIER 2 — Familiar
         ╱                      ╲(can explain, won't surprise you in an interview)
        ╱────────────────────────╲ [Adjacent topics like CV/audio basics]
       ╱                          ╲
      ╱     TIER 1 — Aware         ╲ (heard of it, can describe at high level)
     ╱──────────────────────────────╲ [RL, GNNs, time series, federated, etc.]
    ╱                                ╲
    ──────────────────────────────────
```

---

## Your target by Month 12

| Tier | Coverage % of topics | Examples |
|---|---|---|
| Tier 4 (Expert) | ~10% | GenAI/LLM internals, transformers, RAG, agents, fine-tuning, inference optimization |
| Tier 3 (Practitioner) | ~30% | Classical ML, optimization, regularization, CNNs, eval metrics, ML systems |
| Tier 2 (Familiar) | ~30% | CV beyond OCR, audio/speech, recommender systems, A/B testing |
| Tier 1 (Aware) | ~30% | RL, GNNs, time series, federated, causal inference, quantum ML |

---

## What each tier means in interview terms

### Tier 1 (Aware) — "I've heard of it"

What you can do:
- Define the term in one sentence
- Identify it when mentioned
- Acknowledge if asked, "what would you study to learn this?"

Example: Federated Learning
> "It's a paradigm where multiple devices/servers train models on their local data and only share gradients or model updates (not the data itself). Often used for privacy-preserving ML or when data can't be centralized."

That's it. You don't need deeper.

### Tier 2 (Familiar) — "I can explain the core idea"

What you can do:
- Explain mechanism at high level
- Give 1-2 example use cases
- Compare with related approaches

Example: Recommender Systems (collaborative filtering)
> "Collaborative filtering predicts user preferences based on similar users' preferences. Two main types: user-based ('users like you also liked X') and item-based ('users who liked this also liked Y'). Modern approaches use matrix factorization (factor user-item interaction matrix into latent factors). Hybrid methods combine collaborative + content-based. Cold start is a known problem."

### Tier 3 (Practitioner) — "I've used this in production / I can debate tradeoffs"

What you can do:
- Explain mechanism in depth
- Discuss tradeoffs vs alternatives
- Give specific examples from your own work
- Implement key parts from scratch
- Discuss failure modes

Example: Backpropagation
> "Backprop is the algorithm for computing gradients of a loss with respect to network parameters, using the chain rule of calculus. For a network with parameters θ and loss L:
> - Forward pass: compute predictions and L
> - Backward pass: compute ∂L/∂θ_l for each layer l, propagating from output back
> - Update: θ ← θ - η · ∂L/∂θ
>
> Common implementations use automatic differentiation (autodiff), which computes exact gradients via the computational graph. Frameworks like PyTorch implement reverse-mode autodiff.
>
> Failure modes: vanishing gradients (sigmoid in deep nets), exploding gradients (large initializations + no clipping), bottleneck gradients (skip connections help here).
>
> I implement backprop from scratch in numpy at least once a year as practice."

### Tier 4 (Expert) — "I can teach this and build novel things"

What you can do:
- Derive math from first principles
- Design new variants
- Critique recent research papers in the area
- Have nuanced opinions on contested choices

Example: Transformer attention (your GenAI Tier 4 target)
> Beyond "what is attention" — you can discuss MHA vs MQA vs GQA vs MLA, FlashAttention vs PagedAttention, sliding-window attention, sparse attention, why softmax over alternatives, length generalization issues, the actual gradient flow during training, etc.

### Tier 5 (Pioneer) — Research-level

What you can do:
- Originate new techniques
- Publish papers
- Push field forward

You don't need this for August 2027.

---

## How to climb the pyramid

For any topic, the climb is:

```
Tier 0 (don't know): Read 1 article. Now Tier 1.
Tier 1 → 2: Read a chapter or video. Take notes. Explain it to yourself.
Tier 2 → 3: Implement it. Use it in a project. Discuss tradeoffs.
Tier 3 → 4: Read 3+ research papers. Build a novel variant or extension. Teach others.
Tier 4 → 5: Original research. Years of focus.
```

**Most of your topics are Tier 0 → 3.** GenAI/LLM topics are Tier 3 → 4.

---

## The 80/20 of ML interviews

| Time spent | Topics |
|---|---|
| 30% | Transformer / attention / LLM internals |
| 20% | Classical optimization, regularization, loss functions |
| 15% | Eval metrics |
| 15% | Production ML (deployment, monitoring, A/B testing) |
| 10% | Specific applied areas (NLP, CV, etc.) |
| 10% | Other (RL, GNN, etc.) |

If you nail the top 80% (transformers + classical foundations + evals + prod), you handle most interviews.

---

## Your topic priority list (for time investment)

### Highest ROI (deepen to Tier 3-4):
- Attention + transformers (Tier 4)
- LLMs: tokenization, RoPE, KV cache, etc. (Tier 4)
- Fine-tuning: LoRA, QLoRA, RLHF, DPO (Tier 4)
- RAG architectures + eval (Tier 4)
- Agents + tool calling (Tier 4)
- Inference optimization: vLLM, FlashAttention (Tier 3-4)
- Optimization: Adam, AdamW, schedules (Tier 3)
- Loss functions, regularization, normalization (Tier 3)
- Eval metrics across NLP/CV/general (Tier 3)

### Medium ROI (Tier 2-3):
- CNNs, ResNet, modern CV architectures (Tier 2-3 — your CV work helps)
- Classical ML: trees, ensembles, boosting (Tier 2-3)
- Recommender systems (Tier 2-3, since many companies ask)
- ML systems: deployment, monitoring, A/B testing (Tier 3 — your prod experience helps)

### Lower ROI (Tier 1-2):
- RNN/LSTM (Tier 1-2; rarely deep questions in 2026)
- Reinforcement learning (Tier 1-2; mainly for specific roles)
- Graph neural networks (Tier 1)
- Time series (Tier 1)

---

## Diagnostic test: where are you NOW?

Honestly self-rate on Tier 1-5 for these key topics. Update this every 3 months.

| Topic | Current tier (May 2026) | Target tier (Mar 2027) |
|---|---|---|
| Transformer attention math | 3 | 4 |
| Multi-head attention | 3 | 4 |
| Positional encodings (RoPE, ALiBi) | 2 | 4 |
| Tokenization (BPE) | 2 | 4 |
| KV cache mechanics | 2 | 4 |
| FlashAttention | 1 | 3 |
| vLLM / PagedAttention | 2 | 4 (you have inference exp) |
| LoRA / QLoRA | 3 | 4 |
| RLHF / DPO | 2 | 3-4 |
| RAG architectures | 4 (FinSentinelAI!) | 4 |
| Agent design | 4 (ECHOME!) | 4 |
| Adam optimizer | 3 | 3 |
| Batch norm vs Layer norm | 2 | 3 |
| Classical ML (trees/boosting) | 3 | 3 |
| CNN modern architectures | 3 | 3 |
| RNN/LSTM | 2 | 2 |
| Eval: MMLU, HellaSwag | 2 | 4 |
| A/B testing for ML | 2 | 3 |
| Distributed training | 1 | 3 |
| ML system design | 1 | 4 |

---

## Action: every quarter, run this diagnostic

Mark gaps. Schedule study to close them. Re-test.

---

Next: [`02-classical-ml.md`](./02-classical-ml.md) (stub — expand during Month 2-3 of your study)
