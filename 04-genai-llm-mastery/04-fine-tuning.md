# Fine-Tuning — When, How, Why

The hardest part is knowing **WHEN NOT to fine-tune.** Most "I need to fine-tune" cases are actually "I need better RAG" or "I need better prompts."

---

## When to fine-tune (vs RAG, vs prompting)

| Need | Solution |
|---|---|
| Add new factual knowledge | RAG (almost always) |
| Change behavior / style | Fine-tune (or system prompt) |
| Match a specific output format | Fine-tune (or examples in prompt) |
| Reduce inference cost (smaller model with same quality) | Fine-tune (distillation) |
| Domain-specific reasoning patterns | Fine-tune (with care) |
| Multilingual support not in base model | Fine-tune (continued pre-training) |

### When NOT to fine-tune
- You haven't tried good prompting yet
- You haven't tried RAG
- You don't have a clean, high-quality dataset (<500 examples is risky)
- You can't evaluate the fine-tuned model rigorously
- The base model is doing 90% of what you need

### The fine-tuning failure mode
- You spend weeks fine-tuning
- Result: marginal improvement, can't update behavior, more inference cost (your model not theirs), eval is hard
- Lesson: tried-and-failed RAG approaches are worth weeks before considering FT.

---

## Types of fine-tuning

### Full fine-tuning
- Update all parameters of pre-trained model
- Memory cost: 4x model size (params + gradients + optimizer state)
- For 7B model: ~120GB GPU memory needed → 2x A100 80GB
- Expensive; risks catastrophic forgetting of base capabilities
- Use when: you have huge dataset (100k+ examples) and significant compute

### Parameter-Efficient Fine-Tuning (PEFT)

The umbrella term. Major techniques:

#### LoRA (Low-Rank Adaptation) — most common
- Insert low-rank matrices in linear layers: `ΔW = BA` where rank(BA) << rank(W)
- Only train A and B (small)
- Math: instead of W' = W + ΔW, compute `output = Wx + BAx`
- Typical rank: 8-64
- Memory: 1-2% of full fine-tuning
- Merge LoRA weights back into base for inference (no inference overhead)

```python
# Conceptual LoRA forward
def lora_forward(W, A, B, x, alpha):
    """W: frozen original, A: down-proj, B: up-proj"""
    return W @ x + (alpha / rank) * (B @ (A @ x))
```

#### QLoRA
- Quantize base model to 4-bit (NF4 format)
- LoRA on top of quantized model
- Memory: ~24GB for 7B model fine-tuning
- Trick: dequantize on-the-fly during forward, use original LoRA gradients

#### Other PEFT
- **Prefix tuning:** prepend learned vectors to input
- **Prompt tuning:** learn soft prompt embeddings only
- **Adapter layers:** insert small trainable layers between transformer blocks
- **BitFit:** train only bias parameters

LoRA dominates in 2026. Others are mostly historical or niche.

---

## SFT (Supervised Fine-Tuning)

### Dataset format
Instruction-following format (Alpaca-style):
```json
{
  "instruction": "Translate the following to Tamil",
  "input": "Hello, world",
  "output": "வணக்கம், உலகம்"
}
```

Or chat format (OpenAI/Claude-style):
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of India?"},
    {"role": "assistant", "content": "The capital of India is New Delhi."}
  ]
}
```

### Loss
- Standard cross-entropy on next-token prediction
- Usually: mask the loss on instruction/input tokens (only train on output tokens)

### Dataset size
- 500-1,000 examples: barely works, high variance
- 1k-10k: solid for narrow domains
- 10k-100k: strong, can change behavior substantially
- 100k+: serious, can change broad capabilities

### Quality > quantity
- 1k carefully curated examples > 100k noisy
- LIMA paper (Meta 2023): 1000 high-quality samples → competitive performance

---

## Alignment fine-tuning (RLHF, DPO, etc.)

After SFT, models are helpful but may not be aligned (helpful, honest, harmless). Alignment training adjusts behavior to human preferences.

### RLHF (Reinforcement Learning from Human Feedback)

Pipeline (InstructGPT, original ChatGPT):
1. **SFT** on demonstration data → policy π_SFT
2. **Reward model (RM):** train classifier on preference pairs (A vs B)
3. **PPO:** RL fine-tune π_SFT to maximize RM score, with KL penalty from π_SFT

```
Reward = RM(response) - β · KL(π || π_SFT)
```

Issues:
- Unstable (PPO is finicky)
- Reward hacking (model finds spurious ways to get high RM score)
- Hyperparameter sensitivity
- 4 models in memory (policy, value, reference, reward)

### DPO (Direct Preference Optimization)

Insight: you can derive the optimal policy from preferences DIRECTLY, without explicit reward model.

Loss:
```
L_DPO(θ) = -E[log σ(β log π_θ(y_w|x)/π_ref(y_w|x) - β log π_θ(y_l|x)/π_ref(y_l|x))]
```

Where:
- y_w: preferred response
- y_l: rejected response
- π_ref: reference (SFT) model
- β: scaling factor

In English: increase likelihood of preferred, decrease likelihood of rejected, relative to reference model.

Advantages:
- Simpler (no RM, no PPO)
- More stable
- Comparable quality

Used in: Llama 3 alignment, Mistral, many open models.

### KTO (Kahneman-Tversky Optimization)
- Only need binary feedback (good/bad), not paired comparisons
- Cheaper to collect data
- Comparable to DPO

### IPO (Identity Preference Optimization)
- Addresses DPO's over-optimization (loss can degenerate)
- More robust

### Constitutional AI (Anthropic)
- Hybrid: use AI feedback (not human) for harmlessness
- Self-critique with a "constitution" (set of principles)
- Less human-labeled data needed
- The Claude alignment approach

---

## Fine-tuning practicalities

### Choosing base model
- **Open and strong:** Llama 3, Llama 3.1, Mistral 7B, Mixtral, Qwen2, DeepSeek-V3
- **Small (for cost-sensitive):** Llama 3.2 1B/3B, Phi-3, Qwen2 0.5B
- **License considerations:** check before commercial use (Llama has acceptable use policy; some others MIT)

### Hyperparameters (LoRA SFT defaults)
- Learning rate: 1e-4 to 3e-4
- LoRA rank: 16-32
- LoRA alpha: 32-64 (typically 2x rank)
- LoRA target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Batch size (effective): 32-128 via gradient accumulation
- Epochs: 1-3 (more = overfit)
- Warmup: 3% of steps

### Evaluation
- Held-out eval set
- Domain-specific metrics
- Compare against base model (NOT against itself across iterations)
- Watch for: regression on general capabilities (run on MMLU subset)

### Common pitfalls
- **Catastrophic forgetting:** model loses base abilities. Fix: lower LR, regularization, mix in some original data.
- **Overfitting to training data:** test set looks great, real users see issues. Fix: hold-out validation, smaller LR, fewer epochs.
- **Format leakage:** model only works in exact training format. Fix: vary prompt templates during training.
- **Underfitting:** LR too low or rank too small.

---

## Tools / libraries

### Training
- **TRL** (Hugging Face) — SFT, DPO, PPO, KTO; production-grade
- **Unsloth** — 2x faster LoRA training on consumer GPUs
- **Axolotl** — config-driven; great for experimentation
- **LLaMA Factory** — UI-based fine-tuning

### Quantization for inference
- **AutoAWQ** — Activation-aware Weight Quantization
- **AutoGPTQ** — GPT Quantization
- **bitsandbytes** — for QLoRA training quantization
- **llama.cpp / GGUF** — open-source quantized inference

### Distributed
- **DeepSpeed** — ZeRO stages 1/2/3, offloading
- **FSDP** (PyTorch native) — sharding
- **Accelerate** (HF) — easy multi-GPU
- **Megatron-LM** (Nvidia) — tensor parallel for very large models

---

## When to use which technique (decision tree)

```
Do you want to add knowledge or change behavior?
├── Knowledge → RAG
└── Behavior
    ├── Do you have <500 examples?
    │   ├── Yes → improve prompts; don't fine-tune
    │   └── No
    │       ├── Quality model > smaller cheaper model?
    │       │   ├── Quality matters: full FT or LoRA on big model
    │       │   └── Cost matters: distill into small model
    │       └── Behavior change or alignment?
    │           ├── Behavior (style, format): SFT
    │           └── Alignment (helpful/honest/harmless): SFT then DPO
```

---

## Cost benchmarks (for context)

LoRA fine-tune of 7B Llama on 10k examples (approximate, 2026):

| Hardware | Time | Cost (cloud) |
|---|---|---|
| 1× RTX 4090 (24GB) | 8-12 hrs | ~$10 |
| 1× A100 80GB | 4-6 hrs | ~$15 |
| 4× A100 (FSDP) | 1-2 hrs | ~$25 |
| 8× H100 | 30 min | ~$15-30 |

Full fine-tune of same: 5-10x more.
DPO on top of SFT: another 1-3 hours.

For Lourdu-tier projects: a single A100 + 6 hours = ₹2000 = a real experiment.

---

## Hands-on exercises (Month 6-7)

### Exercise 1: SFT a 1B model on your own writing
- Take a 1B model (Llama 3.2 1B)
- Collect 50-200 of your past tweets / blog posts as training data
- LoRA fine-tune to mimic your voice
- Compare outputs before/after

### Exercise 2: DPO on top
- Generate 100 outputs from your SFT model
- For each prompt, give it both your tweaked response and the original
- Use these as DPO preference pairs
- Train and compare

### Exercise 3: Quantize and serve
- Quantize your fine-tuned LoRA model to INT4 via AWQ
- Serve via vLLM
- Benchmark throughput
- Compare quality (your eval set)

This trilogy makes you genuinely capable, not just literate.

---

## Interview questions on this

1. "When would you fine-tune vs use RAG?" — decision tree above
2. "Explain LoRA. Why does it work?" — low-rank update, freeze base
3. "What's the difference between SFT and RLHF?" — supervised vs preference-based
4. "Walk me through DPO." — see formula above
5. "How would you collect preference data for RLHF?" — pairwise comparisons, raters
6. "What's catastrophic forgetting and how do you prevent it?"
7. "If I have only 100 examples, should I fine-tune?" — probably no, use prompting
8. "How would you choose LoRA rank?" — start at 16, ablate up/down
9. "Why is fine-tuning so popular even though it's expensive?" — model ownership, customization, behavior change

---

## Resources

- **"PEFT: Parameter-Efficient Fine-Tuning"** — Hugging Face docs (foundational)
- **"LoRA: Low-Rank Adaptation of Large Language Models"** — paper (Hu 2021)
- **"QLoRA: Efficient Finetuning of Quantized LLMs"** — paper (Dettmers 2023)
- **"Direct Preference Optimization"** — Rafailov 2023 (must read)
- **"Constitutional AI"** — Bai 2022 (Anthropic)
- **TRL docs** — Hugging Face
- **Unsloth docs** — practical optimization

---

Next: [`05-inference-optimization.md`](./05-inference-optimization.md)
