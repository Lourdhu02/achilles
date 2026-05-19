# 30 Must-Know Papers for GenAI Engineering Interviews

If asked "what papers have you read recently?" — these are your defensible references.

You don't need to memorize them. You need to:
- Know the title and 1-sentence motivation
- Know the key insight
- Have an opinion on its impact
- Be able to answer 1-2 follow-ups

---

## Tier 1: Foundational (must know cold) — 10 papers

### 1. Attention Is All You Need (Vaswani et al., 2017)
- **Key idea:** Transformer architecture; replace RNNs with self-attention
- **Why know:** Every modern LLM derives from this
- **Follow-ups:** Walk through encoder vs decoder; why multi-head; why √d_k

### 2. BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)
- **Key idea:** Masked language modeling + next-sentence prediction for bidirectional understanding
- **Why know:** Encoder-only paradigm; sparked the pretraining era
- **Follow-ups:** Why is BERT bad for generation? Why has it been overtaken by GPT-style?

### 3. Language Models are Few-Shot Learners (GPT-3, Brown et al., 2020)
- **Key idea:** In-context learning; massive scale (175B); decoder-only
- **Why know:** The paper that started the LLM era as we know it
- **Follow-ups:** What is in-context learning? Why does it emerge with scale?

### 4. Chinchilla / Training Compute-Optimal LLMs (Hoffmann et al., 2022)
- **Key idea:** For fixed compute budget, optimal is more data + smaller model than previously thought
- **Why know:** Reshaped how everyone trains LLMs
- **Follow-ups:** What's "Chinchilla-optimal"? Why don't we always follow it now (hint: inference cost)

### 5. Training Language Models to Follow Instructions (InstructGPT, Ouyang et al., 2022)
- **Key idea:** SFT + RLHF pipeline for alignment
- **Why know:** ChatGPT's recipe
- **Follow-ups:** Walk through the three-stage pipeline; reward hacking

### 6. Direct Preference Optimization (Rafailov et al., 2023)
- **Key idea:** RLHF without explicit reward model; closed-form solution
- **Why know:** Replaced RLHF for many open models
- **Follow-ups:** Why is DPO more stable than PPO? What's the math?

### 7. LoRA: Low-Rank Adaptation of LLMs (Hu et al., 2021)
- **Key idea:** Fine-tune via low-rank matrices ΔW = BA on frozen base
- **Why know:** Standard fine-tuning technique
- **Follow-ups:** What rank to use? When LoRA fails?

### 8. QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)
- **Key idea:** 4-bit base + LoRA → fine-tune big models on single GPU
- **Why know:** Democratized fine-tuning
- **Follow-ups:** What's NF4? How does dequantization work in forward pass?

### 9. FlashAttention (Dao et al., 2022)
- **Key idea:** Tile-based attention to reduce HBM I/O
- **Why know:** Standard inference optimization
- **Follow-ups:** What's the memory hierarchy issue? Online softmax?

### 10. ReAct: Synergizing Reasoning and Acting (Yao et al., 2022)
- **Key idea:** Interleave thought + action + observation for agents
- **Why know:** Foundation of modern agent frameworks
- **Follow-ups:** When does ReAct fail? Alternatives?

---

## Tier 2: Architecture & training tricks — 10 papers

### 11. Llama 2 paper (Touvron et al., 2023)
- **Key:** Open weights frontier model + RLHF details
- **Follow-ups:** SFT data quality; safety training

### 12. Llama 3 paper (Meta, 2024)
- **Key:** Scaled to 405B, new tokenizer, more data
- **Follow-ups:** What changed from Llama 2?

### 13. Mistral 7B (Mistral AI, 2023)
- **Key:** Sliding window attention; competitive 7B model
- **Follow-ups:** Why GQA? Why SWA?

### 14. Mixtral 8x7B (Mistral AI, 2023)
- **Key:** Mixture of Experts; active params < total
- **Follow-ups:** How does gating work? Tradeoffs of MoE?

### 15. DeepSeek V3 / R1 (DeepSeek, 2024-2025)
- **Key:** MLA, MoE, ultra-efficient training; R1 = RL-only reasoning
- **Follow-ups:** What's MLA? Why no SFT in R1?

### 16. PaLM / PaLM 2 (Chowdhery et al., 2022-2023)
- **Key:** 540B model; pathways system; SwiGLU; multilingual
- **Follow-ups:** Engineering at scale; multilingual evals

### 17. Gemini family (Google DeepMind, 2023-2024)
- **Key:** Multimodal from the ground up
- **Follow-ups:** How does Gemini handle long context (1M+ tokens)?

### 18. Claude 3 / 4 family (Anthropic, 2024-2026)
- **Key:** Constitutional AI; long context; capabilities + safety
- **Follow-ups:** What's Constitutional AI? RSP (Responsible Scaling Policy)?

### 19. RoPE: Rotary Position Embedding (Su et al., 2021)
- **Key:** Rotation-based relative position encoding
- **Follow-ups:** Math of rotation; length extrapolation tricks (YaRN, NTK)

### 20. GShard / Switch Transformer (Lepikhin et al., 2020 / Fedus et al., 2021)
- **Key:** MoE at scale; expert routing
- **Follow-ups:** Load balancing in MoE training

---

## Tier 3: RAG, agents, alignment — 10 papers

### 21. Retrieval-Augmented Generation (Lewis et al., 2020)
- **Key:** Original RAG paper; retriever + generator
- **Follow-ups:** Why RAG vs fine-tune?

### 22. ColBERT / ColBERTv2 (Khattab & Zaharia, 2020-2021)
- **Key:** Late interaction retrieval; better than single-vector embeddings
- **Follow-ups:** Tradeoffs of late vs early fusion

### 23. Lost in the Middle (Liu et al., 2023)
- **Key:** LLMs attend more to start/end of long contexts
- **Follow-ups:** Mitigation strategies; implications for RAG

### 24. Self-Consistency Improves Chain-of-Thought (Wang et al., 2022)
- **Key:** Sample multiple CoTs; majority vote
- **Follow-ups:** When does this help vs hurt?

### 25. Constitutional AI (Bai et al., 2022)
- **Key:** AI feedback for harmlessness training
- **Why know:** Anthropic's distinctive approach
- **Follow-ups:** Critique-revise loop; tradeoffs vs pure RLHF

### 26. Toolformer (Schick et al., 2023)
- **Key:** LLMs can self-train to use tools
- **Follow-ups:** Tool call frameworks today; advantages of post-hoc fine-tuning

### 27. MemGPT / Letta (Packer et al., 2023)
- **Key:** LLM as OS; memory paging
- **Follow-ups:** Memory tiers; when to use external memory

### 28. Anthropic Sleeper Agents (Hubinger et al., 2024)
- **Key:** Backdoors persist through safety training
- **Why know:** Important alignment paper
- **Follow-ups:** Implications for trust in LLMs

### 29. Reflexion (Shinn et al., 2023)
- **Key:** Self-critique and improve via verbal reinforcement
- **Follow-ups:** When does this help?

### 30. Sparks of AGI (Bubeck et al., 2023)
- **Key:** Microsoft's analysis of GPT-4 capabilities
- **Why know:** Controversial but influential
- **Follow-ups:** Your take on "AGI" framing

---

## How to study these efficiently

### Per paper, 2 hours each:
1. **Read abstract + intro + conclusion** (20 min)
2. **Skim figures and tables** (15 min)
3. **Find one explainer blog or video** (30 min)
4. **Write your own 5-bullet summary** (15 min)
5. **Practice articulating it out loud** in 90 seconds (15 min)
6. **Predict 3-5 follow-up questions** (15 min)
7. **Spaced re-review in 1 week** (10 min)

30 papers × 2 hours = 60 hours. Spread over Months 5-8 = ~3 papers/week. Doable.

---

## "What paper impressed you recently?" — a strong answer

Prepare 2-3 answers across different categories. Example:

> "I've been really impressed by DeepSeek V3's MLA — multi-latent attention. It's a clever way to compress the KV cache that's distinct from MQA/GQA. The latent compression idea generalizes — I think we'll see more variants exploring different compression bottlenecks. It particularly interests me because of my OCR work at Sujanix where we faced similar memory-bandwidth bottlenecks at inference time."

That answer:
- Names a specific paper
- Explains the core idea concisely
- Has a forward-looking opinion
- Connects to your experience

Practice 3 variants like this for different conversation directions.

---

## Tracking your paper reading

Add a `papers.md` to your `04-genai-llm-mastery/` folder (or use this file's checklist). For each:
- [ ] Read
- [ ] Wrote 5-bullet summary
- [ ] Practiced articulating
- [ ] Re-reviewed once

By end of Month 8: all 30 in Tier 1+2+3 done.

---

## Where to find papers + commentary

- **arxiv.org** — official source
- **arxiv-sanity** (Karpathy) — better browsing
- **Papers With Code** — links to implementations
- **Yannic Kilcher YouTube** — paper deep-dives
- **AI Coffee Break** — quick summaries
- **Latent Space podcast** — engineering-focused
- **The Gradient** — magazine-style commentary
- **HF papers daily** — daily new papers with discussion

---

Next: [`08-genai-qa-bank.md`](./08-genai-qa-bank.md)
