# GenAI-Specific Interview Q&A Bank

100+ questions specifically targeting GenAI / LLM Engineer interviews. Pair with `03-ml-fundamentals/06-interview-qa-bank.md` Section 3 for more transformer/LLM theory.

---

## Section A: Architecture & Internals (30 Qs)

1. Walk through what happens when GPT-4 processes the prompt "Hello world".
2. What's the difference between an encoder-only, decoder-only, and encoder-decoder transformer? Examples of each.
3. Why are most modern LLMs decoder-only?
4. Explain pre-norm vs post-norm. Why is pre-norm standard now?
5. What is RMSNorm and why is it preferred over LayerNorm in modern LLMs?
6. Walk through attention computation. Why divide by √d_k?
7. What is the KV cache? Why is it important for inference?
8. Compute the KV cache memory for Llama-70B at 32k context with GQA (8 KV heads).
9. What's MQA, GQA, MLA? When use which?
10. What is RoPE? Why is it popular?
11. How does YaRN extend context length using RoPE?
12. What's ALiBi and how does it compare to RoPE?
13. Why do transformers struggle with long contexts naturally?
14. What's the "lost in the middle" problem?
15. Explain SwiGLU. Why is it used instead of ReLU in FFN?
16. What's MoE? Walk through Mixtral's architecture.
17. How does the gating network work in MoE?
18. What's expert parallelism in distributed training?
19. Explain BPE tokenization with an example.
20. What's the difference between BPE, WordPiece, SentencePiece?
21. Why might 32k vocab not work for Tamil text well?
22. How does the LM head work? Why are embedding weights often tied?
23. What's a context window? Why does it cost so much?
24. How does Gemini handle 1M+ tokens? Architectural choices?
25. Explain speculative decoding.
26. What's lookahead decoding? Medusa?
27. Walk through the forward pass of a transformer block (encoder).
28. Why does scaling depth (more layers) sometimes hurt vs scaling width?
29. What's the role of position 0 in causal attention?
30. Explain the "attention sink" phenomenon and what causes it.

---

## Section B: Training (20 Qs)

31. What's pre-training, SFT, and RLHF? Walk through end-to-end.
32. Why warm up the learning rate? What happens without warmup?
33. Walk through Chinchilla scaling laws.
34. Why do modern LLMs train beyond Chinchilla-optimal data?
35. Explain RLHF in detail. Why is it unstable?
36. What's DPO? Compare to RLHF.
37. Derive the DPO loss intuition.
38. What's reward hacking? Examples?
39. What's KL penalty in RLHF? Why?
40. What's Constitutional AI (Anthropic)?
41. What's RLAIF?
42. Explain instruction-tuning datasets (Alpaca, Dolly, OpenAssistant).
43. What's data deduplication in pre-training? Why?
44. How would you collect preference data for DPO?
45. What's the role of synthetic data in modern training pipelines?
46. Walk through how you'd train a 70B model from scratch (compute, data, time).
47. What's a curriculum learning approach for LLMs?
48. How do you handle catastrophic forgetting during fine-tuning?
49. What's continued pre-training? When use it?
50. What's hyperparameter selection look like for training Llama-3?

---

## Section C: Fine-tuning & Adapters (15 Qs)

51. Walk through LoRA mathematically.
52. Why does low-rank work well for fine-tuning?
53. What's the typical rank for LoRA? How to choose?
54. What's QLoRA? How does it work?
55. What's NF4 quantization?
56. When should you NOT fine-tune?
57. SFT or DPO first? Why?
58. What's prefix tuning? How does it compare to LoRA?
59. What's prompt tuning?
60. When use full fine-tuning vs PEFT?
61. How do you choose target modules for LoRA?
62. What's "catastrophic forgetting" in fine-tuning? Mitigations?
63. How do you evaluate a fine-tuned model vs base?
64. Walk through deploying a LoRA-fine-tuned model in production.
65. Can you stack LoRAs? Use cases?

---

## Section D: RAG (20 Qs)

66. When use RAG vs fine-tuning vs prompting?
67. Walk through a production RAG pipeline.
68. How do you choose chunk size?
69. What's hybrid retrieval? Why is it usually better?
70. Explain RRF (reciprocal rank fusion).
71. What's a reranker? When use one?
72. Compare Cohere Rerank vs bge-reranker.
73. What's HyDE (Hypothetical Document Embeddings)?
74. How do you handle multi-hop questions in RAG?
75. What's GraphRAG?
76. How do you evaluate RAG? RAGAS framework?
77. What's the "lost in the middle" problem in RAG?
78. How would you handle multi-tenant data isolation in RAG?
79. Walk through how FinSentinelAI ensures privacy. [USE YOUR PROJECT]
80. When does RAG fail and what do you do?
81. How do you handle citation accuracy?
82. What's prompt caching and why does it matter for RAG?
83. How do you choose embedding models?
84. What's MTEB benchmark?
85. Design a RAG system that handles 10M documents at 100 QPS.

---

## Section E: Agents (15 Qs)

86. What is an LLM agent vs a regular LLM call?
87. Walk through ReAct paradigm.
88. Compare ReAct vs Plan-and-Execute.
89. What's tool calling? How does it work mechanically?
90. How do you design a tool's schema?
91. What's "tool hallucination"? How do you prevent it?
92. Walk through ECHOME's memory architecture. [USE YOUR PROJECT]
93. How do you handle agent state across turns?
94. What's LangGraph? Why use it over vanilla LangChain?
95. How do you evaluate an agent?
96. What's "agent loop" failure? How do you prevent?
97. Compare CrewAI, AutoGen, LangGraph.
98. How would you design a coding agent (like Aider/Cursor)?
99. What's "Computer Use" (Anthropic)?
100. How do you handle long-horizon planning?

---

## Section F: Inference & Production (15 Qs)

101. Walk through what happens during LLM prefill vs decode.
102. Why is decode memory-bound, not compute-bound?
103. What's PagedAttention?
104. Explain FlashAttention.
105. What's continuous batching?
106. Walk through INT4 vs INT8 quantization tradeoffs.
107. What's AWQ vs GPTQ?
108. Walk through deploying a 70B model in production.
109. What's TGI vs vLLM vs SGLang? When use which?
110. How would you reduce LLM inference cost by 50%?
111. What's prompt caching?
112. How do you handle latency-sensitive LLM apps?
113. Walk through your AWS Lambda + FastAPI ML stack at Sujanix. [USE YOUR EXP]
114. Cold start in serverless ML — how to handle?
115. How do you A/B test a new LLM model in production?

---

## Section G: Evaluation & Safety (10 Qs)

116. How do you evaluate an LLM?
117. What's MMLU? HellaSwag? HumanEval?
118. Why are public benchmarks unreliable?
119. What's LLM-as-judge? Pitfalls?
120. How do you evaluate hallucination?
121. What's TruthfulQA?
122. How do you evaluate a multilingual LLM?
123. What's jailbreaking? Defenses?
124. What's RLHF reward hacking?
125. How would you handle adversarial inputs?

---

## How to drill this bank

### Solo:
- 30 min/day, 5-10 Qs
- Answer out loud (or write)
- Compare against the relevant deep-dive file

### With me (Claude):
- Paste 10 Qs at a time
- Ask: "Grade my answers, then quiz me on follow-ups"

### In mocks:
- Have the interviewer pick 5 random Qs
- Practice in conversational style, not lecture style

---

## Add to this bank

As you do mocks and real interviews, add new questions you encounter here. By Month 12 you should have 250+.

---

End of `04-genai-llm-mastery/` folder. Next big folder: `05-ml-system-design/`.
