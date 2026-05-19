# 15 ML System Design Problems — Brief Solutions

For each: 1-paragraph summary of the design. Use these to practice — try designing each from scratch, then compare.

---

## 1. Design YouTube's recommendations
(Full version: `04-design-recommender.md`)

Two-stage: candidate generation (multiple sources via two-tower, collab filter, trending) → ranking (DLRM-style with rich features) → reranking (diversity, safety). Daily training, online A/B test for new models.

---

## 2. Design Twitter's home feed
Similar two-stage to YouTube. Differences: real-time freshness matters more (sub-minute), social graph features critical, ad injection in ranking. Use TwHIN-style social embeddings.

---

## 3. Design Google's autocomplete
Index: trie of historical queries (with frequencies, location-tagged). User types → trie traversal → top-N completions. Rerank with personalization (user history, time of day). Use language model for novel completions on tail.

---

## 4. Design RAG for 10M PDFs
(Full version: `02-design-rag-at-scale.md`)

Offline: parse → chunk (512 tok) → embed (bge-large) → store (Qdrant + OpenSearch for hybrid). Online: query rewrite → hybrid retrieve → rerank (cross-encoder) → LLM generate → faithfulness check. Multi-tenant ACL.

---

## 5. Design ChatGPT's inference platform
(Full version: `03-design-llm-inference.md`)

Multi-region GPU pools. Per region: vLLM with PagedAttention, continuous batching. Tensor parallel 8x H100 for 70B. Prompt caching for system prompts. Geo-routing + sticky sessions for KV cache reuse. Speculative decoding optional.

---

## 6. Design a fine-tuning platform
(Full version: `05-design-fine-tuning-platform.md`)

Training: K8s-scheduled jobs on GPU pools (different pools per model size). LoRA by default. Inference: multi-LoRA serving on shared base. Strict tenant isolation. Per-hour training + per-token inference billing.

---

## 7. Design Amazon's product recommendations ("Customer also bought")
Item-to-item collaborative filtering. Pre-compute "frequently bought together" via co-occurrence matrix, item embeddings. At query time: lookup top N for the current product. Personalize by reranking with user-specific features.

---

## 8. Design Spotify's Discover Weekly
Weekly batch job. Per user: collaborative filtering candidates + content-based candidates (audio features). Filter heard-before. Rank by likelihood. Diversify by genre/artist. Cache pre-computed per user. Refresh every Monday.

---

## 9. Design fraud detection (Stripe Radar style)
Real-time scoring on every transaction. Features: user history, device fingerprint, network, transaction context. Model: gradient boosting (sub-100ms latency, interpretable). Online learning to adapt to new fraud patterns. Threshold tuned per merchant. Human-in-loop review for borderline.

---

## 10. Design Google Translate
Encoder-decoder transformer. Multilingual via shared embedding space (massive multilingual training). Beam search at inference. Quality eval: BLEU, COMET. Latency: optimize for short translations; cache common phrases. Multi-language pairs: focus on high-volume pairs first.

---

## 11. Design GitHub Copilot
LLM (large code-trained model, e.g., Codex-style) fine-tuned on code. Real-time inference: stream tokens. Context: file + repo context (sometimes RAG-style over codebase). Cache common completions. Filter for safety (no leaked secrets, license violations). Eval: HumanEval offline, A/B online (accept rate).

---

## 12. Design a content moderation pipeline (Reddit / Facebook)
Multi-stage: rule-based filters → ML classifier (fast text classifier) → if borderline, send to bigger LLM for nuanced judgment → if still borderline, human review. Real-time for new posts; batch for re-evaluation. Multiple objectives: hate speech, misinformation, safety. Per-region rule variations.

---

## 13. Design an A/B testing platform
Treatment assignment: hash user ID → deterministic bucket. Track exposure events. Track outcome events. Compute lift with proper statistics (CUPED for variance reduction, sequential testing for early stopping). UI for experiment setup. Compute power calculations. Multi-armed bandit option for online optimization.

---

## 14. Design a real-time speech recognition system (ASR)
Streaming: chunked audio → CTC or Listen-Attend-Spell model → token stream. Latency: < 200ms. Modern: Conformer or RNN-T architecture. Beam search for hypothesis. Multilingual or per-language models. Adapt to accents via fine-tuning.

---

## 15. Design an image search (Google Images, Pinterest visual search)
Index: extract embeddings for all images (CLIP-style). Vector DB (HNSW). Query: image → embedding → ANN search → top-K. Rerank with feature engineering (resolution, freshness, source quality). For text query → image: text encoder → same embedding space → search. Multi-modal CLIP enables this.

---

## How to use this list

### Practice schedule (Months 5-10)

- Week 1-2: pick one design, attempt with framework (no time limit), then read the deep dive
- Week 3+: pick a new one each week, 50-min timed practice
- Months 8-10: rotate through all 15+ in mocks

### Variations on each (interviewers will twist)

For each problem, also think about:
- "Add safety / compliance"
- "Add personalization"
- "Reduce cost by 50%"
- "Migrate from a simpler version"
- "Handle 100x scale"

These are common follow-ups.

---

## Cross-cutting concepts (master these and you can design any of the 15)

- **Two-stage retrieval + ranking** (1, 2, 7, 8, 15)
- **RAG patterns** (4)
- **Streaming inference** (5, 11, 14)
- **Multi-tenant isolation** (5, 6)
- **A/B testing framework** (1, 2, 8, 13)
- **Cost optimization** (5, 6 — everywhere actually)
- **Feature stores + offline-online consistency** (1, 2, 7, 9, 13)
- **Cold start handling** (1, 2, 7, 8)
- **Privacy + safety** (9, 12)

---

## Where to go from here

After mastering these 15, you're done with system design prep. Pivot to:
- Behavioral prep (`06-behavioral/`)
- Final portfolio polish (`07-portfolio/`)
- Application execution (`08-application-strategy/`)

---

End of `05-ml-system-design/` folder.
