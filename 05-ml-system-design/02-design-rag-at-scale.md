# Design: RAG System at Scale

**Problem:** Design a Retrieval-Augmented Generation system to answer questions over 10 million PDFs at 100 QPS with p99 latency < 3 seconds.

This is your SWEET SPOT given FinSentinelAI. Practice this until you can deliver it in 50 minutes flawlessly.

---

## Step 1: Clarify (5 min)

Questions you'd ask:

1. **Domain:** General? Or domain-specific (finance, legal, medical)?
2. **Multi-tenancy:** Are PDFs per-customer/tenant? ACL needed?
3. **Language:** English only? Multilingual?
4. **Update frequency:** PDFs static, or constantly ingested?
5. **Citation requirement:** Must we cite source pages?
6. **LLM choice:** Cloud (OpenAI, Anthropic) or self-hosted?
7. **Cost budget:** USD/query or monthly cap?
8. **Quality bar:** "Mostly right" or "regulated-industry accurate"?

Assumptions you'd make:
- English; multi-tenant; daily ingestion of new PDFs; citations required; quality bar high
- Mix of cloud LLM (Claude) and self-hosted fallback
- Budget: ~$0.01/query

---

## Step 2: Requirements (3 min)

### Functional
- Accept user question + tenant context
- Return: answer + cited sources (page level)
- Handle: multi-hop questions, follow-up questions
- Support: filter by metadata (date, document type, etc.)

### Non-functional
- 100 QPS sustained, 250 QPS peak
- p50 latency < 1.5s, p99 < 3s
- 99.9% uptime
- 10M documents (~ 1-2B chunks)
- 1 TB total embedded data
- Daily: ~50k new documents ingested

### ML/Quality
- Answer accuracy > 85% on internal eval set
- Faithfulness (cited claims supported by source) > 95%

---

## Step 3: Architecture (10 min)

```
                       OFFLINE PIPELINE
                       
PDFs → S3 → [Parser]  → [Chunker] → [Embedder]  → Vector DB
                ↓             ↓            ↓
              extract     metadata    GPU batch
              text+tables  tagging    via Triton

                       ONLINE PIPELINE
                       
User → API GW → [Query Service]
                      ↓
              [Query Rewriter] (LLM)
                      ↓
              [Hybrid Retriever]
                      ↓
              [Reranker]
                      ↓
              [Context Builder]
                      ↓
              [LLM Generator] (Claude/Llama)
                      ↓
              [Post-processor: faithfulness, citation]
                      ↓
                  Response
```

---

## Step 4: Deep Dive

### Component A: Ingestion pipeline

**Data flow:**
1. PDFs arrive at S3 (via API upload or batch transfer)
2. Trigger Lambda → enqueue to SQS for processing
3. Workers (ECS/K8s pods) pull from queue

**Parsing:**
- Use Unstructured.io or LlamaParse for text
- For complex layouts: VLM (Claude Sonnet vision API or local Donut)
- Tables: parse with Table-Transformer or VLM
- Why this matters: financial PDFs have layouts that break naive PyMuPDF

**Chunking:**
- Recursive character splitter, ~512 tokens with 50 token overlap
- Preserve section headers in chunk metadata
- Layout-aware where possible (don't split mid-table)

**Metadata enrichment:**
- Doc ID, page, section, timestamp
- Tenant ID (for ACL)
- Document type (annual report, statement, contract)
- Extracted entities (companies, dates, amounts) via NER

**Embedding:**
- Model: `bge-large-en-v1.5` (open) or `text-embedding-3-large` (closed) — 1024 dim
- Batch on GPU via Triton: 256 chunks/batch
- ~5ms per chunk at scale

**Storage:**
- Vector DB: Qdrant cluster (you've used it) — distributed, sharded by tenant
- Sparse index: OpenSearch with BM25 (for hybrid)
- Metadata: PostgreSQL (joins, filtering)

**Tradeoffs discussed:**
- "Why not Pinecone? Cost. Qdrant self-hosted is 5-10x cheaper at this scale."
- "Why not store everything in OpenSearch? Vector search performance — dedicated vector DB is faster."

### Component B: Retrieval

**Query understanding:**
- Cheap LLM call (Haiku or Llama 8B) to classify query type
- For conversational: rewrite as standalone query
- For multi-hop: decompose to sub-queries

**Hybrid retrieval:**
- Dense: cosine similarity in Qdrant, top 50
- Sparse: BM25 in OpenSearch, top 50
- Fuse via RRF (Reciprocal Rank Fusion)
- Filter by tenant ID BEFORE retrieval (security critical)
- Filter by metadata (date range etc.) BEFORE retrieval

**Reranking:**
- Take top 30 from hybrid
- Cross-encoder reranker (bge-reranker-large on GPU)
- Score each (query, doc) pair
- Keep top 5-10 for context

**Latency budget so far:**
- Query understanding: 200ms
- Hybrid retrieval: 100ms (parallel dense + sparse)
- Reranking: 200ms (30 pairs, batched)
- Total retrieval: ~500ms

### Component C: Generation

**Context construction:**
- Top 5-10 chunks + their metadata
- Order: most-relevant at start AND end (lost-in-the-middle mitigation)
- Format with clear citation tokens [1], [2], etc.

**LLM choice:**
- Primary: Claude Sonnet 4.X (best balance of quality + cost for RAG)
- Fallback: Self-hosted Llama 3.1 70B via vLLM (for cost-sensitive or PII-strict tenants)
- Tier-2 fallback: Claude Haiku for ultra-low-latency requirements

**Prompt template:**
```
You are an expert assistant. Answer the question based ONLY on the context.
Cite sources using [N] notation. If context doesn't have the answer, say so.

Context:
[1] {chunk 1 with metadata}
[2] {chunk 2}
...

Question: {query}

Answer:
```

**Generation params:**
- Temperature 0.1
- Max tokens 800
- Streaming enabled

**Latency:**
- TTFT < 1s
- Total ~1.5-2.5s for typical answer

### Component D: Post-processing

**Faithfulness check:**
- Extract claims from answer
- For each claim, verify against cited context
- Use NLI model or LLM-as-judge
- Flag for human review if faithfulness < threshold

**Citation validation:**
- Parse [N] tokens
- Map back to chunk IDs
- Verify chunks were actually retrieved

**Caching:**
- Embedding cache (query embeddings, LRU, 1M entries)
- Response cache (exact-query match, 1hr TTL) — careful with multi-tenancy

---

## Step 5: Scale (5 min)

### Going to 1000 QPS

**Bottlenecks:**
- LLM cost (Claude API): becomes 10x more expensive
- Embedding compute: need GPU autoscaling
- Reranker compute: same

**Solutions:**
- Move 50%+ traffic to self-hosted Llama 70B (massive cost savings at scale)
- Pre-compute query embeddings for top-N most common queries
- Semantic cache for similar queries (95% threshold)
- Multi-region deployment

### Going to 100M documents

**Bottlenecks:**
- Qdrant index size becomes massive
- Retrieval latency increases

**Solutions:**
- Hierarchical retrieval: first retrieve "section summaries", then drill into chunks
- HNSW index params tuning (ef_search, M)
- Sharding by tenant + topic
- Cold storage for old / rarely-accessed docs

### Cost analysis at scale

At 100 QPS, 80% Claude Sonnet, 20% Llama 70B:
- Claude calls: 100 × 0.8 × $0.015 × 86,400 sec = ~$100k/day for LLM
- Embedding: ~$50/day (cheap)
- Reranker: ~$500/day (self-hosted GPU)
- Storage: ~$100/day (S3 + Qdrant cluster)
- Total: ~$100k+/day = $3M/month

Levers to halve cost:
1. Aggressive prompt caching (Claude has prompt caching API — 90% reduction on system prompt)
2. Self-host 50% of traffic on Llama
3. Tiered LLM: Haiku for easy queries, Sonnet for complex
4. Cache aggressively

---

## Step 6: Iterate (5 min)

### Evaluation strategy
- Golden set: 500 queries (human-curated)
- Daily eval: run on golden set, track scores
- A/B testing: traffic split for new models/components
- User feedback: 👍/👎 buttons captured

### Metrics tracked
- **Quality:** RAGAS scores (faithfulness, relevance, precision, recall)
- **Latency:** p50/p99 end-to-end + per-component
- **Cost:** $ per query, by tenant
- **Business:** Query volume, user satisfaction (NPS), retention

### Monitoring
- Embedding drift (vs reference distribution)
- Retrieval quality (NDCG on production traffic — challenging without labels)
- LLM output quality drift (random sample → LLM-as-judge)

### Future improvements
- Adaptive retrieval (LLM decides k)
- Self-RAG (LLM decides when to retrieve)
- Multi-modal RAG (charts, images)
- Personalization (user-specific reranking)
- Feedback loop: user-marked good answers become training data for cross-encoder

---

## Common follow-ups

1. **"How would you handle multi-tenant isolation?"**
   - ACL at every layer
   - Tenant ID in metadata, filter at retrieval
   - Separate Qdrant collections per tenant for large tenants
   - Audit logs

2. **"What if a tenant uploads sensitive PII?"**
   - PII detection at ingestion (Presidio, AWS Comprehend)
   - Redact or encrypt
   - Compliance: GDPR, CCPA, HIPAA (if healthcare)

3. **"How do you handle hallucinations?"**
   - Faithfulness check post-generation
   - Citation verification
   - "Cannot answer from context" fallback
   - User-visible confidence indicator

4. **"What if a user asks something outside scope?"**
   - Classifier upstream
   - Friendly refusal: "I can only answer questions about your uploaded documents"

5. **"How do you handle queries about recent documents not yet ingested?"**
   - Ingestion SLA visible to users
   - "Documents uploaded < 1 hour ago may not be searchable yet"

6. **"What about adversarial users (prompt injection in PDFs)?"**
   - Sanitize parsed text
   - Separate user content from system prompt structurally
   - Monitor for unusual LLM behavior

7. **"How would you reduce cost by 50%?"**
   - Detailed earlier: prompt caching, model tiering, self-host, semantic cache

8. **"Migration path from a simpler RAG (Chroma + GPT-4) to this?"**
   - Phased: 1) add hybrid, 2) add reranker, 3) migrate vector DB, 4) add post-processing
   - Each phase A/B tested
   - Roll back levers

---

## Your unique angle (use these!)

Connect to FinSentinelAI throughout:
- "When I built FinSentinelAI, we faced [specific issue]. At 10x scale, that issue compounds, so I'd..."
- "We used ChromaDB for FinSentinelAI because it was simpler at our scale. At 10M docs, we'd outgrow it — hence Qdrant."
- "For privacy-first tenants like FinSentinelAI's banking clients, we'd use the self-hosted Llama path exclusively."

Connect to your Sujanix experience:
- "Similar to my AWS Lambda work at Sujanix — burst inference + cost-aware autoscaling."

---

## Practice this design 5+ times until it's natural

Time yourself. Each iteration improve one weak area: clarifying questions, deep dives, scale numbers, cost analysis.

By Month 8, you should be able to deliver this in 45 minutes cleanly, including answering follow-ups.

---

Next: [`03-design-llm-inference.md`](./03-design-llm-inference.md)
