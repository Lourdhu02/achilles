# RAG — Beyond the Toy

Your FinSentinelAI work gives you a head start. This file gets you to "I can architect any production RAG system" depth.

---

## The RAG mental model

```
User Query
   ↓
[Query Understanding / Transformation]
   ↓
[Retrieval] ← Vector DB / Hybrid (sparse + dense) / BM25
   ↓
[Reranking]
   ↓
[Context Construction]
   ↓
[Generation] ← LLM with augmented prompt
   ↓
[Post-processing / Citations]
   ↓
Final Response
```

Each box is a design decision with tradeoffs. Naive RAG skips most of them.

---

## Why RAG (vs. pure LLM, vs. fine-tuning)

| Approach | Pros | Cons |
|---|---|---|
| Pure LLM | Simple, lowest latency | Static knowledge, hallucination, can't update facts |
| RAG | Up-to-date, citable, narrow domain works well | Retrieval quality is bottleneck; complex |
| Fine-tune | Memorize style, narrow scope | Expensive, hard to update, can't cite |
| RAG + Fine-tune | Style + knowledge | Most complex, most expensive |

**Default: start with RAG. Fine-tune only when RAG can't capture style/format.**

---

## Phase 1: Ingestion (the offline part)

### Document parsing
- PDFs: tricky. Use Unstructured.io, PyMuPDF, pdfplumber. For complex layouts: LayoutLM or VLM.
- HTML: BeautifulSoup + readability
- DOCX: python-docx, mammoth
- Code: tree-sitter for AST-aware parsing
- Tables: separate handling (PDF tables are a nightmare)
- Images embedded in docs: extract + OCR (Donut, your SVRT work!) or VLM

### Chunking strategies
- **Fixed-size:** 500-1000 tokens with 10-15% overlap (simplest)
- **Semantic:** split by sentences/paragraphs (recursive character splitter)
- **Layout-aware:** preserve document structure (sections, lists, tables)
- **Sliding window with overlap:** for narrative continuity
- **Sentence-window:** retrieve small chunks, send larger window around match

### Choosing chunk size
- Too small: lose context, retrieve noise
- Too large: embed dilution, exceed LLM context budget
- Sweet spot for English text: 256-512 tokens
- For code: function-level chunks often best

### Metadata enrichment
Every chunk should have:
- Source document ID, page, section
- Timestamp (if temporal queries matter)
- Permissions / ACLs (for multi-tenant)
- Hash for dedup
- Tags / categories (for filtered search)

Don't skip this. Most "RAG failures" trace back to poor metadata.

### Embedding model selection
- **Default open:** `bge-large-en-v1.5`, `E5-large-v2`, `Snowflake/snowflake-arctic-embed-l`
- **Default closed:** `text-embedding-3-large` (OpenAI), `embed-english-v3` (Cohere)
- **Code:** `voyage-code-2`, `codeBERT`
- **Multilingual:** `bge-m3`, `multilingual-e5-large`

Check MTEB leaderboard before committing.

### Vector DB choice
- **Pinecone:** managed, fast, expensive
- **Weaviate:** managed/self-hosted, multimodal
- **Qdrant:** self-hosted-friendly, Rust-fast (you used it!)
- **Chroma:** simple, embedded (you used it for FinSentinelAI)
- **PostgreSQL + pgvector:** if you already have Postgres
- **Milvus:** at very large scale (10B+ vectors)
- **Vespa, OpenSearch w/ KNN:** enterprise scale

For Lourdu-tier projects (<10M vectors): Qdrant or pgvector both fine.

---

## Phase 2: Query handling (the online part)

### Query understanding

**Query classification:**
- Is this a factual question? Conversational? Multi-hop?
- Route to different RAG pipelines or to non-RAG path

**Query expansion:**
- HyDE (Hypothetical Document Embeddings): LLM generates a hypothetical answer, embed that, retrieve similar real docs
- Multi-query: LLM generates 3-5 reformulations, retrieve for each, union
- Step-back questions: zoom out before zooming in

**Query rewriting:**
- For conversational follow-ups: incorporate chat history
- "What about its weight?" → "What is the weight of the Tesla Model S?"
- Often: prompt LLM to rewrite query as standalone

### Retrieval methods

#### Dense retrieval (vector similarity)
- Cosine similarity (most common) or dot product
- Top-k retrieval (k typically 20-50, before reranking)

#### Sparse retrieval (BM25, TF-IDF)
- Lexical matching
- Catches exact terms / acronyms / rare entities that embeddings miss

#### Hybrid retrieval
- Combine dense + sparse
- Methods:
  - **Reciprocal Rank Fusion (RRF):** merge rankings, score = Σ 1/(k + rank_i)
  - **Linear combination:** α · dense_score + (1-α) · bm25_score (normalize first!)
  - **Trained fusion:** learn weights

In practice: hybrid almost always > pure dense for production. **Use it.**

### Filtering
- Apply metadata filters (date range, source, ACL) before / during retrieval
- Most vector DBs support filtered search natively

---

## Phase 3: Reranking

The most underutilized component. Big quality wins here.

### Why rerank
- Embedding retrieval is fast but lossy (256-1024 dim → information loss)
- Cross-encoders see query + doc together → much more accurate relevance
- Tradeoff: cross-encoders are 10-100x slower; use only on top-k from retrieval

### Reranker models
- **Cohere Rerank** (API, easy)
- **bge-reranker-large** (open, strong)
- **ms-marco-MiniLM-L-12-v2** (older, fast)
- **mxbai-rerank-large** (recent strong open)

### Pattern
1. Retrieve top 50-100 via dense (or hybrid)
2. Rerank those 50-100 with cross-encoder
3. Take top 5-10 for context

### Cost analysis
- Embedding retrieval: ~$0.0001 per query
- Reranking 50 docs: ~$0.002-0.005 per query
- Worth it for quality-sensitive applications

---

## Phase 4: Context construction

### Prompt template patterns

```
You are an assistant that answers questions based on the provided context.

CONTEXT:
[doc 1 with citation]
[doc 2 with citation]
[doc 3 with citation]

QUERY: {user_query}

Instructions:
- Answer based ONLY on the context above
- Cite sources using [N] notation
- If context doesn't contain the answer, say "I don't have enough information."

ANSWER:
```

### Considerations
- Lost in the middle: LLMs attend more to start/end of context. Place most-relevant chunks at edges.
- Token budget: balance context size vs cost vs latency
- Compression: for very long contexts, summarize less-relevant docs first

### Citation handling
- Numbered: docs get [1], [2], etc. Response cites them
- Inline links: useful when source URLs available
- Span-level citations: highlight which exact part of the doc supports a claim (harder, see "Verifiable answers")

---

## Phase 5: Generation

### Model choice
- **GPT-4o-mini, Claude Sonnet, Gemini Flash:** good cost/quality
- **GPT-4o, Claude Opus:** when quality matters more
- **Open: Llama-70B, Qwen-72B, Mixtral:** for self-hosted

### Generation hyperparameters for RAG
- Temperature: 0.1-0.3 (low, factual)
- Top-p: 0.9
- Max tokens: 500-1000 (most RAG answers are short-medium)

### Streaming
- Yes, always — feels much faster to user
- Returns first token in <500ms with streaming vs 3-5s without

### Guardrails
- Input: filter harmful queries (use a moderation model)
- Output: filter harmful generations
- Faithfulness check: verify generated claims appear in retrieved context

---

## Phase 6: Post-processing

### Faithfulness verification
- Re-prompt LLM: "Does this answer follow from this context? Yes/No + reason"
- Or: check that named entities, numbers in answer appear in context

### Citation verification
- Did the LLM hallucinate citations? Verify each [N] maps to a real retrieved doc.

### Confidence scoring
- Use retrieval scores
- Use answer self-eval ("rate this answer's confidence 1-10")
- Use ensemble of LLMs for important queries

---

## Evaluation (CRITICAL — RAG without eval is a toy)

### Component-level metrics

**Retrieval quality:**
- Recall@k: % of times the correct doc is in top-k
- MRR (Mean Reciprocal Rank): 1/rank of first correct doc
- NDCG: position-weighted relevance

**Reranker quality:**
- nDCG@k on a labeled set

**Generation quality:**
- Faithfulness: how much of answer is supported by context
- Relevance: does answer address the query
- Coherence: linguistic quality
- Citation accuracy: are citations correct

### End-to-end metrics

**RAGAS framework:**
- Faithfulness, Answer Relevance, Context Precision, Context Recall

**TruLens:**
- Similar with different decomposition

**Custom human eval:**
- Sample 100 queries, manually rate (yes/no for each: correct? Cited correctly? Faithful?)
- Establishes ground truth for automated metrics

### LLM-as-judge
- Use a stronger LLM (GPT-4 or Claude Opus) to judge a weaker LLM's answers
- Pros: scalable
- Cons: judge LLM has biases, can be gamed
- Best practice: validate judge against human ratings first

---

## Multi-modal RAG

### Image-text RAG
- Embed images via CLIP or BGE-M3
- Query: text or image
- Retrieve image+context, send to VLM (GPT-4V, Claude Sonnet)

### Document RAG with vision
- For PDFs with figures, charts, complex layouts
- Use ColPali (visual document retrieval) or DocLayNet
- Pass full page image to VLM, plus extracted text

This is what FinSentinelAI does with VLM extraction from bank statements.

---

## Advanced RAG patterns

### Multi-hop RAG
- Question requires combining info from multiple docs
- Approach: iterative retrieval (retrieve → reason → retrieve more)
- Or: graph-based RAG (entity linking + traversal)

### Self-RAG
- LLM decides WHEN to retrieve, WHAT to retrieve, HOW to use
- Special tokens: <Retrieve>, <Critique>, etc.

### CRAG (Corrective RAG)
- Evaluate retrieval; if low quality, fallback to web search

### Agentic RAG
- LLM agent with retrieval as a tool
- Can plan multi-step retrieval strategies
- Slower but more flexible

### Long-context "RAG-free"
- Pump entire document into 1M-context model (Gemini, Claude)
- Pros: no retrieval errors
- Cons: cost, latency, lost-in-the-middle still
- Hybrid: long context + RAG for filtering

### GraphRAG
- Microsoft's pattern: extract entities + relationships from docs, build knowledge graph
- Retrieve based on graph traversal + community summaries
- Better for global/holistic questions ("what are the main themes?")

---

## Common RAG failure modes (and fixes)

| Failure | Symptom | Fix |
|---|---|---|
| Bad retrieval | LLM hallucinates because context doesn't have answer | Better chunking, better embeddings, hybrid retrieval, reranking |
| Lost in middle | Long context, LLM ignores middle chunks | Reranking + place top results at start/end; or shorter context |
| Hallucinated citations | LLM cites doc that doesn't support claim | Faithfulness check; stricter prompting; verify citations programmatically |
| Stale data | Answers based on outdated docs | Re-ingest pipeline; timestamp filters; show date in citations |
| Multi-tenant leakage | User sees another tenant's data | ACL filters at retrieval level; never rely on prompt-only enforcement |
| Cost blowup | Token cost exceeds budget | Smaller chunks, fewer chunks, cheaper LLM, prompt compression |

---

## Your FinSentinelAI as a teaching example

Walk-through you should be able to give in any interview:

> "FinSentinelAI is a privacy-first RAG platform for financial documents. The core architecture has 5 stages:
>
> 1. **Ingestion:** PDFs and bank statements arrive. We use a local VLM to extract structured data — both text and visual elements like tables, signatures, stamps. This is critical because financial PDFs are layout-heavy.
>
> 2. **Indexing:** Chunks are sized at 512 tokens with 50-token overlap. Each chunk carries metadata: document ID, page, tenant ID for ACL, timestamp, and entity tags extracted via NER. We use Ollama-served embeddings (privacy requirement; data can't leave premises) and ChromaDB for storage.
>
> 3. **Retrieval:** Hybrid — dense via embeddings, sparse via BM25, fused with RRF. Tenant filter applied before retrieval to prevent multi-tenant leakage. Top 20 candidates.
>
> 4. **Generation:** Local LLM via Ollama (privacy-first means no cloud LLM). System prompt enforces citation. We use JWT-encoded session context to enforce per-user permissions.
>
> 5. **Post-processing:** Faithfulness check — every citation is verified against retrieved chunks. If a claim doesn't trace to a chunk, flagged.
>
> Tradeoffs: local LLM is slower and lower-quality than GPT-4o. But for regulated finance customers, it's non-negotiable. We get to ~85% answer accuracy on internal eval set."

Practice this. Adapt to specifics. Use in interviews.

---

## Resources

- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" — original RAG paper (2020)
- "Lost in the Middle" — Liu et al. 2023 (must know)
- "Active Retrieval Augmented Generation" — FLARE
- "Self-RAG" — Asai et al.
- "From Local to Global: A Graph RAG Approach" — Microsoft 2024
- Pinecone learning portal, Weaviate docs, LangChain RAG docs
- Replicate of Chip Huyen's RAG eval framework

---

Next: [`03-agents-deepdive.md`](./03-agents-deepdive.md)
