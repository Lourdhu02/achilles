# Lab 17 — Retrieval for RAG

**Build:** BM25 from scratch, cosine top-k, reciprocal rank fusion, maximal marginal relevance, the standard ranking metrics (recall@k, MRR, nDCG@k), token-window chunking, and the search step of an IVF index. Then embed a real corpus on your GPU and measure BM25, dense and hybrid retrieval on your own questions.
**Time:** 2–3 h for the tests, 4–6 h for the scale-up · **Reads first:** [applied systems §1](../../curriculum/10-applied-llm-systems.md#1-rag-that-actually-works)
**Run:** `pytest labs/17_retrieval` (your code) · `pytest labs/17_retrieval --impl=solution` (reference). The five tests run on CPU in under a second.

Most production LLM systems retrieve before they generate, and most bad RAG answers trace back to retrieval: the passage that held the answer never reached the context. Retrieval is also a place where a method three decades old (BM25) is still a strong baseline, and interviewers probe exactly that: why run BM25 next to embeddings, how RRF works, how you would evaluate the system, how much memory the index needs. This lab builds each piece small enough to check by hand.

---

## 1. BM25

For a query $q$ and document $D$ with $|D|$ tokens:

$$\text{BM25}(q, D) = \sum_{t \in q} \text{idf}(t) \cdot \frac{\text{tf}(t, D)\,(k_1 + 1)}{\text{tf}(t, D) + k_1\big(1 - b + b\,|D|/\text{avgdl}\big)}, \qquad \text{idf}(t) = \ln\Big(1 + \frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5}\Big).$$

- **idf: rare terms count more.** This is Lucene's form. The classic Robertson–Spärck Jones form, $\ln\frac{N - \text{df} + 0.5}{\text{df} + 0.5}$, goes negative for terms in more than half the documents: −0.85 for a term in 3 of 4 documents, against 0.36 with the `1 +`.
- **$k_1$: term-frequency saturation.** At average length with $k_1 = 1.5$, the tf factor is 1.00, 1.43, 1.92 and 2.17 for tf = 1, 2, 5 and 10, approaching $k_1 + 1 = 2.5$. The tenth occurrence of a word adds little. $k_1 = 0$ reduces BM25 to binary matching; a large $k_1$ makes it nearly linear in tf.
- **$b$: length normalization.** With $b = 0.75$, one occurrence scores 1.29 in a document half the average length, 1.00 at the average, and 0.69 at twice the average. $b = 0$ switches normalization off; $b = 1$ applies it fully.

**Worked example** (the test's corpus, query "cat"):

```python
import math
docs = [["the", "cat", "sat"], ["the", "dog", "sat", "down"], ["cats", "and", "dogs"], ["cat", "cat", "food"]]
N, avgdl, k1, b = 4, 13 / 4, 1.5, 0.75
idf = math.log(1 + (N - 2 + 0.5) / (2 + 0.5))     # "cat" is in docs 0 and 3: ln 2 = 0.693
norm = k1 * (1 - b + b * 3 / avgdl)               # both docs have 3 tokens: 1.4135
print(idf * 2 * (k1 + 1) / (2 + norm))            # doc 3, tf = 2 -> 1.0153
print(idf * 1 * (k1 + 1) / (1 + norm))            # doc 0, tf = 1 -> 0.7180
```

Doc 2 contains "cats", which does not match "cat": BM25 matches exact tokens, so tokenization, lowercasing and stemming decide what it can find. For the query "cat sat", doc 0 (1.436) beats doc 3 (1.015): matching two different terms beats repeating one.

## 2. Dense retrieval, and why hybrid wins

A **bi-encoder** (DPR, Karpukhin et al. 2020, [arXiv 2004.04906](https://arxiv.org/abs/2004.04906)) embeds queries and passages separately and scores by cosine or dot product. Passages are embedded once, offline; only the query is embedded at search time. Normalize the vectors and the dot product is the cosine, so any inner-product index works.

| | BM25 (sparse) | Dense bi-encoder |
|---|---|---|
| Matches | exact tokens, weighted by rarity | meaning: paraphrases, synonyms |
| Strong on | identifiers, error codes, names, numbers, rare terms | vocabulary mismatch ("terminate the contract" vs "end the agreement") |
| Weak on | synonyms, paraphrases, other languages | exact strings, rare entities, domains unlike its training data |
| Needs | a tokenizer and an inverted index | an embedding model and a vector index |

The BEIR benchmark (Thakur et al. 2021, [arXiv 2104.08663](https://arxiv.org/abs/2104.08663)) showed BM25 to be a strong zero-shot baseline across 18 datasets, one that many dense retrievers trained on a single dataset fail to beat out of domain. The two methods fail on different queries, so fusing them recovers most of what each finds. That is why hybrid retrieval is the default.

## 3. Reciprocal rank fusion

$$\text{RRF}(d) = \sum_{\text{lists } L} \frac{1}{k + \text{rank}_L(d)}, \qquad k = 60, \text{ ranks starting at } 1$$

(Cormack, Clarke and Büttcher 2009). A document missing from a list gets nothing from it.

**Why ranks, not scores.** BM25 scores are unbounded and their scale changes from query to query with query length and idf, while cosine similarities from one model often sit in a narrow band. Adding raw scores lets whichever system has the larger spread decide, and min-max normalization depends on the extremes of each result list. Ranks are on the same scale in every list and need no calibration.

**Why $k = 60$.** Rank 1 is worth $1/61 = 0.0164$ and rank 10 is worth $1/70 = 0.0143$. A document ranked 10th by both retrievers ($2/70 = 0.0286$) therefore beats one ranked 1st by one retriever and absent from the other (0.0164). With $k = 0$ (plain $1/\text{rank}$) the lone first place wins, 1.0 against 0.2. A large $k$ rewards agreement between retrievers; a small $k$ rewards the top of any single list. 60 is the value from the original paper and the usual default. In the test, fusing [1, 2, 3] and [3, 1, 4] gives doc 1 $1/61 + 1/62 = 0.03252$, doc 3 $1/63 + 1/61 = 0.03227$, doc 2 0.01613 and doc 4 0.01587: the order is [1, 3, 2, 4].

## 4. Maximal marginal relevance

MMR (Carbonell and Goldstein 1998) builds the result list greedily, each time picking

$$\arg\max_{d \notin S}\ \Big[\lambda\,\text{sim}(q, d) - (1 - \lambda)\max_{s \in S}\text{sim}(d, s)\Big].$$

$\lambda = 1$ is plain relevance ranking; $\lambda = 0$ is pure diversity. In the test, four 2-D documents and $q = (1, 0.2)$ give cosine relevances [0.981, 0.995, 0.196, 0.832], so the first pick is always doc 1. Doc 0 is almost a duplicate of doc 1 (cosine 0.995), doc 2 is nearly orthogonal to it (0.100) and doc 3 is in between (0.775). The second pick:

| λ | score of doc 0 | doc 2 | doc 3 | second pick |
|---|---|---|---|---|
| 0.3 (the test) | −0.402 | −0.012 | −0.293 | 2 |
| 0.5 | −0.007 | 0.048 | 0.029 | 2 |
| 0.7 | 0.388 | 0.107 | 0.350 | 0 |

Use MMR when the top results are near-duplicates (versions of one policy, boilerplate repeated across pages): five copies of one passage waste the context window. Tune λ on your eval set.

## 5. Metrics: recall@k, MRR, nDCG

- **recall@k** $= |\text{top-}k \cap \text{relevant}|\,/\,|\text{relevant}|$. The one that matters most for RAG, because the generator reads all $k$ passages: what counts is whether the needed passage got in.
- **MRR** is the mean over queries of $1/\text{rank}$ of the first relevant result (0 if none). Use it when one good result suffices.
- **nDCG@k** handles graded relevance and discounts by position (Järvelin and Kekäläinen 2002):

$$\text{DCG@}k = \sum_{r=1}^{k} \frac{2^{g_r} - 1}{\log_2(r + 1)}, \qquad \text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k},$$

where $g_r$ is the gain of the result at rank $r$ and IDCG is the DCG of the best possible ordering of all judged documents.

**Worked example** (the test): ranking [5, 2, 9, 1], gains {2: 3, 7: 1}, $k = 3$. Only doc 2 appears, at rank 2: $\text{DCG} = 7/\log_2 3 = 4.417$. The ideal ranking puts gain 3 first and gain 1 second: $\text{IDCG} = 7/1 + 1/\log_2 3 = 7.631$, so nDCG@3 = 0.579. With doc 2 at rank 1 it would be 0.917, at rank 3 0.459; it can never reach 1 because doc 7 is not retrieved. In the same test, recall@2 against {2, 1, 7} is 1/3 and the MRR for {9} is 1/3.

All of these are per-query scores, so [lab 15](../15_eval_stats/README.md) applies: with 100 questions and recall@10 near 70%, the 95% CI is ±9.0 points, and two retrievers should be compared paired, on the same questions.

## 6. An IVF index

Brute force compares the query with every vector. An **IVF** (inverted file) index runs k-means to split the vectors into `nlist` cells (the coarse quantizer), stores each vector in the list of its nearest centroid, and at query time scans only the `nprobe` lists whose centroids are nearest the query. A true neighbor in a cell whose centroid is not among the `nprobe` nearest is missed, most often near cell boundaries. On the test's data (2,000 Gaussian vectors in 16 dimensions, `nlist = 20`, 30 queries, recall@10 against exact search):

| nprobe | 1 | 2 | 4 | 8 | 12 | 20 |
|---|---|---|---|---|---|---|
| Vectors scanned | 4.8% | 9.8% | 20% | 40% | 60% | 100% |
| recall@10 | 0.25 | 0.42 | 0.65 | 0.87 | 0.95 | 1.00 |

That is a hard case: isotropic Gaussian data has no cluster structure, so many true neighbors lie across a boundary. On 20 well-separated clusters of the same size, `nprobe = 1` already gives 0.96 and `nprobe = 2` gives 0.997. Real embeddings lie in between, so measure recall against exact search on your own data. Latency grows roughly linearly with the fraction scanned.

**Napkin math at 1 M × 768 float32 vectors.**

- **Memory:** $10^6 \times 768 \times 4$ B = 3.07 GB for the vectors, which IVF-Flat stores in full (1.54 GB in fp16). Centroids for `nlist = 4096` add 12.6 MB, and 8-byte ids add 8 MB. Product quantization at 96 bytes per vector (IVF-PQ) cuts the vectors to 96 MB, at a recall cost usually won back by re-scoring a shortlist with the full vectors.
- **Compute:** brute force costs 768 M multiply-adds per query. IVF with `nlist = 4096` and `nprobe = 16` compares against 4,096 centroids plus 16/4,096 of the vectors: 3.1 M + 3.1 M = 6.1 M multiply-adds, about 125× less, if the lists are balanced.
- **Choosing `nlist`:** of order √N, thousands of lists for 1 M vectors (FAISS's guidelines suggest roughly 4√N–16√N; Johnson, Douze and Jégou 2017, [arXiv 1702.08734](https://arxiv.org/abs/1702.08734)). Train k-means on a sample with at least a few dozen vectors per centroid, then raise `nprobe` until recall meets your target.

## 7. Chunking and end-to-end evaluation

`chunk(tokens, size, overlap)` slides a window of `size` tokens forward by `size − overlap`. On 23 tokens with size 10 and overlap 3 the windows are 0–9, 7–16 and 14–22, the last one shorter. For $N > \text{size}$ there are $\lceil (N - \text{overlap})/(\text{size} - \text{overlap}) \rceil$ windows, and overlap multiplies the tokens you embed and store by $\text{size}/(\text{size} - \text{overlap})$: 1.43× for the test's settings, 1.14× for 512-token chunks with 64 tokens of overlap. In exchange, any span of up to `overlap + 1` tokens lies whole inside at least one window.

Count tokens with the embedding model's tokenizer, so chunks fit its context. Prefer structural boundaries (sections, paragraphs, table rows with their header) where you have them, and keep metadata (document id, section, page) for citations and filters. Chunk size is a hyperparameter: small chunks suit fact lookup, larger ones suit summaries. Sweep it on your eval.

Evaluate the pipeline in three stages. **Retrieval alone:** recall@k on labeled question → relevant-passage pairs; label documents or spans, not chunk ids, so the labels survive re-chunking. **Generation with oracle retrieval:** give the generator the gold passages; if answers are still wrong, the generator or the prompt is at fault. **End to end:** answer correctness, faithfulness (every claim supported by the retrieved passages), citation precision, and abstention on unanswerable questions. Then attribute each failure: gold passage not retrieved (retrieval), retrieved but ignored or contradicted (generation), or not in the corpus at all (the system should have abstained).

## What to implement

| # | function | test | what the test pins down |
|---|---|---|---|
| 1 | `BM25.idf(term)` | `test_bm25_by_hand` | Lucene idf: ln 2 for "cat" (df 2 of N = 4) |
| 2 | `BM25.score(query, i)` | same | 1.0153 for "cat" on doc 3; 0.0 for a term in no document; `search(["cat"], k=2) == [3, 0]` |
| 3 | `dense_topk(query, docs, k)` | `test_dense_mmr_and_fusion` | cosine top 2 is [1, 0] |
| 4 | `reciprocal_rank_fusion(rankings, k=60)` | same | fusing [1, 2, 3] and [3, 1, 4] puts [1, 3] first |
| 5 | `mmr(query, docs, k, lam=0.5)` | same | at λ = 0.3, doc 1 first, then 2 or 3, never the near-duplicate 0 |
| 6 | `recall_at_k(ranked, relevant, k)` | `test_metrics` | 1/3 for the top 2 of [5, 2, 9, 1] against {2, 1, 7} |
| 7 | `mrr(ranked, relevant)` | same | 1/3 for {9}; 0.0 when nothing relevant is retrieved |
| 8 | `ndcg_at_k(ranked, gains, k)` | same | 0.579 for the section 5 example, with gain $2^g - 1$ |
| 9 | `chunk(tokens, size, overlap)` | `test_chunking_covers_everything_with_overlap` | first window 0–9; the second starts 7, 8, 9; the last token (22) is covered; no window longer than 10 |
| 10 | `IVFIndex.search(q, k, nprobe)` | `test_ivf_recall_rises_with_nprobe` | recall@10 exactly 1.0 at `nprobe = nlist = 20`, lower at `nprobe = 1` |

Given: `BM25.__init__` (it builds `tf`, `df` and `avgdl`), `BM25.search`, `kmeans`, and `IVFIndex.__init__` (centroids and the per-cell `lists` of vector ids).

## Tips

> [!TIP]
> Reproduce the section 1 worked example before you write `score`, and when the test fails print the three intermediate values: idf, the length norm, and the tf factor. Almost every BM25 bug shows up in one of them.

- `score` sums only over query terms present in the document (`if t in tf`); a term in no document contributes 0, which the `"zebra"` assertion checks.
- `dense_topk`: normalize both sides, take one matrix-vector product, and use `np.argsort(-sims, kind="stable")` so ties keep index order.
- `mmr`: set the scores of already-picked documents to `-inf`; the redundancy term is the maximum similarity to the picked set.
- `ndcg_at_k`: IDCG uses the best ordering of **all** judged gains, truncated to $k$, including documents that were never retrieved.
- `chunk`: stop as soon as a window reaches the end; otherwise you emit trailing windows that sit entirely inside the previous one.
- `IVFIndex.search`: rank centroids by distance to `q`, concatenate the `nprobe` lists into one candidate array, then rank the candidates exactly. The lists hold global ids, so return those, not positions within the candidate array.

## Common bugs

- **Classic idf without the `1 +`:** negative weights for common terms and a failed idf assertion.
- **RRF with 0-based ranks:** every score shifts, and a rank-0 document gets $1/k$ instead of $1/(k + 1)$.
- **IDCG from the retrieved list only:** it ignores relevant documents that were missed and overstates nDCG (0.631 instead of 0.579 in the test).
- **Linear gain $g$ instead of $2^g - 1$:** both appear in the literature, but the test expects the exponential form (linear gives 0.521).
- **A window step of `size` instead of `size − overlap`**, or an endless loop when `overlap >= size`: check the arguments.
- **Returning candidate positions from `IVFIndex.search`** instead of the global ids in `self.lists`.

## CPU experiments (no GPU needed)

1. **BM25 parameters.** Index a few hundred paragraphs of any text, write 20 questions with known answer paragraphs, and sweep $k_1 \in \{0.5, 1.2, 1.5, 2.0\}$ and $b \in \{0, 0.5, 0.75, 1\}$. Report recall@5 with CIs, then add lowercasing and a crude suffix-stripping stemmer: which change matters more?
2. **Score fusion vs RRF.** Simulate a sparse and a dense system whose scores have different scales, and fuse them by raw sum, by min-max-normalized sum, and by RRF with $k \in \{1, 10, 60, 1000\}$. Which fusion survives one outlier score?
3. **IVF trade-off curve.** On 20 k vectors (random, then clustered), plot recall@10 against the fraction scanned for `nlist` ∈ {16, 64, 256}. The given `kmeans` materializes an N × nlist × d array; switch it to $\lVert x\rVert^2 - 2\,x \cdot c + \lVert c\rVert^2$ before you go larger.
4. **MMR on near-duplicates.** Add lightly edited copies of a few documents, retrieve with and without MMR, and count distinct source documents in the top 5 for λ ∈ {0.3, 0.5, 0.7, 1.0}.
5. **Chunk size.** Split one long document with size ∈ {64, 128, 256, 512} tokens and 10–20% overlap, and measure BM25 recall@5 on your 20 questions.

## GPU scale-up (8 GB): BM25 vs dense vs hybrid on your own questions

Use your own documents (notes, papers, the docs of a library you use) or a BEIR dataset such as SciFact (about 5 k abstracts and 300 labeled test queries). Write 100 real questions and label the passages that answer them; include some with exact identifiers and some paraphrased. A free Colab or Kaggle T4 works too.

Embed with a small open model such as `BAAI/bge-small-en-v1.5` or `intfloat/e5-small-v2` (both 384-dimensional). Read the model card for the pooling and the query and passage prefixes it expects; E5 wants `query: ` and `passage: `.

```python
import torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
name = "BAAI/bge-small-en-v1.5"
tok = AutoTokenizer.from_pretrained(name)
enc = AutoModel.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()

@torch.no_grad()
def embed(texts, bs=256):
    out = []
    for i in range(0, len(texts), bs):
        batch = tok(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        h = enc(**batch).last_hidden_state[:, 0]      # BGE pools with the [CLS] vector; E5 uses mean pooling
        out.append(F.normalize(h.float(), dim=-1).cpu())
    return torch.cat(out).numpy()
```

Napkin math: 50 k chunks of 256 tokens through a 33 M-parameter encoder is about $2 \times 33\text{M} \times 12.8\text{M tokens} = 8.4 \times 10^{14}$ FLOPs, under two minutes at an achieved 10 TFLOP/s. The embeddings take $50{,}000 \times 384 \times 4$ B = 77 MB, so exact search is fine at this size; build your IVF index anyway and measure its recall against exact search.

1. Chunk with your `chunk` on the embedding tokenizer's tokens (for example 256 tokens with 32 of overlap) and embed every chunk once; save the matrix.
2. Retrieve the top 100 with your `BM25` and with `dense_topk`, and fuse them with `reciprocal_rank_fusion`. The pure-Python BM25 scores every document for every query, which is fine at 50 k chunks; build term → document postings if it gets slow.
3. Report recall@10 and MRR for BM25, dense and hybrid, each with a 95% CI, and compare pairs with McNemar on hit@10 (lab 15). Break the results down by question type: identifiers versus paraphrases. Optionally, rerank the fused top 50 with a small cross-encoder such as `cross-encoder/ms-marco-MiniLM-L-6-v2` and measure the gain in recall@5.
4. End to end: answer 50 questions with a small instruct model given the top 5 chunks, grade correctness and faithfulness by hand, and attribute each failure to retrieval or generation (section 7).

## Check yourself

1. A term with idf 2.0 appears twice in a document of average length, with $k_1 = 1.5$. What does it contribute, and what is the most it can contribute as tf grows?
2. Why fuse BM25 and dense results by rank rather than by adding their scores?
3. With $k = 60$, which scores higher: rank 1 in one list only, or rank 10 in both lists? And with $k = 0$?
4. A ranking puts a gain-3 document at rank 2 and misses a gain-1 document entirely. What is nDCG@3?
5. How much memory do 1 M × 768 float32 vectors take in IVF-Flat, and how many multiply-adds does a query cost at `nlist = 4096`, `nprobe = 16`, against brute force?
6. With the same `nlist`, recall@10 at `nprobe = 1` is 0.25 on the test's data but 0.96 on clustered data. Why?
7. Your RAG system answers 30% of questions wrongly. How do you find out whether retrieval or generation is at fault?

<details><summary>Answers</summary>

1. $2.0 \times 2 \times 2.5/(2 + 1.5) = 2.86$. As tf grows, the tf factor approaches $k_1 + 1 = 2.5$, so the contribution approaches 5.0.
2. The scores live on different scales that shift from query to query: BM25 is unbounded and depends on query length and idf, while cosines crowd into a narrow band. A raw sum lets the system with the larger spread decide, and normalization is fragile to outliers. Ranks are comparable across lists without calibration.
3. Rank 10 in both: $2/70 = 0.0286$ against $1/61 = 0.0164$. With $k = 0$ the single first place wins, 1.0 against 0.2.
4. $\text{DCG} = 7/\log_2 3 = 4.417$, $\text{IDCG} = 7 + 1/\log_2 3 = 7.631$, so 0.579.
5. 3.07 GB for the vectors, plus 12.6 MB of centroids and 8 MB of ids. About 6.1 M multiply-adds (3.1 M for centroids, 3.1 M for the 16/4,096 of the vectors scanned) against 768 M, roughly 125× less with balanced lists.
6. Isotropic Gaussian data has no clusters, so a query's true neighbors are spread across many cells and the nearest centroid's cell holds only some of them. With real clusters, the neighbors sit in the query's own cell.
7. Check whether the gold passage was in the top k (recall@k per question), then run the generator with the gold passages (oracle retrieval). Failures fixed by the oracle are retrieval failures; the rest are generation or prompt failures, or questions the corpus cannot answer.
</details>

## Stretch

- Add average precision and MAP, and compare how they and nDCG rank the same set of systems.
- Build an inverted index for BM25 (term → postings with tf), so a query touches only documents that contain its terms, and time it against the brute-force loop.
- Implement IVF-PQ: 96 sub-vectors with 256 centroids each for 768-dimensional vectors (96 bytes per vector); measure recall against memory, with and without re-scoring a shortlist by exact distance.
- Implement HNSW (Malkov and Yashunin, [arXiv 1603.09320](https://arxiv.org/abs/1603.09320)) and compare it with IVF at equal recall.
- Try late interaction with ColBERT (Khattab and Zaharia 2020, [arXiv 2004.12832](https://arxiv.org/abs/2004.12832)) on your scale-up corpus.
