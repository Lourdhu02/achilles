# 05 — Pretraining

Labs: [08 scaling laws](../labs/08_scaling_laws/README.md), [09 parallelism](../labs/09_parallelism/README.md), and scale-up run S1 in [lab 05](../labs/05_transformer/README.md#scale-up-run-s1-pretrain-on-your-rtx-5060).

## 1. Data is the model
Pipeline: crawl (Common Crawl) → HTML text extraction (e.g. trafilatura) → language ID (fastText) → heuristic filters (length, symbol and word ratios, repetition; Gopher/C4 rules) → **model-based quality filtering** (FineWeb-Edu's LLM-annotated classifier; DCLM's fastText classifier) → **deduplication** → PII and toxicity handling → **decontamination** against eval sets → mixing and upsampling → tokenization and packing.
- **MinHash-LSH near-dedup:** Jaccard similarity over n-gram shingles; with b bands of r rows, `P(candidate) = 1 − (1 − s^r)^b`, an S-curve you tune to threshold at s ≈ 0.8.
- **Mixtures matter as much as volume:** code and math upweighting, DoReMi-style learned weights, quality-bucket upsampling.
- **Repetition:** up to ~4 epochs of the same data is nearly as good as fresh data (Muennighoff et al. 2023). Beyond that, returns collapse.
- **Synthetic data** is now central (rephrasing, textbooks, reasoning traces) but risks narrowing diversity. Always keep a real-data baseline.

## 2. Tokenization choices
Vocab size (32k → 128k → 200k+), the pre-tokenization regex, digit handling, and multilingual coverage. See [lab 04](../labs/04_tokenizer/README.md) for the Indic-script case.

## 3. Scaling laws
`L(N, D) = E + A/N^α + B/D^β`. With C = 6ND, **Chinchilla** finds N and D scaling roughly equally, ≈ 20 tokens/param at the compute-optimal point. Kaplan (2020) favored bigger models; the disagreement traces to embedding-parameter counting and fixed LR schedules. **Inference-aware:** if a model will serve trillions of tokens, train it smaller for longer (Llama 3 8B saw ~15T tokens, ~1,900 tokens/param). Method: IsoFLOP sweeps (several N at fixed C, parabola fit to find the minimum, then a power law across C). Downstream task metrics scale much more noisily than loss. Predict loss, then map loss to tasks.

## 4. The standard recipe (2024–26 era)
AdamW (β₂ = 0.95, wd = 0.1) or Muon plus AdamW; warmup then cosine or WSD; clip 1.0; bf16 with fp32 master weights (FP8 at the frontier); init 0.02 with depth scaling, or μP; sequence packing with document masking; batch-size warmup; checkpoints every N steps; a small held-out eval suite run every few thousand steps. **Mid-training/annealing:** during the LR decay, upsample the highest-quality data (math, code, instructions); it moves benchmarks more than anything else per token.

## 5. Distributed training
| strategy | shards | communication | where |
|---|---|---|---|
| DP | batch | all-reduce grads (overlap with backward) | everywhere |
| ZeRO-1/2/3, FSDP | optimizer / + grads / + params | reduce-scatter + all-gather | memory-bound models |
| TP (Megatron) | each matmul (column then row) | 2 all-reduces per layer per pass | within a node (NVLink) |
| SP | layernorm/dropout activations along the sequence | pairs with TP | long sequences |
| PP | layers into stages | point-to-point activations; bubble `(p−1)/(m+p−1)` | across nodes |
| CP / ring attention | sequence for attention | pass KV blocks around a ring | very long context |
| EP | MoE experts | all-to-all tokens | MoE models |

Design order: fit the model (TP within a node, then PP across nodes, or FSDP), then scale throughput with DP. Overlap communication with computation. Check that each GPU's microbatch keeps it compute-bound.

## 6. Stability at scale
Loss spikes come from attention-logit growth, output-logit divergence, bad data batches, too-high LR, or fp8/bf16 overflow. Mitigations: QK-norm, z-loss, lower LR or longer warmup, removing the batch and rewinding to the last checkpoint, embedding norm, and monitoring grad-norm and max-logit dashboards. At thousands of GPUs, **hardware failures are routine**: fast async checkpointing, elastic restarts, detecting silent data corruption and stragglers. Llama 3's report is the best public account.

## 7. Long context
Train short (4–8k), then extend: raise the RoPE base or use YaRN, continue pretraining on long documents, use document masking so packed sequences don't attend across boundaries, and evaluate with retrieval and reasoning at depth (needle tests are necessary but not sufficient).

## 8. Exercise: plan a 7B run on paper
Budget: 2T tokens. Compute = 6 × 7e9 × 2e12 = 8.4e22 FLOPs. On 256 H100s at 40% MFU: 8.4e22 / (256 × 989e12 × 0.4) ≈ 8.3e5 s ≈ 9.6 days. Memory: 16 B/param = 112 GB, so FSDP across 8 GPUs gives 14 GB/GPU plus activations. Layout: FSDP (or TP = 2) within a node, DP across nodes; global batch ~4M tokens; checkpoint every ~2 h. Now defend every number in an interview.

**Read:** Chinchilla; Kaplan; Muennighoff (data-constrained); Sardana (beyond Chinchilla); FineWeb; DCLM; Lee et al. (dedup); Megatron-LM; ZeRO; PyTorch FSDP; Llama 3 herd (infrastructure sections); OLMo 2; HF *Ultra-Scale Playbook*.
