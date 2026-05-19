# Distributed Training Deep Dive

A standalone primer. Critical for Nvidia, Anthropic, OpenAI, Meta interviews — and for understanding any "how would you train X" question.

---

## Why distribute

A 70B model with optimizer state in FP32:
- Params: 70B × 4 bytes = 280 GB
- Gradients: 280 GB
- Adam state (m + v): 560 GB
- Activations (for backward): hundreds of GB depending on sequence + batch

Total for training: 1+ TB just for state, before any activation memory.

Modern GPU max: 80GB (H100 SXM5). One GPU can't fit even the model. Hence:

---

## Parallelism strategies

### 1. Data Parallelism (DP)

**Idea:** Each GPU has a copy of the model. Each processes a different mini-batch. Sync gradients.

```
GPU 1: [Model copy] → forward on batch A → gradients
GPU 2: [Model copy] → forward on batch B → gradients
GPU 3: [Model copy] → forward on batch C → gradients
GPU 4: [Model copy] → forward on batch D → gradients

All-reduce gradients → average → update each GPU's copy
```

**Pros:** Simple. Effective for moderately sized models.
**Cons:** Each GPU holds full model. Doesn't scale to models > GPU memory.

**Communication:** All-reduce per training step. Cost ~O(model_size).

### 2. Tensor Parallelism (TP)

**Idea:** Split each layer's matrix multiplication across multiple GPUs.

For a matmul `Y = X @ W`:
- Split W column-wise: W = [W1, W2]
- GPU 1 computes X @ W1
- GPU 2 computes X @ W2
- Concatenate results

Used heavily in: Megatron-LM, modern LLM training.

**Pros:** Reduces per-GPU memory.
**Cons:** Heavy intra-layer communication (all-reduce after every matmul).
**Best for:** Within a node (NVLink between GPUs).

### 3. Pipeline Parallelism (PP)

**Idea:** Split model into stages. GPU 1 has layers 1-10, GPU 2 has 11-20, etc.

Training pipeline:
```
GPU 1: layers 1-10
GPU 2: layers 11-20
GPU 3: layers 21-30
GPU 4: layers 31-40

Micro-batch flows: GPU 1 → GPU 2 → GPU 3 → GPU 4 → backward
```

**Pros:** Memory-efficient (each GPU has only fraction of layers).
**Cons:** "Pipeline bubbles" — GPUs idle while waiting. Mitigated with micro-batching.

**Best for:** Inter-node (slower communication).

### 4. Sequence Parallelism (SP)

**Idea:** Split the sequence dimension across GPUs (when LayerNorm/Dropout are the bottleneck).

Complementary to TP. Reduces activation memory.

### 5. Expert Parallelism (for MoE)

**Idea:** Different experts on different GPUs. Tokens routed to appropriate GPU.

Used in: Mixtral, DeepSeek-V3.

### 6. Combination: 3D parallelism (or 4D, 5D...)

Modern: combine DP + TP + PP (sometimes + SP, + EP).

E.g., Megatron-LM training a 175B model:
- TP=8 (within node, 8 GPUs)
- PP=8 (across 8 nodes)
- DP=64 (64 copies of the above setup)
- Total: 8 × 8 × 64 = 4096 GPUs

---

## ZeRO (Zero Redundancy Optimizer)

**Problem:** In DP, every GPU has duplicate copies of params, gradients, optimizer state. Wasteful.

**ZeRO Stage 1:** Partition optimizer state across GPUs. Gradient still all-reduced; each GPU updates only its partition.

**ZeRO Stage 2:** Also partition gradients. All-reduce becomes reduce-scatter.

**ZeRO Stage 3:** Also partition params. All-gather params on the fly during forward/backward.

**Memory savings:**
- ZeRO-1: ~4x reduction
- ZeRO-2: ~8x
- ZeRO-3: scales linearly with # GPUs (in theory infinite)

**Cost:** more communication. ZeRO-3 has highest overhead.

**Implementation:** DeepSpeed. PyTorch native FSDP is similar (closer to ZeRO-3).

---

## FSDP (Fully Sharded Data Parallel)

PyTorch native implementation of ZeRO-3-like sharding.

**Key concepts:**
- Wrap layers in FSDP
- Per layer: gather full params before forward, then re-shard
- Forward/backward + reshard
- Reduce-scatter gradients

**Advantages:**
- Native PyTorch
- Easier to use than DeepSpeed for many cases
- Better integration with PyTorch ecosystem

**When to choose FSDP vs DeepSpeed:**
- FSDP: simpler, PyTorch-native, fine for most jobs
- DeepSpeed: more features (offloading to CPU/NVMe, MoE primitives, advanced ZeRO)

---

## Communication primitives

### All-reduce
- Each GPU has data; result is sum (or other reduction) shared across all
- Used in: DP gradient sync, TP after-matmul
- Cost: O(model_size)

### Reduce-scatter
- All GPUs have data; result is reduced and PARTITIONED across GPUs
- Used in: ZeRO gradient handling

### All-gather
- Each GPU has a piece; result is concatenation on all GPUs
- Used in: ZeRO param gathering during forward

### Broadcast
- One GPU has data; sent to all others
- Used in: initial model param distribution

### Cost intuition
- Same node (NVLink): ~hundreds of GB/s
- Cross-node (InfiniBand): ~tens of GB/s
- Optimization: put high-bandwidth comm patterns (TP) within node

---

## Mixed precision training

**FP32 master copy** of weights + **FP16/BF16 forward/backward**:
- Forward: BF16 weights → BF16 activations
- Backward: BF16 gradients
- Optimizer: gradients converted to FP32, applied to FP32 master
- BF16 weights regenerated for next step

**Why:**
- BF16 is 2x faster on tensor cores
- Memory: 2x more space efficient
- FP32 master prevents accumulation error

**BF16 vs FP16:**
- BF16: same exponent range as FP32, less precision in mantissa
- FP16: narrower range, more precision
- BF16 wins for training (less overflow); FP16 OK for inference

**FP8:**
- New on H100
- 2x faster than BF16 in theory
- Still maturing; used in some recent training (DeepSeek-V3)

---

## Gradient accumulation

**Problem:** Want large batch size for training stability, but GPU memory limits real batch per step.

**Solution:**
- Process micro-batch, compute gradients
- DON'T update; keep accumulating
- After N micro-batches, update
- Effective batch size = N × micro-batch

**No quality cost, just slower convergence per step (but bigger steps).**

---

## Checkpoint / activation recomputation

**Problem:** Backward pass needs activations from forward; storing all is memory-prohibitive for deep networks.

**Solution:**
- Store activations at "checkpoint" boundaries only
- During backward, recompute activations between checkpoints

**Cost:** ~33% more compute (one extra forward pass through checkpoint segments).
**Win:** Much less memory.

Standard for large model training.

---

## Bottlenecks at scale

### 1000+ GPU training, common issues:

- **Stragglers:** one slow GPU slows all (sync barrier). Detect and replace.
- **Network failures:** node drops mid-training. Need checkpoint + restart.
- **Convergence:** large effective batch can hurt convergence. Use learning rate scaling laws (linear scaling rule, or square root for LR with warmup).
- **Cost:** $10M+ for top-tier training runs. Mistakes are expensive.

---

## Training a 70B model — concrete example

Llama 3 70B training (approximate):

- **Hardware:** ~2048 H100 GPUs
- **Strategy:** DP × TP × PP combination
  - TP=8 (within node)
  - PP=4 across 4 node groups
  - DP=64 replicas of the above
- **Memory:**
  - Params (BF16): 140GB → sharded
  - Gradients (BF16): 140GB
  - Optimizer state (FP32): 560GB
  - With ZeRO/FSDP: per-GPU memory ~40-60GB total
- **Time:** ~5-7 days on 2048 GPUs for ~15T tokens
- **Cost:** $5-15M
- **Token throughput:** ~500 tokens/sec per GPU × 2048 GPUs = ~1M tokens/sec aggregate
- **Failures:** dozens of stragglers / node failures across the run; automated restart from checkpoint

---

## Training a 1B model — for your portfolio

Doable! Use this as Flagship Project #3 stretch goal.

- Hardware: 4-8 A100 80GB (cloud rental ~$15-30/hr, $5-10k total for ~50-100hr training)
- Strategy: DP only with ZeRO-2 typically suffices
- Data: 200B-1T tokens (depending on Chinchilla optimality)
- Time: 50-200 hours
- Frameworks: Megatron-LM (advanced) or HF transformers + accelerate (simpler)

This shows recruiters: "I've trained an LLM end-to-end, not just used HuggingFace from a CSV."

---

## Common interview questions

1. **"Walk through training a 70B model."** — Use the example above
2. **"What's ZeRO-3?"** — Param sharding via all-gather on demand
3. **"Compare DP, TP, PP."** — Memory/compute/comm tradeoffs
4. **"What's gradient accumulation?"** — As above
5. **"Why mixed precision?"** — Speed + memory, with FP32 master
6. **"What's an all-reduce?"** — Sum then share result
7. **"How would you train if you had only 4 A100s?"** — Smaller model (7B), LoRA, gradient checkpointing
8. **"What's FlashAttention's role in training (vs inference)?"** — Faster forward + backward, less activation memory
9. **"What's a pipeline bubble?"** — Idle time in pipeline parallel; mitigated with micro-batches
10. **"How do you debug a NaN in training?"** — Check loss scaling, learning rate, gradient clipping, mixed precision overflow

---

## Resources

- **Megatron-LM repo** — Nvidia's reference for large model training
- **PyTorch FSDP docs**
- **DeepSpeed docs + paper**
- **"Efficient Large-Scale Language Model Training on GPU Clusters" — Megatron paper**
- **HuggingFace Hub for training tutorials**
- **The "Ultra-Scale Playbook" by HF (2024)** — comprehensive guide
- **"How to Train Really Large Models on Many GPUs" — Lilian Weng blog**

---

## Tradeoffs cheat sheet

| Strategy | Memory savings | Communication overhead | Best for |
|---|---|---|---|
| DP | None | Moderate (all-reduce) | Small models, abundant GPUs |
| TP | Moderate | High (within layer) | Big models, within node |
| PP | High | Low (between stages) | Big models, across nodes |
| ZeRO-1 | 4x | Moderate | Easy to deploy on DP |
| ZeRO-3 / FSDP | Massive | High | Models that don't fit otherwise |
| Activation checkpoint | High | Compute overhead | Memory-constrained, OK with slower |
| Mixed precision | 2x | None | Always useful |

---

End. Next: [`07-common-questions.md`](./07-common-questions.md)
