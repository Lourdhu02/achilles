# Design: Fine-Tuning Platform

**Problem:** Design a fine-tuning service like OpenAI's or Anthropic's, where customers can fine-tune LLMs on their data without managing infrastructure.

Critical for: AWS Bedrock, OpenAI Fine-Tuning, Azure OpenAI, Together.ai, Predibase.

---

## Step 1: Clarify (5 min)

1. **Models supported:** Specific list or "any open model"?
2. **Customer scale:** Few enterprise vs many SMBs?
3. **Data sizes:** Typical fine-tune dataset size?
4. **Multi-tenancy:** Cost vs strict isolation?
5. **Fine-tuning types:** SFT only? DPO? Continued pre-training?
6. **Inference:** Do we also serve the fine-tuned model?
7. **Pricing model:** Per training-hour? Per inference token?

Assumptions:
- Open models: Llama 8B, Llama 70B, Mistral 7B
- Many customers (~1000), most SMB
- Typical dataset: 1k-50k examples
- Strict tenant isolation (data + model)
- SFT + DPO supported
- Yes, we serve fine-tuned models
- Per training-hour + per inference token

---

## Step 2: Requirements (3 min)

### Functional
- Customer uploads dataset (JSONL)
- Customer selects base model + hyperparameters
- Backend trains, returns fine-tuned model ID
- Customer queries fine-tuned model via API
- Track jobs, billing, model registry

### Non-functional
- Time to train (median): 1-4 hours for 1k examples, 4-12 hours for 50k
- Inference latency: < 500ms first token (small models), < 2s (70B)
- Cost: profitable on per-hour training pricing
- Throughput: 100 concurrent training jobs, 10k inference QPS

### Data isolation
- Customer data: encrypted at rest, in transit
- Models: tagged with customer ID, not shared across tenants
- Audit logs: who accessed what

---

## Step 3: Architecture (10 min)

```
                CUSTOMER PORTAL / API
                          ↓
                  [Job Submission]
                          ↓
                  Job Queue (Kafka)
                          ↓
        ┌─────────────────┴────────────────┐
        ▼                                   ▼
TRAINING ORCHESTRATOR              INFERENCE SERVICE
        │                                   │
   ┌────┴────┐                       ┌─────┴─────┐
   ▼         ▼                       ▼           ▼
[Trainer    [Trainer                [Inference   [Multi-LoRA
Llama-8B]   Llama-70B]               Server      Server]
   │          │                       (vLLM)]
   ▼          ▼                          ▼
1xA100      8xA100                    Backend pools
GPU pool    GPU pool                  Sharded by model

   ▼            ▼                          ▼
Model Registry (S3 + metadata DB)
   ▼            ▼                          ▼

Logging + Telemetry → ELK + Grafana
Billing events → DB
```

---

## Step 4: Deep Dive

### Component A: Job submission & validation

**API:**
```
POST /fine-tuning/jobs
{
  "base_model": "llama-3-8b",
  "training_file": "file-abc123",
  "hyperparameters": {"learning_rate": 1e-4, "lora_rank": 16, "epochs": 3, "method": "lora"},
  "validation_file": "file-def456"
}
```

**Validation:**
- File format check (JSONL, valid schema)
- Size check (within plan limits)
- PII scan (optional but advisable)
- Estimate cost + time, return to customer for approval

### Component B: Training orchestrator

**Job queue:**
- Kafka topic per priority tier (premium vs standard)
- Workers consume, claim job, run

**Trainer process:**
- Pulls dataset from S3 (with customer's data encryption key)
- Spins up training pod (K8s) with appropriate GPU
- Uses HuggingFace TRL for SFT/DPO
- LoRA by default (cheaper, faster, smaller artifacts)
- Optionally full FT for premium customers

**GPU pool management:**
- Llama 8B fits on 1xA100 → run on smaller GPU pool
- Llama 70B needs 8xA100 → run on big pool
- Quotas per tier (max concurrent jobs)
- Job scheduler (e.g., Volcano, Slurm) for fair allocation

**Training duration estimation:**
- Learn from past runs by model + dataset size
- ML model predicts duration → improves over time
- Used for billing pre-auth

**Checkpointing:**
- Save every N steps
- Resume on failure (saves customer money on partial restarts)

### Component C: Model registry

**Storage:**
- S3 bucket with customer ID prefix
- LoRA adapters: small (10-100 MB)
- Full fine-tunes: larger (multi-GB)
- Encrypted with customer-specific key

**Metadata DB:**
- Model ID, customer ID, base model, hyperparameters, training metrics
- Status (training, ready, deprecated)
- Cost ledger

### Component D: Inference service (multi-LoRA serving)

**Key insight: serve many LoRA adapters on same base model**
- Base model loaded once (e.g., Llama 8B = 16GB)
- LoRA adapters: ~100MB each
- Memory budget: 1 H100 80GB → 1 base + ~500 LoRA adapters cached
- Hot-swap LoRAs based on request

**Architecture:**
- Multi-LoRA vLLM (supports this natively)
- Customer request → lookup model ID → load LoRA into context → inference

**Latency:**
- First request to a cold LoRA: load 100MB from disk (~500ms-2s)
- Subsequent: cached, no overhead
- Strategy: keep recently-used LoRAs warm; LRU eviction

### Component E: Billing

**Training:**
- Per-hour pricing × actual hours used
- Track: GPU type, GPU count, wall-clock time
- Billed after job completion

**Inference:**
- Per-token (input + output, weighted)
- Tracked at request level
- Pre-paid credits or post-pay invoicing

### Component F: Data security

**Customer data:**
- KMS-managed encryption keys (per-customer)
- Data at rest: encrypted
- Data in transit: TLS
- Data in GPU memory: ephemeral, wiped after job

**Compliance:**
- SOC 2 audit
- GDPR data deletion (right to be forgotten)
- HIPAA mode (for healthcare customers)

---

## Step 5: Scale (5 min)

### To 10x customers

- Reservation system: book GPU time in advance for predictable cost
- Spot instance pool for cost-flexible jobs
- Multi-region (US, EU, Asia) for data residency
- GPU clusters per region

### Cost optimization

- Spot GPUs for queued jobs (interruption-tolerant via checkpointing)
- Smaller LoRA rank by default (rank 8 vs 16 → 2x faster)
- Mixed precision (BF16) for training (default)
- Quantize base models for inference (INT8 → 2x more throughput)

### Performance optimization

- Continuous batching for inference (vLLM)
- Multi-LoRA serving (huge cost win)
- Speculative decoding for cost-sensitive customers
- Prefix caching across customer requests (within same model)

---

## Step 6: Iterate (5 min)

### Eval

- Per-customer: hold out 10% of their data, compute loss after fine-tune
- Aggregate: track success rate (job completed without OOM, NaN, etc.)
- A/B: new optimizers, new hyperparameters → faster training same quality

### Monitoring

- Job failure rate (target <1%)
- Mean training time (track over time)
- GPU utilization
- Customer NPS

### Future improvements

- Auto-hyperparameter tuning
- Bring-your-own-data-format (CSV, plain text, etc.)
- Distillation as a service (fine-tune student from teacher)
- RLHF / DPO from preference feedback
- Multi-model fine-tuning (fine-tune the same dataset on 3 models, customer picks best)

---

## Common follow-ups

1. **"How do you prevent customers from extracting other customers' data via the model?"**
   - Strict tenant isolation: customer A's LoRA never serves customer B's queries
   - Encryption per customer
   - Memory isolation at GPU level (no shared state)

2. **"What if a customer's dataset has poisoning attempts?"**
   - Pre-train data sanity checks
   - Outlier detection on training loss curves
   - Manual review for "weird" jobs

3. **"How do you handle a customer requesting a model that doesn't fit in memory?"**
   - Validate at submit time
   - Queue with FSDP / DeepSpeed for very large jobs
   - Charge differently

4. **"How do you debug a customer's failed training?"**
   - Detailed logs (loss curves, GPU usage, errors)
   - Auto-suggest fixes ("your learning rate is 10x default — likely instability")
   - Support team can re-run with customer permission

5. **"How would you support continued pre-training (not just fine-tuning)?"**
   - Separate orchestrator for longer-running jobs
   - More compute slots (days vs hours)
   - Different pricing tier

6. **"What if a customer wants real-time fine-tuning (online learning)?"**
   - Out of scope for batch service
   - Would require: streaming data, online update queues, very different infra
   - Could be a v2 feature

---

## Tradeoffs

- **LoRA vs Full FT:** LoRA cheaper, smaller, easier to serve; Full FT can reach better quality on complex tasks
- **Multi-LoRA serving:** great for many customers w/ small adapters; bad if customers have unique base models
- **Spot vs Reserved GPUs:** Spot cheaper, interruption risk → mitigated by checkpointing

---

## References

- OpenAI Fine-Tuning API docs
- HuggingFace TRL docs
- Predibase blog (commercial fine-tuning service)
- "S-LoRA: Serving Thousands of Concurrent LoRA Adapters" — paper

---

Next: [`06-distributed-training.md`](./06-distributed-training.md)
