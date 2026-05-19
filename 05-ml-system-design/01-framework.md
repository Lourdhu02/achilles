# The 6-Step ML System Design Framework

Memorize. Use in every interview. Adapt to specifics.

---

## Step 1: CLARIFY (5 minutes)

**Goal:** Scope the problem. Most design problems are underspecified intentionally.

### Standard clarifying questions

1. **Who's the user?** B2C end users? Internal engineers? Other ML systems?
2. **What's the business goal?** Increase engagement? Reduce cost? Improve accuracy?
3. **What's the scale?** Users (DAU, MAU)? Requests/sec? Data size?
4. **What's the latency budget?** ms? Real-time vs near-real-time vs batch?
5. **What's online vs offline?** Real-time predictions or pre-computed?
6. **Any constraints?** Cost cap, privacy/PII, model size constraint, on-device requirement?
7. **What's the success metric?** Business metric (CTR, revenue), ML metric (accuracy, F1), system metric (uptime, latency)?

### Pick 4-6 questions to ask (not all). Show you're thinking about:
- Users and their needs
- Scale and constraints
- Success criteria

### Example
**Q:** "Design Twitter's home feed."

**You:** "A few clarifying questions:
1. Are we ranking for engagement (likes/retweets) or some other metric?
2. What's the scale — say 200M DAU, 100 QPS peak per user feed load?
3. Latency: <500ms p99 OK for feed load?
4. Should we handle the cold-start problem for new users?
5. Do we have user-item interaction history?
6. Are we incorporating safety / content moderation?"

Wait for answers. Take notes.

---

## Step 2: REQUIREMENTS (5 minutes)

### Functional requirements
What the system DOES:
- "Given a user, return top N tweets to show"
- "Tweets ranked by predicted engagement"
- "Filter unsafe content"
- "Personalize based on user history"

### Non-functional requirements (the numbers)
- **Throughput:** 200M DAU × ~20 feed loads/day / 86,400s = 46k QPS average; ~150k peak
- **Latency:** p99 <500ms for feed; per-tweet inference <20ms
- **Storage:** ~500B interactions historical; ~100B daily
- **Availability:** 99.99% (4 nines)
- **Cost:** budget $X/month or per-query

### ML requirements
- Model accuracy/quality target (e.g., NDCG@10 > 0.65)
- Refresh cadence (real-time, hourly, daily?)
- Cold start handling

### Write these on the board. Refer back as you design.

---

## Step 3: ARCHITECTURE (10 minutes)

### Standard ML system blocks
Almost every ML system has these:

```
Data sources → Feature store → Model training → Model registry
                                                       ↓
User request → API gateway → Feature retrieval → Model inference → Post-process → Response
                  ↓                                                    ↓
              Logging                                           Online metrics
```

### Draw top-down

1. **Start with user-facing API:** what's the contract?
2. **Then add: serving layer** (model inference)
3. **Then: feature retrieval** (where features come from at inference time)
4. **Then: data flow** (where features ORIGINALLY come from)
5. **Then: training pipeline** (how the model gets trained / updated)
6. **Then: feedback loop** (how labels / signals flow back)

### Patterns to know

#### Pattern A: Candidate Generation + Ranking (for recommendations)
```
User request →
  [Candidate generation: retrieve top 1000 candidates from billions]
    (collaborative filter, content-based, or vector search)
  →
  [Ranking: score each with rich features]
  →
  [Reranking: business rules (diversity, recency)]
  →
  Top N output
```

#### Pattern B: Two-Stage Inference (for low latency)
```
Online: cheap feature lookup → lightweight inference → response (10ms)
Offline: heavy features + complex model → pre-compute embeddings → update online store (hourly/daily)
```

#### Pattern C: RAG (your sweet spot)
```
Query → Embed → Retrieve top-k → Rerank → Context construction → LLM → Response
```

#### Pattern D: Lambda Architecture
```
Real-time: stream processing (Flink, Kafka Streams) → instant updates
Batch: daily Spark / DBT → comprehensive recompute
Combine for serving
```

#### Pattern E: Multi-armed bandit / online learning
- For when ground truth comes back fast (click, no click)
- Models update continuously
- Used in: ads ranking, content recommendation

---

## Step 4: DEEP DIVE (20 minutes)

This is where you DEMONSTRATE depth. Interviewer steers you toward 2-3 components.

### For each component, cover:

#### Data
- Where does it come from?
- How big?
- Update frequency?
- Schema / format?
- PII / sensitive?

#### Features
- What features matter?
- How are they computed (offline vs online)?
- Feature store choice (Redis, DynamoDB, Feast)?
- Feature freshness (stale = how stale acceptable)?

#### Model
- Architecture (linear, tree-based, NN, transformer?)
- Why this choice given constraints?
- Training: distributed? data size? duration?
- Hyperparameters: high-level approach
- Updates: how often retrained? online learning?

#### Serving
- Synchronous vs async?
- Batching strategy?
- Caching (model output, intermediate)?
- Quantization? Pruning? Distillation?
- Hardware: CPU / GPU / TPU / what kind?

#### Evaluation
- Offline metric on holdout
- Online A/B test plan
- Business metric translation
- Counterfactual measurement (uplift)

---

## Step 5: SCALE (5-10 minutes)

"What if we have 10x the traffic? 100x?"

### Scaling axes
- **Horizontal:** add more replicas (most common)
- **Vertical:** bigger machines
- **Caching:** memoize at every layer
- **Sharding:** partition by user / item / time
- **Async-ification:** move heavy work off the request path

### Bottlenecks at scale
- **Model inference latency:** quantize, batch, distill
- **Feature retrieval:** cache, denormalize, pre-compute
- **Storage:** sharding strategy, archival of old data
- **Cross-region:** read replicas, eventual consistency

### Cost at scale
- Cost per request × QPS × time = monthly bill
- Optimize highest-cost component first
- Sometimes: use cheaper proxy model (rule-based fallback for tail traffic)

---

## Step 6: ITERATE (5-10 minutes)

### Evaluation strategy
- Offline: NDCG@10 / AUC / etc. on holdout
- Shadow: deploy new model alongside old, compare outputs, no user impact
- Canary: route 1% of traffic to new model, monitor
- A/B test: 50/50 with business metrics

### Monitoring
- Latency (p50, p99)
- Error rate
- Model output distribution drift
- Input feature drift
- Business metric drift (CTR, etc.)

### Common production issues
- Model serving slow / down → fallback to cached or rule-based
- Data pipeline breaks → stale features → model degrades silently
- New feature breaks for old users → predictions broken
- Adversarial inputs → safety degradation

### Future improvements
- Better features
- More sophisticated model
- Multi-objective optimization
- Personalization layer
- Active learning loop

---

## Communication tips during the design

### Pacing
- Spend ~5 min on each of steps 1, 2, 5, 6
- ~10 min on step 3
- ~20 min on step 4

### Engage interviewer
- "Should I drill into ranking model or feature engineering?"
- "Does this match what you had in mind?"
- "Want me to discuss eval strategy now or later?"

This shows collaboration, not monologue.

### Handle pushback
- Interviewer: "Why this choice over X?"
- You: "Good question. X has these tradeoffs... but in this context, Y is better because... However, if [constraint changed], I'd reconsider."

Never get defensive. Treat pushback as a tool for going deeper.

### State assumptions explicitly
- "I'm assuming we have GPU capacity for inference. If we're CPU-only, I'd use a smaller distilled model."
- "Assuming data engineering team handles feature pipelines. If not, I'd add..."

---

## Common follow-up questions

After your initial design, expect:

1. "How would you handle the cold-start problem?"
2. "What if the model goes stale — say it was trained 3 months ago?"
3. "How would you debug if CTR dropped 5% suddenly?"
4. "What if we need to add fairness/bias constraints?"
5. "How would you handle multi-region deployment?"
6. "What if we wanted to add a new feature like {X}?"
7. "How would you adversarially robust this system?"
8. "What about privacy / GDPR / data retention?"

Have answers ready.

---

## Practice template (use for every design problem)

Write your own answers in this structure for each practice problem:

```markdown
## Problem: Design X

### Step 1: Clarify
- [Questions I'd ask]
- [Assumptions I'd make if not answered]

### Step 2: Requirements
- Functional: [bullet list]
- Non-functional: [numbers: QPS, latency, scale, availability, cost]
- ML reqs: [accuracy target, refresh, cold start]

### Step 3: Architecture
- [High-level diagram in ASCII or description]
- [Key components and data flow]

### Step 4: Deep dive
#### Component A: [name]
- Data: ...
- Features: ...
- Model: ...
- Serving: ...

#### Component B: [name]
- ...

### Step 5: Scale
- [Bottlenecks]
- [Scaling strategy]

### Step 6: Iterate
- [Eval plan]
- [Monitoring]
- [Future improvements]

### Tradeoffs I considered
- [Choice X vs Y; why I picked X]
- [Choice A vs B; why I picked A]
```

---

## What separates a "hire" from a "lean hire"

Even a complete framework execution gets you to "lean hire". To get "hire" or "strong hire":

- Cite specific real-world systems ("similar to how YouTube does it...")
- Quote real numbers ("YouTube reportedly uses two-tower for candidate gen with billions of items")
- Discuss tradeoffs unprompted
- Connect to your real experience ("at Sujanix, we faced a similar issue with...")
- Show production thinking (failure modes, monitoring, gradual rollout)

You'll get to "hire" by drilling 15-20 design problems with explicit tradeoff analysis.

---

Next: [`02-design-rag-at-scale.md`](./02-design-rag-at-scale.md) — your sweet spot
