# Design: Recommendation System (YouTube / Reels / Feed)

**Problem:** Design YouTube's recommendation system: serve personalized video recommendations to 2B+ users.

The classic ML system design question. Asked at Google, Meta, Netflix, ByteDance, every recommender-heavy company.

---

## Step 1: Clarify (5 min)

1. **Surface:** Homepage feed? Watch-next? Search results?
2. **Goal:** Watch time? Engagement? Diversity? Revenue?
3. **Scale:** Users, videos, requests/sec?
4. **Latency:** Homepage load <200ms? Recommendation refresh after each video?
5. **Personalization:** Pure individual? Cohort-based? New users (cold start)?
6. **Constraints:** Diversity? Safety? Recency boost?

Assumptions:
- Homepage feed
- Goal: watch time + retention (multi-objective)
- 2B users, 1B videos, 100k requests/sec
- p95 < 200ms
- Personalized with cold-start fallback
- Diversity + safety + freshness

---

## Step 2: Requirements (3 min)

### Functional
- Return: 20-50 video IDs ranked
- Personalized per user
- Handles new users (no history) and new videos (no engagement)
- Excludes already-watched, recently-watched
- Respects user preferences (e.g., "not interested" feedback)

### Non-functional
- 100k QPS sustained
- p95 latency 200ms
- 99.99% uptime
- Data: 100B events/day (impressions, clicks, watch time)

### ML
- Online A/B test: test new models vs current champion
- Watch time per session: north-star metric

---

## Step 3: Architecture (10 min)

Classic two-stage:

```
                    OFFLINE
                    
User events → Kafka → Stream processor → Feature store
                                            ↓
                                       Model training
                                            ↓
                                       Model registry
                                            
                    ONLINE
                    
User request → API Gateway
                    ↓
              [Candidate Generation]   ← Pull top 1000 candidates from 1B videos
                    ↓
              [Ranking]                ← Score each with rich features (~10ms)
                    ↓
              [Reranking]              ← Diversity, business rules, ads injection
                    ↓
              [Filter]                 ← Already watched, blocked, age-gated
                    ↓
                Top 20-50 results
```

---

## Step 4: Deep Dive

### Component A: Candidate Generation

Goal: from 1B videos, find ~1000 relevant. Latency budget: ~50ms.

**Multi-source approach:**

#### Source 1: Two-Tower model (collaborative + content)
- User tower: embeds user (history, profile) → 256-dim vector
- Item tower: embeds video (metadata, content embeddings) → 256-dim vector
- Trained: contrastive loss (positive = user watched; negative = sampled)
- Serving: ANN (HNSW) over all 1B items, retrieve top 500

#### Source 2: Collaborative filter (matrix factorization)
- Classic ALS or modern neural CF
- Top 200 candidates from "users like you watched"

#### Source 3: Trending / Popular by region
- Top 100 from "trending in your region in last 24hr"
- Handles new users (cold start)

#### Source 4: Subscriptions / Follows
- Direct: videos from channels user subscribes to
- Always include

#### Source 5: Recent watch context
- "Continue watching" — partially-watched videos
- "Related to last watch" — similar videos to user's recent watch

**Merging:**
- Union all sources → ~1000-2000 candidates
- Dedup
- Pass to ranking

### Component B: Ranking

Goal: score each candidate. Latency budget: ~50ms for ~1000 candidates.

**Model:** Deep neural network (Wide & Deep or DLRM-style)

**Features per (user, item) pair:**
- User features: demographic, watch history aggregates, time on platform, device
- Item features: video metadata (length, language, category, age), embedded content
- Context features: time of day, day of week, device, location
- Interaction features: user-channel affinity, last watch on this channel, time since last engagement

**Architecture:**
- Wide part: linear over crossed features
- Deep part: 3-5 layer MLP over embeddings
- Combined: weighted sum

**Output:** P(watch | impression) AND E[watch_time | watch] → composite score

**Training:**
- Daily / hourly retrain on recent data
- Loss: logistic on click + MSE on watch time (multi-task)

**Serving:**
- Batched inference (1000 candidates at once)
- Model on GPU; ~10-30ms p99
- Feature retrieval from feature store: ~10ms

### Component C: Reranking

Goal: business + UX optimization. Latency budget: ~10ms.

**Reranking layers:**
- **Diversity:** ensure top N has variety (categories, languages)
- **Freshness:** boost recent uploads
- **Safety:** demote borderline content
- **Ads:** inject sponsored content
- **Experiments:** A/B test variants

**Reranker:** typically rule-based + light ML

### Component D: Feature store

**Critical infrastructure:**
- User features: ~5000 features per user, stored in low-latency KV (Bigtable, DynamoDB)
- Item features: ~5000 per video
- Real-time features (just-now interactions): served from Redis
- Historical features: served from offline-trained snapshots

**Tools:** Feast (open-source), or proprietary (Pinterest's, Uber's Michelangelo)

### Component E: Training pipeline

**Data flow:**
- User events → Kafka → Spark streaming → Hive/BigQuery (raw events)
- Daily: Spark job aggregates features, joins with item metadata
- Twice daily: retrain ranker on rolling window
- Continuous: stream-trained candidate generation embeddings

**Negative sampling:**
- Critical for contrastive training
- Hard negatives: similar items the user did NOT watch
- Random negatives: cheap baseline

**Online A/B:**
- 1% traffic to new model
- Track watch time, retention
- Holdout cohort for true causal estimate
- Roll out if positive

---

## Step 5: Scale (5 min)

### Going to 1M QPS

- Cache top recommendations per user (refresh every 5 min)
- Pre-compute candidates for cold users (heavy hitters)
- Shard by user ID hash
- Geo-distributed read replicas for feature store

### Cold start

- New users: trending + diversity sample + collaborative filter via cohort (similar demographics)
- New items: content-based features only (no engagement signals yet); show to small audience initially to gather signals; promote based on early engagement

### Adversarial robustness

- Bot detection (impressions without engagement)
- Click farms detection
- Fake watch time (bot-like behavior)
- Demoting low-quality content sources

---

## Step 6: Iterate (5 min)

### Eval

**Offline:**
- AUC, NDCG@10 on holdout
- Replay simulation
- Cohort comparison

**Online:**
- A/B test with watch time, session length, retention
- Holdout cohort for unbiased estimate
- Long-term: 30-day retention impact

### Monitoring

- Recommendation diversity (Gini coefficient over categories)
- Catalog coverage (% of videos shown to >0 users)
- Engagement metrics by user cohort
- Drift detection (input feature distributions)

### Future improvements

- Multi-task learning (predict click, watch time, like, comment jointly)
- Reinforcement learning (long-term reward optimization)
- Multi-modal embeddings (audio + visual + text)
- Sequence models for context-aware (transformer over user history)

---

## Common follow-ups

1. **"How do you handle the diversity-vs-engagement tradeoff?"**
   - Multi-objective: weighted sum of objectives
   - Calibrate via business preference

2. **"How do you handle exploration vs exploitation?"**
   - Epsilon-greedy: 10% random
   - Thompson sampling (Bayesian bandit)
   - Boost less-shown items proportionally

3. **"What if a creator complains 'my videos aren't recommended'?"**
   - Show their data: watch time, CTR, comparison to similar
   - Often: their content has low engagement signals
   - System works as intended; not personal

4. **"How do you A/B test responsibly?"**
   - Holdout cohort (never sees the variant)
   - Per-user randomization (not per-session)
   - Statistical significance with appropriate multiple-comparison correction
   - 2-week minimum runtime

5. **"What about regulatory / fairness?"**
   - Audit recommendations across demographic groups
   - Bias metrics: equal opportunity, demographic parity
   - Compliance team review for protected categories

6. **"How would you incorporate GenAI?"**
   - LLM for content understanding (better embeddings from video transcripts)
   - LLM for query reformulation (in search)
   - LLM for explainability ("we recommended this because...")
   - Eventually: generative recommendations (LLM proposes items, ranker validates)

---

## Tradeoffs

- **Latency vs Quality:** more expensive models = better quality, higher latency. Two-tower + DLRM is sweet spot.
- **Personalization vs Privacy:** more personal = better recommendation, more data needed
- **Engagement vs Wellbeing:** optimize watch time → potentially harmful spiral. Balance with quality signals.
- **Cold start:** trending/cohort fallbacks vs personalization → tradeoff between new-user UX and overall accuracy

---

## Real-world references (cite in interview)

- **YouTube paper:** Covington, Adams, Sargin 2016 ("Deep Neural Networks for YouTube Recommendations")
- **Twitter (X):** open-sourced algorithm (heuristic + ML rankings)
- **Pinterest:** Pixie Random Walk for candidate generation
- **Netflix:** factorization machines, contextual MAB
- **Spotify:** Discover Weekly approach (collaborative + content)
- **TikTok:** rapid online learning, multi-armed bandit

Mention these in your answer. Shows depth.

---

Next: [`05-design-fine-tuning-platform.md`](./05-design-fine-tuning-platform.md)
