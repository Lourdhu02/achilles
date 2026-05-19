# 05 — ML System Design

The hardest interview round for 2-YOE candidates. Building a project ≠ designing a system that serves 100M users at 99.99% uptime.

**Time investment:** Months 5-10. ~20-30 design problems solved by application time.

---

## Files

1. [`01-framework.md`](./01-framework.md) — The 6-step framework for any ML system design question
2. [`02-design-rag-at-scale.md`](./02-design-rag-at-scale.md) — Production RAG (your sweet spot)
3. [`03-design-llm-inference.md`](./03-design-llm-inference.md) — LLM serving platform
4. [`04-design-recommender.md`](./04-design-recommender.md) — Feed/recommendation ranking
5. [`05-design-fine-tuning-platform.md`](./05-design-fine-tuning-platform.md) — Train + deploy fine-tunes at scale
6. [`06-distributed-training.md`](./06-distributed-training.md) — Train 70B+ models (deep dive)
7. [`07-common-questions.md`](./07-common-questions.md) — 15 design problems with brief solutions

---

## Why this round is hard

You'll be given a vague prompt: "Design Twitter's home feed ranking." Or "Design a RAG system for 10M documents at 100 QPS." You have **50 minutes** to:

- Ask the right clarifying questions
- Define functional + non-functional requirements
- Sketch a high-level architecture
- Drill into 2-3 components
- Discuss scaling, monitoring, failure modes, eval
- Show production thinking

Most candidates fail because:
- They jump to model details before discussing data + serving
- They draw boxes without explaining WHY each box exists
- They forget non-functional reqs (latency, cost, scale)
- They can't discuss tradeoffs
- They run out of time

---

## The 6-Step Framework (briefly — see `01-framework.md` for full)

```
1. CLARIFY      — Ask 4-6 questions to scope the problem
2. REQUIREMENTS — Functional + non-functional, numbers
3. ARCHITECTURE — High-level boxes + data flow
4. DEEP DIVE    — Drill into 2-3 components with detail
5. SCALE        — How does this work at 10x, 100x volume?
6. ITERATE      — Eval, monitoring, A/B testing, future improvements
```

Memorize this. Use it in every mock.

---

## What you'll be asked

### High-frequency design problems for GenAI Eng:
- Design Twitter's home feed
- Design YouTube recommendations
- Design Google's search autocomplete
- Design a RAG system at scale
- Design ChatGPT's inference platform
- Design a fine-tuning platform (for customers)
- Design a content moderation pipeline
- Design fraud detection
- Design an LLM eval platform
- Design Amazon's "Customer also bought"

### Less frequent but possible:
- Design Google Translate
- Design a real-time ASR system
- Design image search (CLIP-style)
- Design a code completion system (Copilot)
- Design a recommendation system for restaurants
- Design a churn prediction system
- Design an A/B testing platform

---

## How to practice

### Stage 1 (Month 5): Learn the framework
- Read `01-framework.md` thoroughly
- Practice ONE design with the framework, no time limit
- Identify which steps you skip naturally

### Stage 2 (Months 5-6): Do worked examples
- Read each `02-` through `05-` deep-dive file
- Try to design BEFORE reading the solution
- Compare your design to the file's; note gaps

### Stage 3 (Months 6-8): Time-pressured practice
- 50-minute design sessions
- Use a whiteboard or Excalidraw
- Pick from `07-common-questions.md`
- Self-debrief after each: what did you miss?

### Stage 4 (Months 7-10): Mocks
- Real human interviewers
- Pramp, interviewing.io, peers
- Aim: 10+ mocks specifically on system design

---

## What "good" looks like

A strong design answer is 60% structure + 40% depth.

### Structure (60%)
- Followed the framework
- Covered all major components
- Drew clear data flow
- Articulated tradeoffs explicitly
- Talked about eval + monitoring

### Depth (40%)
- Cited specific algorithms / techniques
- Quoted real-world numbers (latencies, throughputs)
- Discussed failure modes
- Drew the system at 10x scale
- Mentioned specific tools / frameworks

### Communication
- Diagrammed clearly
- Paced (didn't rush, didn't drag)
- Engaged interviewer (asked check-ins)
- Handled pushback gracefully

---

## Drawing diagrams in remote interviews

Tools:
- **Excalidraw** — fastest for live interviews
- **Whimsical** — better for polished
- **Lucidchart** — heavy but feature-rich
- **draw.io** — free, capable
- **Tablet + pen** — most natural; consider buying an iPad or Wacom

Practice drawing common patterns:
- Client → API gateway → Service → DB
- Lambda architecture (batch + stream)
- Microservices with message queue
- ML inference pipeline (preprocess → model → postprocess)
- RAG pipeline (parse → embed → store → query → rerank → generate)

---

## Common pitfalls

### Pitfall 1: Diving into model details first
Bad: "I'd use a transformer with attention..."
Good: "First, let me understand: what's the user-facing experience? Then we can pick the model later."

### Pitfall 2: Ignoring data
ML systems are 80% data, 20% model. If you don't discuss:
- Where data comes from
- How it's preprocessed
- How it's labeled
- How it's stored
- How features are computed

...you fail.

### Pitfall 3: One-size-fits-all
Bad: "Use Kafka, K8s, Spark, every buzzword."
Good: "For 100 QPS, Kafka is overkill. A simple SQS queue suffices."

Justify every component.

### Pitfall 4: No numbers
"It's at scale" — meaningless.
"100M users, 10 QPS peak, 100ms p99 latency, 1GB/day data" — strong.

### Pitfall 5: Ignoring eval
"Then we deploy the model." STOP.
"How do we know it works? A/B test on 1% of traffic, measure CTR + dwell time + revenue. Roll out gradually if all metrics improve significantly."

### Pitfall 6: Skipping failure modes
"If the model is down" → 5xx error → bad UX.
Discuss: circuit breakers, fallbacks (rules-based, cached responses, simpler model), graceful degradation.

---

## Resources

### Books
- **"Machine Learning System Design Interview" by Ali Aminian** — best book in this space, very practical
- **"Designing Machine Learning Systems" by Chip Huyen** — broader, more foundational
- **"Designing Data-Intensive Applications" by Martin Kleppmann** — system design bible (non-ML); helpful for the system side
- **"AI Engineering" by Chip Huyen** — newer, LLM-focused

### Video courses
- **"Grokking the ML Interview"** by Educative — $60ish
- **"Designing ML Systems" course on Coursera/Udemy** — varies
- **YouTube channels:** Tech Dummies, ByteByteGo, System Design Fight Club

### Articles
- **Eugene Yan's blog** — applieddata.eugene.yan, excellent essays
- **Chip Huyen's blog** — huyenchip.com
- **Vicki Boykis blog** — vickiboykis.com (great real-world ML)
- **/r/MachineLearning case study posts**

### Real-world papers/blogs
- YouTube recommendation paper (Covington et al., 2016)
- Twitter ranking (open-sourced repo)
- Pinterest visual search blog
- Uber Michelangelo platform
- Netflix recommendation blog
- Stripe Radar architecture

Read these. They are the gold standard.

---

## Your unique angle for system design

For each design problem, lean into:
- **GenAI/LLM dimension** if applicable (your edge)
- **Production engineering** (you've shipped stuff)
- **Cost optimization** (your AWS Lambda 30% reduction story)
- **Privacy/security** (FinSentinelAI multi-tenant)

You'll outperform vanilla 2-YOE candidates who can't connect "design" to "real production tradeoffs they've made".

---

Next: [`01-framework.md`](./01-framework.md)
