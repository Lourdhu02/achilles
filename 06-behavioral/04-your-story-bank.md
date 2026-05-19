# Your Story Bank — Active File

**This is YOUR active file. Fill in over Months 6-9. Update through Month 15.**

Goal: 20-25 polished STAR stories, mapped to question types, Amazon LPs, and Anthropic values.

---

## Quick LP/Theme → Story Index (fill in as stories developed)

| Theme | Story Title | LP / Value |
|---|---|---|
| Complex technical project | TBD | Dive Deep, Are Right |
| Failed project | TBD | Learn and Be Curious |
| Disagreement with manager | TBD | Have Backbone |
| Took ownership outside role | TBD | Ownership |
| Handled ambiguity | TBD | Bias for Action |
| Learned new tech fast | TBD | Learn and Be Curious |
| Missed deadline | TBD | Deliver Results |
| Delivered under constraint | TBD | Frugality |
| Mentored someone | TBD | Hire and Develop |
| Influenced without authority | TBD | Earn Trust |
| Said no / pushed back | TBD | Have Backbone |
| Simplified complexity | TBD | Invent and Simplify |
| Fixed unprompted | TBD | Ownership |
| Customer feedback changed approach | TBD | Customer Obsession |
| Turned around struggling project | TBD | Deliver Results |
| Made controversial call | TBD | Are Right |
| Decided to STOP a project | TBD | Deliver Results |
| Scaled something 10x | TBD | Think Big |
| Improved team velocity | TBD | Bias for Action |
| Handled criticism | TBD | Earn Trust |

---

## Story Template (copy for each)

```markdown
## Story #N: [Memorable Title]
**Primary LP/Value:** [LP/Value]
**Secondary LPs/Values:** [...]
**Best for question:** [Question type]
**Talking time:** Target 2-3 min

### Situation (20-30s)
[Context: who, what, when, where; what was at stake]

### Task (15-20s)
[Your specific responsibility / goal in this situation]

### Action (60-90s)
[What YOU did — specific decisions, tradeoffs, technical details]
- Decision 1: [what + why]
- Decision 2: [what + why]
- Roadblock: [...] → How you overcame: [...]

### Result (30-45s)
[Outcome with NUMBERS; impact]
- Quantified outcome: [...]
- Stakeholder impact: [...]

### Learning (15-20s)
[What you'd do differently; what you learned; how it shapes your work now]

### Link to role (optional, 10-15s)
[How this connects to the role/team you're interviewing for]

### Practice notes
- [ ] Practiced out loud 3x
- [ ] Recorded + listened back
- [ ] Got feedback from someone
- [ ] Time check: under 3 min
- [ ] Has at least 2 specific numbers
- [ ] Has a specific learning
```

---

## Pre-filled Story #1: The Sujanix OCR Pipeline (your hero technical story)

**Primary LP:** Insist on Highest Standards
**Secondary LPs:** Dive Deep, Are Right A Lot, Bias for Action, Frugality, Customer Obsession
**Best for question:** "Tell me about a complex technical project" / "Quality bar" / "Hard decision"
**Talking time:** 3 min

### Situation
At Sujanix (Jan 2026–Present), I joined to lead the Transformer-based OCR pipeline for high-stakes government utility automation. The challenge: meter images from real-world deployments had motion blur, low contrast, and inconsistent lighting. The vendor we replaced was at 89% exact-match accuracy. Customer contract required 95%+. Time pressure: 8 weeks to demonstrate viability.

### Task
As the lead ML engineer, I owned model architecture selection, training, optimization, and production deployment. Multiple stakeholders had conflicting priorities: CTO wanted speed, product wanted reliability, ops wanted low edge-inference cost.

### Action
- **Architecture evaluation:** Compared three options — vanilla CRNN, attention-based encoder-decoder, and SVRT (Swin-V2-Regression-Transformer). Ran controlled experiments over 2 weeks. SVRT pulled 4pp ahead.
- **Loss function:** Noticed instability in SVRT training. Traced to CTC loss + class imbalance in character distribution. Implemented FocalCTCLoss — combined focal loss's hard-example focus with CTC's alignment-free decoding. Stabilized training, recovered another 1.5pp.
- **Deployment path:** Made the harder call to do INT8 quantization via ONNX. Lost ~1.5% accuracy initially, but used custom Semantic Guidance Modules to recover most of it. Net positive: deployable on smaller AWS Lambda instances.
- **Communication:** Wrote weekly stakeholder updates with specific tradeoffs. CTO saw cost case, product saw reliability case, ops saw cost case. All three were aligned by week 5.

### Result
- 97% exact-match accuracy in production — exceeded internal target AND contract SLA
- AWS infrastructure cost 30% lower than FP16 deployment
- 99.9% uptime in first 4 production months
- Government contract retained; expansion conversations underway

### Learning
- I'd have run the quantization study in parallel with architecture selection instead of sequentially — would have saved 10 days
- FocalCTCLoss is now reusable across our OCR projects
- Communication cadence matters as much as engineering quality — weekly updates pre-empted stakeholder anxiety

### Link to role
This is the kind of work I'd want to scale — production AI where research quality meets real cost constraints. Your team's work on [X] sounds similar in spirit.

---

## Pre-filled Story #2: SpaceDrift sole-proprietorship founder

**Primary LP:** Ownership
**Secondary LPs:** Bias for Action, Frugality, Earn Trust, Customer Obsession
**Best for question:** "Tell me about taking ownership" / "Operating under constraint" / "What did you do right after college?"

### Situation
After graduating in May 2024, I deliberately chose NOT to take a standard entry-level role. Instead I founded SpaceDrift in Aug 2024 — an MSME-registered sole proprietorship. The bet was that 16 months of solo customer-facing delivery would build skills no junior dev role would: scoping, pricing, customer obsession, technical breadth, and operating under real financial constraint.

### Task
Build a sustainable freelance + engineering business as a solo operator, while staying technically sharp enough to land a strong full-time ML role afterward.

### Action
- **Service mix:** Took on freelance engineering project builds, professional data annotation services, and research support to PhD scholars (problem framing, dataset curation, experimental pipelines). Diversified intentionally because no single line gave me enough income alone.
- **Capacity management:** When workload exceeded my solo capacity, I engaged 3 trusted friend-contractors on a per-project basis. I paid them out of project margin. Taught me real lessons about delegation, scoping clarity, and that managing other people's quality is harder than doing the work yourself.
- **Skill-building loop:** Used the diversity of projects as a training ground — every client taught me a new domain (NLP, CV, RAG). Built ECHOME (agentic AI with 3-tier memory) and FinSentinelAI (privacy-first enterprise RAG) as flagship demonstrations during this period.
- **Financial discipline:** Operated on freelance income — no buffer, no investors. Every customer mattered. Pricing decisions had immediate consequences.

### Result
- Sustained 16 months of independent operation
- Delivered for paying clients across NLP, CV, and RAG domains
- Built two flagship public projects (ECHOME, FinSentinelAI) that became the credibility anchor for my next role
- Built customer-facing skills (scoping, pricing, communication) most fresh grads don't get for 3+ years
- Transitioned to ML Engineer role at Sujanix when the right deeper-engineering opportunity arose

### Learning
- The hardest part of solo founding isn't the work — it's saying no to bad-fit projects. I took on a few that ate weeks of margin. Lesson: scoping discipline matters more than throughput.
- Paying friends to help me work was a real lesson in delegation. Hard to give up control. Harder still to give honest feedback to friends. I learned to separate the working relationship from the friendship explicitly.
- Solo operation forced rigorous customer obsession in a way salaried roles don't — one upset client could swing my month.

### Link to role
This founder experience is why I'm confident about taking on ambiguous, high-ownership work at [Company]. Most candidates with 2 years of standard ML experience haven't had to scope a project from scratch with their own money on the line. That changes how you approach customer-facing engineering work.

---

## Stories to develop (20-30 hours work over Months 6-9)

Pick from the index above. For each:
1. Outline first (15 min)
2. Draft full STAR (30 min)
3. Practice out loud (30 min)
4. Refine after first practice (15 min)
5. Practice again later in week (15 min)

Total: ~2 hours per story × 20 stories = 40 hours. Spread over 3-4 months. Doable.

---

## Anthropic values prep (separate from STAR stories)

### Q1: What's the most important AI safety problem? (write here)

[Your answer — 2 paragraphs]

### Q2: What would you refuse to work on?

[Your answer — 1 paragraph]

### Q3: Why Anthropic specifically?

[Your answer — 1-2 paragraphs]

### Q4: What worries you about current AI development?

[Your answer — 2 paragraphs]

### Q5: How do you balance capability vs safety in your work?

[Your answer using a real project example — 2 paragraphs]

---

## "Why this company" answers (separate file: `05-why-this-company.md`)

---

## Quality-check protocol

Every story should pass these tests:

- [ ] Has at least 2 specific numbers
- [ ] Names specific decisions YOU made
- [ ] Acknowledges a tradeoff
- [ ] Has a genuine learning
- [ ] Could be told in 2-3 minutes without rushing
- [ ] Demonstrates 1-2 LPs clearly
- [ ] Uses "I" appropriately (not all "we")
- [ ] Avoids generic phrases ("worked hard", "best practices")
- [ ] Has a specific situation/timing

If a story fails 3+ of these: rework it.

---

## Drilling schedule

| Month | Focus |
|---|---|
| 6 | Develop 10 core stories |
| 7 | Develop 10 more (all 20 done) |
| 8 | Practice + Anthropic values prep |
| 9-10 | Practice + use in mocks |
| 11-12 | Use in real interviews |
| Ongoing | Add stories from real interview moments |

---

Next: [`05-why-this-company.md`](./05-why-this-company.md)
