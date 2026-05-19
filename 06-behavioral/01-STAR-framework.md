# The STAR Framework

The structure for ANY behavioral story. Internalize.

---

## STAR breakdown

- **S**ituation — Context (who, what, when, where) — 20-30 seconds
- **T**ask — Your specific responsibility / goal — 15-20 seconds
- **A**ction — What YOU did (steps, decisions, tradeoffs) — 60-90 seconds
- **R**esult — Outcome with numbers + reflection — 30-45 seconds

**Total: 2-3 minutes per story.** Tight enough for engagement, long enough for substance.

---

## STAR+ (the upgrade)

Many interviewers prefer STAR with two additions:

- **L**earning — What did you learn? Would you do differently?
- **L**ink — How does this connect to the role you're interviewing for?

Use STARLL when time permits and the question warrants reflection.

---

## Worked example — using YOUR resume

Question: "Tell me about a time you made a technically difficult decision."

### Bad answer
"At Sujanix, I worked on OCR. We had to choose a model architecture and I picked one that worked well."

### Good answer (STAR)

**Situation (25s):** "At Sujanix earlier this year, we were tasked with building a transformer-based OCR pipeline for high-stakes government utility automation. The challenge: meter images came from real-world deployments with motion blur, low contrast, and inconsistent lighting. The vendor we replaced was averaging 89% exact-match accuracy, and contracts required 95%+."

**Task (15s):** "As the lead ML engineer, I owned model architecture selection plus the production deployment path. The CTO wanted a fast solution; product wanted reliability; ops wanted low edge-inference cost."

**Action (75s):** "I evaluated three architectures: vanilla CRNN, an attention-based encoder-decoder, and SVRT — a Swin-V2 Regression Transformer variant. SVRT was newer and less battle-tested, but its Feature Rearrangement Module had strong results on similar messy-input benchmarks.

I ran controlled experiments: same training data, same metrics, three architectures over 2 weeks. SVRT pulled ahead by 4 percentage points but the loss surface was less stable. I traced this to CTC loss struggling with class imbalance in our character distribution. So I implemented FocalCTCLoss — combining focal loss's hard-example focus with CTC's alignment-free decoding. This gave SVRT a clear, stable lead.

I then made the harder call: instead of shipping the FP16 model immediately, I ran a quantization study — INT8 via ONNX. We lost about 1.5% accuracy, but with custom Semantic Guidance Modules we recovered most of it. The result was deployable on smaller AWS Lambda instances."

**Result (30s):** "We hit 97% exact-match accuracy in production — exceeding both our internal target and the contract SLA. AWS infra cost was 30% lower than the FP16 approach would have required, with 99.9% uptime over the first 4 months."

**Learning (15s):** "The lesson: I'd have run the quantization study in parallel with architecture selection instead of sequentially — would have saved 10 days. Also, FocalCTCLoss is now reusable across our other OCR projects."

**Link (15s):** "I bring this up because [Company]'s [team] is solving a similar production-AI challenge where research-grade accuracy meets real-world cost constraints. That intersection is where I do my best work."

Total: ~3 minutes. Specific, numerical, reflective, connected.

---

## When to use which "S" details

- **Senior interviewer:** more business context, ownership, tradeoffs
- **Engineering-focused:** more technical decisions, architecture, code
- **Hiring manager:** results, impact, team dynamics
- **Bar Raiser (Amazon):** more "would-you-do-it-again" reflection

---

## Customizing per company

The SAME story can be framed for different LPs / values:

The OCR story above can be framed for:
- Amazon LPs: Customer Obsession (government contract SLA), Insist on Highest Standards (97% bar), Bias for Action (running parallel experiments), Dive Deep (FocalCTCLoss), Frugality (30% cost reduction), Earn Trust (stakeholders trust delivery)
- Google "Googleyness": dealing with ambiguity (3 stakeholders, different priorities), bias for action (started experiments without full team alignment), humility (acknowledged FP16 would've been "good enough")
- Anthropic values: pushing for higher quality even when "good enough" was acceptable; deploying carefully

One strong story → 5-10 different framings.

---

## Sharpening your "A" section

The Action section is where you DIFFERENTIATE. Most candidates rush this. Don't.

### What to include
- 2-3 specific decisions you made
- WHY each decision (tradeoffs)
- Specific techniques / tools used
- Roadblocks + how you overcame
- Your contribution explicitly (vs team)

### What to AVOID
- "We" without "I"
- Vague phrases ("worked hard", "did my best")
- Generic descriptions ("standard approach")
- Long timeline narration without insights

---

## Question categories + ideal story choices

| Question | Story type to draw from |
|---|---|
| "Tell me about a complex project" | Your proudest technical project |
| "Describe a project that failed" | A genuine failure with learning |
| "Tell me about a disagreement" | Productive disagreement with peer/manager |
| "How do you handle ambiguity?" | A project with unclear requirements |
| "Tell me about ownership" | A project where you stepped beyond scope |
| "Describe leadership" | A time you guided others (even without title) |
| "How do you give/receive feedback?" | Specific instance with outcome |
| "Tell me about a deadline you missed" | Honest answer with learning |
| "Customer obsession" (Amazon) | A specific customer interaction |
| "Earn trust" (Amazon) | A time you built confidence with a skeptical audience |
| "Why this company?" | Your researched, personal answer |

Maintain a mapping (in `04-your-story-bank.md`) of questions → stories.

---

## Behavioral interview rubric (what they're grading)

From inside FAANG-tier interview training:

| Dimension | Strong Hire | Hire | Lean Hire | Lean No Hire | No Hire |
|---|---|---|---|---|---|
| Specificity | Numbers, names, decisions, dates | Specific but light on numbers | Some details | Vague | Generic |
| Ownership | Clear, owned outcome | Owned major parts | Contributed | Vague role | "We" with no "I" |
| Impact | Quantified, large | Quantified, modest | Mentioned | Unclear | None |
| Self-awareness | Genuine reflection + learning | Reflection | Acknowledged some weakness | None | Defensive |
| Communication | Crisp, structured, pacing | Mostly clear | Sometimes meandering | Rambling | Confusing |

Aim for "Hire" on every dimension, "Strong Hire" on 2-3.

---

## How to practice with me (Claude)

### Solo practice
"Give me 5 random behavioral questions. After my answers, score me on the 5 dimensions above. Tell me which dimension to improve."

### Mock interview
"Mock-interview me for 30 min behavioral. You're a Google L4 hiring manager. After 5 questions, give detailed feedback."

### Story refinement
"I'm developing a story for 'tell me about a time you took ownership.' Here's draft 1: [paste]. Critique and suggest tightening."

---

Next: [`02-amazon-LPs.md`](./02-amazon-LPs.md)
