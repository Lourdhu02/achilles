# Amazon, Microsoft, Nvidia — Hiring Deep Dive

Three more League A targets with distinct interview cultures.

---

## AMAZON

### Why Amazon is different: Leadership Principles run the show

Amazon has 16 Leadership Principles (LPs). Every interviewer is trained to ask 2-3 behavioral questions, each mapped to a specific LP. **You will be evaluated on whether your stories demonstrate the LPs.**

If you can't articulate stories that map to LPs, you fail — no matter how good your code is. This is the #1 reason candidates with strong technical skills fail Amazon interviews.

### The 16 Leadership Principles (memorize, internalize):

1. Customer Obsession
2. Ownership
3. Invent and Simplify
4. Are Right, A Lot
5. Learn and Be Curious
6. Hire and Develop the Best
7. Insist on the Highest Standards
8. Think Big
9. Bias for Action
10. Frugality
11. Earn Trust
12. Dive Deep
13. Have Backbone; Disagree and Commit
14. Deliver Results
15. Strive to be Earth's Best Employer
16. Success and Scale Bring Broad Responsibility

See `06-behavioral/02-amazon-LPs.md` for the full story-bank template (you need 2 stories per LP = 32 stories).

### Process for SDE-2 ML / Applied Scientist II

1. **Recruiter screen** — standard, but they'll ask "what LPs resonate with you?"
2. **Online coding assessment (OA)** — 2 LC-medium problems, 90 min, automated grading. Plus a "work simulation" with behavioral multiple-choice.
3. **Phone screen** — 1 hour: 1 coding problem + 1-2 LP-based behavioral questions
4. **Onsite (5 rounds, called "the Loop"):**
   - Coding round 1 (45 min): 1 hard problem + 2-3 LP questions
   - Coding round 2 (45 min): same format
   - System design (60 min): for SDE-2 ML, expect ML system design + 1-2 LP questions
   - Behavioral / hiring manager (45 min): deep LP exploration
   - **Bar Raiser** (45 min): senior engineer from another team, trained to hold the bar; LP questions go deep
5. **Debrief & decision**

### The Bar Raiser
- A senior Amazonian (5+ years, trained for this).
- Has **veto power**.
- Asked to evaluate: "Would this candidate raise the bar of the team they're joining?"
- They will push back, challenge your stories, make you defend numbers.
- **Strategy:** treat Bar Raiser as the most important round. Bring your strongest stories. Be ready to be challenged.

### What ML rounds at Amazon look like

For SDE-2 ML:
- Mostly coding with some ML system design
- Less ML theory than Google
- More "production engineering": deployment, monitoring, A/B testing

For Applied Scientist II:
- More ML depth: explain your projects, design ML systems, sometimes light derivations
- ML system design heavy: design Alexa's NLU, design Rufus shopping assistant, design Bedrock inference
- Slightly less LeetCode pressure (still present)

### Your specific angle for Amazon
- **Bedrock + GenAI:** Amazon Bedrock is their LLM orchestration product. Your work maps directly.
- **Rufus (shopping LLM):** retrieval + ranking + agents. Your RAG work fits.
- **Alexa+ (LLM rewrite):** speech + agents. ECHOME's voice cloning (XTTSv2) maps.
- **LP stories from SpaceDrift founder experience:**
  - Customer Obsession: 12 international clients, 100% satisfaction
  - Ownership: founded and scaled a consultancy
  - Bias for Action: shipping 40+ pipelines fast
  - Earn Trust: stakeholder management
  - Deliver Results: measurable outcomes (97% OCR accuracy, 30% cost reduction)

### Realistic odds for you
**30-40%.** Amazon hires lots, the bar is more "consistent execution" than "extraordinary brilliance". Your founder story + production ML work is a strong fit.

---

## MICROSOFT

### Process overview

For SDE-2 with ML focus / ML Engineer roles:

1. **Recruiter screen** — standard
2. **Coding screen (60 min)** — 1-2 LC-medium problems on a real IDE (you can run code!)
3. **Onsite (4-5 rounds):**
   - Coding round 1: LC-medium
   - Coding round 2: LC-medium-hard or ML implementation
   - ML / Architecture round: project deep dive + ML system design
   - "AsAs" (As-if-an-actual-scenario) / behavioral round
   - (Sometimes) Skip-level manager or HR round
4. **Hiring committee decision**

### What's distinctive about Microsoft

#### a) Real IDE, can run code
- Unlike Google's whiteboard / Meta's CoderPad-without-execution, MS lets you run your code.
- **Lower stress.** Debug as you go.
- **But:** quality bar is higher because you have no excuse for syntax errors.

#### b) "Growth Mindset" culture (post-Nadella)
- They explicitly look for candidates who:
  - Embrace learning from failure
  - Are curious
  - Collaborate across teams
- "Fixed mindset" red flags: defensiveness, blame-shifting, "I already know that".

#### c) Team-based hiring
- Less centralized than Google. Hiring manager has more say.
- This means: team-specific prep matters. Research the team before onsite.

### What ML rounds at Microsoft look like

- Ranges from "explain ML concepts" (basic) to "design Copilot architecture" (deep) depending on team.
- Azure AI / OpenAI integration teams: heavy on LLM ops, RAG, fine-tuning workflows.
- Bing / Search team: search + ranking + GenAI integration.
- Microsoft Research India: more research-oriented.

### Your specific angle for Microsoft
- **Copilot ecosystem:** Microsoft has 100+ Copilot products. They need engineers who can build LLM integrations.
- **Azure OpenAI Service:** Microsoft hosts OpenAI APIs. Engineering opportunity is huge.
- **Hyderabad / Bangalore offices:** Largest non-US Microsoft engineering presence. Many open ML roles.

### Realistic odds for you
**40-50%.** Microsoft hires huge volumes in India for ML. Bar is solid but not Google-tier brutal. Probably your strongest realistic shot.

---

## NVIDIA

### Why Nvidia is unique

Nvidia interviews emphasize **systems** more than any other top company. They want engineers who understand:

- GPU memory hierarchy (HBM bandwidth, L2 cache, register file)
- Compute-bound vs memory-bound operations
- FP16/BF16/FP8/INT8 quantization
- Tensor cores, sparsity
- CUDA basics (not necessarily writing kernels, but understanding cost models)

Less LeetCode-grindy than Google/Meta. More "do you understand WHY FlashAttention exists?"

### Process overview

For Deep Learning Software Engineer / LLM Performance Engineer:

1. **Recruiter screen**
2. **Phone screen (60-75 min):** coding + light systems
3. **Onsite (4-6 rounds):**
   - 1-2 coding rounds (LC-medium, sometimes C++ if low-level role)
   - 1-2 ML rounds (deep into transformers, training dynamics)
   - 1 systems round: explain how attention works at the hardware level, optimize an inference path
   - 1 behavioral
4. **Decision**

### What's tested deeply at Nvidia

- **Attention math:** Q, K, V, scaling factor, softmax, dropout. Why scaling by sqrt(d_k)? What is FlashAttention doing differently?
- **Memory layout:** SRAM vs HBM. Why does FlashAttention tile? Why does PagedAttention exist?
- **Quantization:** When does INT8 work? What is post-training quantization vs QAT? What's the calibration step?
- **Inference optimizations:** KV cache, speculative decoding, continuous batching.
- **Distributed:** Tensor parallel, pipeline parallel, expert parallel (for MoE).

### Your specific angle for Nvidia
- **Your TensorRT + ONNX INT8 quantization work** at Sujanix is unusually strong for a 2-YOE candidate. Lead with it.
- **Edge deployment** = constrained hardware experience.
- **You may need to ramp up:** read FlashAttention paper, PagedAttention paper, ZeRO paper. Do hands-on with Triton Inference Server. See `04-genai-llm-mastery/05-inference-optimization.md`.

### Realistic odds for you
**30-40%.** Less crowded applicant pool because the systems bar filters out many candidates. Your quantization experience is a real edge.

---

## Comparison table (quick reference)

| Company | DSA Bar | ML Depth | System Design | Behavioral | Distinct Quirk |
|---|---|---|---|---|---|
| Amazon | Medium-High | Medium | Medium-High | **Very High (LPs)** | Bar Raiser veto |
| Microsoft | Medium | Medium-High | Medium-High | High (growth mindset) | Real IDE allowed |
| Nvidia | Medium | **Very High** | High (systems-flavored) | Medium | Hardware/systems depth |
| Google | **Very High** | High | High | High (Googleyness) | Hiring committee |
| Meta | **Very High** | High | **Very High (ranking-flavored)** | High (rooting) | Move-fast lens |

---

## Application strategy: Apply in this order

For your timeline (apply Wave 2-3, Month 12-13):

1. **Amazon (apply first)** — most openings, LP-heavy practice transfers to other behavioral interviews
2. **Microsoft** — friendly process, real IDE, lower stress for your second-real-loop
3. **Nvidia** — your TensorRT story is a differentiator; do it after some onsite practice

---

Next: [`05-the-referral-game.md`](./05-the-referral-game.md)
