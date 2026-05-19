# Your Profile — SWOT Analysis

**Audit date:** 19 May 2026
**Based on:** RESUME-TOP-1.pdf + your stated context

Re-audit this file every 3 months. Profile shifts. Plan adapts.

---

## STRENGTHS (lean into these — these are your moat)

### S1. GenAI production credibility (rare for 2 YOE)
- **ECHOME** — LangGraph + three-tier memory (Episodic/Semantic/Procedural) + XTTSv2 voice cloning + CAT/IRT psychometrics. The memory architecture is genuinely novel. Fisher Information for adaptive testing is a credible technical depth signal.
- **FinSentinelAI** — Production RAG with VLMs for PDF/bank statement extraction, JWT session isolation, multi-tenant. This is what every fintech and enterprise wants RIGHT NOW.
- **SVRT/Transformers-OCR @ Sujanix** — 97% exact-match, FocalCTCLoss, ONNX INT8 quantization, edge deployment. Production-flavored. Real customer impact (government utility automation).

> **Why this is a moat:** Most 2-YOE applicants have toy projects. You have systems with real users and real engineering tradeoffs (memory, latency, security). Lead with this.

### S2. Solo-founder story (SpaceDrift)
- MSME-registered sole proprietorship (Aug 2024 – Dec 2025). 16 months running your own business straight out of college. Service mix: engineering project builds + professional data annotation + research support to PhD scholars. Paid 3 friend-contractors per-project when workload exceeded solo capacity.
- The narrative is strong without inflation: real ownership, customer-facing delivery, operating under freelance financial constraint, judgment about when to bring in paid help.
- This differentiates you from a "good IC at TCS" candidate. Anthropic, Stripe, and startup-y arms of FAANG (Google Labs, Meta AI Eng) value the solo-operator story heavily — it signals you can do scope, delivery, and customer communication without supervision.

### S3. Full-stack ML chops
- Training: PyTorch, TensorFlow, LoRA, FocalCTCLoss
- Serving: FastAPI, Docker, ONNX, TensorRT, AWS Lambda
- Storage/retrieval: ChromaDB, Qdrant
- Monitoring: MLflow, W&B, DVC
- This breadth is hard to fake. Drives credibility on system design rounds.

### S4. Kaggle Expert tier
- Signals you do ML, not just describe it. Many candidates claim Kaggle exposure and have a Novice account. Yours is verifiable.

### S5. Customer/business communication
- Customer-facing delivery + technical documentation track record from solo SpaceDrift work. Most engineers can't scope, price, and deliver to paying clients without supervision. Behavioral interviews will favor you if you tell these stories well.

---

## WEAKNESSES (fix these — they are bottlenecks, not death sentences)

### W1. DSA is your biggest blocker [PRIORITY 1]
- **Current state (your self-report):** Intermediate, 50-200 LC, can solve mediums.
- **Required state for FAANG ML Eng phone screen:** Solve 2 LC-mediums in 45 min, clean code, optimal solution, while narrating thought process.
- **Gap:** Substantial. You probably need ~300-500 more focused problems, with timed practice and pattern mastery.
- **Plan:** See `02-dsa/02-90-day-plan.md`. Allocate **40% of your time months 1-3, 25% months 4-6, 15% maintenance after.**
- **Honest:** This is THE round where 2-YOE ML candidates with strong projects most often fail. Do not underestimate.

### W2. No "scale" signal on resume
- Your projects don't communicate: how many users, requests/sec, GPUs, model size, training data size.
- Interviewers will ask. You need numbers.
- **Action:** Audit each project. Retro-extract any scale data you can (even from Sujanix internal stuff if NDA permits). Reframe bullets with numbers. Where you genuinely don't have scale, plan a flagship "scale project" in `07-portfolio/06-new-projects-to-build.md`.

### W3. No publications / no famous OSS contributions
- Closes Research roles. Hurts at applied Anthropic/OpenAI (less, but still).
- **Realistic 15-month options:**
  - (a) Submit 1 workshop paper (NeurIPS workshop, ICLR workshop). Doable if you start NOW. Topic suggestion: novel agent-memory architecture from ECHOME.
  - (b) Make 5-10 meaningful PRs to a top OSS project (vLLM, TRL, LangChain, LlamaIndex, transformers). Less ambitious, more achievable.
  - (c) Skip publications, double down on flagship projects + blog content.
- **Recommendation:** (b) + (c). Skip (a) unless you genuinely enjoy research writing — it's a long road for small interview ROI vs. League A.

### W4. No ML system design experience
- Interview round you've never practiced. Designed for senior+ but asked even at L4/L5 for ML.
- Topics: design YouTube recs, design RAG at scale, design feed ranking, design fine-tuning platform.
- **Plan:** `05-ml-system-design/`. Start Month 5. 1 design problem/week = 25-30 designs by application time.

### W5. Tier-3 undergrad college
- Cannot change. Can route around.
- **Strategy:** Lateral hiring + referrals + strong public work. Skip campus drives entirely (they're closed to you anyway).
- LinkedIn + GitHub + blog will outweigh the college name within 3-4 YOE for almost all companies.

### W6. Resume issues (specific)
- Detailed in `07-portfolio/01-resume-feedback-on-yours.md`. Top issues at a glance:
  - "2+ years of experience" but timeline shows Sept 2024–Present + 2-month internship = closer to ~1.7 years
  - Sujanix dated "Jan 2026 – Present" but resume is dated for Aug 2027 application — make sure your "Present" tense is honest
  - "Delivered 40+ production ML pipelines" from a solo founder consultancy reads as inflated to senior reviewers — quantify per-client work instead
  - Missing GPU/scale numbers (W2 above)
  - "SVRT (Swin-V2-Regression-Transformer)" — not a widely known acronym; senior interviewers might Google it during the interview. Make sure your description holds up.

### W7. Currently in Sujanix (private limited, not a known name internationally)
- Recruiters at top US companies pattern-match on company names.
- Mitigation: GitHub stars + blog reach + Kaggle + LinkedIn engagement become your "name brand" substitute.

### W8. Math depth uncertain
- Resume mentions Linear Algebra + Statistics coursework only. For LLM internals (attention math, optimizer dynamics, calibration), you may need depth.
- Diagnostic in `03-ml-fundamentals/05-math-essentials.md` — take it.

---

## OPPORTUNITIES (market tailwinds working FOR you)

### O1. GenAI hiring boom (2026-2027)
- Every Fortune 500 is hiring "LLM Engineer" / "GenAI Engineer" / "Agent Engineer" — roles that didn't exist 18 months ago.
- Hiring bars at these roles are slightly lower than traditional ML Eng because the talent pool is small. Your edge.

### O2. Anthropic / OpenAI applied roles open up periodically
- They post "ML Engineer, Applied" roles publicly. No publication requirement.
- Smaller numerical bar than Research, but compensation is FAANG-tier+.
- Watch their careers pages weekly.

### O3. India hiring at multinationals expanded post-2024
- Google, Microsoft, Nvidia, Meta have ramped India R&D heavily.
- Specifically for ML/AI: Microsoft Research India, Google DeepMind Bangalore, Adobe AI Lab India, Nvidia Bangalore.
- Many roles that used to be US-only are now India-available.

### O4. Conferences / open-source events
- NeurIPS, ICML, ICLR all have workshop tracks accessible to non-PhDs.
- KaggleX, GitHub Universe, PyTorch Conference, HuggingFace events, AI Engineer World's Fair — these are recruiter-rich.

### O5. AI builder communities are hyper-active
- buildspace, AI Tinkerers, Latent Space Discord, /r/LocalLLaMA, HF community — you can build reputation faster than 5 YOE in a corporate cube.

---

## THREATS (watch these, mitigate where possible)

### T1. AI hiring cools off mid-2026
- If a recession or AI investment slowdown hits in late 2026, FAANG hiring tightens. Window shrinks.
- Mitigation: apply broadly when ready, don't perfectionism-delay.

### T2. You burn out
- 25-30 hrs/week of prep + full-time job for 65 weeks is a marathon. Many quit at Month 4-5.
- Mitigation: scheduled rest days (1/week), monthly off-day, social/family time. See `09-the-grind/02-burnout-prevention.md`.

### T3. Sujanix gets acquired or you lose your job
- Lose the income, lose the credible "currently shipping" signal.
- Mitigation: 6-month emergency fund. Don't quit voluntarily.

### T4. AI plateau / regulatory disruption
- LLMs hit a research plateau, GenAI hiring contracts toward classical ML.
- Mitigation: don't abandon classical ML fundamentals. `03-ml-fundamentals/` stays in the plan.

### T5. Family/health/personal events
- 15 months is long. Life will happen.
- Mitigation: build slack into the schedule. Don't run at 100% all the time.

---

## The plan, in one sentence

**Spend 6 months hardening DSA + classical ML to FAANG bar, 6 months deepening GenAI/system design to League B bar, 3 months applying aggressively with a polished portfolio — all while staying employed and shipping public work monthly.**

Detailed breakdown in `00-START-HERE/04-15-month-roadmap.md`.

---

Next file: [`03-target-companies.md`](./03-target-companies.md)
