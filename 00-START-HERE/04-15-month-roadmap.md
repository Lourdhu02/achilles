# The 15-Month Roadmap

**Start:** 19 May 2026
**Apply window opens:** ~March 2027 (Month 10)
**Decision deadline:** August 2027

This is the **single source of truth** for what you do each month. Print it. Tape it to your wall.

---

## At-a-glance phase map

```
Phase 1: FOUNDATIONS  ─── Month 1–4  (May–Aug 2026)
   DSA hardening + ML fundamentals + project audit + resume rewrite

Phase 2: DEPTH        ─── Month 5–8  (Sep–Dec 2026)
   GenAI/LLM mastery + ML system design + flagship project #1

Phase 3: POLISH       ─── Month 9–11 (Jan–Mar 2027)
   Mock interviews + behavioral + flagship project #2 + first applications

Phase 4: APPLY        ─── Month 12–15 (Apr–Aug 2027)
   Wave 1-4 applications + onsite rounds + negotiation + offer
```

---

## PHASE 1 — FOUNDATIONS (Month 1-4 = May → August 2026)

**Theme:** Plug your biggest gaps. DSA + classical ML + resume + portfolio audit.

**Time allocation:**
| Activity | % of weekly hours |
|----------|---|
| DSA | 40% |
| ML fundamentals (classical + DL) | 25% |
| Portfolio (resume, GitHub, LinkedIn) | 15% |
| GenAI keep-warm (papers, project maintenance) | 15% |
| Public output (blog, OSS) | 5% |

### Month 1 (May–June 2026)
- [ ] Read entire `00-START-HERE/` folder + `01-hiring-process/`
- [ ] Set up: LeetCode account, problem tracker, weekly hour log
- [ ] DSA: 80 problems (mix of Easy/Medium across arrays, strings, hashmaps, two-pointers, binary search). See `02-dsa/02-90-day-plan.md` Week 1-4.
- [ ] ML: Re-do Andrew Ng's "Specialization" — speed-run as refresher. Take written notes.
- [ ] Resume: Read `07-portfolio/01-resume-feedback-on-yours.md`. Rewrite to V2. Get feedback.
- [ ] GitHub: Clean up all repos. Star-able READMEs on ECHOME, FinSentinelAI, Transformers-OCR.
- [ ] LinkedIn: Update headline + about + featured projects.
- [ ] Public output: 1 blog post on Medium/Substack/personal site about ECHOME's memory architecture.

### Month 2 (June–July 2026)
- [ ] DSA: 80 problems (linked lists, stacks/queues, trees, graphs DFS/BFS). Time pressure starts: solve untimed first, then redo timed.
- [ ] ML: Deep dive on transformers — math of attention, multi-head, positional encoding. Read "The Annotated Transformer" + write your own from scratch in PyTorch.
- [ ] Portfolio: Begin **Flagship Project #1** (suggestion: an end-to-end RAG system with eval harness — adds "scale + eval" signal). See `07-portfolio/06-new-projects-to-build.md`.
- [ ] Public: 1 blog post on "RAG evaluation done right" or similar.
- [ ] Network: Join 3 communities (HF Discord, /r/LocalLLaMA, Latent Space Discord).

### Month 3 (July–Aug 2026)
- [ ] DSA: 60 problems (dynamic programming intro, recursion, backtracking). DP is THE wall. Push through.
- [ ] ML: Deep dive on training dynamics — optimizers (Adam vs AdamW vs Lion), learning rate schedules, mixed precision, gradient accumulation.
- [ ] Flagship #1 progress: 50% done. Public repo, README in progress.
- [ ] First mock interview attempt: do a self-conducted one — record yourself solving LC-medium with narration. Painful but eye-opening.
- [ ] Public: 1 blog post.

### Month 4 (Aug–Sept 2026)
- [ ] DSA: 60 problems (advanced DP, graph algorithms, tries, intervals). Begin pattern-based revision — solve only "fresh" problems within patterns.
- [ ] ML: Evaluation, calibration, A/B testing, drift detection — production realities. (You have job experience here; codify it for interviews.)
- [ ] Flagship #1: Complete. Write 1500-word blog post about it. Cross-post to dev.to, Hashnode, X/Twitter.
- [ ] Resume V3 — incorporate Flagship #1, new scale stories.
- [ ] First peer mock interview (Pramp or a friend). Don't aim to pass — aim to feel the format.
- [ ] **Phase 1 retrospective:** Honestly assess. Are you LC-medium-in-30-min yet? If not, extend DSA by 2 weeks before moving on.

**Phase 1 exit criteria (do not skip the audit):**
- ✅ 250-300 LC problems solved, comfortable with all major patterns
- ✅ Can solve LC-medium in 25 min with narration
- ✅ Transformer math + ML fundamentals are tight
- ✅ Resume V3 ready
- ✅ Flagship Project #1 shipped publicly
- ✅ 4+ blog posts

---

## PHASE 2 — DEPTH (Month 5-8 = Sept → December 2026)

**Theme:** Become a GenAI/LLM expert that's hard to ignore. Master system design.

**Time allocation:**
| Activity | % |
|---|---|
| GenAI/LLM mastery (papers, internals, fine-tuning) | 35% |
| ML system design | 25% |
| DSA maintenance | 15% |
| Flagship Project #2 | 15% |
| Public output + community | 10% |

### Month 5 (Sept–Oct 2026)
- [ ] LLM internals: tokenization (BPE/SentencePiece), attention variants (MHA, GQA, MQA, MLA), KV cache, positional encodings (RoPE, ALiBi). Implement in PyTorch from scratch. See `04-genai-llm-mastery/01-llm-internals.md`.
- [ ] System design framework: master the 6-step framework. See `05-ml-system-design/01-framework.md`. Do 4 design problems.
- [ ] DSA: 8 problems/week maintenance, mixed difficulty.
- [ ] Flagship #2 start: pick from `07-portfolio/06-new-projects-to-build.md` (suggestion: an LLM eval framework or an agent benchmark).
- [ ] OSS: First meaningful PR to vLLM / TRL / LangChain. (Will take a while; start small — docs or bug fixes count for first PR.)

### Month 6 (Oct–Nov 2026)
- [ ] LLM internals: training stack — pre-training math (next-token prediction loss), SFT, RLHF, DPO, PPO, KTO. Read papers (see `04-genai-llm-mastery/07-must-know-papers.md`).
- [ ] Fine-tuning: hands-on LoRA + QLoRA + PEFT. Fine-tune a 7B model on a custom dataset. Document costs/numbers.
- [ ] System design: 4 more problems including "design RAG at scale" and "design LLM inference platform".
- [ ] DSA: maintenance + first "company-tagged" practice (Google-tagged, Amazon-tagged).
- [ ] Mock interview: 2 this month (Pramp + interviewing.io if you can afford it).
- [ ] Public: blog post on your fine-tuning experiment with numbers.

### Month 7 (Nov–Dec 2026)
- [ ] LLM internals: inference optimization — vLLM PagedAttention, FlashAttention 2/3, continuous batching, speculative decoding, KV cache quantization, FP8/INT4.
- [ ] Distributed training: data parallel, tensor parallel, pipeline parallel, ZeRO-1/2/3, FSDP. Read papers + understand the why.
- [ ] System design: 4 more problems (recommender, feed ranking, fine-tuning platform).
- [ ] Mock interviews: 2 this month.
- [ ] Flagship #2 progress: 60% done.
- [ ] Behavioral prep starts: outline 10 STAR stories. See `06-behavioral/04-your-story-bank.md`.

### Month 8 (Dec 2026)
- [ ] LLM internals: evaluation deep dive — MMLU, HELM, BIG-Bench, LMSys Arena, custom evals. Build your own eval harness.
- [ ] Safety/alignment: Constitutional AI, RLHF problems, jailbreaks, red-teaming. (Critical for Anthropic interviews.)
- [ ] Flagship #2 ships. Blog + cross-post.
- [ ] Resume V4: incorporate flagship #2, OSS contributions, blog reach numbers.
- [ ] Mock: 3 this month.
- [ ] Outreach: identify 30 people at target companies for warm intros. See `08-application-strategy/03-referral-templates.md`.
- [ ] **Phase 2 retrospective.**

**Phase 2 exit criteria:**
- ✅ Can answer any question about transformer/LLM internals
- ✅ Can design any major ML system in 50 minutes
- ✅ Have published Flagship #1 + #2 with public traction (50+ GitHub stars combined minimum)
- ✅ 3+ OSS PRs merged
- ✅ 10+ blog posts published
- ✅ 5+ mock interviews completed, scoring "hire" on at least half
- ✅ 30+ warm contacts at target companies

---

## PHASE 3 — POLISH (Month 9-11 = Jan → March 2027)

**Theme:** Apply to Wave 1 companies. Iterate on real interview feedback. Finalize portfolio.

**Time allocation:**
| Activity | % |
|---|---|
| Mock interviews + post-mortem | 30% |
| Behavioral prep (stories, Q&A) | 20% |
| Active applications + outreach | 20% |
| Targeted weak-area review (driven by mock feedback) | 20% |
| DSA + GenAI maintenance | 10% |

### Month 9 (Jan 2027)
- [ ] Wave 1 applications submitted (Stripe, Atlassian, Adobe) — see `00-START-HERE/03-target-companies.md`.
- [ ] Mock interviews: 4-6 this month, prioritize ones similar to Wave 1 companies.
- [ ] Behavioral: write out all 16 Amazon LP stories + 10 Anthropic mission-fit responses. Practice out loud.
- [ ] Portfolio polish: every project README is interview-grade.

### Month 10 (Feb 2027)
- [ ] Continue Wave 1 interviews. Debrief every round (good or bad).
- [ ] Submit Wave 2 (Microsoft, Amazon, Nvidia, Databricks).
- [ ] Mock interviews: 4-6.
- [ ] Targeted prep: whatever your mock feedback says is weakest, hammer it.
- [ ] Possible: First offers from Wave 1. If yes — negotiate, stall, use as leverage for Wave 2/3.

### Month 11 (Mar 2027)
- [ ] Submit Wave 3 (Google, Apple, Meta).
- [ ] Continue all open processes. Manage 5-10 concurrent interview pipelines.
- [ ] Mock: 4 (focused on Google/Meta-style).
- [ ] Apply to Wave 4 (League B): Cohere, Hugging Face, Anthropic, OpenAI, xAI, Mistral, Scale.

**Phase 3 exit criteria:**
- ✅ 40+ applications submitted across all waves
- ✅ 10+ phone screens completed
- ✅ 3+ onsites completed
- ✅ 1+ offer received (or in final stages)

---

## PHASE 4 — APPLY (Month 12-15 = Apr → August 2027)

**Theme:** Close. Negotiate. Decide.

### Month 12 (Apr 2027)
- [ ] All onsites in progress. Focus 100% on each one.
- [ ] Continue mocks for any company in the pipeline.
- [ ] Salary research: Levels.fyi, Blind, glassdoor — know the numbers cold.

### Month 13 (May 2027)
- [ ] Manage offers as they come in. Don't accept the first one.
- [ ] Re-interview at companies that rejected you in Wave 1-2 (some hiring cycles re-open).
- [ ] Negotiation prep: read `01-hiring-process/07-comp-negotiation.md`.

### Month 14-15 (June-Aug 2027)
- [ ] Final round + negotiation.
- [ ] Sign offer.
- [ ] **Notice period at Sujanix** — start serving once you have written offer letter.
- [ ] Update memory. Write up your "how I did it" essay (helps the next person; also useful for future interviews).

---

## What if you fall behind?

You will. Plan for it.

**If at end of Month 4 you haven't hit Phase 1 exit criteria:**
- Add 4 more weeks. Push everything back by 1 month. Apply window becomes April 2027.
- This is fine. Better than under-prepped applications.

**If at end of Month 8 you haven't hit Phase 2 exit criteria:**
- Cut Wave 4 (League B). Focus on Wave 1-3 with full preparation.

**If by Month 12 you have no offers in pipeline:**
- This is a signal to re-evaluate. Are you under-applying? Failing rounds? Resume issues? Talk to me, we debug.

**If you get an offer in Wave 1 and panic about lower-tier company:**
- Use it as leverage. Tell Wave 2-3 companies "I have an offer from X, can you expedite your process?" This dramatically accelerates pipelines.

---

## How to read this roadmap

- **Daily:** Check today's tasks against the current month's checklist.
- **Weekly:** Sunday evening, review last week, plan next. Use `09-the-grind/01-weekly-rituals.md`.
- **Monthly:** End of month, update `09-the-grind/03-month-checklist.md` with retrospective. Adjust next month.
- **Quarterly:** Read this entire roadmap again. Update the targets file. Update the SWOT.

---

Next file: [`05-how-to-use-this.md`](./05-how-to-use-this.md)
