# How FAANG (and friends) actually hire ML engineers

The full pipeline, end to end, with what each stage actually does, who decides, and how you optimize for it.

---

## The 8-stage pipeline (almost universal)

```
1. Sourcing            ← Recruiter finds you OR you apply OR referral
2. Resume screen       ← ATS keyword filter + 30-second human glance
3. Recruiter screen    ← 30-min phone call: basic fit, comp expectations, timeline
4. Phone screen        ← 45-60 min technical: coding OR ML basics OR both
5. Onsite (4-6 rounds) ← The real assessment. Coding, ML, system design, behavioral
6. Debrief / Packet    ← Interviewers write feedback, recruiter assembles a packet
7. Hiring committee    ← Senior engineers/managers who didn't interview you read packet, decide
8. Offer + Negotiation ← Recruiter sends offer; you negotiate
```

### Companies that follow this almost exactly:
Google, Meta, Microsoft, Amazon (with LP overlay), Apple, Nvidia, Adobe, Stripe, Databricks, Atlassian.

### Companies that deviate:
- **Anthropic / OpenAI:** Add 1-2 "values"/mission rounds. Sometimes a take-home.
- **Hugging Face:** May skip phone screen if your OSS profile is strong; weight on async work samples.
- **Smaller AI labs:** More likely to do a paid trial week or a research presentation.

---

## STAGE 1: Sourcing — How you get on the radar

Four paths, ranked by conversion rate:

| Path | Conversion to phone screen | Effort |
|------|---|---|
| Internal referral | 30-50% | Find the right person, ask well |
| Recruiter inbound (LinkedIn/email) | 60-80% | Build the inbound profile (LinkedIn, GitHub, blog) |
| Direct apply via careers page | 2-5% | Quick but low yield |
| Public visibility (viral OSS, blog post, conference talk) | 80%+ if hit | High effort, high luck |

**Implication for you:**
- Spend more time on referrals + inbound profile than on direct apply.
- Direct apply is what most people do, which is why most people fail.

---

## STAGE 2: Resume screen — Two layers

### Layer 2a: ATS (Applicant Tracking System)
- A keyword filter. If your resume doesn't contain the JD's keywords, you get auto-rejected before any human sees it.
- **Hack:** For each role you apply to, take the JD. Find 8-12 keywords/phrases that map to your experience. Make sure they appear (truthfully) in your resume.
- Common ATSs: Workday, Greenhouse, Lever, Taleo. Format your resume as a clean PDF with no fancy tables, no images, no two-column layouts. ATSs struggle to parse those.

### Layer 2b: Human screen (30 seconds)
- A recruiter or recruiting coordinator skims your resume for ~30 seconds.
- They are looking for: current/recent company name (brand recognition), title (ML Engineer ≥ Data Scientist ≥ SDE for ML roles), education tier (less important post-2YOE), keywords from the JD, "wow" projects.
- **Your weak spots:** Sujanix is not a known brand internationally. Mitigate with strong project bullet points, public GitHub link, blog domain.
- **Your strong spots:** ML Engineer title, GenAI keywords (RAG, LangGraph, fine-tuning, LoRA), Kaggle Expert, founder story.

### What changes between Lateral hiring vs Campus hiring
- **Lateral (your path):** Recent project work matters most. College is a side note.
- **Campus:** College tier, GPA, internships matter heavily. You're past this.

---

## STAGE 3: Recruiter screen (30 min phone call)

What they do:
- Confirm you actually exist and can speak coherently
- Confirm your interest in the role/company
- Confirm comp expectations are in range (this kills more candidates than you'd think)
- Confirm timeline (start date, notice period)
- Confirm visa/work auth status

What they're judging:
- Communication clarity
- Enthusiasm (do you know what this company does? Did you research the team?)
- Red flags (rambling, can't articulate why this role, asking only about comp upfront)

What to prep:
- 60-second "tell me about yourself"
- "Why this company, why this role" (3-4 specific reasons each, tied to YOUR work)
- Comp expectation: a range, with anchor at high end of market for your level. Say "based on my research, I'd be looking at $X-Y total comp." See `07-comp-negotiation.md`.
- 2-3 questions you want to ask them

**The biggest mistake here:** Saying you "need" a specific salary or being vague about expectations. Recruiter wants a range that's within budget. Be researched.

---

## STAGE 4: Phone screen (45-60 min technical)

Most common formats for ML Eng:

### Format A: Pure coding (Google, Meta, Amazon, Microsoft typically)
- 1-2 LeetCode-style problems
- 45 minutes
- On a shared editor (CoderPad, CodeSignal, Google docs)
- Interviewer wants: clean code, correct, optimal, explained clearly, edge cases handled

**Your prep:** `02-dsa/02-90-day-plan.md` + 10+ mock interviews.

### Format B: ML basics + light coding
- 20 min "tell me about a project from your resume"
- 20 min "implement K-means / write a softmax from scratch / implement attention forward pass"
- 5-10 min Q&A

**Your prep:** `03-ml-fundamentals/06-interview-qa-bank.md` + practice "implement X from scratch" exercises.

### Format C: Take-home + discussion (Anthropic, OpenAI, smaller labs)
- 4-8 hour take-home assignment (could be: train a small model, build a RAG system, evaluate an LLM, debug a notebook)
- 60 min follow-up call to discuss
- You will be evaluated on code quality, decisions, communication

**Your prep:** Treat the take-home as a portfolio piece. Spend 12-16 hours, not 4-8. Write a clean README. Test edge cases. Document tradeoffs.

---

## STAGE 5: Onsite (the real test)

Typically 4-6 rounds, often over 1 day (or 2 half-days for remote).

### Round types (a typical ML Eng onsite has 4-5 of these):

#### a) Coding round (1-2 rounds)
- Same as phone screen but harder. LC-medium to LC-hard.
- Sometimes "real-world coding": "implement a rate limiter", "build a small priority queue with these constraints".
- 45 min each.

#### b) ML / ML breadth round
- "Walk me through a project from your resume." (10 min)
- Then deep technical follow-ups: "Why did you choose X model? What was your eval set? How did you tune hyperparameters? What would you do differently?"
- Sometimes whiteboard math: derive backprop for a simple network, derive cross-entropy loss, explain attention scaling factor.

#### c) ML system design round
- "Design YouTube's recommendation system." / "Design a RAG system that serves 10M users." / "Design fine-tuning infrastructure for Llama-70B."
- 50-60 min. You drive. Interviewer probes.
- See `05-ml-system-design/01-framework.md` for the 6-step framework.

#### d) Behavioral / values round
- "Tell me about a time you disagreed with a teammate." / "Tell me about your biggest failure." / "Why this company?"
- STAR format expected.
- Some companies (Amazon, Anthropic) make this the most important round. Don't dismiss it.

#### e) Hiring manager round
- Combination of behavioral + light technical + "selling you on the role" (if you're strong).
- They are deciding whether they want you on their team specifically.
- ASK QUESTIONS HERE. Show you've researched their team's work.

#### f) (Optional) Research / paper discussion (more common at Anthropic, OpenAI, Research labs)
- "Tell me about a recent paper you read."
- "What do you think of [recent technique X]?"
- They're testing: do you keep up with the field? Can you critically evaluate research?

#### g) (Optional) Bar raiser (Amazon specifically)
- A senior engineer from a different team, trained to hold the "hiring bar" steady across the company.
- They have veto power.
- They will dig deeper than others. Expect harder follow-ups.

---

## STAGE 6: Debrief & packet

After your onsite, every interviewer writes detailed feedback. Typically:

- A score per round (Strong Hire / Hire / Lean Hire / Lean No Hire / No Hire / Strong No Hire)
- Justification with specific evidence ("Candidate solved coding problem in 25 min with optimal solution but missed edge case Y")
- Recommended level (L3 / L4 / L5)

The recruiter assembles a "packet" of these notes + your resume + any work samples.

**What this means for you:**
- Interviewers WRITE about you AFTER the interview. Make their job easy.
- Give them quotable moments: clear explanations, named techniques (use "I applied LoRA" not just "I did fine-tuning"), specific numbers.
- Don't give them ambiguity. If they're unsure between "Hire" and "Lean Hire", they often default to "Lean Hire" — which often loses the packet.

---

## STAGE 7: Hiring committee

**This is the secret stage most candidates don't know about.**

A committee of 4-8 senior engineers/managers, none of whom interviewed you, reads your packet and votes.

- They have ~30 minutes to discuss your case.
- They've never met you. They see only the packet.
- They are looking for: a clear "Hire" signal across multiple rounds, no major red flags.

**Implications:**
- The packet IS you in this stage. Your interviewers' write-ups are your only advocate.
- A "Lean Hire" + "Lean Hire" + "Lean Hire" + "Lean No Hire" usually loses. A "Strong Hire" + "Hire" + "Lean Hire" + "Lean No Hire" usually wins.
- Even ONE strongly negative round is hard to recover from. Aim to be at least "Hire" in every round; "Strong Hire" in 1-2.

**What companies have a formal hiring committee:**
- Google (most rigorous committee process; can override any interviewer)
- Meta (slightly less formal, but committee reviews)
- Amazon (Bar Raiser plays this role; debrief meeting with all interviewers)
- Microsoft (less centralized but team panel meets)
- Apple (team-specific)
- Smaller companies often skip this, decision is more direct from hiring manager.

---

## STAGE 8: Offer + Negotiation

Recruiter calls with verbal offer. Then written offer. Then negotiation window.

- **First offer is almost never the best offer.** Plan to negotiate.
- **Leverage = competing offers.** Especially other offers at the same tier.
- **Total comp = base + bonus + equity + signing bonus.** Negotiate all of them, not just base.
- See `07-comp-negotiation.md` for the full script.

---

## Timing of the funnel

| Stage | Typical time |
|---|---|
| Apply → Recruiter screen | 1-3 weeks (or never, for cold apps) |
| Recruiter → Phone screen | 1-2 weeks |
| Phone screen → Onsite | 1-3 weeks |
| Onsite → Decision | 1-2 weeks |
| Decision → Offer letter | 1 week |
| Offer → Sign | 1-3 weeks (negotiation, comparing offers) |

**Total: 6-12 weeks per company. Often longer.**

This is why you start applying at Month 10 if you want offers by Month 15.

---

## The probability funnel (for a candidate like you, well-prepared)

```
Apply to 150 roles
  → 30-40 callbacks                    (~20-25%)
  → 15-20 phone screens                (~50% of callbacks)
  → 6-10 onsites                       (~50% of phone screens)
  → 2-4 offers                         (~30-40% of onsites)
```

If at any stage your conversion rate is much lower than this, debug:
- Low resume → callback: resume issues or wrong companies
- Low callback → phone screen: scheduling / interest issues; ask recruiter
- Low phone screen → onsite: technical bar (DSA or ML basics); mock more
- Low onsite → offer: any of the rounds; debrief each one

---

## What separates "passes the bar" from "gets the offer"

Three things, in this order:

1. **Technical bar:** clear "Hire" on coding + ML rounds. Non-negotiable.
2. **Communication clarity:** can you explain a complex topic to a non-expert? Can you stay calm under pushback? This is judged in every round.
3. **Mission/team fit:** specifically for Anthropic, OpenAI, smaller labs — do you seem like someone who'd thrive there?

You can have an "exceptional" technical bar but no fit signal and lose. You can have a "solid" technical bar but excellent fit signal and win.

Both matter. Don't neglect either.

---

Next: [`02-anthropic-openai-deepdive.md`](./02-anthropic-openai-deepdive.md)
