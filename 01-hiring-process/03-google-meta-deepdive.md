# Google & Meta — Hiring Deep Dive

The two highest-volume FAANG-ML employers. Different cultures, different processes. Both have India hiring.

---

## GOOGLE

### Process overview (5 rounds total for L4 ML Eng)

1. **Recruiter screen (30 min)** — standard
2. **Phone screen (45-60 min)** — 1-2 LC-medium-or-mixed problems
3. **Onsite (4 rounds, ~45 min each):**
   - Coding round 1
   - Coding round 2
   - "Googleyness & Leadership" (behavioral)
   - ML / system design (depending on team)
4. **Team match round** — informal chats with potential teams
5. **Hiring committee** (you don't attend; they decide)

### The Google interview bar

- **Coding rounds:** 2 problems each 45-min round. Often LC-medium + LC-medium-hard.
- **You will use a Google Doc or shared editor.** No autocomplete, no compilation. Write clean Python by hand.
- **Bar:** Solve correctly, explain reasoning, handle edge cases, discuss complexity. "Almost solved" doesn't pass.

### Famous Google quirks

#### a) Hiring committee — your packet is everything
- Interviewers write detailed feedback after each round.
- A committee of 6-8 senior Googlers reviews packets, not candidates.
- **They look for clarity and consistency:** if 3 interviewers say "good problem solving" but 1 says "communication issues", you might be downleveled.
- Optimize for **legibility of skill**: explain your thought process clearly, name techniques you use, give specific complexity analysis.

#### b) Leveling
- L3 = new grad
- L4 = 2-5 YOE (you fit here)
- L5 = 5-8 YOE senior
- L6 = staff
- You can be down-leveled (offered L4 when applying for L5) — usually for "fine engineer, didn't show scope".

#### c) "Googleyness & Leadership" round
- Not pure behavioral. They probe for:
  - Bias for action (do you ship?)
  - Comfort with ambiguity (Google projects are large, messy)
  - Intellectual humility (do you say "I don't know" or do you bullshit?)
  - Collaboration (have you worked with hard-to-work-with people gracefully?)

- Sample questions:
  - "Tell me about a time you had to make a decision with incomplete information."
  - "Describe a project that failed. Why?"
  - "Tell me about a disagreement with your manager."
  - "How do you handle being wrong?"

#### d) Team match (after onsite)
- Once you pass the bar, recruiter introduces you to 1-3 teams looking for your profile.
- Each team does a 30-min "fit chat".
- You can decline a team if it doesn't fit. They can decline you.
- **Strategy:** Research teams in advance. Know which orgs (Cloud, Search, DeepMind, Gemini) align with your interest. Have informed questions.

### What ML rounds at Google look like

Two flavors:

**ML breadth (most common for L4):**
- "Walk me through your most complex ML project."
- Probes: model choice, eval, deployment, monitoring, problems and fixes.
- Then: "Now design something similar at Google scale."

**ML system design (for senior or platform roles):**
- "Design Google's image search system."
- "Design a multilingual question-answering system."
- 50 minutes. You drive whiteboard. They probe.

### Your specific angle for Google
- **GenAI fit:** Gemini Apps, Vertex AI Agent Builder, Search Generative Experience all need engineers like you.
- **Production rigor:** Sujanix government-grade reliability + AWS Lambda 99.9% uptime story → tells a "I ship and operate" story Google values.
- **Cross-functional:** SpaceDrift founder story → demonstrates collaboration across functions.

### Realistic odds for you
**30-45%** for getting an India L4 offer, with full execution. Highest hiring volume of any League A company.

---

## META

### Process overview (5 rounds total for E4 ML Eng)

1. **Recruiter screen (30 min)**
2. **Phone screen (45 min)** — 2 LC-medium problems, fast pace
3. **Onsite (4 rounds, ~45 min each):**
   - 2x coding rounds (1 LC-medium, 1 LC-medium-hard typically)
   - 1x ML system design
   - 1x behavioral
4. **Hiring committee + leveling**

### The Meta interview bar

- **Coding:** Faster pace than Google. Expect 2 problems in 45 min. Speed and correctness equally important.
- **System design:** "Design Facebook News Feed ranking." "Design Instagram Reels recommendations." Always ML-flavored for ML Eng candidates.
- **Behavioral:** They use "rooting interviews" — deep dives into past projects.

### Famous Meta quirks

#### a) The "rooting interview"
- Interviewer takes a project from your resume.
- Drills DEEP for 30+ minutes: "Why this architecture? Why these hyperparameters? What was your A/B test design? What if metric X had gone down — what would you have done?"
- **Defend every claim on your resume.** If you wrote "improved accuracy by 2.74%", be ready to explain the experiment design, baseline, statistical significance, deployment impact.

#### b) Move-fast culture in interviews
- They actively look for "bias for action" signals.
- Phrases that play well: "I ran a quick experiment to validate", "I shipped a v1 in 2 weeks", "I prioritized impact over polish".
- Phrases that play poorly: "I spent 3 months designing", "I wanted to be perfect", "I waited for sign-off".

#### c) The leveling rubric is rigid
- E3 = entry
- E4 = 2-4 YOE (you)
- E5 = 4-7 YOE
- E6 = staff
- They down-level aggressively if you can't show "scope" — ownership of features end-to-end with measurable impact.

### What you'd be tested on for E4 ML Eng

**Coding rounds (2):**
- Examples: "Implement an LRU cache with O(1) ops." "Find the longest substring with at most K distinct characters." "Implement a thread-safe rate limiter."

**ML system design round:**
- Examples: "Design Instagram's 'For You' page recommender." "Design ranking for Facebook Marketplace search." "Design content moderation pipeline for Reels."
- They want you to discuss: features, candidate generation, ranking model, eval metrics, online experimentation, scaling.

**Behavioral round:**
- "Tell me about a project where you had to convince stakeholders to change direction."
- "Tell me about a time you took ownership of something outside your job."
- "Tell me about a hard technical decision and how you made it."

### Your specific angle for Meta
- **Llama-related work:** If you can demonstrate Llama fine-tuning, evaluation, or inference work (even on a small scale), Meta engineers respect that.
- **Velocity stories:** Your SpaceDrift solo-founder experience — 16 months running an MSME with multi-service delivery under no-buffer freelance income — is exactly the velocity-under-constraint signal Meta wants.
- **A/B-testing literacy:** If you can speak to experiment design crisply (statistical significance, MDE, holdout cohorts), this differentiates you.

### Realistic odds for you
**20-30%** for India. Meta's India presence is small; many roles require relocation. If relocating, US E4 odds are slightly lower (15-25%) but compensation is much higher.

---

## How to prepare specifically for Google AND Meta in parallel

Both companies want:
- Excellent DSA
- Clear ML breadth
- ML system design competence
- Behavioral stories with measurable impact

They differ on:
- Process speed (Meta faster)
- Behavioral lens (Google = collaboration/ambiguity, Meta = ownership/velocity)
- System design framing (Google = "design for global scale", Meta = "design for social/ranking")

### Your prep checklist (Phase 2-3, Month 5-11):
- [ ] Solve 100+ LC problems tagged "google" and 100+ tagged "facebook/meta"
- [ ] Practice 3 ML system designs each (recommender, search, feed ranking)
- [ ] Write down "rooting interview" defenses for each project on your resume
- [ ] Develop one strong story per "Googleyness" axis (ambiguity, collaboration, humility, action)
- [ ] Develop one strong story per Meta core value (move fast, build social value, own it, focus on impact)

---

## Application timing for Google + Meta

- **Apply March-April 2027** (Month 11-12) for August target.
- These take 8-12 weeks per company. Start early.
- Apply through referrals if possible. Otherwise direct.

---

Next: [`04-amazon-microsoft-nvidia.md`](./04-amazon-microsoft-nvidia.md)
