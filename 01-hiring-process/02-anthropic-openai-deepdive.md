# Anthropic & OpenAI — Hiring Deep Dive

The two stretch targets. Different cultures, different processes. Both ferociously selective.

---

## ANTHROPIC

### Company in 30 seconds
Founded 2021 by ex-OpenAI safety researchers (Dario Amodei, Daniela Amodei, etc.). Built Claude (now Opus 4.X, Sonnet 4.X, Haiku 4.X). Mission: build AI that is safe, beneficial, and aligned with human values. ~600 employees as of 2026. HQ San Francisco, London office. Has investors including Google, Amazon ($4B+), Salesforce.

### Why Anthropic interviews differently
Anthropic believes that *who* they hire shapes what AI becomes. They take culture/values screening MORE seriously than any company except maybe early-stage startups. They will reject technically excellent candidates if "values fit" feels off.

### Process (Applied Engineering roles)
Typical pipeline, 5-7 rounds total:

1. **Resume review + recruiter screen (30 min)**
   - Standard, but recruiters here are technical. They will ask follow-ups about projects.

2. **Coding screen (60-75 min)**
   - 1-2 problems. Often "implement something real" — e.g., a small attention mechanism, a debouncer, a token-counting utility. Less LeetCode-style, more "Python competence with real data structures".
   - Pair-programming style. Communication is judged heavily.

3. **Technical onsite — Coding (60-75 min)**
   - Larger, more open-ended problem. Sometimes a 90-min "build this small system" exercise.

4. **Technical onsite — ML / Domain (60-75 min)**
   - Deep dive into your projects.
   - Then: "How would you build X?" where X is something Anthropic is actively working on (RAG infra, eval pipelines, fine-tuning, agent systems).
   - Bring your best ML knowledge here. They will probe deeply.

5. **System design (60 min)**
   - Often involves an LLM system: design a multi-turn agent, design a constitutional AI evaluator, design an inference platform.
   - They want clear thinking and tradeoff articulation.

6. **Values / Mission round (45-60 min)** [THE DECIDER]
   - This is the round that makes or breaks otherwise-qualified candidates.
   - Sample questions:
     - "What do you think the most important AI safety problem is, and why?"
     - "If a customer asked you to build something you considered harmful but legal, what would you do?"
     - "Describe a time you pushed back on a project at work. Why?"
     - "What's your take on RLHF vs. Constitutional AI?"
     - "If you got an offer at Anthropic and a $200k higher offer at a less safety-focused lab, what would you do and why?"

7. **Bar raiser / executive round (45-60 min)**
   - Final values + culture check. Sometimes Dario or another senior leader.

### How to prepare for the Values round
**Read these before any Anthropic interview:**
- [Anthropic's "Core Views on AI Safety"](https://www.anthropic.com/news/core-views-on-ai-safety) (most important)
- "Constitutional AI: Harmlessness from AI Feedback" (paper)
- "Training a Helpful and Harmless Assistant" (paper)
- Dario Amodei's "Machines of Loving Grace" essay
- "Responsible Scaling Policy" (Anthropic's RSP document)

**Develop your own thinking on:**
- What does "AI safety" mean to you, in your own words?
- What kinds of AI systems would you refuse to help build?
- How do you feel about the tradeoff between capability and safety?
- What's your honest opinion on AGI timelines?
- What worries you about current AI development?

**The pattern of strong answers:**
- Specific (not "I care about safety")
- Personal (a story or moment that crystallized your view)
- Nuanced (acknowledge counter-arguments)
- Honest (don't pander; they detect it instantly)

### Anthropic-specific signal boosters
- **Use Claude (their model) for real work and have specific feedback** (positive AND negative — they value critical thinking)
- **Read their MCP (Model Context Protocol) docs**; understand why MCP exists
- **Have an opinion on their recent papers** — they publish actively
- **Try Claude Code, Computer Use, or Anthropic's developer tooling** so you can speak to it

### Anthropic-specific red flags to avoid
- Treating it like just another job ("good comp + interesting work")
- Dismissing safety concerns ("we're not at AGI yet, this is hype")
- Being vague on hard questions
- Coming off as a hype-driven "AI is the future" person without depth
- Lying or exaggerating about projects (they will catch it; they are technically deep)

### Your specific angle for Anthropic
- **FinSentinelAI** is gold: privacy-first, local-deployment, regulated industry. This maps to Anthropic's "deployment safely in sensitive contexts" thesis.
- **ECHOME** + memory architecture: speak to it as an exercise in *trustworthy long-context agents* — directly aligned with what Anthropic researches.
- **Founder/SpaceDrift story:** Frame it as "I learned what it takes to build AI that customers actually trust." Anthropic respects shipping experience.

### Realistic odds for you (Aug 2027)
**15-25%** if you execute the plan AND nail the values round. Not 0%. Not 50%. Treat as a stretch worth attempting, but don't bank on it.

---

## OPENAI

### Company in 30 seconds
The frontier AI lab. Built GPT-5, ChatGPT, DALL-E, Sora, o-series reasoning models. Microsoft holds significant equity (~49%). HQ San Francisco. ~3,000+ employees as of 2026. Ship fast, scale fast, sometimes-controversial.

### Why OpenAI interviews differently
OpenAI optimizes for **velocity** and **scale**. They want engineers who:
- Ship fast and iterate
- Can handle ambiguity (the product is changing weekly)
- Care about end-users (consumer + developer)
- Have strong opinions, weakly held

Less safety-screening than Anthropic (though still present). More "can you build at OpenAI's pace?" filtering.

### Process (Applied / MTS roles)
Pipeline, 5-7 rounds:

1. **Recruiter screen (30 min)** — same as universal
2. **Technical phone screen (60 min)** — coding, often a debugging exercise or "implement X" problem
3. **Onsite (4-5 rounds, often a full day):**
   - Coding (1 round): LC-medium/hard, with cleanup/testing focus
   - ML technical (1 round): deep technical, may involve whiteboard math
   - System design (1 round): ML system or general system design
   - "Hands-on" project round (1 round): may include reviewing a real OpenAI codebase snippet or improving a script
   - Behavioral / hiring manager (1 round)
4. **Sometimes:** Bar-raiser equivalent or a senior leader round

### What OpenAI specifically tests
- **Ship velocity:** "Tell me the fastest you've ever shipped something significant from idea to production. How? Why?"
- **Customer obsession:** "Why is ChatGPT successful? What would you change about it?"
- **Bias to action:** "You see a bug in production at 2am, what do you do?"
- **Technical depth at scale:** "How does the GPT API serve 100M+ users? What are the bottlenecks?"

### OpenAI-specific signal boosters
- **Use their products extensively** — have specific opinions on ChatGPT, the API, Codex, Sora
- **Have built on their APIs** — your work clearly does this (FinSentinelAI uses local LLMs, but you also have GPT/Claude API experience)
- **Read their tech blog** — understand recent posts on training, inference, evals
- **Have shipped fast** — your solo founder story (16 months running SpaceDrift, multi-service delivery under financial constraint) is a velocity signal

### OpenAI-specific red flags
- Being overly cautious ("I'd want to wait until X before...") — they want decisive
- Being underwhelming on customer empathy
- Not having an opinion on their products
- Coming off as risk-averse

### Your specific angle for OpenAI
- **SpaceDrift founder story:** This is your strongest card. Frame: "I built and scaled an AI consultancy. I know how to ship under pressure with ambiguity. I'd love to do that at OpenAI's scale."
- **Production GenAI work** at Sujanix: government-grade reliability + your fintech RAG. Velocity + impact.
- **Kaggle Expert** + diverse domains (CV + NLP + agents): polymath signal.

### Realistic odds for you (Aug 2027)
**15-20%.** Slightly lower than Anthropic because OpenAI has more applicants per role; their offer rate is brutal. But your shipping story plays well.

---

## How to actually apply

### Both companies:
- Watch their careers pages **weekly** starting Month 8.
- New roles often appear and fill within 2-3 weeks.
- Apply DAY OF role being posted. Don't wait.

### Referral path (best route):
- Anthropic and OpenAI both have referral programs.
- Find someone via LinkedIn → Twitter → mutual connections. Don't cold-DM unless your work is genuinely noteworthy.
- Better: become known in the Hugging Face / AI engineer community. People will refer you because they've seen your work.

### Remote-applied caveat:
- Most OpenAI / Anthropic roles are SF or London on-site.
- Some Applied roles are remote. Filter carefully.
- For SF roles: you'd need to relocate. Visa is a constraint (usually O1 or H1B).

---

## What to do RIGHT NOW (next 2 weeks) if you're serious about these

1. Read Anthropic's "Core Views on AI Safety" and write your own 500-word response.
2. Read 3 papers from OpenAI's recent blog (especially anything on reasoning, alignment, scaling).
3. Use Claude (Anthropic) and GPT-5 (OpenAI) for real work. Form opinions.
4. Add Anthropic/OpenAI engineers in Bangalore + SF + London to your LinkedIn network (slowly — 2-3/week).
5. Start writing 1 blog post on a deeply-engaged topic (e.g., "How I'd design a RAG eval framework for production") — this is the kind of content that gets noticed.

---

Next file: [`03-google-meta-deepdive.md`](./03-google-meta-deepdive.md)
