# Anthropic Mission & Values Fit

The unique behavioral round at Anthropic. **More candidates fail here than at the technical rounds.**

---

## Why Anthropic interviews this way

Anthropic believes that *who* they hire shapes what AI becomes. Their thesis:
- The most important AI questions are not purely technical
- Researchers and engineers at frontier labs influence AI trajectory
- Values fit determines whether someone will push for safety vs ship-first vs profit-first

So they screen aggressively for:
- Genuine engagement with AI safety
- Honest, nuanced thinking
- Willingness to push back / not optimize for the wrong things
- Personal investment in the mission (not just comp/role)

---

## The values round structure

A 45-60 min discussion-style interview. Less Q&A, more conversation. Sample topics:

### 1. Your view on AI safety
- "What do you think is the most important AI safety problem?"
- "Are you worried about AI? About what specifically?"
- "How do you think about AGI timelines?"

### 2. Your stance on tradeoffs
- "If a customer wanted you to build something safety-questionable but profitable, what would you do?"
- "How would you handle pressure to ship something faster than feels safe?"
- "Tradeoffs between capability and safety — how do you reason?"

### 3. Your motivations
- "Why Anthropic specifically? You could work at OpenAI for more money."
- "What would you NOT work on?"
- "What do you find most meaningful about AI work?"

### 4. Your reasoning under pressure
- "Tell me about a controversial decision you made."
- "When did you push back on a manager / customer / peer?"
- "How do you handle disagreement?"

### 5. Your engagement with their work
- "What's your view on Constitutional AI?"
- "What do you think about Claude's recent X release?"
- "What's a paper of ours you've engaged with?"

---

## Required reading BEFORE any Anthropic interview

### Top priority
1. **"Core Views on AI Safety"** (Anthropic 2023) — anthropic.com/news/core-views-on-ai-safety
2. **"Constitutional AI: Harmlessness from AI Feedback"** (Bai et al. 2022) — the foundational paper
3. **"Responsible Scaling Policies"** (Anthropic) — their commitments at different capability levels
4. **Anthropic homepage's "Our Research"** and recent blog posts
5. **Dario Amodei's "Machines of Loving Grace"** essay

### Secondary
6. "Training a Helpful and Harmless Assistant" (Bai 2022)
7. "Sleeper Agents" paper (Hubinger 2024) — important alignment paper
8. The "Model Card" for Claude Opus 4 / Sonnet 4 etc.
9. Anthropic's safety research papers (mechanistic interpretability is a big one)

---

## Developing YOUR view (this is what they evaluate)

You can't fake this. They detect it instantly. But you CAN do the work to develop genuine views.

### Worksheet: write 2-paragraph answers for each

**1. What's your view on AI safety?**
Example structure:
- What does safety mean to you specifically? (alignment? misuse? societal effects?)
- A specific concern you care about (concrete, not generic "doom")
- A specific positive vision (what good AI deployment looks like)

**2. What would you refuse to work on?**
- A specific category (e.g., autonomous weapons, mass surveillance, deceptive AI assistants)
- Why personally — connect to your values
- Have you been in a situation that tested this? (sometimes they probe)

**3. Why Anthropic specifically?**
- Their RSP commitment (vs competitors who don't have one)
- Their constitutional AI approach (vs vanilla RLHF)
- The team's reasoning quality (interpretability research, etc.)
- Make it specific. "Smart people working on important things" is generic and weak.

**4. What worries you most about current AI development?**
- A specific concern, with reasoning
- Acknowledge counter-arguments
- Connect to your work

**5. How do you balance capability vs safety in your own work?**
- Use a real example from your projects
- Show you've made tradeoffs (e.g., FinSentinelAI privacy-first choices)

### Writing your answers down

In `your-story-bank.md`, add a section "Anthropic values prep" with these 5 answers. Refine over Months 6-12. Re-read before any Anthropic-related interaction.

---

## Your specific angles (use these)

### Angle 1: FinSentinelAI as a "deploying safely in sensitive contexts" exemplar
> "When I built FinSentinelAI for regulated finance clients, I made a series of choices that were harder than the alternatives. We could have used cloud LLMs — easier to scale, better quality — but our clients needed privacy guarantees. We chose local Llama models + ChromaDB for true data sovereignty. We added a faithfulness verification layer — every cited claim is checked against retrieved context — when most production RAG systems skip this. JWT-based session isolation for multi-tenancy.
>
> These weren't asked for by customers. I felt responsible for them. That's a small-scale version of what Anthropic does — making deployment choices that other labs cut corners on."

### Angle 2: ECHOME memory architecture as "safe long-horizon agents"
> "ECHOME's three-tier memory wasn't just about capability. Long-horizon agents are dangerous if their memory drifts or accumulates errors. I designed explicit consolidation logic so episodic memories distill into semantic facts (which can be reviewed/corrected) rather than implicit, opaque accumulation. This is the kind of work I'd love to do at scale — making agents that humans can trust over long interactions."

### Angle 3: Your founder story as "saying no to wrong work"
> "At SpaceDrift I turned down 2 clients in 2025 because their use cases — [generic example: mass-personalized political ads, surveillance enhancement] — didn't sit right with me. I lost some revenue. Building Anthropic-quality work means saying no sometimes. I'd rather work where the bar is held high."

(Note: only use this if true — replace examples with real ones.)

### Angle 4: Engagement with their work
> "I've used Claude extensively for [specific work, e.g., debugging difficult OCR pipeline issues]. The thing I noticed: Claude is the only model that refused certain edge-case prompts that would have made my work easier in the short term but pushed past comfortable boundaries. I appreciated that. It signaled the company's priorities to me as a user."

---

## Common red flags Anthropic recruiters report

### Red flag 1: Treating safety like a buzzword
Bad: "Yeah, AI safety is super important. I'm into responsible AI."
Better: Specific concern → specific approach you take in your work.

### Red flag 2: Dismissing AGI / alignment concerns
Bad: "I think AGI is overhyped. We're nowhere close."
Better: A nuanced position. Even if you're skeptical, acknowledge the uncertainty.

### Red flag 3: Pure ship-velocity culture
Bad: "I love OpenAI's move-fast culture. Anthropic is too cautious."
Better: Acknowledge the value of velocity AND care about safety. Both can be true.

### Red flag 4: No engagement with their research
Bad: "I read the docs."
Better: "I particularly engaged with [specific paper]. The argument that struck me was..."

### Red flag 5: Pandering / saying what they want to hear
Bad: Performative agreement with everything.
Better: Some friendly disagreement is GOOD. "I think X but I'm not certain — here's the counter-argument."

### Red flag 6: Vague on hard questions
Bad: "It depends" without elaboration.
Better: Honest tension. "I genuinely struggle with this question. On one hand... on the other..."

---

## The compensation question (specifically for Anthropic/OpenAI)

They will ask: "Anthropic salaries are below FAANG/OpenAI for similar roles. How does that factor in?"

This is a test.

### Bad answer
"I'm fine with that as long as comp is in this range..." (sounds like you're only here for money)

### Good answer
"I've thought about this. The marginal $X/year doesn't change my life materially. What does is being part of a team that's pushing toward the safer outcome. Anthropic's mission is one of the few I'd take a salary cut for. Within reason — I'd want comp competitive enough that I'm not sweating it — but I'm not optimizing for it."

(Adjust based on your actual situation. Honesty > polish.)

---

## Practice protocol

### Weeks 1-2 (after starting Anthropic prep)
- Read "Core Views" + Constitutional AI paper
- Write your 5 worksheet answers
- Use Claude for 5+ real work sessions; form opinions

### Weeks 3-4
- Read 3 more Anthropic papers
- Have a mock values conversation with a friend (or me) — they ask, you respond, they push back
- Refine answers

### Ongoing (Months 7-12)
- Read 1 new Anthropic blog post / paper per week
- Follow Anthropic leaders on X/Twitter (Dario Amodei, Jack Clark, Sam Bowman, etc.)
- When you're with friends discussing AI, articulate your views out loud

### Day before Anthropic interview
- Skim Core Views again
- Recent Claude release / paper review
- Revisit your 5 worksheet answers
- Mock with me: "30 min Anthropic values round. Push hard."

---

## Resources

- Anthropic.com/research — all papers
- Anthropic blog
- Dario Amodei interviews (Lex Fridman, Dwarkesh)
- "Concrete Problems in AI Safety" — Amodei et al. 2016 (Dario's pre-Anthropic safety roadmap)
- "AI Alignment: A Comprehensive Survey" (open-source)
- 80,000 Hours podcast episodes with Anthropic researchers

---

Next: [`04-your-story-bank.md`](./04-your-story-bank.md)
