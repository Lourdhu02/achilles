# How to Use This Repo

**Read this once. Refer back when you're confused about what to do.**

---

## The mental model

This repo is your **operating system**, not a textbook.

- A textbook says "here is everything about transformers."
- An OS says "here is what you do today, in this hour, to make progress toward the goal."

When you don't know what to do next, the answer is in here. When the answer isn't in here, ask me (Claude) and we'll add it.

---

## Three modes of reading

### Mode 1: Roadmap mode (every Sunday, 15 minutes)
1. Open `09-the-grind/01-weekly-rituals.md` and `09-the-grind/03-month-checklist.md`
2. Check current month's tasks from `00-START-HERE/04-15-month-roadmap.md`
3. Plan the upcoming week — what topics, how many problems, what mock interview, what blog post
4. Write your weekly plan in your tracker

### Mode 2: Study mode (most weekday/weekend sessions)
1. Pick a study topic from your weekly plan
2. Open the relevant folder (e.g., `04-genai-llm-mastery/`)
3. Read the file, take notes in your own words, implement code where applicable
4. Solve any associated practice problems (e.g., interview Q&As)
5. Update your tracker with hours + what you learned

### Mode 3: Interview mode (when actively applying)
1. Before any interview: skim relevant deep-dive files (e.g., before Amazon onsite, re-read `06-behavioral/02-amazon-LPs.md` + your story bank)
2. After every interview: write a debrief — what was asked, what you said, what you missed. File goes in `08-application-strategy/05-tracker.md`.
3. Update your story bank with any new stories revealed during interview prep.

---

## How to ask Claude (me) for help

I'm part of your prep system. Use me ruthlessly.

### Good prompts that work well:
- **"Quiz me on transformer attention math. Ask 10 questions of increasing difficulty. Tell me when I'm wrong and explain why."**
- **"Mock-interview me. You are a Google L4 ML interviewer. Ask me to design a real-time recommendation system. Push me when I'm shallow. Score me at the end."**
- **"I just finished `04-genai-llm-mastery/02-rag-deepdive.md`. Generate 15 hard interview questions from it. Don't tell me the answers until I attempt each."**
- **"My resume is at `RESUME-TOP-1.pdf`. Compare it against the bullet points I wrote in this draft (paste). Tell me which are weaker, why, and how to rewrite them."**
- **"Give me 5 LC-medium graph problems I haven't seen and walk me through one of them step by step after I attempt it."**

### Bad prompts that waste time:
- "Teach me machine learning" (too broad — pick a topic)
- "What should I study today?" (you have a roadmap — just check it)
- "Am I going to make it?" (probability table is in `01-honest-truth.md`; reread it)

### Memory: I already know about you
Your profile is saved in my memory across sessions:
- Lourdu Raju, ML Engineer at Sujanix, 2+ YOE, GenAI/CV focused, Bengaluru
- Target: GenAI/LLM Engineer at top 10 by Aug 2027
- Constraints: ~25-30 hrs/week, weekend-heavy
- DSA: intermediate

You don't need to re-explain context every conversation. Just say "Continue from where we were" or jump straight into the topic.

---

## Files you create vs. files I create

**I created (in this repo):** All the .md files. These are *templates and frameworks*. They will mostly not change.

**You create (in this repo):**
- `09-the-grind/weekly-log.md` (or similar — your weekly hour tracker)
- `02-dsa/problem-tracker.md` (your solved problems list)
- `06-behavioral/your-story-bank.md` (your STAR stories — I gave you a template, you fill it)
- `08-application-strategy/05-tracker.md` (your application tracker)
- Any notes from study sessions
- Drafts of resume, cover letters, blog posts

**You will both add and modify:**
- `00-START-HERE/02-your-profile-analysis.md` — update every 3 months
- `00-START-HERE/03-target-companies.md` — update when companies change hiring policies
- `04-15-month-roadmap.md` — adjust if you fall behind or get ahead

---

## File naming convention

```
NN-topic-name.md
```

Where NN is the order. If you create new files, follow this.

---

## When to update files

**Add to a file** whenever you learn something useful that future-you should remember:
- Learned a new DSA pattern? Add it to `02-dsa/01-patterns-master-list.md`.
- Found a great paper? Add to `04-genai-llm-mastery/07-must-know-papers.md`.
- Learned a behavioral question pattern? Add to `06-behavioral/06-questions-to-ask-them.md`.

**Edit a file** when:
- It contains outdated info (companies change hiring processes; salaries shift)
- You disagree with the advice after experience
- You've completed a milestone (mark items done in roadmap checklists)

**Don't delete files** — even if you outgrow them. Useful for retrospective.

---

## What if I'm not sure what to do today?

The escalation ladder:

1. **First:** Check the current month section in `04-15-month-roadmap.md`. There's almost always a task there.
2. **Second:** Check your weekly plan from Sunday.
3. **Third:** Pick the lowest-completion area from your trackers and work on that.
4. **Fourth (if truly lost):** Ask me. "I have 3 hours. I've done X, Y, Z this week. What's the highest-leverage thing to do right now?"

Do not spend more than 10 minutes deciding what to do. Decision-paralysis is your enemy.

---

## When to deviate from the plan

The plan is a baseline, not a cage. Deviate intentionally for:

- **Unexpected opportunity** — a referral comes in for a dream role. Pause prep, prep for that interview.
- **Major life event** — health, family. Pause, don't quit. Re-baseline when you return.
- **You're way ahead** — finished Phase 1 in 3 months instead of 4. Move to Phase 2 early. Don't manufacture work to fit the schedule.
- **You're way behind** — finished only 60% of Phase 1 by Month 4. Add 1 month. Push everything back. Better delayed than under-prepared.

Do NOT deviate for:
- "I don't feel like DSA today" → that's exactly when you do DSA
- "I want to learn a new framework I saw on Twitter" → bookmark, return after Phase 2
- "I want to refactor my GitHub README again" → time-boxed 30 min, then move on

---

## The Sunday ritual (60 minutes)

Every Sunday evening, no exceptions:

1. **Review last week (15 min):**
   - Hours logged: actual vs planned
   - DSA problems solved: count
   - Topics covered: list
   - Public output shipped: blog post / OSS PR / etc.
   - Mocks done: count + score
   - What went well? What didn't?

2. **Plan next week (30 min):**
   - Pull from current month checklist
   - Schedule specific time blocks (mornings? evenings? Saturday mornings?)
   - Pick specific files/topics
   - Pick specific LC problems (use tracker to avoid repeats)
   - Set 1 stretch goal

3. **Calibrate (15 min):**
   - Open `01-honest-truth.md` and re-read the probability table
   - Open `04-15-month-roadmap.md` and locate where you are
   - Ask: "Am I on track? If not, what changes this week?"

This 1 hour saves you 10 hours of drift during the week. Non-negotiable.

---

## The one-page mental rotation

When everything feels overwhelming, do this:

1. Pick ONE task from today's list.
2. Set a 25-minute timer.
3. Work on it without checking phone/Slack/anything.
4. When timer ends, 5 minute break.
5. Pick the next task. Repeat.

Pomodoros are clichéd because they work. You won't have motivation every day. You will have momentum if you build it.

---

You now have everything you need to start. Open [`02-dsa/README.md`](../02-dsa/README.md) or [`01-hiring-process/README.md`](../01-hiring-process/README.md) next, depending on what's heaviest in your current month.

Welcome to the grind.
