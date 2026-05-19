# Mock Interview Protocol

A mock interview you didn't debrief is a wasted hour. This file is how you extract maximum value from each mock.

---

## Why mocks matter (no, really)

Solo problem-solving and live interview performance are different skills.

In solo mode, you have:
- Ability to look things up
- Silent thinking
- No interviewer watching
- Pause, reset, restart at will

In an interview, you must:
- Think out loud
- Type while talking
- Handle interruptions
- Recover from being wrong
- Manage time pressure
- Read interviewer cues

You CAN'T train the live-interview skills by solving alone. You need real reps.

---

## The mock interview ladder

### Tier 1: Self-mocks (free, weekly)
- Pick 1-2 LC problems you haven't seen
- Set a 45-min timer
- Record yourself on phone/laptop solving + narrating
- Watch playback
- **Use for:** baseline rep, identifying narration habits, time management

### Tier 2: Peer mocks via Pramp (free, every 2 weeks)
- Pramp.com — peer-to-peer mock interviews
- You're the interviewer for 45 min, then they interview you 45 min
- **Use for:** real "another human watching" pressure, getting/giving feedback
- Quality varies — some peers are excellent, others are weak

### Tier 3: Paid mocks (premium, monthly)
- **interviewing.io** ($200/session) — interviewer is a real FAANG engineer
- **TryExponent** (~$100/session)
- **Igotanoffer.com** (varies)
- **Use for:** calibration against real FAANG bar, intensive feedback
- Spend on these 3-5 times across your prep (~₹20-40k total)

### Tier 4: Reality testing — actual interviews
- Once you reach Month 9-10, Wave 1 applications become your "highest-fidelity mocks"
- Stripe, Atlassian, Adobe phone screens = practice for Google/Meta
- **Use for:** the real deal; debrief is critical

---

## Frequency targets

| Month | Mocks/week | Total mocks by end of month |
|---|---|---|
| 1 | 0.5 (i.e., 2/month) | 2 |
| 2 | 1 | 6 |
| 3 | 1 | 10 |
| 4 | 1 | 14 |
| 5-6 | 1.5 | 26 |
| 7-8 | 2 | 42 |
| 9-11 | 2 + actual interviews | 60+ |

By application time, **60+ mocks is the floor**. Many candidates do 100+.

---

## The mock interview structure (45-60 min)

### Format A: Standard coding mock

```
00-05: Introduction
        Interviewer: "Hi, I'm X, I'll be doing a coding interview today. Tell me briefly about yourself."
        You: 60-second intro

05-08: Problem statement
        Interviewer reads/posts the problem.
        You: Repeat back your understanding to confirm.

08-15: Clarifying questions + brainstorm
        You: ask 2-4 clarifying questions
        You: describe 2 approaches at high level
        You: pick one with justification

15-35: Implementation
        You: code while narrating
        Watch for: bugs, off-by-ones, edge cases

35-40: Test + complexity
        You: walk through with example
        You: state time and space complexity
        You: discuss edge cases

40-45: Follow-up question
        Interviewer might ask: "What if input is too big to fit in memory?" "What about thread safety?"

45-50: Feedback + Q&A
        Interviewer gives feedback
        You ask: "What would you have wanted me to do differently?"
```

### Format B: ML technical mock

```
00-05: Intro
05-15: Project walkthrough — pick one resume project, explain in depth
15-30: ML deep-dive — interviewer probes on a specific area (e.g., RAG, fine-tuning, evals)
30-45: Whiteboard math OR system design lite (e.g., "design eval pipeline for an LLM")
45-50: Follow-ups + feedback
```

### Format C: ML system design mock (later phase)

```
00-05: Intro
05-10: Problem statement + clarifying questions
10-25: Requirements + functional/non-functional, then high-level architecture
25-40: Deep-dive into 2-3 components (model, data, serving, eval)
40-50: Scaling, monitoring, failure modes
50-60: Discussion + feedback
```

---

## How to be the INTERVIEWER (Pramp / peer mocks)

You will be the interviewer for half your peer mocks. Your job:

1. **Pick a problem in advance** — pick a clean LC-medium or LC-hard you know well
2. **Read it slowly to candidate** — let them ask clarifying questions
3. **Wait, watch, don't help too fast** — give 5+ min before any hint
4. **Take notes** — what did they say? where did they struggle?
5. **Don't give the solution** — give incremental hints toward the next step
6. **Feedback at end** — structured: what was good, what to improve, what's blocker

**Bonus:** being the interviewer trains your "interview brain" — you learn what interviewers look for.

---

## How to be the CANDIDATE (the main event)

### Before the mock (30 min prep):
- [ ] Re-read 1 pattern's template
- [ ] Check what you're nervous about
- [ ] Set up environment: clean desk, quiet, water, notepad
- [ ] Open mock platform 5 min before
- [ ] Take 3 deep breaths

### During the mock:

**Narration framework — the "loud thoughts" technique:**

You should say things like:
- "Let me re-read the problem statement... I want to confirm a few things."
- "So the input could be empty, right?"
- "Two approaches come to mind. Approach A is X with O(n²). Approach B is Y with O(n log n). I'll try B."
- "I'll start by..."
- "Wait, let me trace through with the example."
- "I see a bug here — the index would go out of bounds when..."
- "This works. Let me state complexity. Time is O(n log n) because of the sort, space is O(1) excluding output."

**DO NOT:**
- Code silently for 15 minutes
- Say "got it" and write 30 lines
- Skip clarifying questions
- Argue with the interviewer

**DO:**
- Think out loud, even if slowly
- Re-state the problem in your own words
- Confirm assumptions
- Acknowledge "I'm not sure, let me think"
- Show how you handle being stuck (it happens; what matters is the recovery)

### After the mock:

This is where most candidates fail. They feel relief and walk away. Don't.

**Within 30 minutes of the mock, write a debrief:**

```markdown
# Mock #N debrief - YYYY-MM-DD

## Problem(s)
- LC 76 Min Window Substring (medium)

## What I did well
- Asked good clarifying questions about character case sensitivity
- Identified sliding window pattern within 90 seconds
- Code was clean

## What went wrong
- Spent 8 min on a wrong approach before switching
- Made an off-by-one error in window-shrinking that I caught only after testing
- Complexity analysis was slightly wrong (said O(n + m), should have been O(n))
- Got nervous during follow-up and rambled

## Feedback from interviewer
- "Good thinking aloud, but could be tighter."
- "Test cases should include edge cases (empty string, no valid window)"
- "Mention space complexity explicitly"

## Actions for next time
1. Practice 5 more sliding window problems
2. Before testing, list edge cases out loud
3. For complexity, separate time vs space cleanly
4. Practice 2-min "wrap up" responses

## Calibration
- Self-score: 6/10 (would lean-hire but not strong-hire)
- Interviewer score: 6.5/10 ("would pass, not strong")
```

---

## What to track across mocks (in spreadsheet)

| Mock # | Date | Type | Problem | Pattern | Time-to-solve | Bugs found | Hints used | Self-score | Interviewer score |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-05-25 | Self | LC 11 | Two Pointers | 25min | 1 | 0 | 7 | - |
| 2 | 2026-06-08 | Pramp | LC 76 | Sliding Window | 38min | 2 | 1 | 5 | 6 |

After 10 mocks, you should see:
- Time-to-solve decreasing
- Bugs found decreasing
- Hints used decreasing
- Scores increasing

If these trends don't appear, something's wrong with your study process. Diagnose.

---

## Common patterns interviewers grade

Most rubrics include something like:

| Dimension | What it means | Example "Hire" signal |
|---|---|---|
| Problem solving | Approach quality | Considered multiple approaches; picked optimal with justification |
| Coding | Code quality | Clean, runnable code without major bugs; meaningful names |
| Communication | Narration | Clear, paced, responsive to interviewer prompts |
| Testing | Validation | Tested with examples, found own bugs, considered edge cases |
| Complexity | Analysis | Stated correct time/space; could discuss tradeoffs |
| Coachability | Reception of hints | Took hints gracefully, integrated, didn't go silent |

Make sure you're "Hire" on every dimension. Not just "got the right answer."

---

## Specific tactics for FAANG-style mocks

### Google-style:
- Practice on Google Docs format
- Don't expect IDE features
- Pre-write helper functions like `from collections import deque` at the top

### Meta-style:
- Practice tight time management (2 problems in 45 min)
- "Rooting interview" prep: practice answering deep follow-ups on your projects

### Amazon-style:
- ALWAYS pivot behavioral questions to LP-relevant stories
- "Tell me about a time you..." → use STAR + name the LP

### Anthropic / OpenAI:
- Less LeetCode-syntax, more "implement this real-world thing"
- E.g., "Implement a simple token counter for a given LLM tokenizer"
- Practice "build something working from scratch" exercises

---

## When mocks go badly

You'll have bad mocks. Some will be brutal. Don't quit. Instead:

1. **Debrief honestly** — what was the gap? (DSA pattern not mastered? Communication too quiet? Got nervous?)
2. **Target the gap** — if a specific pattern, drill 5 more problems in it. If communication, practice "narrate-out-loud" on solo problems all week.
3. **Schedule the next mock within 7 days** — don't let a bad mock leave a scar; replace it with a better one.

---

## When you're consistently scoring "Hire" in mocks

You're ready. Apply. Don't keep mocking instead of applying.

A common failure mode: candidate hides behind mocks for months when they could be interviewing for real.

---

Next: [`04-coding-during-interview.md`](./04-coding-during-interview.md)
