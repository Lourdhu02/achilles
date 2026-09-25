# 01: Problem selection and customer discovery

Pick a problem that someone already pays to solve, that happens often, and that gets easier for you (not for your competitors) as models improve. Then prove it with past behavior and money, not with opinions.
This file gives the criteria, a scoring rubric with a worked example, a full discovery interview script, and the signals that count.

**Contents:** [What a good problem looks like](#what-a-good-ai-company-problem-looks-like) · [Anti-patterns](#anti-patterns) · [Score your ideas](#score-your-ideas) · [Interview script](#customer-discovery-interview-script) · [Signal ladder](#the-signal-ladder) · [Synthesis](#synthesizing-15-20-interviews) · [Concierge MVP](#concierge-mvp) · [Failure modes](#failure-modes) · [Worksheet](#worksheet)

## What a good AI-company problem looks like

- **Painful and frequent.** People do it daily and dislike it, or it costs real money: labor hours, error rework, compliance penalties, delayed cash.
- **Budget exists.** Someone already pays for it: staff, a BPO vendor, overtime, software. You are replacing existing spend, not asking for a new budget line. New budget lines take a year to create in most enterprises.
- **AI makes it 10x better now,** not 10% better, and **the product gets better as models improve.** If a stronger base model makes your product unnecessary rather than stronger, you are a feature waiting to be absorbed ([absorption test](04-moats-and-flywheels.md#the-absorption-test)).
- **You have unfair access:** you know the workflow, the buyers, the data formats, the language. Candidate areas close to a document-AI background: meter and utility reading, invoice and bank-statement processing for regulated finance (often on-prem), claims and KYC documents, Indic-language forms for small businesses.
- **The workflow is deep:** integration with systems of record, approvals, exceptions and audit trails. Depth is defensible; a chat box is not.
- **Errors are catchable.** In document work a human can verify a field against the source in seconds. Problems where errors are silent and costly (medical advice, legal conclusions) need much more evidence and liability planning before you sell.

## Anti-patterns

- Building a platform before you have one customer.
- A thin wrapper with no workflow, data or distribution advantage.
- Head-on competition with a frontier lab's roadmap: general assistants, generic coding agents, general chat over documents.
- Markets where buyers can't or won't pay: consumer apps in price-sensitive segments without a distribution edge.
- "AI for X" where the people in X don't feel the pain, or the person who feels it has no budget.
- Picking a problem because the technology is interesting to you. Interesting-to-build and worth-paying-for overlap less than engineers expect.

## Score your ideas

Score each idea 1–5 on each criterion, multiply by the weight, and sum. The weights encode a belief: evidence of payment matters more than technical elegance. Change them if you disagree, but change them *before* scoring.

| criterion | weight | 1 means | 5 means |
|---|---|---|---|
| Existing spend | 3 | nobody pays for this today | a named team or vendor is paid for it now |
| Frequency and pain | 2 | quarterly annoyance | daily, costly, visible to management |
| AI delta now | 2 | modest speed-up | replaces most of the manual work at acceptable accuracy |
| Gets better with models | 2 | a better model makes you unnecessary | a better model lowers your cost or raises your accuracy |
| Your unfair access | 2 | no contacts, no domain knowledge | you have worked in it and can get 10 buyer meetings |
| Speed to first payment | 1 | 12+ month procurement | a paid pilot within 60 days is plausible |
| Error tolerance | 1 | silent, costly errors | errors are visible and cheap to fix |

Maximum score: 5 × 13 = 65.

### Worked example (hypothetical ideas, hypothetical scores)

| criterion (weight) | A: utility meter-photo validation for distribution companies | B: bank-statement extraction for small lending firms | C: general "chat with your PDFs" for students |
|---|---|---|---|
| Existing spend (3) | 4 → 12 | 5 → 15 | 1 → 3 |
| Frequency and pain (2) | 4 → 8 | 4 → 8 | 2 → 4 |
| AI delta now (2) | 4 → 8 | 4 → 8 | 3 → 6 |
| Gets better with models (2) | 3 → 6 | 4 → 8 | 1 → 2 |
| Unfair access (2) | 5 → 10 | 3 → 6 | 2 → 4 |
| Speed to first payment (1) | 1 → 1 | 4 → 4 | 3 → 3 |
| Error tolerance (1) | 4 → 4 | 3 → 3 | 3 → 3 |
| **total (of 65)** | **49** | **52** | **25** |

How to read it: A and B are close, so the score alone does not decide. A has the stronger access but a slow, government-heavy sales cycle; B has faster sales and clearer existing spend but weaker access. The next step is 5 interviews for each, not more scoring. C fails on the two things that matter most: nobody pays and a general assistant will absorb it.

> [!TIP]
> The rubric is for ranking your own ideas against each other. It is not a probability of success, and a high score is a reason to interview, not a reason to build.

## Customer-discovery interview script

Goal: learn how the problem is handled today, what it costs, and who pays, **without** pitching. You are not allowed to describe your product until the last five minutes, and only if they ask.

**The three rules** (from [The Mom Test](https://www.momtestbook.com/)):
1. Talk about their life, not your idea.
2. Ask about specific past events, not opinions or hypothetical futures.
3. Talk less, listen more. If you are speaking more than 30% of the time, you are pitching.

**Before the call:** know their role, their company's size, and one specific thing about their workflow. Decide the one thing you most need to learn from this call.

**Script (30 minutes).** The notes in brackets say why each question exists.

```text
Opening (2 min)
  "Thanks for the time. I'm researching how teams like yours handle <workflow>.
   I'm not selling anything today. Could you walk me through how it works for you?"
   [Sets expectations; lowers their guard.]

The last time (10 min)
  1. "Tell me about the last time you had to <do the task>. What happened, step by step?"
     [A specific past event. Vague answers mean the pain is not frequent.]
  2. "How long did that take? How many people touched it?"
     [Converts pain into hours, which converts into money.]
  3. "What went wrong, or what usually goes wrong?"
     [Finds the failure modes your product must handle.]
  4. "What happens downstream when there's an error?"
     [Cost of errors: rework, penalties, delayed payments.]

Current solutions (8 min)
  5. "What have you tried to fix this? Tools, vendors, scripts, extra staff?"
     [No attempts at all is a warning sign: the pain may not be real.]
  6. "What does the current approach cost per month, roughly?"
     [Existing spend is the budget you will replace.]
  7. "Why did you stop using <the thing they tried>?"
     [Tells you the bar you have to clear and the objections you'll face.]

Buying process (6 min)
  8. "If you wanted to change how this is done, who else would be involved?"
     [Maps the champion, budget holder, IT/security and procurement.]
  9. "When did your team last buy a tool or service? How did that go?"
     [Past buying behavior predicts the sales cycle better than stated intent.]
 10. "Are there rules on where this data can go? On-prem, cloud, specific regions?"
     [Deployment constraints decide your architecture and your price.]

Close (4 min)
 11. "Who else should I talk to about this?"   [Referrals are a commitment signal.]
 12. "Can I come back in a few weeks with something to show you?"
 13. If they are clearly in pain and have budget:
     "If we could <outcome, e.g. cut review time in half on your documents>,
      would you run a paid 6-week pilot?"  [The only question that tests money.]
```

**Note sheet (fill within an hour of the call):**

```text
Date · company (size, sector) · person (role, budget authority yes/no)
Last-time story (verbatim quotes, numbers)
Hours per week / people involved / current spend (their words)
Tried before and why it failed
Decision makers named
Deployment or data constraints
Strongest signal reached on the ladder (below)
Surprises (what contradicted my assumptions)
Follow-up promised, by when
```

> [!WARNING]
> Watch for three false positives: compliments ("this is a great idea"), fluff ("we would definitely use something like this"), and feature requests from people with no budget. Write them down, then discount them to zero.

## The signal ladder

Each rung costs the customer more, so each is worth more.

| rung | example | worth |
|---|---|---|
| 0. Compliment | "Love it, keep me posted" | nothing |
| 1. Time | a second meeting; they walk you through their real files | some |
| 2. Reputation | they introduce you to their boss or budget holder | strong |
| 3. Access | they share real (anonymized, consented) data for a test | strong |
| 4. Money, soft | a signed LOI with a price and start date | very strong |
| 5. Money, real | a paid pilot or a purchase order | the goal |

Count, per idea, how many interviewees reached each rung. That table, not your enthusiasm, goes into the [decision gate](README.md#decision-gate-put-it-in-roadmapmd).

## Synthesizing 15–20 interviews

1. Put every "last time" story in one table: steps, time, errors, cost, who paid.
2. Tag recurring pains; count how many interviewees mentioned each **unprompted**.
3. Segment: do the people with the strongest pain share an industry, company size or trigger event? That is the first draft of your ideal customer profile ([05](05-go-to-market.md#define-the-ideal-customer-profile)).
4. Write a one-sentence problem statement with a number in it. Hypothetical example: "Loan officers at small lending firms spend about 40 minutes per application re-typing bank statements; errors delay disbursal by a day."
5. Decide: continue, narrow the segment, or kill. If fewer than 3 of 15 interviewees reached rung 2, the problem is probably not painful enough in that segment.

## Concierge MVP

Do the job semi-manually for 2–3 paying customers before automating it. You deliver the outcome (for example, extracted and verified data) using whatever mix of models, scripts and your own review gets it done. Then automate the steps that repeat.

Why it works: you learn the real edge cases, you are paid while learning, and your first eval set is the corrected outputs you produced. A services background is an advantage here; the risk is staying in services forever ([07](07-team-and-operating.md#the-services-trap)). Set a date by which the concierge must become software.

## Failure modes

- **Interviewing friends and fans.** They are kind. Interview strangers in the target role.
- **Pitching during discovery.** Once you pitch, people start being polite about your idea instead of describing their problem.
- **Talking only to users.** The user feels the pain; the budget holder decides. Reach both.
- **Stopping at 5 interviews** because the first ones were encouraging. Patterns appear around 10–15.
- **Changing the question every call.** Keep a fixed core so answers are comparable; add probes at the end.
- **No kill criteria.** Write them before the first interview (for example, "fewer than 2 paid pilots after 20 interviews and 8 weeks: kill or narrow").

## Worksheet

Fill one per idea; keep the best three.

```text
Problem (one sentence, with a number):
Who has it (role, not company):
How often:
Current solution and its cost (their numbers):
Why now (which model capability unlocked it):
Why it gets better, not obsolete, as models improve:
Why you (access, data, domain):
Deployment constraints (on-prem, residency, languages):
Smallest paid pilot (scope, price, duration):
What you'd have to believe for a ₹10 crore or $1M ARR business (customers x price):
Kill criteria (written before interviewing):
```
