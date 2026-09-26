# Stories: behavioral and values rounds

Behavioral and values rounds test judgment, ownership and honesty through specific past events. Prepare 12–15 true stories that each cover several themes, rehearse them out loud, and be ready for three levels of follow-up questions.
This file has the method, a fill-in template, the story index, the numbers you must verify before telling any story, values and mission prep, and questions to ask.

**Contents:** [Method](#method-star--learning) · [Template](#story-template) · [Story index](#story-index-build-1215-each-covers-several-themes) · [Numbers you must verify](#numbers-you-must-verify) · [Follow-up drill](#follow-up-drill) · [Values and mission prep](#values-and-mission-prep) · ["Why this company?"](#why-this-company) · [Questions to ask](#questions-to-ask-them)

## Method (STAR + learning)

Situation (20 s) → Task (15 s) → **Action (60–90 s: *your* decisions, trade-offs, technical detail)** → Result (numbers) → Learning. Two to three minutes in total.

Each story needs: at least two real, sourced numbers; one trade-off you chose and what you gave up; one honest lesson; and "I" for your actions ("we" only where the team acted). Rehearse out loud and record yourself; cut anything that isn't a decision or a consequence.

## Story template

```text
Title (3–5 words):                         Themes it covers:
Situation: context in two sentences; why it mattered (cost, users, deadline)
Task: what you specifically owned
Action: 3 decisions you made
  1. decision — alternatives considered — why this one
  2. ...
  3. ...
Result: numbers (with source record for each) + what happened next
Learning: what you do differently now, with an example of applying it since
Follow-ups I expect: "How did you measure it?" "What would you change?" "What did others think?"
Source records: <links or file names for every number above>
Short version (60 s) written: yes/no     Long version (5 min, technical) written: yes/no
```

## Story index (build 12–15; each covers several themes)

| theme | candidate story |
|---|---|
| Hard technical problem / quality bar | Sujanix OCR: loss design, quantization trade-off |
| Ownership, ambiguity | SpaceDrift: founding, scoping, pricing |
| Failure / mistake | a SpaceDrift project that was badly scoped (what you'd do differently) |
| Conflict / pushback | stakeholder priorities at Sujanix (cost vs accuracy vs speed) |
| Learning fast | a new domain for a client; now also your from-scratch labs |
| Delegation / leading without authority | managing paid contractors |
| Saying no | turning down bad-fit client work |
| Customer obsession | changing approach after user feedback |
| Research taste (core-AI roles) | the most surprising result from your lab experiments |
| Honesty under pressure | a time you reported a worse number than people hoped for |
| Safety or privacy over speed | FinSentinelAI's local, on-prem design choice |

Map the index against each target company's stated values or principles (see the [company guides](../companies/README.md)) and fill any theme without a story.

## Numbers you must verify

> [!WARNING]
> Earlier drafts of the **Sujanix OCR story** and the **SpaceDrift story** exist in git history (`06-behavioral/04-your-story-bank.md`, commit `de97e92`). They were written from memory rather than from your records, and they contain details (vendor baselines, timelines, test-set sizes, a "1.2M-image" dataset, "~200 req/s") that disagree with the résumé. Treat every number and detail in them as **unverified**. Rewrite each story from your own records before using it.

Before telling any story, fill this table. An interviewer who knows the domain will ask how each number was measured.

| number | story | source record (log, report, email, dashboard) | verified? |
|---|---|---|---|
| OCR exact match before → after | Sujanix OCR | | |
| test-set size and how it was split | Sujanix OCR | | |
| inference cost change and uptime | Sujanix OCR | | |
| number of clients, projects, contractors | SpaceDrift | | |
| revenue or pricing figures (share only if comfortable) | SpaceDrift | | |
| assessment-time reduction | ECHOME | | |

Keep this table consistent with the [résumé open issues](resume/guide.md#open-issues-verify-before-sending-anything). If a number can't be sourced, tell the story without it and say so: "I don't have the exact figure; it was roughly X, measured by Y."

## Follow-up drill

Interviewers dig three levels deep. For each story, practice answering these without notes:

1. **Measurement:** How exactly did you measure the result? What was the baseline? How big was the test set?
2. **Alternatives:** What else did you consider? Why not that?
3. **Counterfactual:** What would have happened if you had done nothing?
4. **Others:** Who disagreed, and what did they argue? What did you learn from them?
5. **Regret:** Knowing what you know now, what would you change?
6. **Transfer:** Where have you applied that lesson since?

## Values and mission prep

Write two paragraphs for each question, in your own words, then pressure-test them with someone who disagrees:

1. What is the most important unsolved problem in AI safety, and why?
2. What would you refuse to build, and where is the line?
3. Describe a time you traded capability or speed for safety or privacy. (FinSentinelAI's local, on-prem design is a genuine example.)
4. What worries you about current AI development? What gives you hope?
5. Why this lab specifically? (Reference its actual research and published policies; read them first. The [company guides](../companies/README.md) list starting points.)

What strong answers look like: specific (a named paper, policy or incident, not "alignment is important"), personal (connected to something you did or decided), nuanced (you can state the strongest opposing view), and honest (you say what you don't know). See [module 09](../curriculum/09-interpretability-and-safety.md) for the technical background.

Common failure modes:
- **Rehearsed platitudes** that could be pasted into any company's interview.
- **Flattering the company** instead of engaging with its choices, including ones you have questions about.
- **False certainty** about timelines or risks. Calibrated uncertainty ("I think X, but I'd change my mind if Y") reads as maturity.
- **No personal evidence:** values claimed but never acted on.

> [!TIP]
> For question 2, prepare one concrete example of work you would decline and one borderline case you would accept with conditions. Interviewers learn more from where you draw a line in a hard case than from the easy refusals.

## "Why this company?"

Template: what they are doing that you care about (a specific product, paper or team) → evidence that you have already engaged with it (you used it, reproduced it, built on it, with a link) → what you would contribute in the first 6 months.

## Questions to ask them

- **Recruiter:** team, level, process and timeline, location, visa policy.
- **Hiring manager:** what success looks like at 6 months; the team's hardest current problem; how research becomes product.
- **Engineers:** how experiments are run and reviewed; the eval culture; on-call.
- **Skip-level:** the team's strategy for the next two years.

Never ask anything you could have read on their site.
