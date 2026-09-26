# 07: Team and operating cadence

A small AI company runs on a weekly loop: look at a handful of honest numbers, pick three priorities, ship, talk to customers, and write down what you learned.
This file covers co-founders and first hires, the services trap, the weekly rhythm, a weekly metrics dashboard template with a hypothetical filled example, and a written-update template.

**Contents:** [Co-founder](#co-founder) · [Hiring](#hiring) · [The services trap](#the-services-trap) · [Weekly operating rhythm](#weekly-operating-rhythm) · [Weekly metrics dashboard](#weekly-metrics-dashboard) · [Written weekly update](#written-weekly-update) · [Focus](#focus) · [Failure modes](#failure-modes)

## Co-founder

A technical founder usually benefits most from a co-founder who owns selling and customer relationships. Test the fit by working together on a real paid pilot for 1–3 months before committing: you learn how the other person handles a difficult customer, a missed deadline and a disagreement about scope, which a dozen coffees will not show you.

Put these in writing before either of you quits a job: roles and final decision rights per area, equity split, vesting (with a cliff), what happens if one person leaves, and how deadlocks are broken. See [06 § Founder essentials](06-fundraising-and-structure.md#founder-essentials).

> [!TIP]
> Ask a prospective co-founder to run three customer calls alone and send you their notes. The quality of those notes predicts more than their résumé does.

## Hiring

- **First hires ship end to end and talk to customers.** A first engineer who can read a customer's error report, find the failure in the eval set and deploy the fix is worth more than a specialist who needs a spec.
- **Hire for clear writing and honest measurement.** In an AI product, someone who reports "the new prompt looks better" without an eval run will cost you customers.
- **Contractors are fine for bounded work** (a connector, a UI, a labeling batch) with a written scope, a fixed deliverable and an IP-assignment clause. Keep the core in-house: the eval suite, the data pipeline, the models that differentiate you, and the customer relationship.
- **Structured interviews.** A work sample close to the real job (debug a failing extraction on real-looking documents, read an eval report and say what to change) beats puzzle questions.

## The services trap

Founders with a services background can fund a product from services revenue. The trap is that services revenue is immediate and certain while product revenue is later and uncertain, so every week the rational short-term choice is more services, and the product never ships.

Symptoms: every customer gets a custom build; the "product" is a folder of per-client scripts; you cannot name the reusable component shipped this month; revenue grows only with hours worked.

Countermeasures:
1. **A fixed time budget:** for example, services capped at a set number of hours per week, product work protected in calendar blocks. Write the cap down and review it monthly.
2. **A dated product milestone:** "a self-serve pilot that a customer can run without us, by <date>". If the date slips twice, decide explicitly whether you are running a services business (a legitimate choice) or a product company.
3. **Only take services work that generalizes:** it uses the product, adds a reusable connector or document type, or produces data you are allowed to learn from. Say no to the rest, politely and early.
4. **Price services to fund product time,** not to win the deal. Underpriced services consume the hours the product needed.

## Weekly operating rhythm

- **Monday (60 min):** review the [dashboard](#weekly-metrics-dashboard); pick the top three priorities for the week; name the one number each priority should move.
- **Daily:** ship something; one customer touchpoint (a call, a review session, a support thread read end to end).
- **Friday (45 min):** send the [written update](#written-weekly-update) to yourself, co-founders, advisers and later investors.
- **Monthly:** runway and "default alive" check ([Paul Graham](https://paulgraham.com/aord.html)); kill or double down on each experiment; re-run the [absorption test](04-moats-and-flywheels.md#the-absorption-test).
- **Quarterly:** re-score the problem against the [rubric](01-problem-selection.md#score-your-ideas) with what you now know; update the unit-economics table from real invoices.

## Weekly metrics dashboard

Keep it to one screen. Every metric has an owner and a definition written next to it, so the number means the same thing every week. Numbers below are a **hypothetical** example for an early document-AI company.

| area | metric | definition | this week | last week | 4-week trend |
|---|---|---|---|---|---|
| Usage | documents processed | completed extractions, all customers | 18,400 | 16,900 | rising |
| Usage | active customers | processed ≥ 1 document this week | 6 | 6 | flat |
| Quality | field accuracy on the golden set | exact match, fixed set, current production build | 98.1% | 97.9% | rising |
| Quality | share sent to human review | documents below the confidence cut-off | 14% | 15% | falling |
| Quality | median time from new failure to fix (days) | first customer report to deployed fix | 6 | 8 | falling |
| Economics | COGS per document | model + infra + review, from invoices and logs | ₹3.60 | ₹3.75 | falling |
| Economics | gross margin | (revenue − COGS) / revenue | 70% | 69% | rising |
| Sales | qualified pipeline | accounts at "discovery done" or later | 11 | 9 | rising |
| Sales | pilots live / converted this quarter | signed pilots; conversions to recurring contracts | 3 / 1 | 3 / 1 | flat |
| Customers | conversations this week | calls or review sessions with users or buyers | 5 | 3 | — |
| Cash | monthly burn / runway (months) | net cash out; cash ÷ burn | ₹5.8L / 14 | ₹5.9L / 14 | flat |

Rules for the dashboard:
- **No vanity metrics.** Sign-ups, demo requests and social followers belong elsewhere.
- **Trend over level.** Early numbers are small and noisy; look at four-week direction, and don't celebrate a single good week.
- **Every red metric gets a named action** in the Monday priorities or an explicit "accepting this for now".

## Written weekly update

```text
Week of <date>
Headline: <one sentence: the most important thing that happened>
Numbers: <link to dashboard>; the 2–3 that moved, and why
Shipped: <features, fixes, eval improvements>
Learned: <from customers and data; include anything that contradicts our plan>
Pipeline: <new qualified accounts, pilots started or converted, lost deals and why>
Next week's top 3: <priority → the number it should move>
Blocked / asks: <specific help needed from advisers or investors: an intro, a decision, a review>
```

> [!TIP]
> Write the "Learned" section first. If it is empty two weeks running, you are building without talking to customers.

## Focus

Say no to custom work that doesn't generalize, to model tweaks the evals don't call for, to conferences before you have customers, and to partnerships that produce press releases but no users. Every "yes" to something outside the top three priorities is a quiet "no" to one of them.

## Failure modes

- **Hiring to escape selling.** Early sales are the founder's job; a salesperson can scale a motion that works, not discover one.
- **Contractors owning the core.** When they leave, the know-how leaves.
- **Metrics that change definition** each week, so trends are meaningless.
- **Skipping the weekly update** when the week went badly, which is when it is most useful.
- **Founder exhaustion.** The [sustainability routines](../../career/sustainability.md) apply to founders at least as much as to job seekers.
