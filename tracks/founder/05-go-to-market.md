# 05: Go-to-market

The first ten customers come from founder-led sales: a narrow ideal customer profile, a hand-built list, specific outreach, and paid pilots with written success criteria and a conversion path.
This file gives the ICP template, an annotated outreach message, the pilot agreement outline, the pipeline you should track, India-specific notes, and the metrics that indicate product-market fit.

**Contents:** [ICP](#define-the-ideal-customer-profile) · [Sales process](#founder-led-sales-process) · [Outreach](#outreach-to-buyers) · [Pilot agreement](#pilot-agreement-outline) · [Pipeline](#pipeline-and-conversion) · [India vs global](#india-vs-global) · [Growth engine](#technical-founders-growth-engine) · [PMF metrics](#metrics-that-indicate-product-market-fit) · [Failure modes](#failure-modes)

## Define the ideal customer profile

```text
Industry / sub-segment:
Company size (employees, revenue, document volume):
Buyer role (budget holder) and user role (feels the pain):
Trigger event (why now: new regulation, volume spike, vendor contract ending, audit finding):
Existing spend we replace (BPO, staff, tool):
Deployment constraint (cloud OK / VPC / on-prem):
Disqualifiers (no budget holder reachable, volume too low, data can't leave premises and we can't deploy there):
```

Write it narrowly. "Mid-sized NBFCs processing over 5,000 loan applications a month, where the credit-operations head owns a BPO contract" is an ICP. "Financial services" is not.

## Founder-led sales process

1. **Build a list of 100 target accounts** that match the ICP. For each, name the user, the likely budget holder and one specific observation about their workflow.
2. **Reach out personally** (message below). Expect a low reply rate; the list size is how you compensate.
3. **Discovery call:** use the [interview script](01-problem-selection.md#customer-discovery-interview-script). Qualify: pain, budget, decision process, timeline, deployment constraints.
4. **Demo on their data,** not on your curated samples. Show the review UI and the error cases, not only the happy path.
5. **Propose a paid pilot** with written success criteria, a price and a conversion date (outline below).
6. **Run the pilot with weekly check-ins.** Sit with users while they work. Fix the top failure cluster each week.
7. **Convert:** at the end of the pilot, present results against the criteria and the pre-agreed conversion terms.

> [!TIP]
> Ask for the budget holder by the second meeting: "Who else would need to be comfortable with this for you to roll it out?" A champion without a budget holder is a friend, not a deal.

## Outreach to buyers

```text
Subject: <their workflow> at <company>

Hi <name>,                                                     
I noticed <specific, public observation: a job posting for data-entry staff,         [1]
a regulator's new reporting rule, their volume from an annual report>.
We help <ICP phrase> cut <task> time by <measured result from a pilot, or            [2]
"we're running pilots to measure how much"> using <one plain-English line on how>.
Would a 20-minute call to understand how your team handles <task> today be useful?   [3]
I'm not pitching on that call; I want to learn how it works for you.                 [4]
<name> · <one line of credibility: shipped document AI in production for <sector>>   [5]
```

Why each line exists:
1. **Specific observation** proves you are not mass-mailing, and shows you understand their world.
2. **Outcome, not technology.** Buyers purchase reduced cost or risk, not models. Never claim a result you have not measured; "we are running pilots to measure it" is honest and fine.
3. **Small, specific ask.** Twenty minutes, one topic.
4. **Lowers the cost of saying yes.** It also commits you to a discovery call, which is what you need at this stage.
5. **One line of credibility,** true and checkable.

Send at most one follow-up, a week later, with something new (a short observation or a relevant result).

## Pilot agreement outline

A pilot is a small paid contract, not a free trial. Put these sections in writing, reviewed by a lawyer before the first one is signed. **Not legal advice.**

```text
1. Parties and dates         start date, end date (4–8 weeks), named owners on both sides
2. Scope                     workflow, document types, volume cap, languages, what is out of scope
3. Success criteria          measurable, agreed before start, measured on data the customer selects:
                             e.g. "≥ 99.0% field accuracy on auto-approved invoices, ≥ 70% of
                             invoices auto-approved, median turnaround < 10 minutes, measured on
                             the last 2 weeks of pilot volume"
4. Measurement method        who labels the ground truth, sample size, how disputes are settled
5. Customer obligations      data access by day X, a named reviewer, weekly 30-minute check-in,
                             feedback within N business days
6. Price and payment         pilot fee (may be credited against the first year), payment terms,
                             taxes (GST) extra
7. Conversion terms          if the criteria are met: annual contract at <price>, start date,
                             volume tiers; decision due within N days of the pilot's end
8. Data and privacy          purpose limitation, where data is stored and processed, retention and
                             deletion at the end, sub-processors (e.g. model API providers),
                             breach notification, whether corrections may be used to improve models
9. Security                  access controls, audit logs, deployment (cloud / VPC / on-prem)
10. IP                       customer owns its data and outputs; you own the product and
                             general improvements
11. Liability and warranty   caps, no guarantees beyond the success criteria, human review remains
                             the customer's responsibility where agreed
12. Termination              either side with notice; what happens to data
```

> [!WARNING]
> Pilots without a price and a conversion clause become "pilot purgatory": they extend, they generate compliments, and they never become revenue. If a customer refuses to write down success criteria, they are not ready to buy.

## Pipeline and conversion

Track every account through these stages in a spreadsheet, weekly:

| stage | exit criterion |
|---|---|
| target | matches ICP; contact identified |
| contacted | outreach sent |
| discovery done | pain, budget holder and process known |
| demo on their data | customer data processed; results shown |
| pilot proposed | written proposal sent |
| pilot signed | signed and paid (or invoiced) |
| pilot complete | results measured against criteria |
| converted | annual or recurring contract signed |

Compute the conversion rate between each pair of stages once you have enough accounts (a few dozen). The stage with the worst conversion is your bottleneck; fix it before adding more accounts at the top.

## India vs global

- **Indian enterprise and government:** long cycles, formal procurement, relationships, and a strong preference for on-prem or local deployment. Central-government purchases often run through the Government e-Marketplace (GeM). Startup India lists public-procurement relaxations for DPIIT-recognized startups; check the current terms for any tender before relying on them. Experience delivering to a government customer is directly useful here.
- **Global SMB and mid-market:** faster, more self-serve and content-led, priced in USD. Many Indian AI founders build in India for customers abroad; this needs cross-border payment setup and attention to the customer's data-protection law.
- **Pick one motion first.** Running enterprise sales and self-serve at once splits a small team's attention.

## Technical founder's growth engine

Public writing and open source: benchmarks, eval write-ups, a free tool that solves one slice of the problem (for example, an open-source document-type classifier). This builds trust with technical buyers and attracts hires. Your research write-ups from this repo double as marketing; see [career/public-presence.md](../../career/public-presence.md).

## Metrics that indicate product-market fit

In order of importance:
1. **Retention and usage depth** by cohort: are customers processing more documents in month 3 than in month 1?
2. **Expansion:** net revenue retention above 100% (existing customers pay more over time).
3. **Organic pull:** referrals and inbound requests you did not generate.

Revenue from pilots that don't convert is not product-market fit. Neither are logos from free pilots.

## Failure modes

- **An ICP so broad that no message fits anyone.**
- **Demoing on curated data.** The customer's worst documents will appear in week one of production; show them in the demo.
- **Discounting to win the first deal** so deeply that the conversion price later feels like a price increase.
- **Selling custom work** to close a deal. See [07 § The services trap](07-team-and-operating.md#the-services-trap).
- **Stopping outreach during a big pilot.** Pipeline takes months to rebuild; keep one outreach block every week.

Further reading: Paul Graham, [Do Things that Don't Scale](https://paulgraham.com/ds.html).
