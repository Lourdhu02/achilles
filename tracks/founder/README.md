# Track: AI founder

Most AI startups die because nobody pays for what they built, not because the model was weak. This track is a method for the other path: find a painful, paid-for workflow, prove demand with money before code, build eval-first, and keep the unit economics honest.
It is written for a technical founder in India who has already sold ML services and shipped document AI in production, and it works for anyone with a workflow they know from the inside.

**Contents:** [Your starting edge](#your-starting-edge) · [Sequence](#sequence) · [Templates](#templates) · [Decision frameworks](#decision-frameworks) · [Rules](#rules) · [Failure modes](#failure-modes-that-kill-technical-founders) · [Lessons from post-mortems and essays](#lessons-from-post-mortems-and-essays) · [Decision gate](#decision-gate-put-it-in-roadmapmd)

> [!NOTE]
> Every number in this track that is not a formula is **hypothetical** and labelled as such. The numbers exist to show the arithmetic. Replace them with your measured token counts, current provider prices and real customer quotes before you decide anything.

## Your starting edge

You have two things most first-time AI founders lack.

| asset | why it matters | how to use it |
|---|---|---|
| 16 months running a sole-proprietorship ML services business (scoping, pricing, delivery, paid contractors) | You have already sold, delivered and collected. You know what a real buyer objection sounds like. | Services revenue can fund discovery; past clients are your first interview list. Watch the services trap ([07](07-team-and-operating.md#the-services-trap)). |
| Document AI / OCR shipped to a production customer | Document workflows are a large, boring, paid-for category where accuracy, audit trails and on-prem deployment matter more than a chat UI. | Start discovery in document-heavy workflows you have seen up close. Your eval and quantization experience is a cost and privacy advantage. |
| Core-AI depth from this repo's labs | You can decide build vs buy vs train from evidence, not fashion. | Use the [build vs buy vs train](02-ai-product-engineering.md#build-vs-buy-vs-train) framework and [lab 03](../../labs/03_napkin_math/README.md) for costs. |

What you probably lack, and should plan for: a repeatable sales motion, a co-founder who owns selling, and experience raising money. The track addresses each.

## Sequence

Each step ends in an artifact. Do not start the next step until the artifact exists.

| # | step | file | artifact that ends the step | rough time (part-time) |
|---|---|---|---|---|
| 1 | Earn the insight: pick 3 candidate problems, score them | [01](01-problem-selection.md) | 3 scored idea sheets | 2–4 weeks |
| 2 | Validate: 15–20 problem interviews, then ask for money | [01](01-problem-selection.md#customer-discovery-interview-script) | interview notes plus a signal ladder; 1+ paid pilot or LOI | 4–8 weeks |
| 3 | Build eval-first: frontier API baseline, your own models only if evals or economics demand it | [02](02-ai-product-engineering.md) | an eval set of 100–300 real examples and a baseline score | 2–4 weeks |
| 4 | Make the unit economics work at small scale | [03](03-unit-economics.md) | a filled unit-economics table with measured costs | 1 week, then monthly |
| 5 | Design a moat that compounds with usage | [04](04-moats-and-flywheels.md) | a flywheel diagram with one measured velocity number | ongoing |
| 6 | Win the first 10 paying customers yourself | [05](05-go-to-market.md) | signed pilots with written success criteria | 3–9 months |
| 7 | Choose structure and funding | [06](06-fundraising-and-structure.md) | an incorporated entity (when needed) and a funding decision | when the evidence says so |
| 8 | Run the company on a weekly cadence | [07](07-team-and-operating.md) | a weekly metrics dashboard and written updates | ongoing |

## Templates

| template | where |
|---|---|
| Idea scoring rubric (with a worked example) | [01 § Score your ideas](01-problem-selection.md#score-your-ideas) |
| Customer-discovery interview script and note sheet | [01 § Interview script](01-problem-selection.md#customer-discovery-interview-script) |
| Eval set and failure taxonomy starter | [02 § Eval-driven loop](02-ai-product-engineering.md#the-eval-driven-loop) |
| Unit-economics table (per unit and per customer) | [03 § Template](03-unit-economics.md#unit-economics-table-template) |
| Pilot agreement outline with success criteria | [05 § Pilot agreement](05-go-to-market.md#pilot-agreement-outline) |
| Customer outreach message (annotated) | [05 § Outreach](05-go-to-market.md#outreach-to-buyers) |
| One-page pitch | [06 § One-page pitch](06-fundraising-and-structure.md#one-page-pitch-template) |
| Weekly metrics dashboard and written update | [07 § Dashboard](07-team-and-operating.md#weekly-metrics-dashboard) |

## Decision frameworks

| decision | where | the one-line version |
|---|---|---|
| Is this problem worth a company? | [01](01-problem-selection.md#score-your-ideas) | budget exists, pain is frequent, better models make you stronger, you have unfair access |
| Build vs buy vs train | [02](02-ai-product-engineering.md#build-vs-buy-vs-train) | buy (API) until evals, privacy or cost at *measured* volume force otherwise |
| Human review threshold | [02](02-ai-product-engineering.md#reliability-for-probabilistic-systems) | pick the confidence cut-off from a measured accuracy-vs-coverage curve |
| Price per seat, per usage or per outcome | [03](03-unit-economics.md#pricing) | price against the labor you replace; keep a floor above your cost |
| Is it a moat? | [04](04-moats-and-flywheels.md#the-absorption-test) | does the next frontier model make you stronger or redundant? |
| Bootstrap vs raise | [06](06-fundraising-and-structure.md#bootstrap-vs-raise) | raise only when money converts into growth you have already seen |
| Which company structure | [06](06-fundraising-and-structure.md#company-structures-in-india) | sole proprietorship to learn; private limited company before equity, ESOPs or investors |

## Rules

- **Keep your job until you have evidence:** signed pilots or paying customers, plus 12 or more months of personal runway. **Read your employment contract's IP, moonlighting and non-solicitation clauses before building anything on the side**, and never use your employer's data, code, hardware or customers.
- Work on a problem, not on "an AI startup". "We use model X" is not a company.
- Every week: talk to at least one customer, ship one thing, measure one number.
- Money is the only honest signal. Compliments, waitlists and "send me a deck" are not demand.
- Write your kill criteria before you start, not after you are attached.

> [!TIP]
> Put three recurring blocks in your calendar before anything else: one customer conversation, one eval run, one written weekly update. If those three happen every week for six months, you will know whether you have a company.

## Failure modes that kill technical founders

| failure mode | early symptom | countermeasure |
|---|---|---|
| Building before selling | a demo exists; no one has been asked for money | ask for a paid pilot in the second conversation with a qualified buyer |
| The wrapper | a competitor, or the model provider, ships the same feature | pass the [absorption test](04-moats-and-flywheels.md#the-absorption-test); go deeper into the workflow |
| Pilot purgatory | pilots extend, never convert | written success criteria, a price and a conversion date in every pilot ([05](05-go-to-market.md#pilot-agreement-outline)) |
| Margin collapse from human review | gross margin falls as volume grows | measure review minutes per unit weekly; make it a first-class metric ([03](03-unit-economics.md)) |
| The services trap | every customer needs custom work; the product never ships | time budget for services, a product milestone, and a "no" list ([07](07-team-and-operating.md#the-services-trap)) |
| Vibes-based quality | "it looks better" after each change | an eval suite in CI ([02](02-ai-product-engineering.md#the-eval-driven-loop)) |
| Selling to the wrong person | great meetings, no budget holder in the room | map user, champion, budget holder and blocker for every account |
| Founder burnout | skipped weekly reviews, sleep debt | the [sustainability routines](../../career/sustainability.md) apply doubly to founders |

## Lessons from post-mortems and essays

Read these before you write code. The lesson line is a summary; read the source for the argument.

| source | lesson for this track |
|---|---|
| Rob Fitzpatrick, [The Mom Test](https://www.momtestbook.com/) (book) | Ask about past behavior and money already spent, never about hypothetical futures. The basis of the [interview script](01-problem-selection.md#customer-discovery-interview-script). |
| Paul Graham, [How to Get Startup Ideas](https://paulgraham.com/startupideas.html) (2012) | The best ideas come from noticing problems in a world you live in; a small group that wants something urgently beats a large group that mildly wants it. |
| Paul Graham, [Do Things that Don't Scale](https://paulgraham.com/ds.html) (2013) | Recruit early users by hand and serve them manually; the concierge MVP in [01](01-problem-selection.md#concierge-mvp). |
| Paul Graham, [Default Alive or Default Dead?](https://paulgraham.com/aord.html) (2015) | At current growth and burn, do you reach profitability before the money runs out? Check it monthly ([07](07-team-and-operating.md#weekly-operating-rhythm)). |
| Martin Casado and Matt Bornstein (a16z), [The New Business of AI](https://a16z.com/the-new-business-of-ai-and-how-its-different-from-traditional-software/) (2020) | AI companies often have lower gross margins than classic SaaS because of compute and human-in-the-loop work, and can look more like services businesses. Written before LLM APIs, and still the right warning for [03](03-unit-economics.md). |
| Hamel Husain, [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/index.html) (2024) | Products that stall almost always lack a real evaluation system; build task-specific evals from your own failure analysis. |
| Yan, Bischof, Frye, Husain, Liu, Shankar, [What We've Learned From A Year of Building with LLMs](https://applied-llms.org/) (2024) | Tactical, operational and strategic lessons from practitioners; the strategy part argues that the model is rarely the moat. |
| Schluntz and Zhang (Anthropic), [Building effective agents](https://www.anthropic.com/research/building-effective-agents) (2024) | Prefer simple, composable workflows over autonomous agents until the task demands autonomy. |
| Glass Box Medicine, [Why I shut down my bootstrapped health AI startup after 7 years](https://glassboxmedicine.com/2026/02/21/why-i-shut-down-my-bootstrapped-health-ai-startup-after-7-years-a-founders-postmortem/) (2026) | A founder's post-mortem of a bootstrapped health AI company. The author's summary: building the AI was the smaller part of the challenge; workflow integration, sales and a sustainable business model were the larger part. |
| SimpleClosure, [State of Startup Shutdowns 2025](https://simpleclosure.com/blog/insights/state-of-startup-shutdowns-2025/) | Aggregate data on shutdowns from a company that runs wind-downs. Read it for base rates, not for your own odds. |
| Sam Altman, [Startup Playbook](https://playbook.samaltman.com/) | A compact guide to idea, product, execution and hiring from the YC perspective. |

> [!IMPORTANT]
> Post-mortems suffer from survivorship bias in reverse: they are written by founders who can explain their failure, often with hindsight. Use them to generate hypotheses about your own risks, then test those hypotheses with your own customers.

## Decision gate (put it in [ROADMAP.md](../../ROADMAP.md))

The roadmap has two gates: **Gate A** at the end of March 2027 (about month 6) and **Gate B** at the end of June 2027 (about month 9). At each, compare two bodies of evidence and move your hours toward the stronger one. It is fine to do a frontier-lab stint first and found later with better insight, savings and network.

**Pre-register the thresholds now.** Write the numbers below into your journal before Gate A. If you choose them after seeing the evidence, you will choose the ones that confirm what you already want.

| evidence | founder side | research-engineer side |
|---|---|---|
| strongest signal | signed paid pilots or paying customers | offers or final-round onsites |
| medium signal | LOIs with a price and a start date; a buyer introducing you to their budget holder | technical screens passed; referrals from people who know your work |
| weak signal | interviews where the pain was specific and costed; a concierge MVP someone paid for | recruiter replies; public write-ups with engagement from practitioners |
| not a signal | compliments, waitlists, "send me a deck", social media likes | course certificates, stars on a tutorial repo |

A decision sheet for each gate (copy into the journal):

```text
Gate: A / B            Date:
Pre-registered thresholds (written on <date>):
  Founder: ____ paid pilots or ____ LOIs with price by this gate
  Research engineer: ____ onsites or ____ offers by this gate
Evidence (founder):    pilots __  LOIs __  interviews __  paying customers __  MRR __
Evidence (research):   screens passed __  onsites __  offers __  write-ups __  merged PRs __
Personal runway (months at current spend): __
Decision: emphasize founder / research engineer / sequence (which first, until when)
What would change this decision before the next review:
```

No probabilities are invented here. The gate compares observed evidence against thresholds you set in advance.
