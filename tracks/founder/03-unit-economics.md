# 03: Unit economics of AI products

An AI product has a marginal cost per unit of work (tokens, GPU time, human review) that classic SaaS mostly does not. You must know that cost per unit, what you charge per unit, and what it costs to win and keep a customer, before you scale anything.
This file gives the formulas, a fully worked hypothetical example, a fill-in table, the cost levers in order of impact, and pricing guidance.

**Contents:** [Cost per task](#cost-per-task) · [Worked example](#worked-example-document-extraction) · [Template](#unit-economics-table-template) · [Levers](#levers-roughly-in-order-of-impact) · [Customer-level metrics](#customer-level-metrics-you-must-know-cold) · [Pricing](#pricing) · [Failure modes](#failure-modes)

> [!IMPORTANT]
> **All prices and volumes below are hypothetical.** Provider prices change several times a year and differ by model. Plug in current prices from the provider's pricing page and token counts you have *measured* on your own eval set. Assumed exchange rate for the examples: ₹85 = $1.

## Cost per task

$$
\text{cost per task} = \sum_{\text{calls}} \left(t_\text{in}\,p_\text{in} + t_\text{out}\,p_\text{out}\right) - \text{cache savings} + \text{retries} + \text{OCR and infra} + \text{human review}
$$

- **Reasoning models** bill their hidden "thinking" tokens as output tokens. Measure them per task; they can exceed the visible output many times over.
- **Retries** (schema failures, timeouts, validation re-runs) are a multiplier on model cost. Measure the retry rate.
- **Human review** is usually the largest cost that founders forget. It is (share of units reviewed) × (minutes per review) × (loaded cost per minute).

## Worked example: document extraction

A hypothetical product extracts fields from invoices for a mid-sized firm. The customer currently pays a BPO vendor ₹18 per document (hypothetical). You charge ₹12 per document.

**Baseline (first pilot).** Hypothetical API prices: $3 per million input tokens, $15 per million output tokens.

| cost line | calculation | per document |
|---|---|---|
| model: input | 6,000 tokens × $3/M = $0.0180 | |
| model: output | 800 tokens × $15/M = $0.0120 | |
| model: retries | $0.030 × 1.10 (10% of documents re-run) = $0.033 | ₹2.81 |
| OCR, storage, hosting, monitoring | measured, hypothetical | ₹0.40 |
| human review | 20% reviewed × 2 min × ₹5/min (₹300/hour loaded) | ₹2.00 |
| **COGS** | | **₹5.21** |
| **gross margin** | (12 − 5.21) / 12 | **56.6%** |

The 20% review share comes from the confidence threshold chosen in [02](02-ai-product-engineering.md#reliability-for-probabilistic-systems).

**After three levers.** Cache the stable 2,000-token prefix (instructions and schema), hypothetically billed at 10% of the input price when cached; tighten the output schema to 500 tokens; improve calibration and the review UI so 12% of documents go to review at 1.5 minutes each.

| cost line | calculation | per document |
|---|---|---|
| model: input | 4,000 × $3/M + 2,000 × $0.30/M = $0.0126 | |
| model: output | 500 × $15/M = $0.0075 | |
| model: retries | $0.0201 × 1.10 = $0.0221 | ₹1.88 |
| OCR, storage, hosting, monitoring | unchanged | ₹0.40 |
| human review | 12% × 1.5 min × ₹5/min | ₹0.90 |
| **COGS** | | **₹3.18** |
| **gross margin** | (12 − 3.18) / 12 | **73.5%** |

What the example teaches: the human-review line fell from ₹2.00 to ₹0.90, which is most of the improvement. Better calibration and a faster review UI moved margin more than any model choice. Measure review minutes per document every week.

## Unit-economics table template

Copy into a spreadsheet. Fill with measured numbers; mark any estimate with "(est.)".

| line | unit (per document / call / ticket) | per customer per month | source and date |
|---|---|---|---|
| price | | | contract |
| model cost (incl. reasoning tokens, retries) | | | provider invoice ÷ units |
| cache and batch discounts | | | provider invoice |
| OCR, retrieval, hosting, storage | | | cloud bill ÷ units |
| human review (share × minutes × cost/min) | | | review-tool logs |
| payment and support costs | | | |
| **COGS** | | | |
| **gross margin %** | | | |
| CAC (see below) | — | | sales costs ÷ new customers |
| CAC payback (months) | — | | CAC ÷ monthly gross profit per customer |
| monthly logo churn | — | | |

## Levers, roughly in order of impact

1. **Fewer tokens.** Send only relevant pages (classify or retrieve first), shorten prompts, use structured outputs, cache stable prefixes.
2. **Less human review.** Calibrate confidence, improve the review UI, fix the top failure cluster. See the worked example above.
3. **Routing and cascades.** A cheap model first; escalate on low confidence. Expected cost is $c_\text{small} + P(\text{escalate}) \cdot c_\text{large}$. Hypothetical: ₹0.40 + 0.3 × ₹1.88 = ₹0.96, against ₹1.88 for always using the large model. Only keep it if the evals show no quality loss on the combined system.
4. **Batch APIs** for work that can wait hours. Several providers price batch requests at a large discount (often around half); check current terms.
5. **Fine-tune a small model** once volume is high and the task is stable ([02 § Build vs buy vs train](02-ai-product-engineering.md#build-vs-buy-vs-train)).
6. **Self-host** when volume is steady, privacy requires it, or a small tuned model beats the API on your evals:

$$
\$/\text{M tokens} = \frac{\text{GPU \$/hour}}{\text{tokens/s} \times 3600} \times 10^6
$$

At a hypothetical $2/hour and 1,500 tokens/s aggregate throughput, that is $0.37 per million tokens at *full* utilization, and $1.23 per million at 30% utilization. Utilization is the business: an idle GPU still bills. Practice the arithmetic in [lab 03](../../labs/03_napkin_math/README.md).

## Customer-level metrics you must know cold

| metric | definition | healthy early-stage range (rule of thumb) |
|---|---|---|
| Gross margin | (revenue − COGS) / revenue | AI apps often run below classic SaaS; aim to improve it every quarter |
| CAC | sales and marketing spend (including founder time, pilot discounts, travel, integration work) ÷ new customers | — |
| CAC payback | CAC ÷ monthly gross profit per customer | under 12–18 months |
| LTV | monthly gross profit per customer ÷ monthly churn | — |
| LTV/CAC | | above 3 |
| Burn and runway | net monthly cash out; cash ÷ burn | know it to the week |
| Default alive? | at current growth and burn, do you reach profitability before cash runs out? ([Paul Graham](https://paulgraham.com/aord.html)) | yes, or a plan that makes it yes |

### Worked example: two segments (hypothetical)

Using the improved COGS of ₹3.18 and price of ₹12 (gross profit ₹8.82 per document):

| | mid-market customer | small business |
|---|---|---|
| documents per month | 20,000 | 1,500 |
| revenue per month | ₹2,40,000 | ₹18,000 |
| gross profit per month | ₹1,76,400 | ₹13,230 |
| CAC | ₹5,00,000 (4-month cycle: travel ₹50k, pilot discount ₹1L, integration ₹1.5L, founder time valued at ₹2L) | ₹60,000 (demos, onboarding) |
| CAC payback | 2.8 months | 4.5 months |
| monthly churn | 2% | 5% |
| LTV | ₹88,20,000 | ₹2,64,600 |
| LTV/CAC | 17.6 | 4.4 |

How to read it honestly: the mid-market numbers look excellent because the example assumes a large customer and low churn. With five customers you cannot measure churn at all, and one lost customer is 20% of revenue. Cap lifetime at 36 months for planning (LTV ₹63.5 lakh, LTV/CAC 12.7) and track customer concentration: no single customer above about a quarter of revenue if you can help it.

## Pricing

| model | fits when | risk |
|---|---|---|
| per seat | the value is per user (a copilot for analysts) | misaligned when AI replaces seats |
| usage-based (per call, per page) | cost scales with usage; buyers understand the unit | revenue is volatile; customers ration usage |
| **outcome-based** (per document processed, per ticket resolved) | automation that replaces labor; you can define "done" | disputes over what counts; you carry the accuracy risk |
| platform fee + usage | enterprise; on-prem deployments with fixed support costs | longer negotiations |

Guidance:
- **Price against the labor you replace,** not against your cost. If the customer pays ₹18 per document today, ₹12 saves them a third and leaves you margin. Your cost sets the floor; their alternative sets the ceiling.
- **Never run free pilots indefinitely.** Charge something, even at a discount, with success criteria and a conversion price written in ([05 § Pilot agreement](05-go-to-market.md#pilot-agreement-outline)).
- **Minimum commitments** (a monthly minimum or annual prepay) smooth revenue and filter out unserious buyers.
- **Reprice when models get cheaper?** Pass some savings on at renewal if competitors force it; keep the rest as margin. Do not cut prices proactively in the first year.
- **India:** quote exclusive of GST and state it on every proposal (most software services attract 18% GST as of September 2026; confirm with your chartered accountant). Enterprise buyers may deduct TDS on payments; plan cash flow for it.

## Failure modes

- **Unit cost measured on the demo, not on production traffic.** Real documents are longer, messier and retried more.
- **Forgetting review and support time** in COGS. It hides until volume grows, then margin collapses.
- **Reasoning-token surprises** after switching to a reasoning model. Measure before switching.
- **Pricing per seat for an automation product.** Success reduces seats, so your revenue falls as you deliver value.
- **Counting pilot revenue as recurring revenue.** It is not ARR until the contract recurs.
- **Ignoring customer concentration.** One large customer can be most of your revenue and all of your risk.

Background: Casado and Bornstein, [The New Business of AI](https://a16z.com/the-new-business-of-ai-and-how-its-different-from-traditional-software/) (a16z, 2020), on why AI gross margins and scaling differ from classic software.
