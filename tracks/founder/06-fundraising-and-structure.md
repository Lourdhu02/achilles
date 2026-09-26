# 06: Structure, compliance and fundraising

How to choose a legal structure in India, what DPIIT recognition and the data-protection law mean for an AI startup, how to decide between bootstrapping and raising, and a one-page pitch template.
Most founders need a private limited company only when they are about to take investment, grant equity or sign contracts that require one; until then, evidence of demand matters more than paperwork.

> [!WARNING]
> **Not legal or tax advice.** Facts below were checked in **September 2026** and change often, sometimes by notification with immediate effect. Before acting, verify on the official sources (Startup India, the Ministry of Corporate Affairs, the Income Tax Department, MeitY) and consult a chartered accountant and a startup lawyer.

**Contents:** [Company structures](#company-structures-in-india) · [DPIIT recognition](#dpiit-startup-recognition) · [Tax notes](#tax-notes) · [Data protection](#data-protection-dpdp-act-2023) · [Bootstrap vs raise](#bootstrap-vs-raise) · [Funding paths](#funding-paths) · [Founder essentials](#founder-essentials) · [One-page pitch](#one-page-pitch-template) · [Failure modes](#failure-modes)

## Company structures in India

| structure | limited liability | can raise equity from investors | ESOPs | DPIIT-eligible entity type | fits when |
|---|---|---|---|---|---|
| Sole proprietorship | no (you are the business) | no | no | no | services work, concierge MVPs, learning what customers buy |
| One Person Company (OPC) | yes | not while it is an OPC; convert to a private limited company to add shareholders | check with your CA | check; the recognition rules list private limited companies, and an OPC is a form of private company | a solo founder who wants limited liability before raising |
| Limited Liability Partnership (LLP) | yes | rarely; venture investors generally do not invest in LLPs | no standard ESOP mechanism | yes | a services firm or a partnership with no plans to raise |
| **Private limited company** | yes | yes (equity, CCPS, convertible notes) | yes | yes | the default for a startup that will raise money or grant equity |
| US parent company with an Indian subsidiary ("flip") | yes | yes, familiar to US investors | yes, at the parent | the Indian subsidiary can be recognized | only when a specific investor or market requires it; get tax advice first |

Notes:
- A sole proprietorship cannot issue shares, grant ESOPs or take venture investment. Moving from one to a private limited company is normal; plan how contracts, IP and bank accounts will move across.
- On flips: several large Indian startups that once moved their parent abroad have since moved it back to India ("reverse flips"), often paying significant tax to do so. Understand the exit and tax path before you flip, not after.
- Foreign investment into an Indian company brings FEMA reporting and pricing rules. Your chartered accountant should handle filings for every round.

## DPIIT startup recognition

DPIIT (the Department for Promotion of Industry and Internal Trade) recognizes eligible entities as startups under the Startup India programme. A gazette notification of **4 February 2026** replaced the 2019 framework. As reported by the government and summarized by practitioners ([PIB release](https://www.pib.gov.in/PressReleasePage.aspx?PRID=2224069&reg=3&lang=1), [Startup India recognition page](https://www.startupindia.gov.in/content/sih/en/startupgov/startup_recognition_page.html)), the main criteria as of September 2026 are:

| criterion | general startup | deep-tech startup |
|---|---|---|
| maximum age from incorporation | 10 years | 20 years |
| turnover ceiling in any financial year | ₹200 crore | ₹300 crore |
| entity types | private limited company, registered partnership firm, LLP, and (newly) cooperative society | same |
| other conditions | working on innovation or improvement of products, services or processes, with potential for employment or wealth creation; not formed by splitting up or reconstructing an existing business | same, plus the deep-tech criteria in the notification |

Benefits listed by Startup India include self-certification under some labour and environmental laws, faster and cheaper patent and trademark filing, relaxations in public procurement, eligibility to apply for the income-tax holiday below, and access to government-backed funds. Recognition is free and applied for online; a sole proprietorship is not eligible.

> [!TIP]
> Apply for recognition soon after you incorporate a private limited company. It costs little, and procurement relaxations and the tax holiday application depend on it.

## Tax notes

- **Tax holiday (Section 80-IAC of the Income-tax Act, 1961):** DPIIT-recognized startups incorporated before **1 April 2030** can apply to an Inter-Ministerial Board for a deduction of 100% of profits for any 3 consecutive years out of the first 10. It needs a separate certificate, not just recognition.
- **"Angel tax"** (Section 56(2)(viib), which taxed share premiums above fair value) was abolished for all classes of investors by the Finance (No. 2) Act, 2024. Ask your CA whether anything still applies to rounds raised before the abolition took effect.
- **New Income-tax Act.** The Income-tax Act, 2025 replaced the 1961 Act from April 2026 and renumbers its provisions. Ask your CA for the current section references for the benefits above.
- **GST:** register when you cross the threshold or when customers require it; most software services attract 18% GST (confirm for your service).

## Data protection: DPDP Act, 2023

India's Digital Personal Data Protection Act, 2023 applies to digital personal data processed in India, and to processing outside India connected with offering goods or services to people in India. The **DPDP Rules were notified on 14 November 2025** with phased commencement ([PIB document](https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/nov/doc20251117695301.pdf)):

| phase | when | what switches on |
|---|---|---|
| 1 | 14 November 2025 | the Data Protection Board and its provisions |
| 2 | 12 months later (November 2026) | consent-manager registration and obligations |
| 3 | 18 months later (around 13 May 2027) | most substantive obligations: notice and consent, security safeguards, breach notification, data-principal rights, retention and erasure |

What it means for a document-AI startup, in plain terms (verify with a lawyer):
- **Role.** When you process documents for a business customer, that customer is usually the *data fiduciary* (it decides why and how data is processed) and you are a *data processor* acting under a contract. The fiduciary stays accountable and will push obligations onto you contractually: security, deletion, breach reporting, sub-processor approval.
- **Security safeguards and breaches.** Expect to show access controls, encryption, logging and a breach-response plan. The Act allows penalties of up to ₹250 crore for a failure to take reasonable security safeguards.
- **Retention and erasure.** Delete data when the purpose is served or the contract ends; build deletion into the product, including backups and derived training data you are not entitled to keep.
- **Children's data** needs verifiable parental consent; avoid it unless it is your core market.
- **Cross-border transfer** is allowed except to countries the government restricts by notification. Your model API provider's data location matters to regulated customers even when the law permits the transfer.
- **Sector rules stack on top.** Banks, NBFCs, insurers and securities firms have regulators (RBI, IRDAI, SEBI) whose outsourcing, localization and security requirements will be written into your contracts.

## Bootstrap vs raise

| question | points to bootstrap | points to raise |
|---|---|---|
| Is growth limited by money or by learning? | learning: you don't yet know what sells | money: you know the sales motion works and need people to run it |
| Do you have revenue? | yes, services or early customers fund the product | not yet, but the evidence (pilots, LOIs) is strong |
| Is the market winner-take-most? | no, many profitable niches | yes, speed decides the outcome |
| What does the product need up front? | modest compute; API-based | large data collection, model training or certifications before the first sale |
| Your personal situation | limited risk tolerance; you want control | you can live on a reduced salary for 2+ years and accept dilution and investor governance |

**Services-to-product** is a common middle path for founders with a services background: fund the product from services revenue. The risk is that services swallow the product ([07 § The services trap](07-team-and-operating.md#the-services-trap)). Mitigate with a fixed weekly time budget for product work and a dated product milestone.

### Worked example: sizing a pre-seed round (hypothetical numbers)

| monthly cost | amount |
|---|---|
| 2 founders at reduced salaries | ₹2,00,000 |
| 2 engineers | ₹2,40,000 |
| cloud and model APIs | ₹60,000 |
| legal, accounting, tools | ₹40,000 |
| sales travel and events | ₹60,000 |
| **monthly burn** | **₹6,00,000** |

Raise for the milestones that make the next round (or profitability) possible, plus a buffer: 20 months to reach them + 3 months of buffer = 23 months × ₹6 lakh = ₹1.38 crore, so ask for about ₹1.5 crore. State the milestones it buys (for example, "10 paying customers and ₹X monthly recurring revenue"), not just the runway.

## Funding paths

- **Bootstrap / services-to-product:** see above.
- **Angels and pre-seed:** investors bet on the team, the insight and early evidence. Instruments vary: equity and compulsorily convertible preference shares (CCPS) are common in India; the post-money SAFE is common in the US. Ask your lawyer which fits the investor and structure.
- **Accelerators:** Y Combinator and India-focused programmes. Valuable for discipline, a network and a deadline. Read each programme's current terms.
- **Government schemes:** Startup India lists seed funds and fund-of-funds programmes; check current eligibility.
- **Seed and Series A:** seed investors want early traction; Series A investors want a repeatable go-to-market motion and a path to strong unit economics ([03](03-unit-economics.md)).

## Founder essentials

- **Founder vesting** (typically 4 years with a 1-year cliff), even for a solo founder who may add co-founders later. In India this is commonly done through the shareholders' agreement rather than the ESOP plan; ask your lawyer.
- **A clean cap table:** no dead equity held by people who left, no informal promises of shares.
- **An ESOP pool** (commonly around 10% at the early stage) for the first hires.
- **A written co-founder agreement:** roles, equity, vesting, decision process, what happens if someone leaves.
- **IP assigned to the company,** including anything written before incorporation. Make sure **nothing is owned by your current or former employer**: check your employment contract before you write product code.

## One-page pitch template

```text
Company · one-line description (what it does, for whom, the outcome)

Problem       <who> spends <number> on <task>; <what goes wrong>. (numbers from interviews)
Why now       <capability that became good or cheap enough>, <regulation or market change>
Product       <what it does, in the customer's words>; demo link or two screenshots
Evidence      <paid pilots, conversion, retention, revenue>; one customer quote with permission
Market        bottom-up: <number of target accounts> × <annual contract value> = <SAM>
Business      price per unit, gross margin now and target, CAC payback (measured or labelled estimate)
Moat          what compounds with usage (data, integrations, evals, distribution)
Why us        domain access + core-AI depth + what you have already shipped
Team          founders, key hires, advisers
Ask           amount, instrument, and the milestones it buys in <N> months
```

Rules: every number has a source; estimates are labelled; no top-down market sizes ("1% of a ₹X trillion market"); one page means one page.

## Failure modes

- **Incorporating first, validating later.** Compliance costs money and attention every month.
- **Raising before evidence,** then spending the round on building instead of learning what sells.
- **Over-raising at a high valuation** that the next round cannot justify.
- **Informal equity promises** to early helpers, which later block a clean round.
- **Ignoring data-protection terms** until a bank's security review stops the deal. Prepare a security and data-handling document before your first regulated customer asks for it.
