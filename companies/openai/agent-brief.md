# Quarterly refresh brief: OpenAI

A reusable research brief for refreshing the volatile facts in this folder once a quarter (next due: December 2026).
Give the prompt below to a person or to any research tool that can browse the web, then apply the result to [README.md](README.md), [reading-list.md](reading-list.md) and [projects.md](projects.md).

## What goes stale

The facts below were checked in **September 2026**. Rows marked *secondary* rest on press reports, not OpenAI's own pages; upgrade them to primary sources when possible.

| Fact | Value as of September 2026 | Used in | Primary source to re-check |
|---|---|---|---|
| Latest frontier model | GPT-6 "Astra" rollout reported 3 Sep 2026 (*secondary*: CNBC) | README, at a glance | https://help.openai.com/en/articles/9624314-model-release-notes, https://openai.com/news/ |
| Latest system card | GPT-5 system card on arXiv (2601.03267); newer cards on openai.com | Reading list #15 | https://openai.com/news/ (filter: safety), arXiv search "OpenAI system card" |
| Corporate structure | Nonprofit OpenAI Foundation controlling a for-profit PBC (Oct 2025 recapitalization) | README, at a glance | https://openai.com/our-structure/ |
| Headcount | About 4,500 early 2026; target about 8,000 by end 2026 (*secondary*: FT via CNBC, 21 Mar 2026) | README, at a glance | Company statements; label any third-party estimate |
| Safety and research organization | Safety folded into research; head of Safety Systems departed; Preparedness risk areas distributed (*secondary*: July 2026 reports) | README, where core-AI people work | https://openai.com/safety/, https://openai.com/news/, job listings |
| Preparedness Framework version | v2 (April 2025); a revision reported August 2026 (*secondary*) | README, what they value | https://openai.com/safety/ and the framework PDF linked there |
| Interview stages | Résumé review → ~30 min recruiter call → skills-based assessment (pair coding, take-home or tests) → 4–6 h finals with 4–6 people over 1–2 days, virtual by default | README, the hiring process | https://openai.com/interview-guide/ |
| Residency | Six-month, full-time, paid, San Francisco; 2026 cohort applications closed | README, signals | https://openai.com/residency/ |
| Open challenges | Parameter Golf ran 18 Mar – 30 Apr 2026 (16 MB artifact, 10 min on 8×H100, FineWeb bpb) | README, signals; projects #7 | https://github.com/openai/parameter-golf, https://openai.com/news/ |
| Open-weight models | gpt-oss-120b and gpt-oss-20b (Aug 2025, Apache 2.0) | README; reading list #28; projects #9 | https://github.com/openai/gpt-oss, https://huggingface.co/openai |
| India offices | New Delhi (announced Aug 2025); Mumbai and Bengaluru planned for late 2026 (*secondary*: Feb 2026 reports) | README, at a glance and from India | https://openai.com/global-affairs/ and https://openai.com/news/ (search "India") |
| India roles | Applied AI Engineer (Delhi, Mumbai), Solutions Engineer (Delhi, Mumbai, Bangalore); no research roles seen | README, from India | https://openai.com/careers/search/ (filter by location) |
| Remote and visa policy | Core roles SF-based; sponsorship not stated publicly per role | README, from India | Each job listing; recruiter |
| Values on careers page | Last reviewed list emphasized AGI focus, intensity, scale, shipping | README, what they value | https://openai.com/careers/ |

## The prompt

Copy everything in the block, fill in the dates, and run it.

```text
You are refreshing a careers guide about joining OpenAI as a core AI / research engineer
(research engineer, member of technical staff, research scientist, applied AI engineer).
Today is <DATE>. The guide was last verified in <LAST MONTH YEAR>.

Rules:
- Primary sources first: openai.com (careers, careers/search, interview-guide, residency,
  charter, our-structure, safety, news, research index, system cards), model-spec.openai.com,
  help.openai.com model release notes, github.com/openai, huggingface.co/openai, arxiv.org.
- Use press or candidate reports only when no primary source exists. Label them
  "secondary", name the outlet and date, and prefer outlets with editorial standards
  (Reuters, FT, Bloomberg, CNBC, The Information, TechCrunch, major Indian business press).
- For every fact give: the value, the exact URL, the page's publication or "last updated"
  date, and a short verbatim quote (under 30 words) that supports it.
- If you cannot verify a fact, say "unverified" and do not guess. Never invent URLs,
  numbers, team names, interview questions or quotes.
- Report changes only, then a full table.

Tasks:
1. Models and products: list frontier models, open-weight models, and major products
   (ChatGPT, API, Codex, agents, Sora) released since <LAST MONTH YEAR>, each with its
   announcement URL and whether a system card was published (link it; note any arXiv ID).
2. Research areas: from the research index, system cards and recent papers, list the
   research areas with visible output this quarter (reasoning/RL, pretraining, safety,
   interpretability, CoT monitoring, evals, preparedness). Name 5-10 new papers or posts
   with URL, date and a one-line summary.
3. Organization: any reported reorganization of research, safety systems, preparedness,
   alignment or model behavior teams. Primary statements first; otherwise label secondary.
4. Hiring: re-read the interview guide and summarize every stage, duration and stated
   expectation. Note any changes vs the previous summary: <PASTE CURRENT TABLE>.
   Note any policy on AI tool use in interviews.
5. Programs: status of the Residency (open/closed, dates, eligibility, location, pay if
   stated), internships, fellowships, and any public competition or challenge
   (like Parameter Golf) that is framed as a hiring or talent channel.
6. Careers listings: count open roles in Research, Safety, Scaling/Infrastructure,
   Applied AI/Engineering and Go-To-Market. List every role located in India with title,
   city and URL. List roles marked remote and which countries they allow.
7. India: office status (New Delhi, Mumbai, Bengaluru), any India-specific initiatives,
   partnerships or data-centre plans, with dates and sources.
8. Structure and size: the current corporate structure and the most recent headcount
   figure with its source and date.
9. Governance documents: current versions and dates of the Charter, Model Spec and
   Preparedness Framework. Summarize what changed since <LAST MONTH YEAR>.
10. Open-source: new or archived repositories under github.com/openai relevant to
    research engineers (evals, agents, models, interpretability, benchmarks).
11. Reading list: propose up to 5 additions (OpenAI-authored, arXiv or openai.com, with
    ID or URL and year) and any items that are now superseded.

Output format:
- "Changes since last refresh": bullet list, each with source URL and date.
- A table with columns: fact | new value | old value | URL | page date | quote | primary/secondary.
- "Could not verify": bullet list.
```

## Source checklist

Tick each one during the refresh. Record the date you checked it.

**OpenAI primary**

- [ ] Careers overview and job search: https://openai.com/careers/ and https://openai.com/careers/search/
- [ ] Interview guide: https://openai.com/interview-guide/
- [ ] Residency: https://openai.com/residency/
- [ ] Charter: https://openai.com/charter/
- [ ] Structure: https://openai.com/our-structure/
- [ ] Safety hub (Preparedness Framework, safety evaluations): https://openai.com/safety/
- [ ] Newsroom and research index: https://openai.com/news/ and https://openai.com/research/index/
- [ ] Model release notes: https://help.openai.com/en/articles/9624314-model-release-notes
- [ ] Model Spec: https://model-spec.openai.com/
- [ ] GitHub: https://github.com/openai (sort by recently updated)
- [ ] Hugging Face: https://huggingface.co/openai
- [ ] arXiv: search for OpenAI-authored papers since the last refresh

**Secondary (label clearly)**

- [ ] Major business press for reorganizations, headcount and India plans
- [ ] Candidate reports on the interview process (weigh as anecdotes; note the date of each report)
- [ ] Levels.fyi for compensation (record date and level; never copy figures into the guide without a date)

## Applying the results

1. Update the dated facts in [README.md](README.md) (at a glance, where core-AI people work, the hiring process, signals, from India) and change every "as of" date you touched.
2. Add or retire items in [reading-list.md](reading-list.md); keep it between 15 and 30 items and keep exactly five "read first".
3. If a new open challenge, benchmark or open-weight model appears, check whether a project in [projects.md](projects.md) should change.
4. Run `python3 tools/check_links.py` from the repository root.
5. Update the "Value as of" column in the table above and set the next due date.
