# Quarterly refresh brief: Anthropic

A reusable research brief for refreshing the volatile facts in this folder once a quarter (next due: December 2026).
Give the prompt below to a person or to any research tool that can browse the web, then apply the result to [README.md](README.md), [reading-list.md](reading-list.md) and [projects.md](projects.md).

## What goes stale

The facts below were checked in **September 2026**. Each row names the section that depends on it and the primary source to re-check.

| Fact | Value as of September 2026 | Used in | Primary source |
|---|---|---|---|
| Latest model launches | Claude Fable 5.1 and Mythos 5.1 (1 Sep 2026); Claude Opus 5.5 (22 Sep 2026) | README, at a glance | https://www.anthropic.com/news |
| Research teams | Alignment, Economics, Interpretability, Societal Impacts, Frontier Red Team | README, where core-AI people work | https://www.anthropic.com/research |
| Job departments and example titles | See README table | README, where core-AI people work | https://www.anthropic.com/jobs |
| RSP version | v3.4, effective 8 July 2026 | README, what they value; reading list | https://www.anthropic.com/responsible-scaling-policy |
| Company values list | Seven values on the careers page | README, what they value | https://www.anthropic.com/careers |
| Interview tools and policies | Google Meet; Colab and CodeSignal; lookups allowed; reapply after 12 months; no internships | README, the hiring process | https://www.anthropic.com/careers |
| Candidate AI guidance | Last updated 10 July 2025 | README, the hiring process | https://www.anthropic.com/candidate-ai-guidance |
| Performance take-home | 2-hour limit; AI allowed for it; public repo | README; projects 9 | https://www.anthropic.com/engineering/AI-resistant-technical-evaluations |
| Visa sponsorship | "We sponsor visas and green cards for eligible roles" | README, from India | https://www.anthropic.com/careers |
| Fellows Program | Tracks, locations, eligibility (US/UK/Canada work authorization; no visa sponsorship), stipend and compute | README, signals | https://alignment.anthropic.com (latest Fellows post) and the jobs board |
| India presence | Bengaluru office opened 16 Feb 2026; India listings: Applied AI Architect (Bangalore, Mumbai), Customer Success (Bangalore) | README, at a glance and from India | https://www.anthropic.com/news, https://www.anthropic.com/jobs |
| Headcount | Not published; Revelio Labs estimate about 3,950 (March 2026) | README, at a glance | Company statements first; label any third-party estimate |
| Remote policy | Most staff in the Bay Area; flexibility for people further away | README, from India | https://www.anthropic.com/careers |

## The prompt

Copy everything in the block, fill in the date, and run it.

```text
You are refreshing a careers guide about joining Anthropic as a core AI / research engineer.
Today is <DATE>. The guide was last verified in <LAST MONTH YEAR>.

Rules:
- Use primary sources first: anthropic.com (careers, jobs, research, news, engineering,
  candidate-ai-guidance, responsible-scaling-policy, system-cards), alignment.anthropic.com,
  transformer-circuits.pub. Use press or candidate reports only when no primary source
  exists, and label them as secondary.
- For every fact, give: the value, the exact URL, the publication or "last updated" date
  shown on the page, and a short verbatim quote that supports it.
- If you cannot find or open a source, say "not verified" and do not guess.
- Report only changes and confirmations. Do not rewrite the guide.

Check each item:
1. Latest Claude model launches since <LAST MONTH YEAR> (names and dates).
2. Research teams named on anthropic.com/research, and any new or renamed team.
3. Job departments on anthropic.com/jobs; for AI Research & Engineering, Safeguards,
   Software Engineering - Infrastructure, Compute and Applied AI, list current example titles
   for pretraining, RL, inference, interpretability, alignment, safeguards, research
   infrastructure, applied AI and forward-deployed roles, with locations.
4. Every role listed in India (city, title, department). Any core research or engineering
   role in India is a major change: flag it.
5. Responsible Scaling Policy: current version, effective date, and a one-paragraph summary
   of changes since the last version.
6. Careers page: values list, interview format and tools, AI-use rules, reapplication
   policy, internships, visa sponsorship wording, remote and office expectations.
7. Candidate AI guidance: "last updated" date and any change in rules for applications,
   take-homes, preparation and live interviews.
8. Anthropic Fellows Program: whether applications are open, tracks, locations, eligibility
   and work-authorization rules, stipend, compute budget, cohort dates, and any published
   outcome statistics.
9. Engineering and research posts since <LAST MONTH YEAR> that describe hiring,
   evaluations, interpretability, alignment, RL environments or inference. Give title,
   date, URL and one line on why a candidate should read it.
10. Offices: any new office announcement, especially in India or Asia-Pacific.
11. Headcount: any company-stated figure; otherwise one labelled third-party estimate.
12. Whether each URL in the "Sources" section of the guide still resolves.

Output format:
- A table with columns: item | previous value | current value | source URL | date on page | quote | status (unchanged / changed / not verified).
- Then a list of proposed edits, each naming the file and section to change.
- Then up to five new reading-list candidates, each with year, URL and "what to extract".
```

## Source checklist

Tick each one; note the date shown on the page.

**Company pages**
- [ ] https://www.anthropic.com/careers
- [ ] https://www.anthropic.com/jobs (filter: AI Research & Engineering; Safeguards; Software Engineering - Infrastructure; Compute; Applied AI; and location India)
- [ ] https://www.anthropic.com/candidate-ai-guidance
- [ ] https://www.anthropic.com/company

**Research and policy**
- [ ] https://www.anthropic.com/research
- [ ] https://www.anthropic.com/responsible-scaling-policy (version, effective date, redline)
- [ ] https://www.anthropic.com/system-cards (latest system card)
- [ ] https://www.anthropic.com/constitution
- [ ] https://alignment.anthropic.com (Fellows posts and alignment research notes)
- [ ] https://transformer-circuits.pub (new interpretability papers and monthly updates)

**News and engineering**
- [ ] https://www.anthropic.com/news (model launches, offices, programs)
- [ ] https://www.anthropic.com/engineering (hiring, evals, infrastructure posts)

**India**
- [ ] Newsroom search for "India", "Bengaluru", "Mumbai"
- [ ] Jobs board filtered by India locations
- [ ] Current Indian and destination-country visa rules via official government sites (see [career/visa-and-relocation.md](../../career/visa-and-relocation.md))

**Secondary, label if used**
- [ ] Candidate-reported interview processes (for example Glassdoor, Exponent, IGotAnOffer). Use only for the shape of the loop, never for specific questions.
- [ ] Reputable press (for example Reuters, CNBC, Business Standard) for office and headcount news.

## Applying the result

1. Update every "as of" date you re-verified, and only those.
2. Change facts only where the refresh gives a primary source with a date and a quote.
3. If a team or role title disappeared, remove it from the README table rather than leaving a stale name.
4. Add at most five reading-list items per quarter; remove one for each you add if the list exceeds 30.
5. Run `python3 tools/check_links.py` from the repository root.
