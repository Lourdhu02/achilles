# Quarterly research brief: Google DeepMind and Google's AI teams

A reusable brief for refreshing this guide every quarter (January, April, July, October) and again before you apply. Follow it yourself or hand it to a research agent or a friend.
It lists every volatile fact in [README.md](README.md), [reading-list.md](reading-list.md) and [projects.md](projects.md), the value recorded in September 2026, and where to re-verify it.

## Contents

- [Rules](#rules)
- [Baseline: volatile facts as of September 2026](#baseline-volatile-facts-as-of-september-2026)
- [Open questions to research each quarter](#open-questions-to-research-each-quarter)
- [Method](#method)
- [Output format](#output-format)
- [Copy-paste brief](#copy-paste-brief)

## Rules

1. **Primary sources first**: Google, Google DeepMind, Google Research and Google Cloud pages, official GitHub repositories, arXiv, Hugging Face model cards. Candidate reports (Glassdoor, blogs, forums) are allowed only as labelled reports, never as fact.
2. **Date everything volatile** as "as of <Month Year>". If a fact cannot be re-verified, mark it "verify" or remove it. Never guess a number, a URL, a team name or a date.
3. **Keep headings stable.** Other files link to these anchors: in README.md `#at-a-glance`, `#where-core-ai-people-work`, `#what-they-value`, `#the-hiring-process`, `#signals-ranked`, `#90-day-plan`, `#tips-and-common-mistakes`, `#from-india`, `#sources`; in projects.md `#1-lab-05-gpt-in-jax-on-a-free-tpu` and `#2-a-rigorous-gemma-fine-tune`. Change body text, not these headings.
4. **Style**: GitHub Markdown, sentence-case headings, no emoji, tables for comparisons, alert blocks for warnings. Plain, direct English.
5. **Finish with** `python3 tools/check_links.py` from the repository root and fix any broken links in `companies/google/`.

## Baseline: volatile facts as of September 2026

Compare each row with what you find. Record changes in the output table.

| Area | Recorded value (Sep 2026) | Where to verify |
|---|---|---|
| GDM offices | London, Bay Area, Bangalore, Cambridge (US), Montreal, New York City, Paris, Tokyo, Toronto, Zurich; HQ London and Mountain View | https://deepmind.google/careers/ |
| Where GDM roles are posted | GDM careers page (Greenhouse job boards) and some on Google Careers | https://deepmind.google/careers/ · https://www.google.com/about/careers/applications/ |
| Bengaluru research leadership and naming | Manish Gupta, Senior Director at GDM, leads GDM research teams across India and Japan; Google Research keeps an India research lab page | https://research.google/teams/india-research-lab/ · GDM and Google Research blogs · recent Indian press |
| Google's described hiring process | Structured interviews scored on general cognitive ability, role-related knowledge, leadership, Googleyness; hiring committee decides | https://www.google.com/about/careers/applications/how-we-hire/ |
| Google SWE loop (reported) | Phone screen(s); ~4–5 onsite rounds (DSA, Googleyness and leadership, design at L5+, ML round for ML roles); committee; team matching | Recruiter; recent candidate reports (label them) |
| GDM RE loop (reported) | Recruiter call, a rapid "quiz" (maths, stats, CS, ML), two coding rounds with runnable code, ML rounds, team/manager conversations | Recruiter; recent candidate reports (label them) |
| Student Researcher Program | BS/MS/PhD students and some pre-academic researchers; ~12–24 weeks; per-cycle openings on Google Careers | https://deepmind.google/student-researcher-program/ |
| TPU Research Cloud | Free Cloud TPU quota after acceptance; rolling invitations; expected to share research publicly and give feedback; other Google Cloud services are billed | https://sites.research.google/trc/about/ · https://sites.research.google/trc/faq/ |
| Google AI Principles | Revised February 2025; three headings (bold innovation; responsible development and deployment; collaborative progress); earlier "applications we will not pursue" list removed | https://ai.google/principles/ |
| Frontier Safety Framework | Published and periodically updated by GDM; record the current version and date | GDM safety and responsibility pages |
| Latest open models | Gemma 4 (report July 2026, arXiv 2607.02770), Apache 2.0; E2B, E4B, 12B, 26B-A4B (MoE), 31B; up to 256K context; 262K vocabulary | https://huggingface.co/google · https://ai.google.dev/gemma |
| Latest Gemini generation | Gemini 2.5 report (arXiv 2507.06261) is the latest *arXiv* report used in the reading list; later generations documented in model cards | GDM model pages and model cards; arXiv |
| TPU numbers used in README | v5e 16 GB, 8.2e11 B/s, 1.97e14 bf16 FLOP/s; v5p 96 GB, 2.8e12 B/s, 4.59e14; v6e 32 GB, 1.6e12 B/s, 9.20e14 | https://jax-ml.github.io/scaling-book/tpus/ · https://cloud.google.com/tpu/docs |
| JAX stack | JAX (jax-ml), Flax (NNX API), Optax, Orbax, Grain, MaxText (AI-Hypercomputer org), Tunix (post-training: SFT, LoRA, DPO, PPO, GRPO), Pallas kernels in `jax/experimental/pallas/ops/tpu/` | The GitHub repos linked in README Sources |
| JAX on Windows | GPU builds Linux-only; use WSL2; check Blackwell (RTX 50-series) support in current `jaxlib` | JAX installation docs |
| Free compute | Kaggle and Colab offer GPU and TPU runtimes with quotas that change | Kaggle and Colab documentation |
| AI Research Foundations | Free eight-course GDM curriculum on Google Skills; labs on GitHub; India partnership (NASSCOM, IISc) announced July 2026 | https://www.skills.google/collections/deepmind · https://github.com/google-deepmind/ai-foundations |
| Visas | L-1 needs one continuous year with the employer abroad in the last three years; H-1B rules changed substantially in 2025 | USCIS; UK Home Office; [career/visa-and-relocation.md](../../career/visa-and-relocation.md) |
| Re-application wait | Reported as often 6–12 months | Recruiter only |

## Open questions to research each quarter

- Has any team moved between Google Research, Google DeepMind and Google Cloud, or been renamed? Update the org table in README "At a glance".
- Are there new core-AI openings in Bengaluru or Hyderabad (research engineer, research scientist, SWE-ML on Gemini, Cloud AI or ML infrastructure)? Note titles and teams, not counts that will be stale next week.
- Has the interview format changed: new assessments, in-person versus virtual final rounds for India-based candidates, a policy on AI tools during interviews, changes to team matching?
- New programmes: student researcher cycles, pre-doctoral or residency-style programmes, fellowships open to applicants in India. Only add one if an official page describes it as current.
- New Google or GDM papers worth adding to [reading-list.md](reading-list.md): new Gemini or Gemma reports, scaling, inference and safety papers. Keep the list at 15–30 items; replace rather than append.
- New or changed open-source tools that affect [projects.md](projects.md): JAX sharding APIs, Flax, Tunix, Pallas, Gemma releases and licences, Kaggle/Colab TPU availability.
- Any Gemma-themed Kaggle competitions currently open (a portfolio opportunity worth adding to the 90-day plan).

## Method

1. Read the three files end to end and list every sentence containing a date, number, name, URL or "as of".
2. Verify each against the source column above. Fetch official pages directly; use web search only to find the right official page.
3. Check every external link still resolves and still says what the guide claims.
4. For each open question, spend at most a few searches; if you cannot find a primary source, record "no primary source found" rather than a guess.
5. Draft edits. Keep changes minimal: update values and dates, add or remove rows, do not restructure sections.
6. Run `python3 tools/check_links.py`.

## Output format

Return a single report with these sections:

```text
## Changes
| File | Section | Old | New | Source URL | Checked on | Confidence |

## Unchanged but re-verified
<list of baseline rows confirmed, one line each>

## Could not verify
<fact, what you tried, recommendation: keep with "verify" or remove>

## Proposed additions
<new papers, programmes, projects; each with a primary source>

## Link check
<output summary of tools/check_links.py>
```

## Copy-paste brief

```text
You are refreshing the Google DeepMind / Google AI careers guide in companies/google/
(README.md, reading-list.md, projects.md) of the Achilles repository
(https://github.com/Lourdhu02/achilles). Today is <DATE>.

Goal: every volatile fact is correct and dated "as of <Month Year>".
Use companies/google/refresh-brief.md: check each row of the baseline table, answer the
open questions, and follow the rules (primary sources first; label candidate reports;
never invent numbers, URLs, names or dates; keep the listed headings unchanged; no emoji).

Reader: an ML engineer in Bengaluru with about two years of experience and an 8 GB GPU,
targeting research engineer or ML software engineer roles at Google DeepMind,
Google Research or Google Cloud AI.

Deliver: the report in the "Output format" section, then apply the edits,
then run python3 tools/check_links.py and fix broken links in companies/google/ only.
```
