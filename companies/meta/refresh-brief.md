# Meta AI guide: quarterly research brief

A reusable brief for refreshing the Meta guide ([README.md](README.md), [reading-list.md](reading-list.md), [projects.md](projects.md)) once a quarter. Work through it yourself or hand it to a research agent.
It lists the current baseline, the questions to re-check for each section, where to look, and the output format, so each refresh takes two to three hours and leaves an audit trail.

## Contents

- [When to run it](#when-to-run-it)
- [Rules for sources](#rules-for-sources)
- [Baseline to re-verify (as of September 2026)](#baseline-to-re-verify-as-of-september-2026)
- [Questions by section](#questions-by-section)
- [Where to look](#where-to-look)
- [Output and checklist](#output-and-checklist)
- [Copy-paste brief](#copy-paste-brief)
- [Refresh log](#refresh-log)

## When to run it

- **Quarterly:** early January, April, July and October.
- **Immediately** after any of these: a new Meta model release (open or closed), a reorganization or leadership change in Meta Superintelligence Labs (MSL) or FAIR, a change to Meta's published interview guidance, a PyTorch library being launched, paused or archived, or a US/UK visa rule change.
- **Before you apply:** re-check the hiring-process and "From India" sections regardless of the last refresh date.

## Rules for sources

1. **Primary first.** Meta's own pages (metacareers, Meta Newsroom, the AI blog, model cards, GitHub READMEs) outrank everything else. A model card or README is the best evidence of what exists and what license it has.
2. **Press only where there is no primary source**, mainly org charts, leaders, headcount and layoffs, which Meta rarely publishes. Label these "reported" and name the outlet.
3. **Interview-format reports** from prep sites and candidate write-ups are labelled "reported" and never presented as Meta's official process. Prefer patterns reported by several independent sources.
4. **Date every volatile fact** as "as of <Month Year>" and link it.
5. **If you can't verify it, cut it** or mark it "verify". Never carry forward a claim just because it was in the previous version.
6. **Don't record** individual compensation figures, leaked interview questions, or names of non-public employees.

## Baseline to re-verify (as of September 2026)

Each row is a claim the guide currently makes. Re-check every row; update the guide where the answer changed and note it in the [refresh log](#refresh-log).

| # | Claim in the guide | Source type | Re-check question |
|---|---|---|---|
| 1 | MSL has four groups announced Aug 2025: TBD Lab (frontier models), FAIR, Products and Applied Research, MSL Infra | Reported | Do these groups still exist under these names? Any merged, split or renamed? |
| 2 | Leaders: Alexandr Wang (Chief AI Officer, TBD Lab), Rob Fergus (FAIR), Nat Friedman (PAR), Aparna Ramani (MSL Infra) | Reported | Has anyone left or changed role? |
| 3 | An Applied AI engineering group formed in Mar 2026, about 6,500 engineers and PMs | Reported | Is it still separate from MSL? Who leads it, and does it hire ML engineers in India? |
| 4 | About 600 MSL roles cut in Oct 2025; TBD Lab spared | Reported | Any further cuts or hiring freezes since? |
| 5 | Muse Spark (Apr 8, 2026) is proprietary; first MSL model | Primary + press | New versions? Any change to its availability (API, weights)? |
| 6 | Muse Glimmer 30B (Aug 2026): open weights, Apache 2.0, distilled from Muse Spark | Primary (model card) | New open-weight releases? License changes? Any official statement on opening larger models? |
| 7 | Meta's "Advanced AI Scaling Framework" decides which models count as frontier | Primary (model card mention) | Is the framework document public? Link it and summarize the open/closed decision rule. |
| 8 | The last Llama release was Llama 4 (Scout, Maverick; Apr 2025) | Primary | Any further Llama release, or official end of the Llama brand? |
| 9 | Active: `pytorch/pytorch`, `pytorch/torchtitan` (TitanRL added Aug 2026), `pytorch/ao`. Not active: torchtune (stopped Jul 2025), torchforge (paused) | Primary (READMEs, issue #2883) | Check each README's status banner and recent commit activity. Any new library that merits a signal entry? |
| 10 | Official process: recruiter call, ~45-min technical screen, full loop of coding, design and behavioral; ML loops have up to six 45-min interviews | Primary (metacareers) | Has Meta's prep page changed the rounds or durations? |
| 11 | An AI-enabled coding round (~60 min, CoderPad, multi-file codebase) replaced one onsite coding round for some candidates from Oct 2025, mostly E4–E5 | Reported | Is it now standard, official, or dropped? Any official Meta page about it? |
| 12 | E4 is the usual target for 2–4 years of experience; team matching after the loop for many E3–E5 roles | Reported | Still accurate? Does MSL hire through the general pool or directly? |
| 13 | Research internships mainly target PhD students; no verified AI residency for non-PhD engineers | Unverified | Is any residency or early-career research program open? Link it if so. |
| 14 | India: offices in Gurugram, Mumbai, New Delhi, Hyderabad and Bengaluru; Bengaluru engineering office opened in 2025 (enterprise engineering, per press); no known MSL or FAIR research teams in India | Mixed | Current count and type of ML / AI roles listed in India on metacareers. Any MSL, PyTorch or Applied AI roles in India? |
| 15 | US: US$100,000 fee on certain new H-1B petitions introduced Sep 2025; lottery-based selection | Primary (US government) | Current fee, exemptions and selection method. Cross-check [career/visa-and-relocation.md](../../career/visa-and-relocation.md). |
| 16 | FAIR research sites include Paris and London; most MSL roles in Menlo Park, New York and the Seattle area | Mixed | Which locations appear on current MSL and FAIR postings? |

## Questions by section

Mirror the guide's sections so updates land in the right place.

**At a glance** (README): Has the timeline gained an entry? Is "What they build" still accurate for frontier models, research, frameworks, ranking and hardware? Add new rows at the bottom of the timeline table; don't rewrite history.

**Where core-AI people work**: Which job titles appear most often in current AI postings (Software Engineer, Machine Learning; Research Engineer; Research Scientist; AI kernels / MTIA)? Any new team names in postings that should be listed? Are the worked examples still based on the most relevant public model card? If Meta publishes a new open model card with architecture details, redo the KV-cache worked example with its numbers.

**What they value**: Any change to Meta's published company values? Any new official statement on open weights (a blog post or letter from leadership)? Update the open-weights bullet list with dated entries.

**The hiring process**: Re-read metacareers' SWE prep, technical-screen prep and ML full-loop prep pages. Record any change in rounds, durations or advice. Then check three or more independent recent candidate reports for the AI-enabled coding round and ML system design formats.

**Signals, ranked**: Re-rank only if the evidence changed, for example a library being paused, a new residency program, or a new PyTorch contribution program. Keep the list honest about what can be done from India.

**90-day plan**: Update only if a referenced library or tool changed (for example torchtitan's debug config name or the torch.compile logging flags).

**From India**: Search metacareers with location India; note role families and counts. Check for any Meta AI research presence in India. Re-check visa facts.

**Reading list**: Has Meta published a technical report for Muse Spark or a newer model? A new PyTorch systems paper (compiler, distributed, low precision)? Any FAIR paper that became central? Keep 15–30 items and five "read first"; drop the weakest item when adding one.

**Projects**: Do all the libraries and checkpoints named still exist and work (torchao config names, the LayerSkip and Llama 3.2 checkpoints on Hugging Face, torchtitan's debug model)? Does any project need a new model because licensing or availability changed?

## Where to look

| What | Primary source | Notes |
|---|---|---|
| Roles, locations, interview prep | [metacareers.com](https://www.metacareers.com/), especially the [SWE prep blog](https://www.metacareers.com/blog/preparing-for-your-software-engineering-interview-at-meta/), [technical-screen prep](https://www.metacareers.com/swe-prep-techscreen/), [ML full-loop prep](https://www.metacareers.com/ML-prep-onsite/) | Filter jobs by team keywords: "Superintelligence", "FAIR", "PyTorch", "MTIA", "machine learning" |
| Company announcements | [Meta Newsroom](https://about.fb.com/news/) | Reorganizations are often announced by internal memo first; check whether an official post followed |
| Models and research | [ai.meta.com/blog](https://ai.meta.com/blog/), Meta's Hugging Face organizations ([meta-models](https://huggingface.co/meta-models), [meta-llama](https://huggingface.co/meta-llama), [facebook](https://huggingface.co/facebook)) | Model cards give architecture, license, release date and authoring org |
| PyTorch libraries | [pytorch/torchtitan](https://github.com/pytorch/torchtitan), [pytorch/ao](https://github.com/pytorch/ao), [pytorch/pytorch](https://github.com/pytorch/pytorch), [meta-pytorch](https://github.com/meta-pytorch) | Read the README status banner and "Latest News"; check commit dates |
| Papers | arXiv and Hugging Face paper pages | Confirm authors are Meta before listing a paper as Meta's |
| Org and leadership | Established outlets (for example Reuters, CNBC, The Information) | Label as "reported"; prefer articles that quote the memo |
| Visa | Official US, UK and EU government immigration sites | Never rely on a prep site for visa facts |

Search queries that work well:

- `Meta Superintelligence Labs reorganization <month> <year>`
- `site:metacareers.com machine learning interview prep`
- `Meta AI-enabled coding interview <year>`
- `Meta open weights model <month> <year>`
- `Meta Bengaluru engineering hiring <year>`
- `torchtitan release`, `torchao release notes`

## Output and checklist

**Output.** Edit the three guide files in place, then append one row to the [refresh log](#refresh-log): date, what changed, and the sources used. If nothing changed in a section, update its "as of" date only after actually re-checking it.

**Checklist before you finish**

- [ ] Every row of the baseline table re-checked; changed rows updated in the guide and in the table.
- [ ] Every new or changed fact has a date and a link; press-based facts say "reported".
- [ ] The important note at the top of the README carries the new "as of" month.
- [ ] Signals and projects name only active libraries.
- [ ] Reading list still has 15–30 items and five "read first".
- [ ] `python3 tools/check_links.py` passes for `companies/meta/`.

## Copy-paste brief

```text
Task: refresh the Meta AI career guide in companies/meta/ (README.md, reading-list.md,
projects.md) for a core-AI engineering audience (ML engineers, many outside the US).

1. Read companies/meta/refresh-brief.md, especially the "Baseline to re-verify" table.
2. For each baseline row, find the current answer. Prefer primary sources: metacareers.com,
   about.fb.com/news, ai.meta.com/blog, Meta's Hugging Face model cards, and the GitHub READMEs
   of pytorch/torchtitan, pytorch/ao, pytorch/pytorch and meta-pytorch repositories.
   Use press only for org structure, leaders and headcount, and label it "reported".
3. Update the guide files where anything changed. Date every volatile fact
   ("as of <Month Year>") and link its source. Remove anything you cannot verify.
4. Check the reading list for new Meta technical reports or central papers, and check that
   every library and checkpoint named in projects.md still exists and is maintained.
5. Do not include compensation figures, leaked interview questions or private names.
6. Run python3 tools/check_links.py and fix broken links in companies/meta/.
7. Append a row to the refresh log in refresh-brief.md: date, changes, sources.
Report: what changed, what you could not verify, and what still needs a human check.
```

## Refresh log

| Date | Changes | Main sources |
|---|---|---|
| Sep 2026 | First version: MSL structure and 2025–26 reorganizations, Muse Spark (closed) and Muse Glimmer (open, Apache 2.0), PyTorch library status (torchtune stopped, torchforge paused, torchtitan and torchao active), official and reported interview process, India routes | Muse Glimmer model card; torchtitan, torchao and torchforge READMEs; torchtune issue #2883; metacareers prep pages; press coverage listed in the README's sources |
