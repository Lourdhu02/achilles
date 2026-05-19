# GitHub Optimization

Recruiters and interviewers look at your GitHub. Make sure they see what you want.

---

## What recruiters do on GitHub

1. **Click your profile** → see contribution graph, pinned repos, bio
2. **Click pinned repos** → judge README quality
3. **Glance at code** → judge style, naming, organization
4. **Look at commits** → recent activity? frequency?
5. **Check followers/stars** → social proof

You have ~60 seconds to convince. Optimize accordingly.

---

## Profile setup

### 1. Profile README (`github.com/Lourdhu02/Lourdhu02`)

Create a repo named exactly your username. The README there shows on your profile.

Content:
```markdown
### Hi, I'm Lourdu Raju 👋

ML Engineer in Bengaluru. I build production GenAI and Computer Vision systems.

**Currently:** Architecting Transformer-based OCR pipelines at Sujanix.
**Previously:** Founded SpaceDrift, delivering ML systems for international clients.

**Flagship projects:**
- 🧠 [ECHOME](https://github.com/Lourdhu02/echome) — Agentic AI with three-tier memory architecture (LangGraph + CAT/IRT)
- 💼 [FinSentinelAI](https://github.com/Lourdhu02/fin-sentinal.ai) — Privacy-first enterprise RAG platform
- 🔍 [Transformers-OCR](https://github.com/Lourdhu02/transformers-ocr) — High-accuracy industrial OCR (97% exact-match)

**Writing:** [yourblog.com](https://...)  
**Reach out:** [LinkedIn](https://linkedin.com/in/lourdhu) | b.lourdhuraju1234@gmail.com
```

Keep concise. Lead with your strongest signal.

### 2. Profile photo
- Professional headshot
- High-resolution, well-lit
- Smile, eye contact
- Single subject (just you)
- Background: neutral, not distracting

### 3. Bio
"ML Engineer | GenAI · Computer Vision | Kaggle Expert"

### 4. Pinned repos (max 6)
Pin in order of importance:
1. ECHOME
2. FinSentinelAI
3. Transformers-OCR
4-6. Your future flagship projects + best Kaggle notebooks

### 5. Contribution graph
- Green daily (at least 5 days/week)
- Make commits on your own repos, blog code, OSS PRs
- Quality over quantity (don't fake commits)

---

## Per-repo README quality

This is THE single biggest GitHub win. Most repos have weak READMEs. Yours should be strong.

### Template for a flagship repo README

```markdown
# Project Name
> One-line description that sells it

[![License](https://img.shields.io/...)]() [![Python](https://img.shields.io/...)]() [![Stars](https://img.shields.io/github/stars/Lourdhu02/repo)]()

## What it does
2-4 sentences. Why does this exist? What problem does it solve?

## Quick start
```bash
git clone https://github.com/Lourdhu02/repo.git
cd repo
pip install -r requirements.txt
python main.py
```

## Architecture
[Diagram — Excalidraw or Mermaid]

## Key technical decisions
- Decision 1: rationale + tradeoff
- Decision 2: rationale + tradeoff
- Decision 3: rationale + tradeoff

## Results / Benchmarks
[Tables or charts with numbers]

## How to extend
[1-3 examples for users to customize]

## Acknowledgements
- [Library / paper / inspiration]
- [Library / paper / inspiration]

## License
MIT (or whatever)
```

### What makes a great README

- **Hero image or diagram** at the top (Excalidraw screenshot is fine)
- **Quick-start that actually works** (test on a clean machine)
- **Architectural reasoning** — not just "what" but "why"
- **Benchmarks if applicable** — with reproducible code
- **GIF of the system in action** (huge engagement win)
- **Badges** (build status, license, stars) — looks professional
- **Star-worthy structure** — clean, scannable, well-formatted

### What kills a README

- Wall of code in the middle
- No setup instructions
- Outdated commands
- "TODO" everywhere
- Half-finished thoughts
- No explanation of decisions

---

## Code quality

### Naming
- Functions: `verb_object_form` — `parse_pdf_pages`, not `pdf_func`
- Classes: `NounPhrase` — `PDFParser`, not `pdf_p`
- Variables: meaningful — `user_query` not `q`

### Structure
- Files <500 lines
- Functions <50 lines
- Classes that do ONE thing
- Tests in `tests/` directory
- Configs in `configs/` not hardcoded

### Documentation
- Module docstrings explaining purpose
- Function docstrings explaining purpose + args + returns + raises
- Inline comments for non-obvious WHY (not WHAT)

### Don't
- Commit large files (>10MB; use Git LFS or external)
- Commit secrets / API keys (use `.env` + `.gitignore`)
- Commit raw datasets (use scripts to download)
- Push broken code
- "Final-final-v2" branches — use proper branches

---

## OSS contributions strategy

By Month 8, target 3-5 merged PRs to top-tier OSS.

### Where to contribute
- **vLLM** — inference engine, your area
- **TRL (Hugging Face)** — fine-tuning, your area
- **LangChain** / **LangGraph** — you use them; contribute
- **LlamaIndex** — RAG
- **transformers (HF)** — the big one; smaller PRs ok
- **sentence-transformers** — embedding models
- **PEFT (HF)** — LoRA / fine-tuning

### What to start with
- **Documentation fixes** — typos, clarifications
- **Bug reports with reproduction** — sometimes maintainers ask you to fix
- **Tests for existing code** — high acceptance rate
- **Small bug fixes** — find via "good first issue" labels
- **Examples in docs** — adding worked examples

### Then escalate to
- **New features (small)** — discussed with maintainers first
- **Performance optimizations**
- **New tutorials / cookbooks**

### Anti-patterns
- Massive untested PRs
- "I rewrote everything" PRs (rejected)
- Drive-by drive-by-style contributions without context

---

## Activity rhythm

### Daily (5 min)
- Push a commit (real, not fake)
- Could be: blog code, learning project, flagship-in-progress

### Weekly (1-2 hr)
- Polish 1 README
- Engage with OSS: issues, discussions, small PRs
- Update profile README if needed

### Monthly (4-6 hr)
- Audit pinned repos — still relevant?
- Bigger OSS PR
- Refactor / improve a flagship project

---

## Stretch targets (if you achieve these, you're TOP 1%)

- 100+ followers
- 50+ stars across your repos
- 5+ merged PRs in top-tier OSS
- 1+ project with 50+ stars (real signal)
- 1+ trending day on Hacker News / Reddit /r/LocalLLaMA

Realistic for you by Month 12 with focus.

---

Next: [`04-linkedin-strategy.md`](./04-linkedin-strategy.md)
