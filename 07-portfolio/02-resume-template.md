# ML Engineer Resume Template

The canonical structure for ML / AI engineer resumes targeting FAANG and top AI labs.

---

## Structure (one page)

```
┌─────────────────────────────────────────────────────────┐
│ NAME                                                     │
│ Tagline (1 line, 4-7 words)                             │
│ Location | Phone | Email | LinkedIn | GitHub | (extras) │
├─────────────────────────────────────────────────────────┤
│ PROFESSIONAL SUMMARY (3 lines max)                       │
├─────────────────────────────────────────────────────────┤
│ TECHNICAL SKILLS (4-5 lines categorized)                 │
├─────────────────────────────────────────────────────────┤
│ EXPERIENCE (most space — 50% of resume)                  │
│   Job 1 (most recent, 4-5 bullets)                       │
│   Job 2 (3-4 bullets)                                    │
│   Job 3 (2-3 bullets, only if relevant)                  │
├─────────────────────────────────────────────────────────┤
│ KEY PROJECTS (2-4 projects, 3 bullets each)              │
├─────────────────────────────────────────────────────────┤
│ ACHIEVEMENTS & CERTIFICATIONS (4-5 lines)                │
├─────────────────────────────────────────────────────────┤
│ EDUCATION (2-3 lines)                                    │
└─────────────────────────────────────────────────────────┘
```

---

## Section-by-section

### Header
- Name: 18-24pt, bold
- Tagline: 11-12pt, italic optional
- Contact: 10pt, separated by `|` or `·`

### Summary
3 lines, no fluff. Pattern:
1. Role + experience years + specialty
2. Specific technical depth signals
3. Current focus / unique trait

### Skills
Categorize. Don't list 100 keywords. Patterns:
- Languages
- ML/DL frameworks
- GenAI/LLM-specific tools (your edge)
- Domain-specific (CV, NLP, etc.)
- Infrastructure/MLOps

Keep keywords specific. "Python" yes, "Programming Languages" no. "PyTorch" yes, "Frameworks" no.

### Experience
Most-important section. Per role:

```
COMPANY NAME | Location                                Date – Date
Title
• [Action verb] [specific work] [measurable outcome]
• [Action verb] [specific work] [measurable outcome]
• [Action verb] [specific work] [measurable outcome]
• [Action verb] [specific work] [measurable outcome]
```

### Action verbs (use these, vary them)
Architected, Built, Designed, Developed, Engineered, Implemented, Optimized, Reduced, Improved, Led, Orchestrated, Deployed, Productionized, Shipped, Scaled, Established, Owned, Drove, Validated, Benchmarked, Refactored

### Bullet patterns

Pattern A: Action + What + Result
"Optimized inference latency by replacing FP16 with INT8 quantization via ONNX; reduced p99 from 280ms to 95ms with <1% accuracy loss."

Pattern B: Action + What + Why + Result
"Implemented FocalCTCLoss combining focal loss with CTC decoding to address class imbalance in character distribution; stabilized training and improved exact-match by 1.5%."

Pattern C: Owned X end-to-end
"Owned end-to-end OCR pipeline: data spec, model training, INT8 quantization, FastAPI/Lambda deployment, monitoring; achieved 97% exact-match in production at 99.9% uptime."

### Numbers
At least 50% of bullets should contain a specific number:
- % improvement
- $ saved
- Throughput / latency / scale
- Team size / users impacted

If you don't have numbers, the bullet is weaker. Retro-extract where possible.

### Key Projects
Use for projects:
- Personal projects (ECHOME etc.)
- OSS contributions worth showcasing
- Hackathons / academic projects (rarely — only if standout)

Per project: title, link, 3 bullets max.

### Achievements
- Kaggle ranks
- Certifications (only ones with brand value)
- Awards
- Publications (link to arXiv / paper)
- Talks (link to slides)

Keep tight.

### Education
- Degree, institution, location, graduation date
- GPA: only if 8.0+/10 or 3.7+/4.0
- Relevant coursework: only if entry-level / no work experience

---

## What NOT to include

- Photo (privacy + unconscious bias triggers)
- Hobbies (unless directly relevant)
- References ("available upon request" is implicit)
- Date of birth
- Marital status
- Religion
- Salary expectations or history
- Long paragraphs (use bullets)
- Skill ratings (5/5 PyTorch — meaningless)
- "Excellent communication skills" type fluff
- Buzzwords without substance (synergy, paradigm-shift)

---

## ATS optimization

- PDF format
- Standard fonts (Calibri, Arial, Helvetica)
- No images, no graphics
- No two-column layouts (some ATS misread)
- Use standard section headings ("Experience", not "My Journey")
- Mirror keywords from JD where they apply truthfully
- Save as: `FirstLast_Resume.pdf` (e.g., `LourduRaju_Resume.pdf`)

---

## Length

**ONE PAGE** for ≤7 YOE. Two pages only if:
- Significant publications/talks/patents
- 8+ years experience with distinct chapters

You're at 2 YOE → one page, period.

---

## Tools

### LaTeX (best ATS-compatibility + clean output)
- Overleaf with **AltaCV** or **JakeGutierrez** or **Awesome-CV** templates
- Reusable across versions
- Predictable rendering

### Markdown + Pandoc
- Write in markdown
- Convert via `pandoc resume.md -o resume.pdf`
- Clean, version-controllable

### Word
- Use only if you must
- Microsoft "Resume Designer" templates often ATS-friendly
- Save as PDF, never .docx

### Avoid
- Canva (cute but ATS often chokes)
- Pretty templates with graphics
- Multi-column complex layouts

---

## Pre-submit checklist

Before sending to ANY application:
- [ ] PDF
- [ ] One page
- [ ] All bullets have action verb + result
- [ ] At least 50% have numbers
- [ ] No grammatical errors (read aloud)
- [ ] No "we" in bullets
- [ ] All links work (test each)
- [ ] Keywords from JD appear (truthfully)
- [ ] Filename: `LourduRaju_Resume.pdf`
- [ ] Looks good in Chrome + Edge + Acrobat (test rendering)

---

## Iteration cadence

| When | Action |
|---|---|
| Today | V2 rewrite from V1 + feedback |
| End of Month 1 | Get external feedback; finalize V2 |
| End of Month 4 | V3 — add Flagship Project #1 |
| End of Month 8 | V4 — add Flagship #2, OSS work, Anthropic papers if any |
| Month 11 | V5 — final pre-application; per-target tweaks |

---

Next: [`03-github-optimization.md`](./03-github-optimization.md)
