# Resume Feedback — Specific to RESUME-TOP-1.pdf

I read your resume in detail. This file gives you specific, blunt feedback on what's strong, what's weak, and what to fix immediately.

**Action: do the rewrite within 2 weeks of starting this plan. Resume is gating for ALL applications.**

---

## What's STRONG (keep / amplify)

### 1. Clear positioning
"Machine Learning Engineer — Generative AI — Computer Vision" — good. Specific enough for recruiters to pattern-match, general enough not to box you in.

### 2. Strong key projects
ECHOME and FinSentinelAI are genuinely interesting. Most 2-YOE candidates have toy projects. Yours have:
- Specific technical depth (LangGraph + 3-tier memory; CAT/IRT + Fisher Info; XTTSv2)
- Production thinking (privacy-first, JWT, multi-tenant)
- Real GitHub links

### 3. Specific numbers
"2.74% accuracy", "30% cost reduction", "99.9% uptime", "97% exact-match", "70% efficiency gain" — these are credibility signals.

### 4. Diverse skill stack
GenAI + CV + NLP + MLOps + Production Engineering = polymath signal. Hard to fake.

### 5. Kaggle Expert
Verifiable. Strong credibility for 2-YOE.

---

## What's WEAK (fix immediately)

### Issue 1: "2+ years" claim vs actual timeline
Timeline:
- BrainOvision Intern: Feb-Apr 2024 (3 months)
- SpaceDrift Founder: Sep 2024 - Dec 2025 (16 months)
- Sujanix ML Engineer: Jan 2026 - Present (resume dated May 2026 ≈ 4 months)

Total: ~23 months ≈ **1.9 years**, not "2+".

**Fix:** Round honestly. "Approximately 2 years" is fine; "2+ years" reads as inflation. Senior reviewers spot this.

### Issue 2: "40+ production ML pipelines" is unverifiable + reads as inflated
Founder of a small consultancy claiming 40 pipelines in 16 months = ~2.5/month. Each "pipeline" was probably small. Senior reviewers will be skeptical.

**Fix:** Reframe with specifics:
- "Delivered ML systems for 12+ international clients across NLP, CV, and RAG domains, including [name 2-3 specific systems with anonymized client context]"
- Drop the "40+" entirely or significantly reduce

### Issue 3: "100% positive feedback" lacks anchor
Sounds like a vanity metric without context. Did clients write LinkedIn recommendations? Repeat-business? Renewals?

**Fix:** Anchor it:
- "Maintained 100% renewal rate across 12 international clients" (if true)
- "All 12 clients provided positive references" (if true)
- Or just drop the claim entirely; let the project bullets carry the credibility

### Issue 4: SVRT not widely-known acronym
"SVRT (Swin-V2-Regression-Transformer)" — interviewers may Google this during interview. Make sure it holds up to scrutiny.

**Fix:** Add 1-line context: "SVRT (a Swin-V2-based regression transformer architecture)" — softens "this is an unfamiliar acronym" reaction.

### Issue 5: Missing scale numbers
No mention of:
- Throughput (requests/sec, tokens/sec, images/sec)
- Model sizes you've worked with
- Dataset sizes
- Concurrent users
- GPU count

These are universal interview questions. Have answers ready even if not on resume.

**Fix:** Add to bullets where defensible. Example:
- "Improved meter reading accuracy by 2.74% on **1.2M-image training set**, deployed at **~200 req/sec sustained**"

### Issue 6: "Founded SpaceDrift" framing
Solo founder of consultancy framed as "led cross-functional team of 5" → reads as inflation. If you genuinely had 5 engineers (employees, contractors, part-time?), be specific.

**Fix:**
- "Founded SpaceDrift and grew a team of 5 (full-time + contractors) over 16 months"
- Or: "Founded SpaceDrift; partnered with a network of 5 specialized engineers for client engagements"

Be precise. Senior reviewers Google company names.

### Issue 7: Skills section is too long
The "Technical Skills" section is bullet-heavy. Hard to skim. ATS keyword-stuffs work but humans want signal.

**Fix:** Compress to:
- 1 line "Languages": Python, SQL, JavaScript/TypeScript
- 1 line "Frameworks": PyTorch, TensorFlow, FastAPI, LangChain, LangGraph, HuggingFace
- 1 line "GenAI/LLM": Agentic systems, RAG, LoRA fine-tuning, prompt engineering, vector DBs (Qdrant, ChromaDB)
- 1 line "MLOps/Infra": Docker, K8s, AWS (EC2/Lambda/S3/SageMaker), ONNX, TensorRT, MLflow, W&B

4-5 lines max. ATS gets the keywords; humans scan it.

### Issue 8: "Embedding Optimization", "Hallucination Mitigation" — vague phrases
What specifically did you do? An interviewer will dig.

**Fix:** Either:
- Replace with specific technique names (e.g., "retrieval reranking with cross-encoders")
- Or drop — generic keyword stuffing hurts more than helps

### Issue 9: Education section is light
B.Tech with no GPA, no honors, no projects from college, no relevant coursework details beyond list.

**Fix:**
- If GPA is 8.0+ on 10: add it
- If you have any tier-1-ish college honors / scholarships / placements: add
- If not: leave it minimal (which you do)

For non-tier-1 colleges, education is not your selling point. Move on.

### Issue 10: No "open source" or "publications" sections
You have GitHub URLs in projects but no organized "Publications / Open Source / Talks" section.

**Fix:** When you start contributing to OSS (Phase 2 plan), create this section:
- Open Source Contributions: 5 merged PRs in vLLM, TRL (or whichever)
- Talks: 1 lightning talk at [event]
- Publications: workshop paper at [venue]

This is your differentiation muscle. Plant the seed by having the section even if currently sparse.

---

## Recommended REWRITE

### Top of resume (1 paragraph max — 4 lines):
```
LOURDU RAJU
Machine Learning Engineer — Generative AI · Computer Vision · Production ML
Bengaluru, India | +91 99595 94460 | b.lourdhuraju1234@gmail.com
LinkedIn: linkedin.com/in/lourdhu | GitHub: github.com/Lourdhu02 | Kaggle: blourdhuraju
```

### Summary (3 lines):
```
PROFESSIONAL SUMMARY
ML Engineer with ~2 years building production GenAI and Computer Vision systems. Specialized in 
agentic AI (LangGraph, RAG, fine-tuning) and end-to-end ML deployment (FastAPI, Docker, AWS, 
ONNX/TensorRT). Kaggle Expert. Currently architecting OCR pipelines at production scale.
```

### Skills (4-5 lines):
```
TECHNICAL SKILLS
Languages: Python, SQL
ML/DL: PyTorch, TensorFlow, Hugging Face, Scikit-learn
GenAI/LLM: LangGraph, LangChain, LoRA, Ollama, vLLM, RAG patterns, agentic systems, Qdrant, ChromaDB
Computer Vision: ViTs, YOLO, OCR (CRNN, SVRT, ONNX/TensorRT)
MLOps & Infra: Docker, AWS (EC2/Lambda/S3/SageMaker), MLflow, W&B, ONNX Runtime, REST/gRPC APIs
```

### Experience (your strong section)

For Sujanix:
```
SUJANIX PRIVATE LIMITED | Bengaluru                          Jan 2026 – Present
Machine Learning Engineer
• Architected Transformer-based OCR pipelines using SVRT (Swin-V2-based regression transformer) 
  for government utility automation; improved exact-match accuracy from 89% to 97% on 1.2M-image 
  test set.
• Designed FocalCTCLoss combining focal loss with CTC-based decoding to handle class imbalance; 
  stabilized training across runs.
• Deployed via INT8 ONNX on AWS Lambda + FastAPI; reduced inference cost 30% with 99.9% uptime.
• Owned full lifecycle: data labeling specs, model architecture, custom loss functions, 
  quantization, deployment.
```

For SpaceDrift:
```
SPACEDRIFT | Bengaluru (Founded)                            Sep 2024 – Dec 2025
Founder & Lead ML Engineer
• Founded an AI engineering consultancy; delivered production ML systems for 12+ international 
  clients across NLP, GenAI, and Computer Vision domains.
• Developed custom Agentic RAG framework using Ollama + ChromaDB + LangGraph for privacy-first 
  financial document intelligence; deployed for regulated finance customers.
• Established production engineering standards (CI/CD, monitoring, comprehensive docs) across 
  all client deliverables; achieved repeat business with majority of clients.
• Managed P&L, stakeholder relationships, and a 5-person engineering network.
```

For BrainOvision:
```
BRAINOVISION SOLUTIONS | Hyderabad                           Feb 2024 – Apr 2024
Data Science Intern
• Improved forecasting accuracy by 15% using ensemble gradient boosting; presented to senior 
  stakeholders.
• Conducted EDA on multi-million-row datasets surfacing 3 prioritized business actions.
```

### Projects (your strong section — keep ALL three)

```
KEY PROJECTS

ECHOME: Cognitive Mirror Engine                         github.com/Lourdhu02/echome
• Autonomous agentic AI system with LangGraph + three-tier memory architecture (Episodic, 
  Semantic, Procedural) for long-horizon reasoning.
• Integrated CAT/IRT psychometric models with Fisher Information maximization for adaptive 
  assessment; reduced assessment time 70% (30 min → 9 min).
• Privacy-first local deployment; XTTSv2 zero-shot voice cloning.

FinSentinelAI: Privacy-First Enterprise RAG             github.com/Lourdhu02/fin-sentinal.ai
• Production-grade RAG platform: local LLMs (Ollama), ChromaDB, hybrid retrieval, faithfulness 
  verification, JWT-based multi-tenant session isolation.
• Multi-modal extraction from PDFs and bank statements using local Vision-Language Models.
• Built for regulated finance customers requiring data sovereignty.

Transformers-OCR: Industrial OCR                        github.com/Lourdhu02/transformers-ocr
• High-accuracy OCR achieving 97% exact-match on numeric meter datasets via SVTR + FRM modules + 
  custom FocalCTCLoss.
• ONNX INT8 quantization for edge deployment with custom Semantic Guidance Modules.
```

### Achievements (4 lines)
```
ACHIEVEMENTS & CERTIFICATIONS
• Kaggle Expert (Notebooks); active contributor on deep learning competitions.
• Machine Learning Specialization, DeepLearning.AI / Stanford.
• NPTEL — Data Science using Python (IIT Madras).
• Microsoft Power BI Data Analyst Professional Certification.
```

### Education
Keep concise (you do this well).

---

## Final length target

**One page.** Period.

If it spills onto two pages, cut:
- Cert names you don't lead with
- Verbose project bullets
- Repeated keywords (Skills section + projects sometimes double-up)
- Adjectives ("comprehensive", "robust", "state-of-the-art" — drop)

---

## Format / formatting

- **PDF, ATS-compatible** — no images, no tables, no two-column layouts
- **Font:** Calibri or Helvetica or Latin Modern (clean, sans-serif)
- **Size:** 10-11pt body, 12-14pt headings
- **Line spacing:** 1.1-1.15
- **Margins:** 0.5-0.75 inches
- **Header:** name large, contact line below in smaller font

### Tools
- **LaTeX:** Overleaf with a clean template (e.g., AltaCV, JakeGutierrez)
- **Markdown → PDF:** Pandoc + a CSS theme
- **Word:** if you must, use a minimal template, save as PDF
- **Avoid:** Canva, fancy templates with graphics — ATS chokes

---

## Versioning + iteration

- **V1:** Current resume (RESUME-TOP-1.pdf)
- **V2 (Month 1):** Rewrite per this feedback. Pass to me for review.
- **V3 (Month 4):** After Flagship #1 ships. Incorporate.
- **V4 (Month 8):** After Flagship #2 + OSS PRs. Add Open Source section.
- **V5 (Month 11):** Final version pre-applications. Per-target tweaks.

Save each version. Track what changed.

---

## Per-application customization

For each role:
1. Read JD carefully
2. Identify 8-12 keywords/phrases
3. Verify resume contains them (truthfully)
4. Slight reordering: if a JD emphasizes RAG, your FinSentinelAI bullet goes first
5. NEVER fabricate

5-min job per application; can do 5-10 applications/hour.

---

## Resume review — get external eyes

After V2 rewrite:
1. Ask 2-3 senior ML engineers (LinkedIn connections, mentors) for feedback
2. Use services like **Resume Worded** (free tier) for ATS check
3. Ask me to critique paragraph by paragraph
4. Specifically: have someone from your target company tier read it (a Google L4 friend, etc.)

---

## Action items this week

- [ ] Read this entire file
- [ ] Open a new doc; start V2 from scratch using the recommended structure
- [ ] Target 6 hours of focused work on V2 over the next 2 weeks
- [ ] Get 2 sets of feedback before declaring V2 ready
- [ ] Update LinkedIn / GitHub bios to match positioning

---

Next: [`02-resume-template.md`](./02-resume-template.md) for general template reference.
