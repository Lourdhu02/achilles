# Quarterly research brief: NVIDIA's AI teams

A reusable brief for refreshing [README.md](README.md), [reading-list.md](reading-list.md) and [projects.md](projects.md) once a quarter. Hand it to a research agent, or work through it yourself in about three hours.
Last full refresh: September 2026.

## Instructions

You are updating a guide for an ML engineer (about two years of experience, based in Bengaluru, India, with an 8 GB RTX 5060 laptop GPU) who wants a core-AI engineering role on one of NVIDIA's deep-learning software, inference, performance or research teams.

Rules:

1. **Primary sources first**: NVIDIA's careers site and job postings, NVIDIA's developer and technical blogs, NVIDIA Research pages, the GitHub repositories listed below, arXiv, and official documentation. Candidate reports (Glassdoor, Blind, Reddit, LeetCode discussions) may be used only for the shape of an interview loop and must be labelled "reported".
2. **Never invent** team names, interview questions attributed to specific people, numbers, quotes or URLs. If you cannot verify something, say so and leave it out.
3. **Date every volatile fact** ("as of <Month Year>") and give its source link.
4. **Report changes as a diff**: for each section of the README, list what changed, what is new and what to delete, with sources.

## Questions to answer

### Teams and products

1. What are the current names and scopes of NVIDIA's AI software projects: CUDA libraries (cuBLAS, cuDNN, CUTLASS and the CuTe DSL, any new tile-based CUDA programming model), TensorRT and TensorRT-LLM, NeMo and Megatron-Core (and any successor or companion libraries), Transformer Engine, Dynamo, Triton Inference Server? Any renames, merges or deprecations this quarter?
2. What are the latest releases of TensorRT-LLM, Megatron-LM, NeMo, CUTLASS, Transformer Engine and Dynamo, and what changed that a candidate should know (new quantization formats, new parallelism modes, new hardware support)?
3. Does TensorRT-LLM (and the others) officially support GeForce Blackwell (compute capability 12.0) and Windows or WSL2? Which quantization formats run on sm_120?
4. What has NVIDIA Research published this quarter on LLMs, efficient architectures, low-precision training and inference? What is the newest Nemotron release, and is there a technical report?
5. What new GPU architectures, whitepapers or programming-model changes (CUDA major versions) were announced, and which interview topics do they add?

### Hiring

6. Does NVIDIA's careers site describe the interview process, or provide candidate guidance? Quote and link it.
7. Search current postings (titles such as "Deep Learning Software Engineer", "Senior Software Engineer, TensorRT-LLM", "Deep Learning Performance Engineer", "Developer Technology Engineer", "Research Scientist", "Solutions Architect") in India and in the US. Summarise the recurring items under "What we need to see" and "Ways to stand out from the crowd". Which skills are rising or falling compared with last quarter?
8. Which of those roles are open in Bengaluru, Hyderabad, Pune or other Indian cities right now? Count them by city and area if possible.
9. What do recent (last 6 months) candidate reports say about loop structure for these roles? Label everything as reported and note sample sizes.
10. Are there new-college-graduate, intern or early-career programs for India this cycle? Deadlines?

### Learning resources

11. Which Deep Learning Institute courses are currently free, and which are relevant to CUDA, LLM inference or training? Any changes to certifications?
12. Upcoming GTC events (including in India) and whether sessions are free to watch on demand.

### India and mobility

13. Any news this quarter about NVIDIA's India offices (new campuses, headcount, partnerships with Indian companies or government programs)? Link primary sources or major news outlets only.
14. Any changes to US or European work-visa rules that affect an Indian engineer hired by or transferring within NVIDIA? Link official government sources.

### Reading list and projects

15. Are there new papers or official docs that should replace an item in the reading list (for example a newer Nemotron report, a new architecture whitepaper, a new FlashAttention version)? Propose at most three swaps, each with a reason.
16. Do all project instructions still work on an RTX 5060 and on Colab (tool versions, supported GPUs, Colab GPU types)? List anything that broke.

## Output format

```
## Summary of changes (5 bullets max)
## Section-by-section diff
### At a glance
- change: ... | source: <link> | as of: <Month Year>
...
## Unverified or conflicting claims
## Links checked (URL, status, date)
```

## Sources to check every time

- NVIDIA careers and its Workday job portal (search by title and location)
- [NVIDIA Research](https://research.nvidia.com/) and [NVIDIA on Hugging Face](https://huggingface.co/nvidia)
- NVIDIA Technical Blog (developer.nvidia.com/blog)
- Release pages: [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/releases), [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/releases), [NeMo](https://github.com/NVIDIA/NeMo/releases), [CUTLASS](https://github.com/NVIDIA/cutlass/releases), [Transformer Engine](https://github.com/NVIDIA/TransformerEngine/releases), [Dynamo](https://github.com/ai-dynamo/dynamo/releases)
- CUDA Toolkit release notes and the CUDA C++ Programming Guide (docs.nvidia.com/cuda)
- arXiv listings for NVIDIA-authored papers in cs.LG, cs.CL and cs.DC

## After the refresh

- Update the "as of" dates in [README.md](README.md).
- Run `python tools/check_links.py` from the repo root.
- Note in the commit message which facts changed and which could not be verified.
