# More labs and paths

The five largest employers of core-AI engineers have their own guides: [Anthropic](anthropic/README.md), [OpenAI](openai/README.md), [Google](google/README.md), [Meta](meta/README.md) and [NVIDIA](nvidia/README.md). This guide covers the rest of the map: other frontier and open-weight labs, the open-source and inference companies, India-based labs, academic research-assistant paths, and how to judge a startup offer.
Everything about teams, offices and hiring here is a snapshot **as of September 2026**. Re-check the linked careers pages before you act on any of it.

## Contents

- [Comparison table](#comparison-table)
- [How to read this guide](#how-to-read-this-guide)
- [Microsoft AI and Microsoft Research](#microsoft-ai-and-microsoft-research)
- [Apple](#apple)
- [xAI](#xai)
- [Mistral AI](#mistral-ai)
- [DeepSeek](#deepseek)
- [Qwen (Alibaba)](#qwen-alibaba)
- [Moonshot AI](#moonshot-ai)
- [Hugging Face](#hugging-face)
- [Cohere](#cohere)
- [Allen Institute for AI (Ai2)](#allen-institute-for-ai-ai2)
- [Together AI and the inference companies](#together-ai-and-the-inference-companies)
- [India-based labs and IndiaAI](#india-based-labs-and-indiaai)
- [Academic groups in India and research-assistant paths](#academic-groups-in-india-and-research-assistant-paths)
- [Startups as a path](#startups-as-a-path)
- [Sources](#sources)

## Comparison table

As of September 2026. "Openness to remote/India" is about core-AI roles specifically, not the company's whole job board.

| Organization | Focus | Where they hire | Openness to remote / India | Strongest signal to show |
|---|---|---|---|---|
| Microsoft AI (MAI) | Its own frontier models (MAI family) and Copilot; a superintelligence team since November 2025 | US and UK hubs | Low for model roles; India roles are mostly product engineering | Large-scale training or post-training work with measured results |
| Microsoft Research India | Systems for LLMs, algorithms, ML for societal impact | Bengaluru | High: in India, with a fellows program for new graduates | A systems paper or project reproduced end to end (e.g., an inference scheduler) |
| Apple (foundation models) | On-device models, MLX; next-generation server models built with Google's Gemini | Mostly US, some Europe | Low for model roles; India offices are mostly other engineering | Efficient inference: quantization, small models, on-device latency |
| xAI (part of SpaceX since 2026) | Grok models, very large training clusters | US (Bay Area, Memphis), London | Low; in-person culture | Exceptional shipped engineering, stated concisely |
| Mistral AI | Open-weight and commercial models, enterprise deployment | France first; also UK, US, Germany, Singapore | Medium; a Bengaluru centre was reported to be under discussion in 2026 | Open-source work on inference, fine-tuning or evals of their models |
| DeepSeek | Efficient frontier models (MoE, MLA, RL for reasoning) | Hangzhou and Beijing, China | Very low for non-Chinese residents | Not a realistic hire for most readers; study and reproduce their work |
| Qwen (Alibaba) | Open-weight model family (dense, MoE, multimodal) | China; Alibaba also hires in Singapore and elsewhere | Low for the core team | Contributions to the Qwen ecosystem; multilingual evaluation |
| Moonshot AI | Kimi models, long context, Muon at scale | China | Very low; little evidence of India hiring | Study their open reports; reproduce Muon results |
| Hugging Face | Open-source libraries, open models and datasets | Remote-first, many countries | High: remote is normal | Merged pull requests to their libraries; public models and write-ups |
| Cohere (merging with Aleph Alpha) | Enterprise and sovereign LLMs; Cohere Labs does open multilingual research | Canada, US, UK, Germany and other hubs | Medium; Cohere Labs community is open to anyone | Multilingual research or Aya contributions; enterprise RAG systems |
| Ai2 | Fully open models (OLMo, Tülu, Molmo) | Seattle, some remote | Medium; predoctoral program is US-based | Reproducible training or post-training research with released code |
| Together AI and inference companies | Inference speed, kernels, serving platforms | US, some remote | Medium to high for strong kernel and systems engineers | Kernel work with benchmarks; contributions to vLLM or SGLang |
| Sarvam AI and IndiaAI-funded labs | Indian-language LLMs (Sarvam-30B and -105B are open-weight), speech, sovereign AI | Bengaluru and other Indian cities | High: in India | Indic tokenizer or eval work; speech or translation models |
| IISc, IITs, AI4Bharat | Research, Indic datasets and models | Bengaluru, Chennai, Mumbai, Delhi and others | High | A reproduced paper and a clear research question emailed to the right faculty member |

## How to read this guide

For each organization you get: what they do now, where they hire, how people get noticed, tips, and links. Two rules apply everywhere.

1. **Signals travel.** A merged pull request to vLLM helps you at Together AI, Mistral, NVIDIA and every inference team. An honest evaluation of Indic performance helps at Sarvam, Cohere Labs and Google Research India. Pick signals that serve several targets. The labs in this repo map to them: kernels ([lab 06](../labs/06_attention_kernels/README.md)), inference ([lab 07](../labs/07_kv_cache_sampling/README.md), [lab 13](../labs/13_quantization/README.md), [lab 14](../labs/14_speculative_decoding/README.md)), post-training ([lab 11](../labs/11_dpo/README.md), [lab 12](../labs/12_grpo/README.md)), evals ([lab 15](../labs/15_eval_stats/README.md)).
2. **Location decides more than skill.** Many labs here hire model researchers only in a few cities. If you want to stay in India, prioritize the rows marked high, and treat the others as targets for open-source collaboration, which can later become a relocation offer. See [career/visa-and-relocation.md](../career/visa-and-relocation.md).

> [!IMPORTANT]
> Job boards change weekly. "Where they hire" below is what the organization's careers pages and public announcements showed as of September 2026. Always filter the live board by location before you plan around a lab.

## Microsoft AI and Microsoft Research

**What they do now.** Microsoft has two relevant groups.

- **Microsoft AI (MAI)**, led by Mustafa Suleyman since March 2024, trains Microsoft's own models (the MAI family: text, voice and image models, first shown publicly in August 2025 and expanded in 2026). In November 2025 Suleyman announced the MAI Superintelligence Team, and in March 2026 Microsoft hired former Ai2 CEO Ali Farhadi and several senior Ai2 and University of Washington researchers into it. The goal is less dependence on OpenAI's models. Model roles sit mainly in the US and UK.
- **Microsoft Research (MSR)** runs labs worldwide. The Phi small-model series came from MSR ([Phi-4 technical report](https://arxiv.org/abs/2412.08905)). **MSR India** in Bengaluru has done well-known LLM-systems work: [Sarathi-Serve](https://arxiv.org/abs/2403.02310) (chunked prefills for throughput-latency trade-offs, the idea behind [lab 07](../labs/07_kv_cache_sampling/README.md)'s chunked prefill), [vAttention](https://arxiv.org/abs/2405.04437) (KV-cache memory management without PagedAttention) and [Vidur](https://arxiv.org/abs/2405.05465) (an LLM-inference simulator).

Microsoft also employs many engineers in India (the India Development Center in Hyderabad, Bengaluru and Noida) on Azure AI and Copilot product work. These are applied roles, but some sit close to model serving.

**Where they hire (as of September 2026).**

| Path | Where | Who it's for |
|---|---|---|
| MSR India Research Fellow | Bengaluru, 1–2 years | BTech/MTech graduates (not PhDs). The 2026 cycle opened 7 January and closed 15 February 2026, with a July start |
| MSR India researcher / research SDE | Bengaluru | Usually PhD for researcher; strong systems engineers for research SDE roles |
| Microsoft AI model teams | US and UK | Experienced training, post-training and infrastructure engineers |
| Azure AI / Copilot engineering | Hyderabad, Bengaluru, Noida and elsewhere | Applied ML and systems engineers |

**How people get noticed.** MSR India fellows are selected through a formal application. Beyond that, what gets attention is a strong project in the lab's research area. An inference-systems researcher will read a blog post that reproduces Sarathi-Serve's throughput-latency curve on a single GPU and explains where it differs from the paper. For Microsoft AI, the path is the same as at any frontier lab: demonstrated large-scale training or post-training experience.

**Tips.**

- For the fellows program, name one or two MSR India projects in your statement and say concretely what you would build. Generic interest in "AI research" does not stand out.
- MSR India's systems group publishes at venues such as OSDI, ASPLOS and MLSys. Read their recent papers before applying and implement a small piece of one.
- An internal transfer from an India engineering role to a research group is possible but not guaranteed. Don't join a product team only for that.

**Links.** [MSR India lab](https://www.microsoft.com/en-us/research/lab/microsoft-research-india/) · [Research Fellows program](https://www.microsoft.com/en-us/research/academic-program/research-fellows-program-at-microsoft-research-india/) · [Microsoft careers](https://careers.microsoft.com/) · [Sarathi-Serve](https://arxiv.org/abs/2403.02310) · [Microsoft AI](https://microsoft.ai/)

## Apple

**What they do now.** Apple's foundation-model work powers Apple Intelligence. The [2025 tech report](https://arxiv.org/abs/2507.13575) describes two models: a roughly 3B-parameter on-device model that uses KV-cache sharing and 2-bit quantization-aware training, and a server model built on a Parallel-Track Mixture-of-Experts (PT-MoE) transformer that runs on Apple's Private Cloud Compute. (The 2024 predecessor is [Apple Intelligence Foundation Language Models](https://arxiv.org/abs/2407.21075).) Third-party developers can call the on-device model through the Swift Foundation Models framework, which supports guided generation, tool calling and LoRA adapters. Apple also maintains [MLX](https://github.com/ml-explore/mlx), an array framework for Apple silicon.

The organization changed in 2026. In December 2025 Apple announced that its AI chief, John Giannandrea, would retire and that Amar Subramanya (previously at Google, where he worked on Gemini, and briefly at Microsoft) would become vice president of AI, leading Apple Foundation Models, ML research, and AI safety and evaluation. In early 2026 Apple and Google announced that the next generation of Apple Foundation Models would be based on Gemini models and Google's cloud technology. On-device efficiency, adaptation and evaluation remain Apple's own work; how much frontier-scale pretraining Apple still does in-house is less clear, so ask in interviews.

**Where they hire (as of September 2026).** Model research and engineering roles are mostly in Cupertino, Seattle and New York, with some in Europe. Apple has large offices in Hyderabad and Bengaluru, but core model roles there are rare; filter [jobs.apple.com](https://jobs.apple.com/) by location and team ("Machine Learning and AI") to check.

**How people get noticed.** Apple publishes less than the other labs and is quieter about hiring. Signals that fit: work on small, efficient models (quantization, distillation, on-device latency), contributions to MLX or MLX-based model ports, and papers or projects on privacy-preserving ML.

**Tips.**

- Build the efficiency story: quantize a small model with [lab 13](../labs/13_quantization/README.md), measure quality with confidence intervals ([lab 15](../labs/15_eval_stats/README.md)), and report latency and memory on real hardware.
- Contribute to MLX or port a model to it. It's open source and the maintainers are Apple engineers.
- Expect a strong emphasis on confidentiality in interviews. Talk about what you built and measured; don't speculate about Apple's internals.

**Links.** [Apple Machine Learning Research](https://machinelearning.apple.com/) · [Apple foundation models tech report 2025](https://arxiv.org/abs/2507.13575) · [MLX on GitHub](https://github.com/ml-explore/mlx) · [Apple Jobs](https://jobs.apple.com/)

## xAI

**What they do now.** xAI builds the Grok models and trains them on its Colossus clusters in Memphis. It released open weights for Grok-1 (2024) and Grok-2 (2025). In February 2026 SpaceX acquired xAI in an all-stock deal, and Grok is now developed as SpaceX's AI division; SpaceX went public in June 2026. The group has a reputation for small teams, very fast iteration and long hours.

**Where they hire (as of September 2026).** Mostly in person in the Bay Area and Memphis, plus London. Remote core-model roles are uncommon. After the merger, AI roles may be listed on SpaceX's careers pages as well as [x.ai/careers](https://x.ai/careers); check both.

**How people get noticed.** By evidence of exceptional work, stated briefly. Lead with the single most impressive thing you built, with numbers (speed-up, scale, cost). Public engineering at scale (training runs, kernels, large systems) counts more than credentials.

**Tips.**

- Prepare one paragraph about your best piece of work: the problem, what you did, the measured result. No adjectives.
- Expect practical coding and systems questions; see [coding-interviews.md](../tracks/research-engineer/coding-interviews.md).
- Weigh the culture honestly. Intense, in-person environments suit some people and burn out others; see [career/sustainability.md](../career/sustainability.md).

**Links.** [xAI careers](https://x.ai/careers) · [Grok-1 weights](https://github.com/xai-org/grok-1)

## Mistral AI

**What they do now.** Mistral, founded in Paris in 2023, trains open-weight and commercial models and sells enterprise deployment (including on-premises). Its early papers are short and practical: [Mistral 7B](https://arxiv.org/abs/2310.06825) (grouped-query attention plus sliding-window attention) and [Mixtral of Experts](https://arxiv.org/abs/2401.04088) (sparse mixture-of-experts, 8 experts with top-2 routing). Its reasoning work is described in the [Magistral](https://arxiv.org/abs/2506.10910) report.

**Where they hire (as of September 2026).** France is the centre of research. Mistral has legal entities in France, the US, the UK and Singapore, and teams in Germany too; Singapore has had applied-science and research-engineer openings. In 2026 Indian media reported that Mistral was discussing a Bengaluru centre that would start with engineering and later add research; treat that as unconfirmed until roles appear on its job board.

**How people get noticed.** Open-source work around their models: fine-tuning recipes, inference optimizations, evaluations, integrations. A good bug report or a merged fix in [mistral-common](https://github.com/mistralai/mistral-common) (their tokenization and request-formatting library) or in vLLM's support for Mistral models shows you understand their architecture.

**Tips.**

- Implement sliding-window attention and MoE routing yourself (extend [lab 05](../labs/05_transformer/README.md)); be ready to discuss load balancing and expert capacity.
- Applied AI engineer roles (working with enterprise customers) are an easier entry point than research scientist, and they are real core-AI work: fine-tuning, evals, deployment.
- If relocation to France is the path, read the French Talent Passport rules on the official site before interviews; see [career/visa-and-relocation.md](../career/visa-and-relocation.md).

**Links.** [Mistral careers](https://mistral.ai/careers) · [Mistral docs](https://docs.mistral.ai/) · [Mistral 7B](https://arxiv.org/abs/2310.06825) · [Mixtral](https://arxiv.org/abs/2401.04088)

## DeepSeek

**What they do now.** DeepSeek, based in Hangzhou and funded by the quantitative fund High-Flyer, publishes some of the most technically detailed reports in the field. [DeepSeek-V3](https://arxiv.org/abs/2412.19437) combined multi-head latent attention (MLA, introduced in DeepSeek-V2), fine-grained MoE with auxiliary-loss-free load balancing, multi-token prediction and FP8 training. [DeepSeek-R1](https://arxiv.org/abs/2501.12948) showed that reinforcement learning with verifiable rewards (GRPO) can produce long chain-of-thought reasoning. In February 2025 they open-sourced infrastructure such as [FlashMLA](https://github.com/deepseek-ai/FlashMLA) (MLA decoding kernels) and DeepEP (expert-parallel communication).

**Where they hire (as of September 2026).** In China. Public reporting describes a preference for young graduates of top Chinese universities. It is not a realistic target for most readers of this guide.

**How people get noticed.** For most readers, the value of DeepSeek is educational, and the signal you build is for other employers: reproducing a DeepSeek idea at small scale is one of the strongest portfolio projects available.

**Tips.**

- Reproduce GRPO on a verifiable task ([lab 12](../labs/12_grpo/README.md)) and report what happens to response length and accuracy over training.
- Compute MLA's KV-cache saving against GQA for a concrete config ([lab 03](../labs/03_napkin_math/README.md)), then implement it in your lab 05 model.
- Read their kernel repos to learn how production MoE and attention kernels are organized.

**Links.** [DeepSeek on GitHub](https://github.com/deepseek-ai) · [DeepSeek-V3](https://arxiv.org/abs/2412.19437) · [DeepSeek-R1](https://arxiv.org/abs/2501.12948) · [FlashMLA](https://github.com/deepseek-ai/FlashMLA)

## Qwen (Alibaba)

**What they do now.** The Qwen team at Alibaba releases one of the most widely used open-weight model families: dense and MoE language models from under 1B to hundreds of billions of parameters, plus vision-language, audio and coding models. The [Qwen3 technical report](https://arxiv.org/abs/2505.09388) describes hybrid thinking and non-thinking modes in one model and broad multilingual coverage. Qwen models are a common base for fine-tuning research because of their permissive licences and range of sizes.

**Where they hire (as of September 2026).** The core team is in China. Alibaba Cloud hires internationally (for example in Singapore), mostly for cloud and applied roles.

**How people get noticed.** Contributions to the Qwen ecosystem (fine-tuning recipes, quantized releases, evaluations) are visible to the team, which is active on GitHub and Hugging Face. As with DeepSeek, most readers will turn Qwen work into signal for other employers.

**Tips.**

- Qwen models in the 0.5B–4B range fit an 8 GB GPU for LoRA ([lab 10](../labs/10_lora/README.md)) and GRPO experiments; they are the usual base in small-scale reasoning papers.
- Evaluate Qwen on Indian languages with confidence intervals ([lab 15](../labs/15_eval_stats/README.md)) and compare tokenizer fertility ([lab 04](../labs/04_tokenizer/README.md)); that result is useful to Indian labs too.

**Links.** [Qwen on GitHub](https://github.com/QwenLM) · [Qwen3 technical report](https://arxiv.org/abs/2505.09388) · [Qwen on Hugging Face](https://huggingface.co/Qwen)

## Moonshot AI

**What they do now.** Moonshot AI, based in Beijing, builds the Kimi assistant and models. Its open-weight [Kimi K2](https://arxiv.org/abs/2507.20534) is a large MoE model trained with the Muon optimizer plus a stabilizing modification (MuonClip). Its earlier report [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) showed Muon working at scale.

**Where they hire (as of September 2026).** In China, through [careers.kimi.ai](https://careers.kimi.ai/). Its 2026 international push was a user and developer ambassador program, not overseas research hiring; there was no evidence of India-based or remote core-model roles.

**How people get noticed.** As with DeepSeek, treat Moonshot as a source of reproducible ideas.

**Tips.**

- [Lab 02](../labs/02_training_core/README.md) has you implement Muon. Extend it: compare Muon and AdamW on your lab 05 model at equal compute, with several seeds, and write up the result.
- Read the K2 report's section on training stability (attention-logit growth and how they clip it) and test whether QK-norm has a similar effect at small scale.

**Links.** [Moonshot careers](https://careers.kimi.ai/) · [Moonshot AI on GitHub](https://github.com/MoonshotAI) · [Kimi K2](https://arxiv.org/abs/2507.20534) · [Muon is Scalable](https://arxiv.org/abs/2502.16982)

## Hugging Face

**What they do now.** Hugging Face runs the Hub (models, datasets, Spaces) and maintains core open-source libraries: [Transformers](https://github.com/huggingface/transformers), [TRL](https://github.com/huggingface/trl) (post-training), PEFT, Accelerate, datasets, and text-generation-inference. Its science team trains open models (the SmolLM series), builds datasets (FineWeb) and publishes practical guides such as the [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) on distributed training.

**Where they hire (as of September 2026).** Remote-first across many countries; many postings accept remote candidates within broad regions. Check each posting's location restrictions before applying from India.

**How people get noticed.** Hugging Face hires heavily from its open-source contributors. The most direct route is sustained, high-quality contributions to one library: fixing real bugs, adding a model architecture, improving docs with runnable examples. Public models, datasets and write-ups on the Hub also count.

**Tips.**

- Pick one library and go deep for 2–3 months. TRL is a natural fit after [lab 11](../labs/11_dpo/README.md) and [lab 12](../labs/12_grpo/README.md), because you will already understand DPO and GRPO losses at the level of the code.
- Start with issues labelled "good first issue", then move to real bugs. Write clear pull-request descriptions with a minimal reproduction; maintainers remember those.
- Publish your lab results as Hub model cards or Spaces; see [career/public-presence.md](../career/public-presence.md).

**Links.** [Hugging Face jobs](https://apply.workable.com/huggingface/) · [Transformers](https://github.com/huggingface/transformers) · [TRL](https://github.com/huggingface/trl) · [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)

## Cohere

**What they do now.** Cohere builds LLMs for enterprises and governments (the Command models, plus Embed and Rerank for retrieval) with an emphasis on secure, private and sovereign deployment. In April 2026 it announced it would combine with Germany's Aleph Alpha, and the two signed a definitive merger agreement in September 2026; the combined company keeps the Cohere name, with headquarters in Toronto and Berlin and a research office in Heidelberg. Its research arm, **Cohere Labs** (formerly Cohere For AI), runs an open-science community and leads the Aya project on massively multilingual models ([Aya model paper](https://arxiv.org/abs/2402.07827)).

**Where they hire (as of September 2026).** Toronto, San Francisco, New York, London, Montreal, Paris, Seoul and, after the merger, Germany. No India office was listed. Cohere Labs' open-science community is open to researchers anywhere, and it is the realistic entry point from India.

**How people get noticed.** Two routes. Engineering: strong retrieval and enterprise-LLM work ([lab 17](../labs/17_retrieval/README.md) plus a real evaluation). Research: joining Cohere Labs' community, contributing to multilingual projects, and applying to its Scholars Program when it opens.

**Tips.**

- Indian-language data and evaluation is a direct fit for Aya-style work. A well-documented dataset or benchmark for one Indian language can lead to co-authorship.
- For engineering roles, show a retrieval system with an eval: recall@k, nDCG and a reranker ablation with confidence intervals.

**Links.** [Cohere careers](https://cohere.com/careers) · [Cohere Labs](https://cohere.com/research) · [Aya model paper](https://arxiv.org/abs/2402.07827)

## Allen Institute for AI (Ai2)

**What they do now.** Ai2, a non-profit in Seattle, releases fully open models: weights, training data, code, intermediate checkpoints and logs. The main lines are OLMo (pretraining; [OLMo 2](https://arxiv.org/abs/2501.00656)), Tülu (open post-training recipes; [Tülu 3](https://arxiv.org/abs/2411.15124), which introduced reinforcement learning with verifiable rewards as a named recipe), OLMoE (open MoE) and Molmo (open vision-language). OLMo 3 followed in late 2025.

**Where they hire (as of September 2026).** Mostly Seattle; check each posting for remote eligibility. The Predoctoral Young Investigator program takes recent graduates for research positions before a PhD. Leadership changed in 2026: former CEO Ali Farhadi and several senior researchers moved to Microsoft AI in March. Ask current staff how the open-model roadmap has changed since.

**How people get noticed.** Because everything is open, you can do real research on their artifacts: analyze OLMo checkpoints, reproduce a Tülu ablation, find a data issue. That work speaks directly to the team.

**Tips.**

- Use the intermediate OLMo checkpoints for a small interpretability or training-dynamics study ([lab 16](../labs/16_interpretability/README.md)). Few labs give you checkpoints across training; Ai2 does.
- Read [open-instruct](https://github.com/allenai/open-instruct), their post-training code, after labs 11 and 12. Compare their DPO and RLVR implementations with yours.

**Links.** [Ai2 careers](https://allenai.org/careers) · [OLMo](https://allenai.org/olmo) · [OLMo on GitHub](https://github.com/allenai/OLMo) · [Tülu 3](https://arxiv.org/abs/2411.15124)

## Together AI and the inference companies

**What they do now.** Together AI runs a cloud for training, fine-tuning and serving open models, and does research on efficient inference and kernels. Tri Dao, author of FlashAttention, is its chief scientist; [FlashAttention-3](https://arxiv.org/abs/2407.08608) (Hopper-specific attention) involved Together researchers. Similar companies competing on inference speed and developer experience include Fireworks AI, Baseten and Modal; the open-source serving engines [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) sit underneath much of this market.

**Where they hire (as of September 2026).** Mostly the US (San Francisco Bay Area), with some remote roles for strong systems and kernel engineers. Check [together.ai/careers](https://www.together.ai/careers).

**How people get noticed.** Performance work with numbers. A Triton or CUDA kernel with a roofline analysis, a merged vLLM or SGLang pull request, or a benchmark write-up that explains why a serving configuration is faster.

**Tips.**

- Finish [lab 06](../labs/06_attention_kernels/README.md) and measure your kernel against PyTorch's scaled dot-product attention on your GPU. Report achieved bandwidth or FLOP/s as a fraction of peak ([lab 03](../labs/03_napkin_math/README.md)).
- Contribute to vLLM or SGLang. Both have active issue trackers with performance and correctness bugs; a merged fix is visible to every inference company.
- This path overlaps heavily with [NVIDIA](nvidia/README.md). Prepare both together.

**Links.** [Together AI careers](https://www.together.ai/careers) · [FlashAttention-3](https://arxiv.org/abs/2407.08608) · [vLLM](https://github.com/vllm-project/vllm) · [SGLang](https://github.com/sgl-project/sglang)

## India-based labs and IndiaAI

**What they do now.**

- **Sarvam AI** (Bengaluru, founded in 2023 by Vivek Raghavan and Pratyush Kumar, both previously involved with AI4Bharat) builds models and speech systems for Indian languages: LLMs, translation, speech recognition and text-to-speech. In April 2025 it was the first company selected under the IndiaAI Mission to build a sovereign foundation model. In 2026 it released open-weight [Sarvam-30B](https://huggingface.co/sarvamai/sarvam-30b) and [Sarvam-105B](https://huggingface.co/sarvamai/sarvam-105b) under Apache 2.0. The 105B model card describes a DeepSeek-style design: MLA-style attention, 128 routed experts with top-8 routing plus one shared expert, auxiliary-loss-free load balancing, about 10.3B active parameters, 128K context, and a focus on 22 Indian languages. These are the same ideas as in DeepSeek-V3 (above), so the DeepSeek tips double as Sarvam interview preparation.
- **The IndiaAI Mission** (approved March 2024) funds shared GPU compute (tens of thousands of subsidized GPUs), datasets (AIKosh) and sovereign foundation models. As of 2026, twelve teams were shortlisted to build foundation models: Sarvam AI, Soket AI, Gnani.ai, Gan.AI, Avataar AI, the IIT Bombay-led BharatGen consortium, GenLoop, Zenteq, Intellihealth, Shodh AI, Fractal Analytics and Tech Mahindra Maker's Lab. BharatGen received the largest allocation. Sarvam, BharatGen and Gnani showed models at the India AI Impact Summit in February 2026. IndiaAI is a funder, not an employer, but its list tells you which Indian teams have compute to train models.
- **Krutrim** (Ola's AI company) built Indic LLMs and an AI cloud. It shrank sharply through 2025 and 2026 (reported layoffs cut it from over 500 staff to under 200, with another round in July 2026) and refocused on AI cloud and enterprise services. As of September 2026 it is not a strong target for model-building roles.
- **Global labs in India.** Google DeepMind and Google Research have Bengaluru teams (see the [Google guide](google/README.md)); Microsoft Research India is covered above; NVIDIA has large Indian engineering sites (see the [NVIDIA guide](nvidia/README.md)).

**Where they hire (as of September 2026).** Bengaluru mainly, with the IndiaAI-funded teams spread across Bengaluru, Mumbai, Delhi NCR and other cities. They hire ML engineers, research engineers, speech engineers and data specialists for Indian languages. Check each team's funding and compute before you join; the gap between the best-funded and the rest is large.

**How people get noticed.** Indic-specific competence is scarce and valued: tokenizer efficiency on Indian scripts, evaluation in Indian languages, speech data quality, low-resource fine-tuning. A public, reproducible project in one of these areas is a direct signal.

**Tips.**

- Measure tokenizer fertility (tokens per word) of several open models on Hindi, Telugu, Tamil and Kannada text ([lab 04](../labs/04_tokenizer/README.md)). The differences are large and they drive cost and context length.
- Build a small evaluation in one Indian language with a clear scoring rule and confidence intervals ([lab 15](../labs/15_eval_stats/README.md)). Publish it on the Hub.
- Ask about compute in interviews: how many GPUs, owned or rented, and whether IndiaAI-subsidized compute is involved. It tells you what the team can actually train.

**Links.** [Sarvam AI](https://www.sarvam.ai/) · [Sarvam on Hugging Face](https://huggingface.co/sarvamai) · [Sarvam-105B model card](https://huggingface.co/sarvamai/sarvam-105b) · [IndiaAI](https://indiaai.gov.in/) · [BharatGen](https://bharatgen.com/)

## Academic groups in India and research-assistant paths

**What they do now.** Indian academic groups do strong work in NLP, ML theory, systems and Indic language technology.

- **AI4Bharat** at IIT Madras builds open datasets and models for Indian languages, for example [IndicTrans2](https://arxiv.org/abs/2305.16307) for translation across all 22 scheduled languages.
- **IISc Bengaluru** (the Department of Computational and Data Sciences, the Department of Computer Science and Automation, and others) and the **IITs** have ML, NLP and systems faculty who take project staff.
- **BharatGen**, the government-funded consortium led by IIT Bombay (see above), builds Indian multilingual and multimodal foundation models and hires research staff through its academic partners.

**Paths in.**

| Path | What it is | Notes |
|---|---|---|
| Project associate / research assistant | Paid position on a faculty member's funded project, typically 1–2 years | Advertised on institute and department job pages; also by direct email to faculty |
| MS by research (IISc, IITs) | Research master's with a stipend | Admission usually needs GATE or an institute test; check each institute |
| Corporate predoctoral programs | MSR India Research Fellows; Google Research India predoctoral researchers | Competitive; see above and the [Google guide](google/README.md) |
| AI4Bharat roles | Research engineers and associates on open Indic models | Posted on AI4Bharat's site |

**How people get noticed.** Faculty respond to specific emails. Read two of their recent papers, reproduce one result or find a limitation, and write 150 words that say what you did and what you'd like to try next. Attach a link, not a CV-only message.

**Tips.**

- A 1–2 year RA position that produces one good paper can be worth more for a research career than the same time in a product job. It also strengthens PhD applications abroad.
- Ask before you join: who you will work with day to day, what compute you get, and whether RAs are co-authors on papers.
- See [career/outreach.md](../career/outreach.md) for email templates.

**Links.** [AI4Bharat](https://ai4bharat.iitm.ac.in/) · [AI4Bharat on GitHub](https://github.com/AI4Bharat) · [IISc CDS](https://cds.iisc.ac.in/) · [MSR India Research Fellows](https://www.microsoft.com/en-us/research/academic-program/research-fellows-program-at-microsoft-research-india/)

## Startups as a path

An early-stage AI startup can give a core-AI engineer more ownership in a year than a large lab gives in three. It can also give you a wrapper around someone else's API and equity worth nothing. Evaluate the offer like an investor.

**1. Is the job core AI?** Ask what the team trained, fine-tuned, evaluated or served in the last quarter, and on what hardware. A company that only calls hosted APIs can be a good business, but you will build prompts and product, not models. Both are valid; know which one you are joining.

**2. How long does the money last?** Ask directly: cash in the bank, monthly burn, and when they plan to raise.

$$\text{runway (months)} = \frac{\text{cash}}{\text{monthly net burn}}$$

Under 12 months means the next round decides your job. 18–24 months is the usual target after a raise.

**3. What is the equity worth?** Work through the arithmetic. Suppose you get options on 0.10% of the company, the company raises two more rounds that each dilute existing holders by 20%, and it is later sold for $500M:

$$0.10\% \times (1 - 0.20)^2 = 0.064\%, \qquad 0.064\% \times \$500\text{M} = \$320{,}000$$

That is before the exercise cost (strike price times number of options), taxes, and the liquidation preferences of investors, which are paid first. At a smaller sale, preferences can take most or all of the proceeds. Treat equity as a lottery ticket whose odds you can partly assess, and compare offers on cash.

**4. Terms to get in writing.** Number of options and the fully diluted share count (so you can compute your percentage), strike price, vesting schedule (four years with a one-year cliff is common), the exercise window after you leave, and whether any acceleration applies. In India, ESOPs are taxed as a perquisite at exercise and again as capital gains on sale, with deferral rules for eligible startups; ask a tax adviser before you exercise. More in [career/offers-and-negotiation.md](../career/offers-and-negotiation.md).

**5. Who will you learn from?** The first engineers you work beside matter more than the brand. Ask to meet them and ask what they shipped recently.

> [!WARNING]
> Red flags: no clear answer on runway; refusal to tell you the total share count; "we'll sort out the equity paperwork later"; a GPU budget that cannot support the models they claim to train; an exploding offer. Any one of these is a reason to pause.

> [!TIP]
> A good startup year can be a strong step toward a frontier lab: you will have owned an end-to-end system, with real users and real failure modes. Document what you built and measured as you go, so it becomes portfolio material ([portfolio.md](../tracks/research-engineer/portfolio.md)). If you are thinking of starting a company yourself, use the [founder track](../tracks/founder/README.md).

## Sources

Volatile facts were checked against these pages as of September 2026. Re-verify before acting.

- Microsoft Research India, [Research Fellows program](https://www.microsoft.com/en-us/research/academic-program/research-fellows-program-at-microsoft-research-india/) (eligibility, 2026 dates)
- [Microsoft Research India](https://www.microsoft.com/en-us/research/lab/microsoft-research-india/)
- Agrawal et al., [Sarathi-Serve](https://arxiv.org/abs/2403.02310); Prabhu et al., [vAttention](https://arxiv.org/abs/2405.04437); Agrawal et al., [Vidur](https://arxiv.org/abs/2405.05465); Abdin et al., [Phi-4](https://arxiv.org/abs/2412.08905)
- Apple, [Apple Intelligence Foundation Language Models](https://arxiv.org/abs/2407.21075) (2024) and [2025 tech report](https://arxiv.org/abs/2507.13575)
- Jiang et al., [Mistral 7B](https://arxiv.org/abs/2310.06825) and [Mixtral of Experts](https://arxiv.org/abs/2401.04088); Mistral AI, [Magistral](https://arxiv.org/abs/2506.10910)
- DeepSeek-AI, [DeepSeek-V3](https://arxiv.org/abs/2412.19437) and [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- Qwen Team, [Qwen3 technical report](https://arxiv.org/abs/2505.09388)
- Kimi Team, [Kimi K2](https://arxiv.org/abs/2507.20534); Liu et al., [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
- Üstün et al., [Aya model](https://arxiv.org/abs/2402.07827)
- OLMo Team, [OLMo 2](https://arxiv.org/abs/2501.00656); Lambert et al., [Tülu 3](https://arxiv.org/abs/2411.15124)
- Shah et al., [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- Gala et al., [IndicTrans2](https://arxiv.org/abs/2305.16307)
- [IndiaAI](https://indiaai.gov.in/); DD News, [IndiaAI Mission shortlists 12 teams for indigenous AI models](https://ddnews.gov.in/en/indiaai-mission-shortlists-12-teams-for-indigenous-ai-models-minister-updates-rajya-sabha/); MediaNama, [Govt is funding 12 AI orgs to develop indigenous models](https://www.medianama.com/2026/04/223-centre-funds-12-ai-projects-sovereign-models-bharatgen-4x-next-highest-allocation-rs-1000-crore/) (April 2026)
- Sarvam AI, [Sarvam-105B model card](https://huggingface.co/sarvamai/sarvam-105b) (architecture, licence, languages)
- Inc42, [Krutrim cuts nearly half of remaining workforce](https://inc42.com/buzz/krutrim-cuts-nearly-half-of-remaining-workforce-in-fresh-layoffs/) (2026)

News used for 2025–2026 organizational changes (secondary sources; confirm on company pages):

- CNBC, [Microsoft forms superintelligence team under Suleyman](https://www.cnbc.com/2025/11/06/microsoft-forms-superintelligence-team-under-ai-head-mustafa-suleyman-.html) (November 2025); GeekWire, [Microsoft hires former Ai2 CEO Ali Farhadi and key researchers](https://www.geekwire.com/2026/microsoft-hires-former-ai2-ceo-ali-farhadi-and-key-researchers-for-suleymans-ai-team/) (March 2026)
- Apple press release via Business Wire, [AI leadership change](https://www.businesswire.com/news/home/20251201260097/en) (December 2025)
- Wikipedia, [xAI (company)](https://en.wikipedia.org/wiki/XAI_(company)) (SpaceX acquisition, February 2026)
- TechCrunch, [Cohere acquires, merges with German-based startup](https://techcrunch.com/2026/04/24/cohere-acquires-merges-with-german-based-startup-to-create-a-transatlantic-ai-powerhouse/) (April 2026); SiliconANGLE, [Cohere and Aleph Alpha agree to merge](https://siliconangle.com/2026/09/16/cohere-and-aleph-alpha-agree-to-merge-in-reported-20b-deal/) (September 2026)
- Storyboard18, [Mistral AI in talks to set up Bengaluru GCC](https://www.storyboard18.com/brand-marketing/mistral-ai-in-talks-to-set-up-bengaluru-gcc-with-phased-expansion-plan-88076.htm) (2026)
