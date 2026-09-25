# Papers, by module

For each paper, extract: *claim · evidence · compute · what I'd test next.* ★ = read in full and reproduce something.

## Transformers (04)
★ Vaswani et al. 2017, *Attention Is All You Need* · Radford et al. 2019 (GPT-2) · Brown et al. 2020 (GPT-3) · ★ Su et al. 2021, *RoFormer* (RoPE) · Shazeer 2019 (MQA) · Ainslie et al. 2023 (GQA) · Shazeer 2020, *GLU Variants* · Zhang & Sennrich 2019 (RMSNorm) · Xiong et al. 2020 (pre-LN) · Touvron et al. 2023 (LLaMA, Llama 2) · ★ Llama Team 2024, *The Llama 3 Herd of Models* · DeepSeek-AI 2024, DeepSeek-V2 (MLA) and ★ DeepSeek-V3 · Fedus et al. 2021 (Switch Transformers) · Gu & Dao 2023 (Mamba) · Chen et al. 2023 (position interpolation) · Peng et al. 2023 (YaRN) · Dosovitskiy et al. 2020 (ViT) · Radford et al. 2021 (CLIP) · Liu et al. 2023 (LLaVA)

## Optimization and training (03)
Kingma & Ba 2014 (Adam) · ★ Loshchilov & Hutter 2019 (AdamW) · Yang et al. 2022, *Tensor Programs V* (μP) · McCandlish et al. 2018 (critical batch size) · Micikevicius et al. 2017 (mixed precision) · Wortsman et al. 2023, *Small-scale proxies for large-scale Transformer training instabilities* · Jordan 2024 (Muon, blog) and Liu et al. 2025, *Muon is Scalable for LLM Training* · Hägele et al. 2024 (WSD/cooldowns)

## Pretraining and scaling (05)
Kaplan et al. 2020 · ★ Hoffmann et al. 2022 (Chinchilla) · Muennighoff et al. 2023 (data-constrained) · Sardana et al. 2023 (beyond Chinchilla-optimal) · Besiroglu et al. 2024 (Chinchilla replication) · ★ Penedo et al. 2024 (FineWeb) · Li et al. 2024 (DCLM) · Lee et al. 2021 (dedup) · Xie et al. 2023 (DoReMi) · Groeneveld et al. 2024 and OLMo Team 2024 (OLMo, OLMo 2) · Shoeybi et al. 2019 (Megatron-LM) · ★ Rajbhandari et al. 2020 (ZeRO) · Narayanan et al. 2021 (PTD-P) · Korthikanti et al. 2022 (sequence parallelism, activation memory) · Liu et al. 2023 (Ring Attention) · Zhao et al. 2023 (PyTorch FSDP)

## Post-training (06)
★ Ouyang et al. 2022 (InstructGPT) · Christiano et al. 2017 · Schulman et al. 2017 (PPO), 2015 (GAE) · ★ Rafailov et al. 2023 (DPO) · Azar et al. 2023 (IPO) · Ethayarajh et al. 2024 (KTO) · Meng et al. 2024 (SimPO) · Gao et al. 2022 (RM overoptimization) · Zhou et al. 2023 (LIMA) · Bai et al. 2022 (HH-RLHF; Constitutional AI) · ★ Shao et al. 2024 (DeepSeekMath, GRPO) · ★ DeepSeek-AI 2025 (DeepSeek-R1) · Yu et al. 2025 (DAPO) · Liu et al. 2025 (Dr. GRPO) · Lambert et al. 2024 (Tülu 3) · Lightman et al. 2023 (Let's Verify Step by Step) · Wei et al. 2022 (CoT) · Wang et al. 2022 (self-consistency) · Snell et al. 2024 (test-time compute) · Muennighoff et al. 2025 (s1) · Hinton et al. 2015 (distillation) · Agarwal et al. 2024 (on-policy distillation, GKD) · ★ Hu et al. 2021 (LoRA) · Dettmers et al. 2023 (QLoRA) · Liu et al. 2024 (DoRA)

## Inference (07)
★ Pope et al. 2022 (*Efficiently Scaling Transformer Inference*) · Milakov & Gimelshein 2018 (online softmax) · ★ Dao et al. 2022 (FlashAttention), Dao 2023 (FA-2), Shah et al. 2024 (FA-3) · ★ Kwon et al. 2023 (vLLM / PagedAttention) · Yu et al. 2022 (Orca) · Zheng et al. 2024 (SGLang) · Agrawal et al. 2024 (Sarathi-Serve) · Zhong et al. 2024 (DistServe) · ★ Leviathan et al. 2023 and Chen et al. 2023 (speculative decoding) · Cai et al. 2024 (Medusa) · Li et al. 2024 (EAGLE) · Dettmers et al. 2022 (LLM.int8) · Xiao et al. 2023 (SmoothQuant) · ★ Frantar et al. 2023 (GPTQ) · Lin et al. 2024 (AWQ) · Rouhani et al. 2023 (MX formats) · Tillet et al. 2019 (Triton)

## Evaluation and research (08)
★ Miller 2024 (*Adding Error Bars to Evals*) · Chen et al. 2021 (Codex; pass@k) · Zheng et al. 2023 (LLM-as-a-judge; MT-Bench) · Chiang et al. 2024 (Chatbot Arena) · Hendrycks et al. 2021 (MMLU) · Rein et al. 2023 (GPQA) · Jimenez et al. 2024 (SWE-bench) · Jain et al. 2024 (LiveCodeBench) · Dubois et al. 2024 (length-controlled AlpacaEval) · Biderman et al. 2024 (lessons from lm-eval-harness) · Kwa et al. 2025 (METR, task time horizons)

## Interpretability and safety (09)
★ Elhage et al. 2021 (*A Mathematical Framework for Transformer Circuits*) · ★ Olsson et al. 2022 (induction heads) · ★ Elhage et al. 2022 (*Toy Models of Superposition*) · Bricken et al. 2023 (*Towards Monosemanticity*) · Templeton et al. 2024 (*Scaling Monosemanticity*) · Gao et al. 2024 (scaling SAEs) · Meng et al. 2022 (ROME) · Wang et al. 2022 (IOI) · Anthropic 2025 (*On the Biology of a Large Language Model*) · Amodei et al. 2016 (*Concrete Problems in AI Safety*) · Hubinger et al. 2024 (Sleeper Agents) · Greenblatt et al. 2024 (Alignment Faking) · Sharma et al. 2023 (sycophancy) · Zou et al. 2023 (GCG) · Greshake et al. 2023 (indirect prompt injection) · Baker et al. 2025 (CoT monitoring and obfuscation)

## Applied systems (10)
Lewis et al. 2020 (RAG) · Karpukhin et al. 2020 (DPR) · Khattab & Zaharia 2020 (ColBERT) · Cormack et al. 2009 (RRF) · Liu et al. 2023 (Lost in the Middle) · Yao et al. 2022 (ReAct) · Schick et al. 2023 (Toolformer) · Kusupati et al. 2022 (Matryoshka) · Jégou et al. 2011 (product quantization) · Malkov & Yashunin 2016 (HNSW)

**Staying current:** arXiv (cs.CL, cs.LG) via the Hugging Face daily papers page; lab blogs (Anthropic, OpenAI, Google DeepMind, Meta, DeepSeek, Qwen, AI2); GPU MODE. Re-check the frontier sections of modules 05–07 every quarter.
