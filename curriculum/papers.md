# Papers, by module

The papers behind each module, with links (arXiv where one exists). Each module starts with a short **Read first** list, in order: read those in full, and reproduce one figure from at least one. Skim the rest when a module or lab needs them.

For every paper, write four lines in your journal ([paper-notes template](../journal/templates/paper-notes.md)): **claim · evidence · compute · what I would test next.**

**Contents:** [03 Optimization and training](#03-optimization-and-training) · [04 Transformers](#04-transformers) · [05 Pretraining and scaling](#05-pretraining-and-scaling) · [06 Post-training](#06-post-training) · [07 Inference](#07-inference) · [08 Evaluation and research](#08-evaluation-and-research) · [09 Interpretability and safety](#09-interpretability-and-safety) · [10 Applied LLM systems](#10-applied-llm-systems) · [Staying current](#staying-current)

> [!TIP]
> Read a paper in three passes: (1) abstract, figures and conclusion, 10 minutes, decide whether to continue; (2) the method and the main experiment, noting every number you would need to reproduce it; (3) the appendix, which is where the hyperparameters and the caveats live.

---

## 03 Optimization and training

**Read first**
1. Loshchilov and Hutter 2019, [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101) (AdamW): why weight decay is not L2 regularization under Adam. [Lab 02](../labs/02_training_core/README.md) makes it bit-exact.
2. McCandlish et al. 2018, [*An Empirical Model of Large-Batch Training*](https://arxiv.org/abs/1812.06162): the critical batch size and the gradient noise scale.
3. Wortsman et al. 2023, [*Small-scale proxies for large-scale Transformer training instabilities*](https://arxiv.org/abs/2309.14322): reproduce big-model instabilities on a small GPU.

**Then**
- Kingma and Ba 2014, [*Adam*](https://arxiv.org/abs/1412.6980)
- Yang et al. 2022, [*Tensor Programs V*](https://arxiv.org/abs/2203.03466) (μP: transfer hyperparameters across widths)
- Micikevicius et al. 2017, [*Mixed Precision Training*](https://arxiv.org/abs/1710.03740)
- Jordan 2024, [*Muon*](https://kellerjordan.github.io/posts/muon/) (blog) and Liu et al. 2025, [*Muon is Scalable for LLM Training*](https://arxiv.org/abs/2502.16982)
- Hägele et al. 2024, [*Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations*](https://arxiv.org/abs/2405.18392) (WSD schedules and cooldowns)

## 04 Transformers

**Read first**
1. Vaswani et al. 2017, [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
2. Su et al. 2021, [*RoFormer*](https://arxiv.org/abs/2104.09864) (RoPE): derive the rotation yourself.
3. Llama Team 2024, [*The Llama 3 Herd of Models*](https://arxiv.org/abs/2407.21783): the most complete public account of a modern dense recipe.
4. DeepSeek-AI 2024, [*DeepSeek-V3 Technical Report*](https://arxiv.org/abs/2412.19437): MLA, fine-grained MoE, FP8 training.

**Then**
- Radford et al. 2019, [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2)
- Brown et al. 2020, [*Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165) (GPT-3)
- Shazeer 2019, [*Fast Transformer Decoding: One Write-Head is All You Need*](https://arxiv.org/abs/1911.02150) (MQA)
- Ainslie et al. 2023, [*GQA*](https://arxiv.org/abs/2305.13245)
- Shazeer 2020, [*GLU Variants Improve Transformer*](https://arxiv.org/abs/2002.05202)
- Zhang and Sennrich 2019, [*Root Mean Square Layer Normalization*](https://arxiv.org/abs/1910.07467)
- Xiong et al. 2020, [*On Layer Normalization in the Transformer Architecture*](https://arxiv.org/abs/2002.04745) (pre-LN)
- Touvron et al. 2023, [*LLaMA*](https://arxiv.org/abs/2302.13971) and [*Llama 2*](https://arxiv.org/abs/2307.09288)
- DeepSeek-AI 2024, [*DeepSeek-V2*](https://arxiv.org/abs/2405.04434) (multi-head latent attention)
- Fedus et al. 2021, [*Switch Transformers*](https://arxiv.org/abs/2101.03961)
- Gu and Dao 2023, [*Mamba*](https://arxiv.org/abs/2312.00752)
- Chen et al. 2023, [*Extending Context Window of Large Language Models via Position Interpolation*](https://arxiv.org/abs/2306.15595)
- Peng et al. 2023, [*YaRN*](https://arxiv.org/abs/2309.00071)
- Dosovitskiy et al. 2020, [*An Image is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929) (ViT)
- Radford et al. 2021, [*Learning Transferable Visual Models From Natural Language Supervision*](https://arxiv.org/abs/2103.00020) (CLIP)
- Liu et al. 2023, [*Visual Instruction Tuning*](https://arxiv.org/abs/2304.08485) (LLaVA)

## 05 Pretraining and scaling

**Read first**
1. Hoffmann et al. 2022, [*Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556) (Chinchilla); then Besiroglu et al. 2024, [*Chinchilla Scaling: A replication attempt*](https://arxiv.org/abs/2404.10102), for how fragile the fit is. [Lab 08](../labs/08_scaling_laws/README.md).
2. Penedo et al. 2024, [*The FineWeb Datasets*](https://arxiv.org/abs/2406.17557): data quality as the main lever.
3. Rajbhandari et al. 2020, [*ZeRO*](https://arxiv.org/abs/1910.02054): the memory accounting behind every large run. [Lab 09](../labs/09_parallelism/README.md).

**Then**
- Kaplan et al. 2020, [*Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361)
- Muennighoff et al. 2023, [*Scaling Data-Constrained Language Models*](https://arxiv.org/abs/2305.16264)
- Sardana et al. 2023, [*Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*](https://arxiv.org/abs/2401.00448)
- Li et al. 2024, [*DataComp-LM*](https://arxiv.org/abs/2406.11794) (DCLM)
- Lee et al. 2021, [*Deduplicating Training Data Makes Language Models Better*](https://arxiv.org/abs/2107.06499)
- Xie et al. 2023, [*DoReMi*](https://arxiv.org/abs/2305.10429)
- Groeneveld et al. 2024, [*OLMo*](https://arxiv.org/abs/2402.00838) and OLMo Team 2024, [*2 OLMo 2 Furious*](https://arxiv.org/abs/2501.00656)
- Shoeybi et al. 2019, [*Megatron-LM*](https://arxiv.org/abs/1909.08053)
- Narayanan et al. 2021, [*Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*](https://arxiv.org/abs/2104.04473) (PTD-P)
- Korthikanti et al. 2022, [*Reducing Activation Recomputation in Large Transformer Models*](https://arxiv.org/abs/2205.05198) (sequence parallelism)
- Liu et al. 2023, [*Ring Attention*](https://arxiv.org/abs/2310.01889)
- Zhao et al. 2023, [*PyTorch FSDP*](https://arxiv.org/abs/2304.11277)

## 06 Post-training

**Read first**
1. Ouyang et al. 2022, [*Training language models to follow instructions with human feedback*](https://arxiv.org/abs/2203.02155) (InstructGPT)
2. Rafailov et al. 2023, [*Direct Preference Optimization*](https://arxiv.org/abs/2305.18290): derive the closed form. [Lab 11](../labs/11_dpo/README.md).
3. Shao et al. 2024, [*DeepSeekMath*](https://arxiv.org/abs/2402.03300) (GRPO) and DeepSeek-AI 2025, [*DeepSeek-R1*](https://arxiv.org/abs/2501.12948). [Lab 12](../labs/12_grpo/README.md).
4. Hu et al. 2021, [*LoRA*](https://arxiv.org/abs/2106.09685). [Lab 10](../labs/10_lora/README.md).

**Then**
- Christiano et al. 2017, [*Deep Reinforcement Learning from Human Preferences*](https://arxiv.org/abs/1706.03741)
- Schulman et al. 2017, [*PPO*](https://arxiv.org/abs/1707.06347) and 2015, [*GAE*](https://arxiv.org/abs/1506.02438)
- Azar et al. 2023, [*A General Theoretical Paradigm to Understand Learning from Human Preferences*](https://arxiv.org/abs/2310.12036) (IPO)
- Ethayarajh et al. 2024, [*KTO*](https://arxiv.org/abs/2402.01306)
- Meng et al. 2024, [*SimPO*](https://arxiv.org/abs/2405.14734)
- Gao et al. 2022, [*Scaling Laws for Reward Model Overoptimization*](https://arxiv.org/abs/2210.10760)
- Zhou et al. 2023, [*LIMA*](https://arxiv.org/abs/2305.11206)
- Bai et al. 2022, [*Training a Helpful and Harmless Assistant with RLHF*](https://arxiv.org/abs/2204.05862) and [*Constitutional AI*](https://arxiv.org/abs/2212.08073)
- Yu et al. 2025, [*DAPO*](https://arxiv.org/abs/2503.14476) and Liu et al. 2025, [*Understanding R1-Zero-Like Training*](https://arxiv.org/abs/2503.20783) (Dr. GRPO)
- Lambert et al. 2024, [*Tülu 3*](https://arxiv.org/abs/2411.15124)
- Lightman et al. 2023, [*Let's Verify Step by Step*](https://arxiv.org/abs/2305.20050)
- Wei et al. 2022, [*Chain-of-Thought Prompting*](https://arxiv.org/abs/2201.11903) and Wang et al. 2022, [*Self-Consistency*](https://arxiv.org/abs/2203.11171)
- Snell et al. 2024, [*Scaling LLM Test-Time Compute Optimally*](https://arxiv.org/abs/2408.03314) and Muennighoff et al. 2025, [*s1*](https://arxiv.org/abs/2501.19393)
- Hinton et al. 2015, [*Distilling the Knowledge in a Neural Network*](https://arxiv.org/abs/1503.02531) and Agarwal et al. 2024, [*On-Policy Distillation of Language Models*](https://arxiv.org/abs/2306.13649) (GKD)
- Dettmers et al. 2023, [*QLoRA*](https://arxiv.org/abs/2305.14314) and Liu et al. 2024, [*DoRA*](https://arxiv.org/abs/2402.09353)

## 07 Inference

**Read first**
1. Pope et al. 2022, [*Efficiently Scaling Transformer Inference*](https://arxiv.org/abs/2211.05102): the napkin math of serving. [Lab 03](../labs/03_napkin_math/README.md).
2. Dao et al. 2022, [*FlashAttention*](https://arxiv.org/abs/2205.14135). [Lab 06](../labs/06_attention_kernels/README.md).
3. Kwon et al. 2023, [*Efficient Memory Management for LLM Serving with PagedAttention*](https://arxiv.org/abs/2309.06180) (vLLM)
4. Leviathan et al. 2023, [*Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192). [Lab 14](../labs/14_speculative_decoding/README.md).
5. Frantar et al. 2023, [*GPTQ*](https://arxiv.org/abs/2210.17323). [Lab 13](../labs/13_quantization/README.md).

**Then**
- Milakov and Gimelshein 2018, [*Online normalizer calculation for softmax*](https://arxiv.org/abs/1805.02867)
- Dao 2023, [*FlashAttention-2*](https://arxiv.org/abs/2307.08691) and Shah et al. 2024, [*FlashAttention-3*](https://arxiv.org/abs/2407.08608)
- Yu et al. 2022, *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022; continuous batching)
- Zheng et al. 2024, [*SGLang: Efficient Execution of Structured Language Model Programs*](https://arxiv.org/abs/2312.07104)
- Agrawal et al. 2024, [*Sarathi-Serve*](https://arxiv.org/abs/2403.02310) (chunked prefill)
- Zhong et al. 2024, [*DistServe*](https://arxiv.org/abs/2401.09670) (prefill–decode disaggregation)
- Chen et al. 2023, [*Accelerating Large Language Model Decoding with Speculative Sampling*](https://arxiv.org/abs/2302.01318)
- Cai et al. 2024, [*Medusa*](https://arxiv.org/abs/2401.10774) and Li et al. 2024, [*EAGLE*](https://arxiv.org/abs/2401.15077)
- Dettmers et al. 2022, [*LLM.int8()*](https://arxiv.org/abs/2208.07339)
- Xiao et al. 2023, [*SmoothQuant*](https://arxiv.org/abs/2211.10438)
- Lin et al. 2024, [*AWQ*](https://arxiv.org/abs/2306.00978)
- Rouhani et al. 2023, [*Microscaling Data Formats for Deep Learning*](https://arxiv.org/abs/2310.10537) (MX formats)
- Tillet et al. 2019, *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations* (MAPL 2019)

## 08 Evaluation and research

**Read first**
1. Miller 2024, [*Adding Error Bars to Evals*](https://arxiv.org/abs/2411.00640): the statistics of [module 08 §2](08-evaluation-and-research.md#2-statistics-error-bars-or-it-didnt-happen). [Lab 15](../labs/15_eval_stats/README.md).
2. Chen et al. 2021, [*Evaluating Large Language Models Trained on Code*](https://arxiv.org/abs/2107.03374): HumanEval and the unbiased pass@k estimator.
3. Zheng et al. 2023, [*Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*](https://arxiv.org/abs/2306.05685): judge biases and agreement.
4. Biderman et al. 2024, [*Lessons from the Trenches on Reproducible Evaluation of Language Models*](https://arxiv.org/abs/2405.14782)

**Benchmarks**
- Hendrycks et al. 2020, [*MMLU*](https://arxiv.org/abs/2009.03300) and Wang et al. 2024, [*MMLU-Pro*](https://arxiv.org/abs/2406.01574)
- Rein et al. 2023, [*GPQA*](https://arxiv.org/abs/2311.12022) and Phan et al. 2025, [*Humanity's Last Exam*](https://arxiv.org/abs/2501.14249)
- Cobbe et al. 2021, [*GSM8K*](https://arxiv.org/abs/2110.14168), Hendrycks et al. 2021, [*MATH*](https://arxiv.org/abs/2103.03874), Glazer et al. 2024, [*FrontierMath*](https://arxiv.org/abs/2411.04872)
- Jimenez et al. 2023, [*SWE-bench*](https://arxiv.org/abs/2310.06770) and Jain et al. 2024, [*LiveCodeBench*](https://arxiv.org/abs/2403.07974)
- Zhou et al. 2023, [*IFEval*](https://arxiv.org/abs/2311.07911)
- Yao et al. 2024, [*τ-bench*](https://arxiv.org/abs/2406.12045), Mialon et al. 2023, [*GAIA*](https://arxiv.org/abs/2311.12983), Xie et al. 2024, [*OSWorld*](https://arxiv.org/abs/2404.07972), Chan et al. 2024, [*MLE-bench*](https://arxiv.org/abs/2410.07095), METR 2024, [*RE-Bench*](https://arxiv.org/abs/2411.15114)
- Kwa et al. 2025, [*Measuring AI Ability to Complete Long Tasks*](https://arxiv.org/abs/2503.14499) (METR time horizons)
- Liang et al. 2022, [*Holistic Evaluation of Language Models*](https://arxiv.org/abs/2211.09110) (HELM)
- Chiang et al. 2024, [*Chatbot Arena*](https://arxiv.org/abs/2403.04132) and Singh et al. 2025, [*The Leaderboard Illusion*](https://arxiv.org/abs/2504.20879)

**Contamination, judges and harness sensitivity**
- Oren et al. 2023, [*Proving Test Set Contamination in Black Box Language Models*](https://arxiv.org/abs/2310.17623)
- Shi et al. 2023, [*Detecting Pretraining Data from Large Language Models*](https://arxiv.org/abs/2310.16789) (Min-K% probability)
- Yang et al. 2023, [*Rethinking Benchmark and Contamination for Language Models with Rephrased Samples*](https://arxiv.org/abs/2311.04850)
- Zhang et al. 2024, [*A Careful Examination of Large Language Model Performance on Grade School Arithmetic*](https://arxiv.org/abs/2405.00332) (GSM1k)
- Wang et al. 2023, [*Large Language Models are not Fair Evaluators*](https://arxiv.org/abs/2305.17926) (position bias)
- Panickssery et al. 2024, [*LLM Evaluators Recognize and Favor Their Own Generations*](https://arxiv.org/abs/2404.13076)
- Dubois et al. 2024, [*Length-Controlled AlpacaEval*](https://arxiv.org/abs/2404.04475)
- Sclar et al. 2023, [*Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design*](https://arxiv.org/abs/2310.11324)
- Alzahrani et al. 2024, [*When Benchmarks are Targets*](https://arxiv.org/abs/2402.01781)

**Research method and statistics**
- Card et al. 2020, [*With Little Power Comes Great Responsibility*](https://arxiv.org/abs/2010.06595)
- Bouthillier et al. 2021, [*Accounting for Variance in Machine Learning Benchmarks*](https://arxiv.org/abs/2103.03098)
- Madaan et al. 2024, [*Quantifying Variance in Evaluation Benchmarks*](https://arxiv.org/abs/2406.10229)
- Dodge et al. 2019, *Show Your Work: Improved Reporting of Experimental Results* (EMNLP 2019)

## 09 Interpretability and safety

**Read first**
1. Elhage et al. 2021, [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html): QK/OV circuits and composition.
2. Olsson et al. 2022, [*In-context Learning and Induction Heads*](https://arxiv.org/abs/2209.11895). [Lab 16](../labs/16_interpretability/README.md).
3. Elhage et al. 2022, [*Toy Models of Superposition*](https://arxiv.org/abs/2209.10652)
4. Bricken et al. 2023, [*Towards Monosemanticity*](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
5. Greshake et al. 2023, [*Not what you've signed up for: indirect prompt injection*](https://arxiv.org/abs/2302.12173)

**Features and sparse autoencoders**
- Cunningham et al. 2023, [*Sparse Autoencoders Find Highly Interpretable Features in Language Models*](https://arxiv.org/abs/2309.08600)
- Templeton et al. 2024, [*Scaling Monosemanticity*](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
- Gao et al. 2024, [*Scaling and evaluating sparse autoencoders*](https://arxiv.org/abs/2406.04093) (TopK)
- Rajamanoharan et al. 2024, [*Gated SAEs*](https://arxiv.org/abs/2404.16014) and [*JumpReLU SAEs*](https://arxiv.org/abs/2407.14435)
- Bussmann et al. 2024, [*BatchTopK Sparse Autoencoders*](https://arxiv.org/abs/2412.06410)
- Lieberum et al. 2024, [*Gemma Scope*](https://arxiv.org/abs/2408.05147)
- Dunefsky et al. 2024, [*Transcoders Find Interpretable LLM Feature Circuits*](https://arxiv.org/abs/2406.11944)
- Chanin et al. 2024, [*A is for Absorption*](https://arxiv.org/abs/2409.14507)
- Karvonen et al. 2025, [*SAEBench*](https://arxiv.org/abs/2503.09532)
- Kantamneni et al. 2025, [*Are Sparse Autoencoders Useful? A Case Study in Sparse Probing*](https://arxiv.org/abs/2502.16681)

**Circuits, patching, probing and steering**
- Wang et al. 2022, [*Interpretability in the Wild: a Circuit for Indirect Object Identification*](https://arxiv.org/abs/2211.00593) (IOI)
- Meng et al. 2022, [*Locating and Editing Factual Associations in GPT*](https://arxiv.org/abs/2202.05262) (ROME)
- Geva et al. 2020, [*Transformer Feed-Forward Layers Are Key-Value Memories*](https://arxiv.org/abs/2012.14913)
- Nanda et al. 2023, [*Progress measures for grokking via mechanistic interpretability*](https://arxiv.org/abs/2301.05217)
- Conmy et al. 2023, [*Towards Automated Circuit Discovery*](https://arxiv.org/abs/2304.14997) (ACDC)
- Zhang and Nanda 2023, [*Towards Best Practices of Activation Patching*](https://arxiv.org/abs/2309.16042) and Heimersheim and Nanda 2024, [*How to use and interpret activation patching*](https://arxiv.org/abs/2404.15255)
- Kramár et al. 2024, [*AtP\*: An efficient and scalable method for localizing LLM behaviour to components*](https://arxiv.org/abs/2403.00745)
- Makelov et al. 2023, [*Is This the Subspace You Are Looking For?*](https://arxiv.org/abs/2311.17030)
- Marks et al. 2024, [*Sparse Feature Circuits*](https://arxiv.org/abs/2403.19647)
- Ameisen et al. 2025, [*Circuit Tracing*](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) and Lindsey et al. 2025, [*On the Biology of a Large Language Model*](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- Alain and Bengio 2016, [*Understanding intermediate layers using linear classifier probes*](https://arxiv.org/abs/1610.01644); Hewitt and Liang 2019, *Designing and Interpreting Probes with Control Tasks* (EMNLP 2019); Belinkov 2021, [*Probing Classifiers*](https://arxiv.org/abs/2102.12452)
- Burns et al. 2022, [*Discovering Latent Knowledge Without Supervision*](https://arxiv.org/abs/2212.03827) and Marks and Tegmark 2023, [*The Geometry of Truth*](https://arxiv.org/abs/2310.06824)
- Belrose et al. 2023, [*Eliciting Latent Predictions with the Tuned Lens*](https://arxiv.org/abs/2303.08112)
- Turner et al. 2023, [*Steering Language Models With Activation Engineering*](https://arxiv.org/abs/2308.10248); Panickssery et al. 2023, [*Steering Llama 2 via Contrastive Activation Addition*](https://arxiv.org/abs/2312.06681); Zou et al. 2023, [*Representation Engineering*](https://arxiv.org/abs/2310.01405)
- Arditi et al. 2024, [*Refusal in Language Models Is Mediated by a Single Direction*](https://arxiv.org/abs/2406.11717)
- Sharkey et al. 2025, [*Open Problems in Mechanistic Interpretability*](https://arxiv.org/abs/2501.16496)

**Alignment failure modes**
- Amodei et al. 2016, [*Concrete Problems in AI Safety*](https://arxiv.org/abs/1606.06565)
- Skalse et al. 2022, [*Defining and Characterizing Reward Hacking*](https://arxiv.org/abs/2209.13085)
- Sharma et al. 2023, [*Towards Understanding Sycophancy in Language Models*](https://arxiv.org/abs/2310.13548)
- Turpin et al. 2023, [*Language Models Don't Always Say What They Think*](https://arxiv.org/abs/2305.04388); Lanham et al. 2023, [*Measuring Faithfulness in Chain-of-Thought Reasoning*](https://arxiv.org/abs/2307.13702); Chen et al. 2025, [*Reasoning Models Don't Always Say What They Think*](https://arxiv.org/abs/2505.05410)
- Baker et al. 2025, [*Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation*](https://arxiv.org/abs/2503.11926) and Korbak et al. 2025, [*Chain of Thought Monitorability*](https://arxiv.org/abs/2507.11473)
- Hubinger et al. 2024, [*Sleeper Agents*](https://arxiv.org/abs/2401.05566)
- Greenblatt et al. 2024, [*Alignment faking in large language models*](https://arxiv.org/abs/2412.14093)
- van der Weij et al. 2024, [*AI Sandbagging: Language Models can Strategically Underperform on Evaluations*](https://arxiv.org/abs/2406.07358)
- Meinke et al. 2024, [*Frontier Models are Capable of In-context Scheming*](https://arxiv.org/abs/2412.04984)
- Denison et al. 2024, [*Sycophancy to Subterfuge*](https://arxiv.org/abs/2406.10162)
- Betley et al. 2025, [*Emergent Misalignment*](https://arxiv.org/abs/2502.17424)
- Greenblatt et al. 2023, [*AI Control: Improving Safety Despite Intentional Subversion*](https://arxiv.org/abs/2312.06942)

**Misuse, security and governance**
- Wei et al. 2023, [*Jailbroken: How Does LLM Safety Training Fail?*](https://arxiv.org/abs/2307.02483)
- Zou et al. 2023, [*Universal and Transferable Adversarial Attacks on Aligned Language Models*](https://arxiv.org/abs/2307.15043) (GCG)
- Anil et al. 2024, [*Many-shot jailbreaking*](https://www.anthropic.com/research/many-shot-jailbreaking)
- Perez et al. 2022, [*Red Teaming Language Models with Language Models*](https://arxiv.org/abs/2202.03286)
- Sharma et al. 2025, [*Constitutional Classifiers*](https://arxiv.org/abs/2501.18837)
- Wallace et al. 2024, [*The Instruction Hierarchy*](https://arxiv.org/abs/2404.13208); Chen et al. 2024, [*StruQ*](https://arxiv.org/abs/2402.06363)
- Debenedetti et al. 2024, [*AgentDojo*](https://arxiv.org/abs/2406.13352) and 2025, [*Defeating Prompt Injections by Design*](https://arxiv.org/abs/2503.18813) (CaMeL)
- Beurer-Kellner et al. 2025, [*Design Patterns for Securing LLM Agents against Prompt Injections*](https://arxiv.org/abs/2506.08837)
- Shevlane et al. 2023, [*Model evaluation for extreme risks*](https://arxiv.org/abs/2305.15324) and Phuong et al. 2024, [*Evaluating Frontier Models for Dangerous Capabilities*](https://arxiv.org/abs/2403.13793)
- Mitchell et al. 2019, [*Model Cards for Model Reporting*](https://arxiv.org/abs/1810.03993)
- Anthropic, [*Responsible Scaling Policy*](https://www.anthropic.com/responsible-scaling-policy) (current version)

## 10 Applied LLM systems

**Read first**
1. Lewis et al. 2020, [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401)
2. Anthropic 2024, [*Building Effective Agents*](https://www.anthropic.com/engineering/building-effective-agents)
3. Liu et al. 2023, [*Lost in the Middle*](https://arxiv.org/abs/2307.03172)
4. Yao et al. 2022, [*ReAct*](https://arxiv.org/abs/2210.03629)

**Retrieval**
- Karpukhin et al. 2020, [*Dense Passage Retrieval*](https://arxiv.org/abs/2004.04906)
- Reimers and Gurevych 2019, [*Sentence-BERT*](https://arxiv.org/abs/1908.10084) and Wang et al. 2022, [*Text Embeddings by Weakly-Supervised Contrastive Pre-training*](https://arxiv.org/abs/2212.03533) (E5)
- Khattab and Zaharia 2020, [*ColBERT*](https://arxiv.org/abs/2004.12832)
- Nogueira and Cho 2019, [*Passage Re-ranking with BERT*](https://arxiv.org/abs/1901.04085)
- Cormack et al. 2009, *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods* (SIGIR 2009)
- Kusupati et al. 2022, [*Matryoshka Representation Learning*](https://arxiv.org/abs/2205.13147)
- Malkov and Yashunin 2016, [*HNSW*](https://arxiv.org/abs/1603.09320) and Jégou et al. 2011, *Product Quantization for Nearest Neighbor Search* (IEEE TPAMI)
- Gao et al. 2022, [*Precise Zero-Shot Dense Retrieval without Relevance Labels*](https://arxiv.org/abs/2212.10496) (HyDE)
- Anthropic 2024, [*Contextual Retrieval*](https://www.anthropic.com/news/contextual-retrieval)
- Thakur et al. 2021, [*BEIR*](https://arxiv.org/abs/2104.08663) and Muennighoff et al. 2022, [*MTEB*](https://arxiv.org/abs/2210.07316)

**RAG evaluation**
- Es et al. 2023, [*Ragas*](https://arxiv.org/abs/2309.15217) and Saad-Falcon et al. 2023, [*ARES*](https://arxiv.org/abs/2311.09476)
- Asai et al. 2023, [*Self-RAG*](https://arxiv.org/abs/2310.11511)

**Agents, tools and structured output**
- Schick et al. 2023, [*Toolformer*](https://arxiv.org/abs/2302.04761) and Patil et al. 2023, [*Gorilla*](https://arxiv.org/abs/2305.15334)
- Yang et al. 2024, [*SWE-agent: Agent-Computer Interfaces*](https://arxiv.org/abs/2405.15793)
- Anthropic 2025, [*Effective context engineering for AI agents*](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) and [*Writing effective tools for agents*](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Willard and Louf 2023, [*Efficient Guided Generation for Large Language Models*](https://arxiv.org/abs/2307.09702) (Outlines) and Dong et al. 2024, [*XGrammar*](https://arxiv.org/abs/2411.15100)
- Tam et al. 2024, [*Let Me Speak Freely?*](https://arxiv.org/abs/2408.02442)

**Cost and routing**
- Chen et al. 2023, [*FrugalGPT*](https://arxiv.org/abs/2305.05176)
- Ong et al. 2024, [*RouteLLM: Learning to Route LLMs with Preference Data*](https://arxiv.org/abs/2406.18665)

---

## Staying current

- **Daily:** skim [Hugging Face daily papers](https://huggingface.co/papers) (arXiv cs.CL and cs.LG, filtered by the community). Read titles; open at most two.
- **Weekly:** lab research blogs (Anthropic, OpenAI, Google DeepMind, Meta, DeepSeek, Qwen, AI2), the [Transformer Circuits Thread](https://transformer-circuits.pub), GPU MODE lectures.
- **Quarterly:** re-check the frontier sections of modules 05–07 and the benchmark table in [module 08](08-evaluation-and-research.md#12-the-benchmark-landscape-as-of-september-2026).
- A paper is not "read" until you can state its claim, the evidence and one experiment you would run next. Write those four lines, or it didn't happen.
