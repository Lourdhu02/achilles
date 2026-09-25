# 06 — Post-training: SFT, preferences, RL

Labs: [10 LoRA](../labs/10_lora/README.md), [11 DPO](../labs/11_dpo/README.md), [12 GRPO](../labs/12_grpo/README.md).

## 1. SFT
Train on (prompt, response) with loss **only on response tokens** (ignore_index on the prompt). Use chat templates with special tokens that user text can never produce. Packing with correct masking. Quality ≫ quantity (LIMA: ~1k curated examples teach the format). SFT is forward-KL imitation: it covers every behavior in the data, including the bad ones.

## 2. Parameter-efficient fine-tuning
LoRA: `W + (α/r)·BA`, B initialized to 0, so training starts exactly at the base model. It trains ~0.1–1% of parameters and merges for free at inference. QLoRA: a frozen NF4 4-bit base with bf16 LoRA adapters, the way to fine-tune 7–8B on 8 GB. Target all linear layers for best quality. LoRA learns less and forgets less than full fine-tuning; know when that is the right trade. Serve many adapters on one base (S-LoRA, Punica).

## 3. Reward models and RLHF with PPO
Reward model: Bradley–Terry on human comparisons, `−log σ(r(y_w) − r(y_l))`. RLHF objective: `max E[r(x, y)] − β·KL(π‖π_ref)`. **PPO**: clipped surrogate `min(ρA, clip(ρ, 1±ε)A)`, a learned value function, advantages from GAE, a per-token KL penalty, and four models in memory (policy, reference, reward, value). Failure modes: reward hacking and overoptimization (Gao et al.: true reward peaks and then falls as KL grows), length exploitation, sycophancy.

## 4. DPO and its family
The KL-regularized objective has the closed form `π*(y|x) = π_ref(y|x)·exp(r(x,y)/β)/Z(x)`. Invert it for r and substitute into Bradley–Terry; **Z cancels**, giving
`L_DPO = −log σ(β[log π(y_w)/π_ref(y_w) − log π(y_l)/π_ref(y_l)])`.
It needs no reward model and no sampling, and training is stable. Weaknesses: it is offline (no exploration), both log-likelihoods can *decrease* ("likelihood displacement"), and it can exploit length. Variants: **IPO** (squared loss, resists overfitting deterministic preferences), **KTO** (unpaired thumbs up/down), **SimPO** (length-normalized, reference-free, with a margin), **ORPO**, and online or iterative DPO (sample fresh pairs from the current policy).

## 5. RL with verifiable rewards: GRPO and friends
For math, code and other checkable tasks, the reward is a program: the answer checker, unit tests. **GRPO** (DeepSeekMath): sample G completions per prompt, use the *group-normalized* reward as the advantage (no value network), a PPO-style clipped ratio, and a KL penalty via the k3 estimator. DeepSeek-R1 showed that large-scale RLVR produces long chain-of-thought reasoning; R1-Zero trained with RL only, without SFT first.
Known fixes, each an interview topic:
- **Length bias:** per-sequence mean normalization rewards long wrong answers less harshly, so use a token-level mean (DAPO) or a constant normalizer (Dr. GRPO).
- **Std normalization** over-weights easy or hard groups (Dr. GRPO drops it).
- **Entropy collapse:** use clip-higher (an asymmetric ε, from DAPO).
- **Zero-signal groups:** if all G samples are right (or all wrong) the advantages are zero, so filter and resample them (dynamic sampling).
- **Reward hacking:** format and reward exploits, test-case special-casing in code. Audit samples by reading them.
- Infrastructure: rollouts dominate cost. Separate inference workers (vLLM/SGLang) from trainers, sync weights, and manage off-policy staleness.

## 6. Reasoning and test-time compute
Chain of thought; self-consistency (majority vote); best-of-N with a verifier; outcome vs **process** reward models (Let's Verify Step by Step); search over steps; budget forcing (s1). Compute-optimal test-time scaling (Snell et al. 2024): on easy and medium problems, extra inference compute can beat a bigger model. Distill reasoning traces into small models (the R1-distill models). Monitor the chain of thought; don't optimize directly against a CoT monitor, which teaches the model to obfuscate (Baker et al. 2025).

## 7. Distillation and AI feedback
Logit KD with temperature; sequence-level KD (train on teacher samples); **on-policy distillation** (student samples, teacher scores its tokens, reverse KL), which is cheap and strong. Constitutional AI and RLAIF: a model critiques and ranks outputs against written principles.

## 8. An 8 GB practical path
Qwen2.5-0.5B-Instruct: LoRA SFT → DPO on 500 pairs → GRPO on arithmetic with a verifier. Every step evaluated with CIs. That is a portfolio piece that shows the whole modern stack.

**Read:** InstructGPT; PPO; GAE; DPO; IPO; KTO; SimPO; Gao et al. (RM overoptimization); DeepSeekMath (GRPO); DeepSeek-R1; DAPO; Dr. GRPO; Tülu 3 (RLVR); Let's Verify Step by Step; Snell et al. 2024; LoRA; QLoRA; GKD (on-policy distillation); Constitutional AI; Nathan Lambert's *RLHF Book*.
