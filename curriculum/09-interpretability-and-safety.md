# 09 — Interpretability and safety

Lab: [16 interpretability](../labs/16_interpretability/README.md). Essential for Anthropic-style roles and for anyone deploying agents.

## 1. Mechanistic interpretability
- **Residual stream view:** layers read and write linear subspaces of a shared stream. Attention heads decompose into QK circuits (where to look) and OV circuits (what to move).
- **Induction heads:** a previous-token head in an early layer plus a head that finds "the token after the previous occurrence of the current token" (K-composition). They drive much of in-context learning and form in a visible phase change during training.
- **Superposition:** when features are sparse, models pack more features than dimensions into almost-orthogonal directions, so single neurons are polysemantic (*Toy Models of Superposition*).
- **Sparse autoencoders / dictionary learning:** reconstruct activations as sparse combinations of learned directions (`MSE + λ‖f‖₁`, or TopK/JumpReLU). They yield interpretable features. Watch dead latents, feature splitting, reconstruction loss vs L0, and downstream-loss impact.
- **Causal methods:** activation patching (clean → corrupted) localizes computation; attribution patching approximates it with gradients; path patching isolates circuits (the IOI circuit). Steering vectors and representation engineering add directions to change behavior. Attribution graphs / circuit tracing (Anthropic 2025) trace features through replacement models.
- **Caveats:** probes find correlations, not use; interpretability illusions exist; always intervene causally.

## 2. Alignment failure modes to be able to discuss concretely
Reward hacking and specification gaming; sycophancy (it is rewarded by human raters); unfaithful chain of thought; alignment-faking behavior in evaluations (Greenblatt et al. 2024); backdoors that persist through safety training (Sleeper Agents); sandbagging; obfuscation when a CoT monitor is optimized against.

## 3. Misuse and security (agent builders must master this)
Jailbreaks (role-play, many-shot, adversarial suffixes like GCG); **prompt injection**, especially indirect injection via retrieved documents, web pages and tool outputs. Defenses in depth: least-privilege tools, human confirmation for irreversible actions, separating trusted instructions from untrusted data, input/output classifiers (Constitutional Classifiers), and monitoring. Assume any text the model reads can be adversarial.

## 4. Governance and practice
Dangerous-capability evals (cyber, bio, autonomy), red teaming, responsible scaling policies and frontier safety frameworks, system and model cards, staged deployment. Know Anthropic's RSP and at least one other lab's framework well enough to compare them.

## 5. Your position
Frontier labs ask what you think and why. Write, in [career/stories.md](../career/stories.md#values-and-mission-prep), 2 paragraphs each on: the most important unsolved safety problem; what you would refuse to build; how you have traded capability for safety in your own work (FinSentinelAI's local, privacy-first design is a real example). Be specific, honest and nuanced; pandering is detected quickly.

**Read:** A Mathematical Framework for Transformer Circuits; In-context Learning and Induction Heads; Toy Models of Superposition; Towards Monosemanticity; Scaling Monosemanticity; OpenAI's TopK SAEs (Gao et al. 2024); ROME; IOI; Concrete Problems in AI Safety; Sleeper Agents; Alignment Faking; Towards Understanding Sycophancy; Greshake et al. (indirect prompt injection); Constitutional Classifiers. Course: **ARENA** (free; interpretability and RL chapters).
