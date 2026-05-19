# 03 — ML Fundamentals

Your ML knowledge needs to be **deep where it matters, broad where it doesn't.**

You can't be a research scientist on every topic. But you must be able to:
- Explain core concepts crisply
- Implement key algorithms from scratch
- Reason about tradeoffs
- Defend choices in your projects

---

## Files in this folder

1. [`01-knowledge-pyramid.md`](./01-knowledge-pyramid.md) — What to know cold vs. broadly vs. just-recognize
2. [`02-classical-ml.md`](./02-classical-ml.md) — Linear/logistic regression, trees, SVM, clustering, ensembles
3. [`03-deep-learning.md`](./03-deep-learning.md) — Neural net basics, CNNs, RNNs, optimization, regularization
4. [`04-transformers-deepdive.md`](./04-transformers-deepdive.md) — Attention math, multi-head, positional, training dynamics
5. [`05-math-essentials.md`](./05-math-essentials.md) — Linear algebra, probability, info theory, calculus you'll need
6. [`06-interview-qa-bank.md`](./06-interview-qa-bank.md) — 200+ ML interview Q&A, the practice bank

---

## The mental model

ML interviews probe at 3 depths:

```
DEPTH 1 — Definition: "What is gradient descent?"
DEPTH 2 — Mechanism: "Walk me through how Adam differs from SGD with momentum"
DEPTH 3 — Tradeoffs: "When would you NOT use Adam? Why?"
```

By Month 6, you should be **Depth 3 on every fundamental topic**.

---

## Topics by importance for your target roles (GenAI Eng)

### Tier 1: Must know at Depth 3 (every concept rock-solid)

- **Backpropagation** — forward + backward pass, gradient computation, chain rule
- **Optimization** — SGD, momentum, Adam, AdamW, learning rate schedules, warmup, cosine decay
- **Loss functions** — Cross-entropy (binary + multiclass + multi-label), MSE, focal loss, contrastive
- **Attention + transformers** — full math, multi-head, positional encoding, masking, KV cache
- **Tokenization** — BPE, SentencePiece, WordPiece, vocabulary effects
- **Embeddings** — what they are, how they're trained, contrastive learning
- **Eval metrics** — Precision/recall/F1, ROC/AUC, mAP, BLEU/ROUGE/METEOR, perplexity, accuracy@k

### Tier 2: Must know at Depth 2 (mechanism clear)

- **CNNs** — convolutions, pooling, receptive fields, common architectures (ResNet, EfficientNet)
- **RNNs / LSTMs / GRUs** — vanishing gradients, gating mechanisms (less critical now, but asked)
- **Regularization** — L1/L2, dropout, batch/layer norm, data augmentation
- **Generative models** — VAEs, GANs, diffusion (at conceptual level)
- **Classical ML** — linear/logistic regression, decision trees, random forests, gradient boosting (XGBoost, LightGBM)
- **Clustering** — K-means, DBSCAN, hierarchical
- **Bayesian basics** — Bayes rule, naive bayes, MAP/MLE

### Tier 3: Depth 1 (just recognize and explain at high level)

- **Reinforcement learning** — basic terminology (state, action, reward, policy, Q-learning)
- **Graph neural networks** — GCN, GAT, message passing
- **Time series** — ARIMA, Prophet, basic forecasting
- **Recommender systems** — collaborative filtering, content-based, hybrid
- **Causal inference** — at high level

### Tier 4: Don't even bother (unless interviewing for a specialist role)

- Advanced theory: PAC learning, VC dimension, statistical learning bounds
- Specialized: federated learning, neuromorphic computing, quantum ML

---

## Study order (matches the 15-month roadmap)

### Months 1-4 (Phase 1)
- Andrew Ng's Coursera "ML Specialization" — speed-run for refresher (covers Tier 1-2 classical)
- "Deep Learning Specialization" by Andrew Ng — for DL fundamentals
- Implement: linear regression, logistic regression, simple neural net (forward + backward) FROM SCRATCH in numpy
- Build: 1-page cheatsheets for each Tier 1 topic

### Months 5-8 (Phase 2)
- Focus shifts to GenAI/LLM mastery (`04-genai-llm-mastery/`)
- Maintain ML fundamentals via Q&A bank (`06-interview-qa-bank.md`)
- Implement: from-scratch transformer (encoder + decoder)
- Read: "The Annotated Transformer" + "Illustrated Transformer"

### Months 9-15 (Phase 3-4)
- Active recall via Q&A bank
- Mock interview drilling on ML breadth questions
- Per-company review: e.g., before Google interview, refresh attention math; before Meta, refresh ranking metrics

---

## How to study (most efficient method)

### The Feynman technique
1. Pick a topic (e.g., "How does backpropagation work?")
2. Explain it OUT LOUD to an imaginary 5-year-old (or write it in plain English)
3. When you get stuck, that's your gap — go back to source material
4. Re-explain after closing source

This is 10x more effective than passive reading.

### Implement from scratch
You'll be asked to implement key algorithms in interviews. Common ones:

- **Linear regression** with gradient descent (no scikit-learn)
- **K-means**
- **Logistic regression**
- **Forward + backward pass** for a 2-layer MLP
- **Single-head attention**
- **Multi-head attention**
- **Softmax + cross-entropy** with manual gradient

Have these in a personal `from_scratch/` folder. Practice them every few weeks.

### Build a personal cheatsheet
For every topic in Tier 1, write a 1-page personal cheatsheet covering:
- One-sentence definition
- Math notation
- When you'd use it
- When you wouldn't
- Common gotchas

Review these in the week before any interview.

---

## Common ML interview traps

### Trap 1: "What's the difference between L1 and L2 regularization?"
- Bad answer: "L1 adds absolute values, L2 adds squares."
- Good answer: "L1 promotes sparsity (drives weights exactly to zero), L2 promotes small weights but not zero. L1 is non-differentiable at zero (need subgradient), L2 is smooth. Use L1 for feature selection, L2 when you want all features to contribute weakly."

### Trap 2: "How would you handle imbalanced classes?"
- Bad: "Oversampling."
- Good: Discuss multiple approaches (class weights, oversampling, undersampling, SMOTE, focal loss, threshold tuning) AND when each is appropriate AND their failure modes.

### Trap 3: "Why does Adam usually work better than SGD?"
- Bad: "It's adaptive."
- Good: Discuss adaptive learning rates per-parameter, momentum-like effect via first moment, RMSProp-like effect via second moment, AND when SGD-with-momentum outperforms (often in CNN training to final performance), AND bias correction terms.

### Trap 4: "Explain attention."
- Bad: "It's Q, K, V dot products with softmax."
- Good: Motivation (capturing long-range dependencies that RNNs struggle with), math (scaled dot product, why scaling), multi-head intuition (different "perspectives"), positional encoding (transformers lose order), self-attention vs cross-attention.

### Trap 5: "What's overfitting?"
- Bad: "When the model memorizes training data."
- Good: Bias-variance tradeoff, train/val/test split, signs of overfitting (low train loss + high val loss), regularization strategies, AND how to detect/fix.

---

## When you find a gap

You'll discover gaps constantly. Process:

1. Note the gap in a single file `gaps.md` (or per-topic note)
2. Schedule a 30-min slot in the week to study it
3. When you study it, write a cheatsheet entry
4. Quiz yourself 1 week later

Don't let gaps accumulate. They compound into "I don't know enough" anxiety.

---

## Resources

### Free:
- **Andrew Ng's ML & DL Specializations** (Coursera) — gold standard for fundamentals
- **fast.ai Practical Deep Learning** — hands-on, code-first
- **3Blue1Brown Neural Networks playlist** — best math intuition videos
- **Stanford CS229 (Andrew Ng's lectures)** — deeper theory
- **Stanford CS224N (NLP)** — for NLP specifically
- **Stanford CS231N (CV)** — for CV specifically
- **The "Illustrated" series** by Jay Alammar — illustrated transformer, illustrated BERT, etc.

### Paid (worth it):
- **"Deep Learning" by Goodfellow, Bengio, Courville** — the textbook ($60)
- **"Pattern Recognition and Machine Learning" by Bishop** — classical reference
- **"Elements of Statistical Learning" by Hastie/Tibshirani** — statistical ML reference
- **fast.ai's "Practical Deep Learning"** book version

### Q&A and interview-specific:
- **"Machine Learning Interviews" book by Khang Pham** (free PDF online)
- **"Designing Machine Learning Systems" by Chip Huyen**
- **"Acing the AI Interview"** (various, search latest)
- **"Machine Learning System Design Interview" by Ali Aminian** — also covers ML breadth

---

Next: [`01-knowledge-pyramid.md`](./01-knowledge-pyramid.md) for what depths to target per topic.
