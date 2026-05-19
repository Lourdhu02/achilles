# Deep Learning — Concise Reference

**Status: outline. Expand depth during Months 2-4.**

Pair with `06-interview-qa-bank.md` Section 2.

---

## The fundamental loop

```
1. Forward pass: input → through layers → predictions
2. Compute loss: compare predictions to targets
3. Backward pass: compute gradients ∂L/∂θ
4. Update: θ ← θ - η · ∇L (with chosen optimizer)
5. Repeat over mini-batches and epochs
```

---

## Activation functions

| Activation | Formula | When use | Issues |
|---|---|---|---|
| Sigmoid | 1/(1+e^-x) | Output for binary classification | Vanishing gradients, saturates |
| Tanh | (e^x - e^-x)/(e^x + e^-x) | Hidden in RNNs (historical) | Vanishing gradients (less severe than sigmoid) |
| ReLU | max(0, x) | Standard hidden activation | Dying ReLU (zero gradients for x<0) |
| Leaky ReLU | max(0.01x, x) | Fix dying ReLU | Hyperparameter to tune |
| ELU | x if x>0 else α(e^x - 1) | Smooth alternative | Slightly more compute |
| GELU | x · Φ(x) (Gaussian CDF) | Transformers (default) | Slightly more compute than ReLU |
| Swish/SiLU | x · σ(x) | Modern alternative to ReLU | Smooth, self-gated |
| SwiGLU | (xW + b) · σ(xV + c) | Modern transformer FFN | Doubles parameter count of FFN |
| Softmax | exp(x_i)/Σexp(x_j) | Final layer for multiclass | Numerical stability needed (subtract max) |

---

## Loss functions

### Regression
- **MSE:** `(1/n) Σ (y - ŷ)²` — standard, smooth, sensitive to outliers
- **MAE:** `(1/n) Σ |y - ŷ|` — robust to outliers, not differentiable at 0
- **Huber:** MSE near 0, MAE far — best of both
- **Quantile loss:** for quantile regression

### Classification
- **Binary cross-entropy:** `-y log(p) - (1-y) log(1-p)`
- **Categorical cross-entropy:** `-Σ y_i log(p_i)` for one-hot y
- **Sparse categorical CE:** same, with y as integer class index (memory-efficient)
- **Focal loss:** `-α (1-p)^γ log(p)` — downweights easy examples (for imbalanced)
- **Label smoothing:** soften targets from 1.0 to 0.9, distribute 0.1 across other classes (regularizing)

### Embedding learning
- **Contrastive loss:** pulls similar pairs together, pushes dissimilar apart
- **Triplet loss:** anchor, positive, negative — relative distances
- **InfoNCE / NT-Xent:** SimCLR-style contrastive (used in CLIP, etc.)
- **Cosine embedding loss**

### Specialized
- **CTC loss:** for OCR, ASR (alignment-free sequence prediction) — you used **FocalCTCLoss**!
- **Dice loss:** for segmentation (handles class imbalance)
- **IoU loss:** for object detection
- **KL divergence:** for distillation, variational methods

---

## Optimizers

### SGD with momentum
```
v ← βv + (1-β)·∇L  (momentum term, typically β=0.9)
θ ← θ - η · v
```

Pros: Final accuracy often best (for CNN training). Predictable.
Cons: Slow convergence, sensitive to LR.

### RMSProp
```
s ← βs + (1-β)·(∇L)²
θ ← θ - η · ∇L / (√s + ε)
```

Idea: per-parameter adaptive LR. Larger gradients get smaller LRs.

### Adam (most common default)
```
m ← β₁·m + (1-β₁)·∇L            (1st moment, momentum)
v ← β₂·v + (1-β₂)·(∇L)²         (2nd moment, RMSProp-like)
m̂ ← m / (1 - β₁^t)             (bias correction)
v̂ ← v / (1 - β₂^t)
θ ← θ - η · m̂ / (√v̂ + ε)
```

Defaults: β₁=0.9, β₂=0.999, ε=1e-8.

### AdamW
Adam + decoupled weight decay (don't apply L2 to the adaptive update; apply directly to θ).
```
θ ← θ - η·(m̂/√v̂ + λθ)
```

**Why prefer AdamW:** Cleaner separation of regularization from gradient signal. Standard for transformer training.

### Lion (recent)
Sign-based momentum optimizer, faster + less memory than AdamW. Some papers report better.

### Comparing
- **For CNNs (image classification):** SGD+momentum often beats Adam in final accuracy
- **For transformers:** AdamW is default
- **For sparse/embedding-heavy:** Adagrad / AdaFactor

---

## Learning rate schedules

| Schedule | Description | When use |
|---|---|---|
| Constant | LR fixed | Rare in production; baseline |
| Step decay | Drop LR by factor every N epochs | Older CNNs |
| Cosine | Smooth cosine annealing | Modern default, transformers |
| Cosine with warmup | Linear ramp up, then cosine | Transformer training (must-have) |
| Polynomial | LR(t) = LR_0 · (1 - t/T)^p | LARS-style, large batch |
| One-cycle | LR up then down, momentum opposite | fast.ai default |
| ReduceLROnPlateau | Drop when val metric plateaus | Adaptive, no manual schedule |

**Warmup matters for transformers:** First few thousand steps with very small LR avoids early instability when gradient noise is high.

---

## Regularization techniques

### Dropout
- Randomly zero out p% of activations at each forward pass during training
- At inference: scale activations by (1-p), OR use "inverted dropout" during training
- Effect: implicit ensemble of subnetworks

### Batch Normalization (BN)
- Normalize activations within a batch: `(x - μ_B) / σ_B`
- Learnable scale γ and shift β: `γ · (x - μ) / σ + β`
- Effect: stabilizes training, allows higher LR, mild regularization
- Issues: small batch sizes → noisy stats; doesn't work in RNN well

### Layer Normalization (LN)
- Normalize across features WITHIN a single sample
- No batch dependency
- Standard in transformers

### Group Normalization (GN)
- Split features into groups, normalize within group
- Works for very small batches (e.g., 1-2)

### Weight decay
- L2 regularization on parameters
- In Adam, use AdamW (decoupled)

### Label smoothing
- Targets: 1 → 0.9 (true class), 0 → 0.1/(K-1) (others)
- Effect: softer training signal, often improves generalization + calibration

### Data augmentation
- Image: random crop, flip, color jitter, mixup, cutmix, augmix
- Text: back-translation, synonym replacement, EDA
- Audio: SpecAugment

### Stochastic depth
- Randomly drop entire residual blocks during training (like dropout but for layers)

### MixUp
- Linearly interpolate two training examples and their labels
- `x = λ·x_i + (1-λ)·x_j, y = λ·y_i + (1-λ)·y_j`

### Early stopping
- Track val loss, stop when it stops improving for N epochs

---

## CNNs

### Convolution operation
- Slide a kernel over input, dot-product at each position
- Parameters: kernel size, stride, padding, dilation, # filters

### Pooling
- Max pooling: take max in each window
- Average pooling: take mean in each window
- Global average pooling: pool over whole spatial dim (modern replacement for FC layer before classifier)

### Receptive field
- The input region a particular neuron "sees"
- Grows as you go deeper
- Dilated convolutions grow receptive field without more params

### Common architectures
- **LeNet:** classic, small
- **AlexNet:** 2012 ImageNet, ReLU + dropout introduced
- **VGG:** simple, deep (16-19 layers), uniform 3x3 convs
- **GoogLeNet / Inception:** parallel paths with different kernel sizes
- **ResNet:** skip connections, 50/100/152 layer variants
- **DenseNet:** dense connections (each layer to all subsequent)
- **EfficientNet:** scaled depth + width + resolution
- **ConvNeXt:** modern revival of CNNs competitive with ViTs

### ResNet skip connection math
```
y = F(x, {W_i}) + x
```
Why: identity mapping is always available; gradients flow through skip; deeper networks become trainable.

---

## RNNs (still asked, increasingly historical)

### Vanilla RNN
- `h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b)`
- Vanishing gradient problem: products of derivatives shrink exponentially through time
- Exploding gradient: opposite extreme; fixed by clipping

### LSTM (Long Short-Term Memory)
Four gates: forget (f), input (i), candidate (g), output (o)
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

c_t = f_t · c_{t-1} + i_t · g_t      (cell state)
h_t = o_t · tanh(c_t)                 (hidden state)
```

Key: cell state c provides a "highway" for gradients, mitigating vanishing.

### GRU
Simpler LSTM variant: combine forget+input → "update gate"; merge h and c.
- Update gate z_t, reset gate r_t
- Two gates instead of four
- Comparable performance, fewer params

### When still relevant
- Constrained-resource scenarios
- Streaming inference
- Sometimes still used as encoder layers in mixed architectures

---

## Generative models (high level)

### Autoencoders / VAEs
- Encoder: x → z (latent)
- Decoder: z → x'
- VAE: z is distribution; loss = reconstruction + KL divergence to prior

### GANs
- Generator G: noise → fake data
- Discriminator D: real vs fake
- Minimax game: G fools D; D distinguishes
- Failure modes: mode collapse, training instability

### Diffusion models
- Forward process: progressively add noise to data
- Reverse process: learn to denoise step by step
- Implementation: U-Net predicts noise at each step
- DDPM, DDIM, Stable Diffusion, Flow Matching

---

## Training stability and tricks

- **Mixed precision (fp16/bf16):** train with lower precision, master copy of weights in fp32
- **Gradient accumulation:** accumulate gradients over N micro-batches before update (simulate large batch)
- **Gradient checkpointing:** trade compute for memory; recompute activations during backward
- **Loss scaling (for fp16):** multiply loss by large factor to prevent gradient underflow
- **EMA (Exponential Moving Average) of weights:** smoother final model
- **Stochastic weight averaging (SWA):** average weights from multiple late-training checkpoints

---

## Common DL pitfalls (interview gotchas)

1. **Forgetting `.eval()` mode** — BN/dropout behave differently
2. **Not using `with torch.no_grad():`** during eval — wastes memory
3. **Data leakage** — fitting scaler on full data before split
4. **Mismatch in loss + activation** — using softmax+CE when you should use logits+CE (numerical stability)
5. **Reinitializing optimizer state** between epochs (kills Adam's momentum)
6. **Forgetting to shuffle DataLoader** for training
7. **Wrong axis for mean/sum** in custom losses

---

End. Next: `04-transformers-deepdive.md` (the file you'll spend the most time on).
