# Classical ML — Concise Reference

**Status: outline. Expand depth as needed during Month 2-3.**

---

## How to use this file

This is the reference card for classical ML topics. For each:
- One-paragraph definition
- Key equations
- When to use / when not to use
- Gotchas to mention in interviews

Pair this with `06-interview-qa-bank.md` Section 1 for practice questions.

---

## Linear Regression

**Definition:** Predicts continuous target as a linear combination of features. Loss: MSE.

**Math:**
- Model: `y = Xβ + ε`
- Closed form: `β = (X^T X)^(-1) X^T y`
- Gradient: `∇L = (2/n) X^T (Xβ - y)`

**Assumptions:**
- Linearity, independence, homoscedasticity, normality of residuals, no multicollinearity

**When to use:** Baseline. Inferential analysis. Quick prototype.
**When NOT:** Non-linear relationships, high-dim sparse data, when prediction > interpretability is the goal.

**Gotchas:**
- Sensitive to outliers
- Multicollinearity inflates variance of estimates
- Closed form is O(d²n + d³); GD is better for large d
- Regularize for high-dim: Ridge (L2), Lasso (L1)

---

## Logistic Regression

**Definition:** Linear model + sigmoid for binary classification.

**Math:**
- `P(y=1|x) = σ(w^T x + b)` where `σ(z) = 1/(1+e^(-z))`
- Loss: Binary cross-entropy `-y·log(p) - (1-y)·log(1-p)`
- Gradient: `(σ(w^T x) - y) · x`

**Multiclass extension:** Softmax regression / multinomial logistic
- `P(y=k|x) = exp(w_k^T x) / Σ exp(w_j^T x)`

**When to use:** Baseline classification, interpretable, calibrated probabilities, low-dim.
**When NOT:** Highly non-linear boundaries, very high-dim, when calibration isn't important.

**Gotchas:**
- Always calibrate if probabilities matter
- Threshold tuning matters for imbalanced classes
- Class weights or upweighting for imbalance

---

## Decision Trees

**Splitting criterion:**
- Classification: Gini = Σ p_i(1-p_i); Entropy = -Σ p_i log p_i
- Regression: MSE / variance reduction

**Pros:** Interpretable, handles non-linearity, no need to scale features.
**Cons:** Easily overfits, unstable (small data change → big tree change), can't extrapolate.

**Control overfitting:** max_depth, min_samples_split, min_samples_leaf, max_features, pruning.

---

## Random Forest

**Idea:** Bagging of decision trees, each trained on bootstrap sample + random subset of features.

**Why it works:** Reduces variance (high-variance base learners + averaging). Decorrelation via feature randomness.

**Hyperparameters:** n_estimators, max_features (usually sqrt(d)), max_depth.

**When NOT:** Need extreme interpretability, very high-dim sparse text (XGBoost or linear better).

---

## Gradient Boosting (XGBoost, LightGBM, CatBoost)

**Idea:** Sequentially add trees that fit residuals of previous prediction. Each tree minimizes a loss + complexity penalty.

**XGBoost specifics:**
- Second-order Taylor approximation of loss
- Built-in regularization (L1, L2)
- Handles missing values natively
- Tree growth: level-wise (XGB) or leaf-wise (LightGBM)

**LightGBM specifics:**
- Histogram-based (faster than exact)
- Leaf-wise growth (faster, but can overfit small datasets)
- Better for very large datasets

**CatBoost specifics:**
- Ordered boosting (less overfit to target)
- Native handling of categorical features

**Hyperparameters (XGBoost):** learning_rate (eta), n_estimators, max_depth, min_child_weight, subsample, colsample_bytree, gamma.

**Tuning order:** learning_rate + n_estimators → max_depth + min_child_weight → subsample/colsample → gamma/lambda.

**When to use:** Tabular data, ranking, structured features. Often beats deep learning on tabular.
**When NOT:** Image, text, audio (DL wins). Need streaming-friendly model.

---

## SVM (Support Vector Machine)

**Idea:** Find hyperplane that maximizes margin between classes.

**Math:**
- Linear SVM: minimize `(1/2)||w||²` subject to `y_i(w^T x_i + b) ≥ 1`
- Soft margin: add slack variables ξ_i, minimize `(1/2)||w||² + C Σ ξ_i`

**Kernel trick:** Implicit feature mapping. Common kernels:
- Linear: `K(x, y) = x^T y`
- RBF: `K(x, y) = exp(-γ||x-y||²)`
- Polynomial: `K(x, y) = (γ x^T y + r)^d`

**Pros:** Strong theoretical foundation, effective in high-dim, kernel flexibility.
**Cons:** Doesn't scale well (O(n²) to O(n³) for kernels), no native probabilities (need Platt scaling).

**When to use in 2026:** Mostly classical settings (small-medium structured data), some bioinformatics.
**When NOT:** Large datasets, deep features available, need calibrated probabilities.

---

## K-Means

**Algorithm:**
1. Initialize K centroids (kmeans++ smart init recommended)
2. Assign each point to nearest centroid
3. Update centroids to mean of assigned points
4. Repeat until convergence

**Choosing K:**
- Elbow method (within-cluster SSE vs K)
- Silhouette score
- Gap statistic
- Domain knowledge

**When fails:**
- Non-globular clusters (elongated, donut-shaped)
- Different cluster densities
- Different cluster scales (always normalize!)
- High dimensionality (curse)

**Variants:** K-Medoids (PAM, more robust to outliers), MiniBatch KMeans (scalable).

---

## DBSCAN

**Idea:** Density-based clustering. Two hyperparameters: ε (radius), minPts (min neighbors to be "core").

**Pros:** No K needed, handles arbitrary cluster shapes, identifies outliers naturally.
**Cons:** Sensitive to ε, struggles with varying density.

**When use:** Spatial data, anomaly detection, irregular cluster shapes.

---

## Naive Bayes

**Math:**
`P(y | x_1, ..., x_d) ∝ P(y) Π P(x_i | y)`

The "naive" assumption: features are conditionally independent given class.

**Variants:**
- Multinomial NB (text classification)
- Gaussian NB (continuous features)
- Bernoulli NB (binary features)

**When use:** Text classification baseline, very fast, works surprisingly well.
**When NOT:** Strong feature correlations, need calibrated probabilities.

---

## Feature Engineering Patterns

### Encoding categoricals:
- **One-hot:** simple but blows up dim
- **Label encoding:** for ordinal categories only (or trees that don't care)
- **Target encoding:** mean of target per category; risk of leakage (use k-fold)
- **Frequency encoding:** count or % occurrence
- **Embedding:** learn dense rep (for high-cardinality)
- **Hash encoding:** for very high cardinality (collision risk)

### Handling missingness:
- Drop (only if <5% and random)
- Impute mean/median/mode (simple)
- Impute by KNN, regression, MICE (sophisticated)
- Indicator feature for "was missing" (often valuable)
- Use models that handle it natively (XGBoost, LightGBM)

### Scaling:
- **Standardization** (zero mean, unit variance): for distance-based, linear, NN
- **Min-max normalization**: for bounded features, neural net input
- **Robust scaling** (IQR-based): when outliers present
- **No scaling needed**: for tree-based models

### Other:
- **Log transform:** for skewed positive features
- **Box-Cox / Yeo-Johnson:** for general skew
- **Binning:** continuous → discrete (can capture non-linearity in linear models)
- **Polynomial features:** explicit interactions
- **Interaction features:** product of two features

---

## Evaluation Metrics for Classical ML

### Classification

| Metric | When to use | Math |
|---|---|---|
| Accuracy | Balanced classes | (TP+TN)/(TP+TN+FP+FN) |
| Precision | Cost of FP is high | TP/(TP+FP) |
| Recall | Cost of FN is high | TP/(TP+FN) |
| F1 | Balance of P and R | 2PR/(P+R) |
| AUC-ROC | Threshold-independent ranking | Area under TPR vs FPR curve |
| AUC-PR | Imbalanced classes | Area under P vs R curve |
| Log loss | Probabilistic predictions | -y·log(p) - (1-y)·log(1-p) |
| MCC | Balanced metric, all 4 cells | See Matthews coeff. formula |

### Regression

| Metric | When |
|---|---|
| MSE | Standard, penalizes large errors |
| RMSE | Same units as target |
| MAE | Robust to outliers |
| MAPE | Percentage-based; fails near zero |
| R² | Variance explained |

### Calibration

- Reliability diagram (predicted prob vs actual)
- Brier score
- Expected Calibration Error (ECE)

When models are poorly calibrated: Platt scaling (sigmoid), isotonic regression.

---

## Imbalanced classification

When positive class is rare (1-5%):

### Data-level:
- Random oversampling minority
- SMOTE (synthetic samples via interpolation)
- ADASYN (focus on hard examples)
- Random undersampling majority
- Tomek links / Edited Nearest Neighbor

### Algorithm-level:
- Class weights in loss
- Focal loss (downweights easy examples)
- Threshold tuning post-hoc

### Eval-level:
- Don't use accuracy
- Use AUC-PR, recall@k, F-beta

---

## Bias-Variance Tradeoff

```
Total error = Bias² + Variance + Irreducible noise
```

**High bias (underfit):** model too simple. Train + val both poor.
- Fix: more capacity, more features, less regularization

**High variance (overfit):** model too complex. Train good, val poor.
- Fix: more data, simpler model, regularization, ensembling

---

## Curse of dimensionality

In high-dim:
- Distance measures lose meaning (all points become "far")
- Data becomes sparse
- More data required to estimate density
- Many algorithms degrade (KNN, K-means, SVM RBF)

Mitigations: feature selection, dimensionality reduction (PCA, t-SNE, UMAP), regularization.

---

## When to expand this file

Each section above is a one-pager. Expand to deeper notes when:
- You're studying that topic specifically in your weekly plan
- An interview revealed a gap
- A specific company asks deep questions in that area

Don't pre-emptively expand everything. Just-in-time depth is more efficient.

---

End. Next: `03-deep-learning.md` (similar structure, similarly intentional brevity).
