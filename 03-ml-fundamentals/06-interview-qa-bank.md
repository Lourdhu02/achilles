# ML Interview Q&A Bank

The single largest practice resource in this repo. **200+ questions** organized by topic.

**How to use:**
1. Don't read passively. Pick a topic. Try to answer OUT LOUD before reading my notes.
2. After answering, compare. Identify your gaps.
3. Mark questions you struggled on. Re-attempt in 2 weeks.
4. Use this as a quizzing source — paste 10 questions into a chat with me (Claude) and have me grade your answers.

**Difficulty markers:** [E]asy, [M]edium, [H]ard

---

## SECTION 1: Classical ML

### Linear & Logistic Regression

1. [E] What is linear regression? When does it fail?
2. [E] Derive the closed-form solution for linear regression. What's its computational complexity? Why might we prefer gradient descent?
3. [M] Why is the loss function for linear regression usually MSE? What if outliers are present?
4. [M] What's logistic regression? Why use sigmoid? Why not just use linear regression for binary classification?
5. [M] Derive the gradient of the logistic regression loss function (binary cross-entropy).
6. [M] What is multicollinearity? How does it affect linear regression? How do you detect and handle it?
7. [H] When would you use Ridge vs Lasso vs Elastic Net? Walk through the geometric intuition.
8. [H] What is the LASSO's effect on feature selection mathematically?

### Decision Trees & Ensembles

9. [E] How does a decision tree decide where to split?
10. [M] What's the difference between Gini impurity and entropy? When does the choice matter?
11. [M] Why are decision trees prone to overfitting? How do you prevent it?
12. [M] Explain Random Forest in 60 seconds. Why is it more robust than a single tree?
13. [M] What is bagging vs boosting? Pros/cons?
14. [H] Walk through how XGBoost works. What makes it different from GBM?
15. [H] How does LightGBM differ from XGBoost? When would you prefer one over the other?
16. [H] What is feature importance in XGBoost? Multiple ways to measure it?

### SVM & Kernel Methods

17. [M] What's the intuition behind SVM? What's the "margin"?
18. [M] What's the kernel trick? Give two examples of kernels.
19. [H] How does SVM handle non-separable data?
20. [H] When would you NOT use SVM in 2026? (Hint: deep learning has subsumed many SVM use cases)

### Clustering

21. [E] How does K-means work? What's its convergence criterion?
22. [M] How do you choose K in K-means? Multiple methods?
23. [M] What's the difference between K-means and K-medoids?
24. [M] When does K-means fail? (Hint: non-globular clusters, different scales)
25. [H] Explain DBSCAN. When is it preferable to K-means?
26. [H] What's the difference between hierarchical clustering's "single linkage" vs "complete linkage"?

### Probabilistic Models

27. [E] State Bayes theorem. Give an example.
28. [M] What is Naive Bayes? Why "naive"?
29. [M] What's the difference between MLE and MAP estimation?
30. [H] Explain the EM algorithm at a high level.

### Feature Engineering

31. [E] What are common types of features? (numerical, categorical, ordinal, text, image, time)
32. [M] How do you encode categorical features? Tradeoffs of one-hot vs label vs target encoding?
33. [M] How do you handle missing data? Multiple strategies + their failure modes.
34. [M] What's normalization vs standardization? When use each?
35. [H] What's feature crossing? Give an example useful for recommendation systems.
36. [H] How do you handle high-cardinality categorical features?

---

## SECTION 2: Deep Learning Foundations

### Neural Network Basics

37. [E] What is a neural network at the highest level? Why does it work?
38. [E] What are activation functions? Why do we need them? List 5 common ones.
39. [M] Why is ReLU preferred over sigmoid for hidden layers in modern networks?
40. [M] What are vanishing and exploding gradients? Why do they happen?
41. [M] Derive the backprop equations for a 2-layer MLP.
42. [H] What is the universal approximation theorem? What are its limitations?

### Optimization

43. [E] What is gradient descent? What are batch GD, SGD, and mini-batch GD?
44. [M] Explain momentum in optimization. Why does it help?
45. [M] What is RMSProp? What's its core idea?
46. [M] What is Adam? Combine of which two ideas?
47. [M] What's the difference between Adam and AdamW?
48. [M] When would you prefer SGD with momentum over Adam? (Hint: image classification often does)
49. [H] Explain warmup. Why is it especially important for transformer training?
50. [H] What is cosine learning rate schedule? Why is it popular?
51. [H] What's gradient clipping? When do you need it?

### Regularization

52. [E] What is overfitting? Three ways to detect it.
53. [E] What is dropout? Why does it work?
54. [M] What's the difference between L1 and L2 regularization? Geometric intuition?
55. [M] What's batch normalization? Why does it accelerate training?
56. [M] What's layer normalization? Why is it preferred over BN for transformers?
57. [M] What's early stopping?
58. [H] Why is dropout sometimes "turned off" at inference time? What's the math?
59. [H] What is label smoothing? Why does it improve generalization?

### CNNs

60. [E] What is a convolution? Why are CNNs good for images?
61. [M] What's a receptive field? How does it grow through a CNN?
62. [M] What's pooling? Max vs average — when to use which?
63. [M] What's the role of 1x1 convolutions? (Hint: channel mixing, dimensionality reduction)
64. [M] Explain ResNet skip connections. Why do they help?
65. [H] Walk through the EfficientNet scaling philosophy.
66. [H] How does dilation in convolutions work? When is it useful?

### RNNs and Variants

67. [E] What's an RNN? Why does it suffer from vanishing gradients?
68. [M] How do LSTMs solve the vanishing gradient problem? Walk through the gates.
69. [M] What's a GRU? How does it differ from LSTM?
70. [H] Are RNNs still relevant in 2026? When would you still use one?

---

## SECTION 3: Transformers & LLMs (THIS IS YOUR AREA)

### Attention

71. [E] What is self-attention? What problem does it solve?
72. [M] Walk through the attention formula: Attention(Q, K, V) = softmax(QK^T / √d_k) V. Explain each step.
73. [M] Why divide by √d_k in the attention formula? What happens if you don't?
74. [M] What's the difference between encoder self-attention, decoder self-attention, and cross-attention?
75. [M] What is multi-head attention? Why is it better than single-head?
76. [H] Explain causal masking. Where in the network is it applied? Why?
77. [H] What is the complexity of vanilla attention? Why is it problematic for long sequences?

### Attention Variants

78. [M] What is multi-query attention (MQA)? Why does it help inference?
79. [M] What is grouped-query attention (GQA)? Why is it a compromise between MHA and MQA?
80. [H] What is multi-latent attention (MLA, as in DeepSeek)? Why is it interesting?
81. [H] Explain FlashAttention. What problem does it solve? How?
82. [H] What is PagedAttention (vLLM)? How does it handle KV cache?

### Positional Encoding

83. [E] Why do transformers need positional encoding?
84. [M] What's the original sinusoidal positional encoding? Why those frequencies?
85. [M] What is learned positional encoding? Tradeoffs vs sinusoidal?
86. [M] What is RoPE (rotary position embedding)? Why is it popular?
87. [H] What is ALiBi (Attention with Linear Biases)? Compare with RoPE.

### Tokenization

88. [E] Why do LLMs need tokenization?
89. [M] Explain BPE (byte-pair encoding). Walk through the algorithm.
90. [M] What's the difference between BPE, WordPiece, and SentencePiece?
91. [M] What is "unicode normalization" in tokenization? Why does it matter?
92. [H] Why does a vocabulary size of 32k-128k tend to be optimal? Tradeoffs of larger vs smaller vocab?

### LLM Architecture

93. [E] What's the difference between encoder-only, decoder-only, and encoder-decoder transformers?
94. [M] Why are most modern LLMs decoder-only (GPT-style)?
95. [M] What's a "context window"? Why does it matter?
96. [M] What's pre-norm vs post-norm? Why has pre-norm become standard?
97. [H] Explain Mixture of Experts (MoE). What problem does it solve? Tradeoffs?
98. [H] What is SwiGLU? Why is it the default FFN activation now?

### Training (Pre-training, SFT, RLHF)

99. [E] What's the pre-training objective for a GPT-style LLM?
100. [M] What's the difference between pre-training and fine-tuning?
101. [M] What is supervised fine-tuning (SFT)? When do you use it?
102. [M] What is RLHF? Walk through the steps: reward model → PPO.
103. [H] What is DPO (Direct Preference Optimization)? Why is it preferred over RLHF in some cases?
104. [H] What is RLAIF? How does it differ from RLHF?
105. [H] Compare PPO vs DPO vs KTO vs IPO. Tradeoffs of each.

### Fine-tuning Techniques

106. [E] What is full fine-tuning? Why is it expensive?
107. [M] What is LoRA? How many parameters does it train relative to full fine-tuning?
108. [M] What is QLoRA? What's the "Q" part doing?
109. [M] What is PEFT? What does it include besides LoRA?
110. [H] Explain prefix tuning vs prompt tuning vs LoRA. When use each?
111. [H] What is "catastrophic forgetting" in fine-tuning? How do you mitigate?

### Inference Optimization

112. [E] What is the KV cache? Why is it important?
113. [M] Why does inference get slower for longer outputs in LLMs? How does KV cache help?
114. [M] What is continuous batching? Why is it better than static batching?
115. [M] What is speculative decoding? Walk through the algorithm.
116. [H] What is INT8 quantization? AWQ vs GPTQ?
117. [H] Explain FP8 quantization. When is it useful?
118. [H] What's distillation in the LLM context? Examples?

### Evaluation

119. [E] What's perplexity? Pros/cons as an LLM metric?
120. [M] How would you evaluate a chatbot's response quality?
121. [M] What is MMLU? HellaSwag? HumanEval? What does each measure?
122. [M] How does BLEU score work? Limitations?
123. [H] How would you design a custom eval for "is this RAG system's answer faithful to the source"?
124. [H] What is "LLM as a judge"? Pitfalls?

### Generation

125. [E] What is greedy decoding? When does it fail?
126. [E] What is temperature in generation? What does temp=0 vs temp=1 vs temp=2 do?
127. [M] What's top-k sampling? Top-p (nucleus)?
128. [M] What's the difference between top-p and top-k? When use each?
129. [H] What is beam search? Why is it less used for LLMs vs classical NMT?
130. [H] What is contrastive search?

---

## SECTION 4: GenAI Applications

### RAG

131. [E] What is RAG? Why use it over pure LLM?
132. [M] Walk through a RAG pipeline end to end.
133. [M] How do you choose chunk size for RAG?
134. [M] What's the role of the embedding model in RAG? How do you evaluate it?
135. [M] What is "hybrid retrieval"? Why use it?
136. [H] How do you evaluate a RAG system? Multiple dimensions.
137. [H] What is "lost in the middle" in long-context RAG? How do you mitigate?
138. [H] How do you handle multi-hop questions in RAG?
139. [H] What's a reranker? When do you use one?

### Agents

140. [E] What is an LLM agent? How is it different from a regular LLM call?
141. [M] Explain the ReAct paradigm.
142. [M] What's "tool calling" / function calling? When does it fail?
143. [H] How do you design an agent's memory architecture? (You have ECHOME experience — articulate it!)
144. [H] What is Plan-and-Execute vs ReAct? Tradeoffs?
145. [H] How do you evaluate agent correctness vs efficiency vs robustness?

### Safety, Alignment, Hallucination

146. [E] What's a hallucination in LLMs?
147. [M] How do you reduce hallucinations? Multiple methods.
148. [M] What is Constitutional AI?
149. [H] What's the difference between alignment, safety, robustness, and fairness?
150. [H] Explain "jailbreaks" in LLMs. Defense strategies?

---

## SECTION 5: ML Systems & Production

### Deployment

151. [E] What's the difference between batch and real-time inference?
152. [M] What is ONNX? What problem does it solve?
153. [M] What's TensorRT? When use it?
154. [M] How do you handle model versioning in production?
155. [H] What is canary deployment? Shadow deployment? When use each?
156. [H] How would you A/B test a new ML model in production?

### Monitoring & Drift

157. [E] What is data drift? Concept drift?
158. [M] How do you detect drift?
159. [M] What's PSI (Population Stability Index)?
160. [H] How do you monitor an LLM in production? What metrics?

### Scaling

161. [M] What is data parallelism vs model parallelism?
162. [M] Explain ZeRO (1, 2, 3). What does each stage shard?
163. [M] What is tensor parallelism? Pipeline parallelism?
164. [H] How does FSDP work?
165. [H] How would you train a 70B model? What infrastructure?

---

## SECTION 6: Math Foundations

166. [E] What does it mean for vectors to be orthogonal? Linearly independent?
167. [M] What is matrix multiplication, geometrically?
168. [M] What is an eigenvector? Eigenvalue?
169. [M] What is SVD? Where does it appear in ML?
170. [M] What is the dot product, geometrically? Why is it used in attention?
171. [H] Explain PCA mathematically.
172. [H] What is the Gram matrix? Where does it appear?

### Probability

173. [E] What's the difference between probability and likelihood?
174. [M] What's expectation, variance, covariance?
175. [M] What's the central limit theorem? Why does it matter?
176. [M] What's the difference between Gaussian and Multinomial distributions?
177. [H] What is KL divergence? Why is it asymmetric?
178. [H] Explain mutual information.

### Calculus

179. [E] What is a gradient? What's the chain rule?
180. [M] What's a Jacobian? Hessian?
181. [H] What is a Taylor expansion? Where do we use it in ML?

---

## SECTION 7: System Design (preview — full content in `05-ml-system-design/`)

182. [M] Design YouTube's recommendation system.
183. [M] Design Twitter's home feed ranking.
184. [M] Design Google's autocomplete.
185. [H] Design a RAG system to answer questions over 10M PDFs.
186. [H] Design a fine-tuning platform for fine-tuning 70B models.
187. [H] Design an A/B testing platform for ML models.

---

## SECTION 8: Behavioral / Project Deep-Dives

(These come up in EVERY interview — practice them.)

188. Walk me through your most complex project in 5 minutes.
189. What was the hardest technical problem you solved? How did you approach it?
190. What's a project that failed? What did you learn?
191. Tell me about a time you disagreed with a coworker. (Specifically, project-related.)
192. How do you stay current with ML research?
193. Why are you leaving your current company?
194. What questions do you have for me?

Cover these deeply in `06-behavioral/04-your-story-bank.md`.

---

## SECTION 9: "Curveballs"

These appear randomly. Don't be caught off guard.

195. [M] If you had a 10x faster GPU tomorrow, what would change about how we build LLMs?
196. [M] Why is BERT not as popular for generation as GPT?
197. [H] Is in-context learning "real learning"? What does the research say?
198. [H] What's "emergent ability" in LLMs? Is it real?
199. [H] Why do LLMs sometimes "regret" their answers when re-prompted?
200. [H] Where do you think LLMs will be in 2 years?
201. [H] What's the most exciting paper you've read in the last 6 months? Why?

---

## How to drill this bank effectively

### Method 1: Self-quiz cycle
1. Pick 10 random questions
2. Answer out loud (record yourself)
3. Compare against reference answers (some you'll need to research; use this as a learning opportunity)
4. Mark weak ones for re-review
5. Repeat with different 10 questions next session

### Method 2: With me (Claude)
- Paste any subset of questions to me with: "Drill me on these. After my answer, score me 1-10 and tell me what's missing."
- I'll grade you with no mercy.

### Method 3: Topic-focused
- Pick one topic (e.g., Section 3: Transformers)
- Answer ALL questions in that section in one session
- This builds depth, not breadth

### Method 4: Mock interview prep
- Day before an ML interview: do 15-20 random questions
- Focus on "explain to a colleague" pace and clarity

---

## What to do if you genuinely don't know an answer

This is fine. The right move:

1. Say "I don't know" honestly
2. Make a guess at what it might mean from the question
3. Ask for a hint
4. Reason from first principles
5. Note it for later study

Interviewers prefer "I don't know but here's how I'd find out" over a confident-wrong answer.

---

This bank will grow. As you encounter new questions in mocks/interviews, ADD them here. By Month 12, you should have 300+ questions logged.
