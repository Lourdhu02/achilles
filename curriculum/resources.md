# Resources (curated: fewer, better)

The courses, books and long-form writing worth your time alongside the labs, most of them free. Each entry says what it is for and when in the curriculum to use it. Paper lists live in [papers.md](papers.md); visual guides and PDFs per topic live in the [library](../library/README.md).

**Contents:** [Courses](#courses-in-the-order-to-take-them) · [Free books](#free-books) · [Paid books worth buying](#paid-books-worth-buying) · [Long-form writing](#long-form-writing-worth-rereading) · [Tools to learn](#tools-to-learn) · [Communities](#communities-contribute-dont-just-lurk) · [If you have no GPU](#if-you-have-no-gpu) · [Anti-list](#anti-list)

> [!TIP]
> Pick **one** course per phase and finish it, doing the exercises. Starting five courses and finishing none is the most common way self-study fails. Where a course and a lab cover the same thing, do the lab first, then use the course to fill gaps.

---

## Courses, in the order to take them

| # | Course | Cost | Use it for | Pairs with |
|---|---|---|---|---|
| 1 | Karpathy, [*Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html), including *Let's reproduce GPT-2 (124M)* | Free | The build-from-scratch spine: backprop, language models, tokenizers, GPT | [Labs 01](../labs/01_autograd/README.md), [04](../labs/04_tokenizer/README.md), [05](../labs/05_transformer/README.md) |
| 2 | Stanford [CS336, *Language Modeling from Scratch*](https://stanford-cs336.github.io/) | Free (lectures and assignments online) | University-level version of this curriculum; do its assignments alongside labs 04–09 | Modules [04](04-transformers.md)–[07](07-inference.md) |
| 3 | [GPU MODE lectures](https://github.com/gpu-mode/lectures) and Discord | Free | Kernels, Triton, CUDA, profiling | [Lab 06](../labs/06_attention_kernels/README.md), [module 02](02-compute-and-hardware.md) |
| 4 | [OpenAI *Spinning Up in Deep RL*](https://spinningup.openai.com) plus [Sutton and Barto](http://incompleteideas.net/book/the-book-2nd.html) (chapters 1–6 and 13) | Free | Policy gradients before RLHF, DPO and GRPO | [Module 06](06-post-training.md), [labs 11](../labs/11_dpo/README.md)–[12](../labs/12_grpo/README.md) |
| 5 | MIT 6.5940, [*TinyML and Efficient Deep Learning Computing*](https://efficientml.ai) (Song Han) | Free | Quantization, pruning, distillation, efficient inference | [Module 07](07-inference.md), [lab 13](../labs/13_quantization/README.md) |
| 6 | [ARENA](https://github.com/callummcdougall/ARENA_3.0) | Free | Hands-on mechanistic interpretability (TransformerLens, induction heads, SAEs) and RL for LLMs | [Module 09](09-interpretability-and-safety.md), [lab 16](../labs/16_interpretability/README.md) |
| 7 | [Hugging Face Learn](https://huggingface.co/learn) (LLM, deep RL and agents courses) | Free | Practical library skills: `transformers`, `datasets`, TRL, agents | [Module 10](10-applied-llm-systems.md) |

**Fill-in courses, when you need them**

- **Math:** MIT 18.06 [*Linear Algebra*](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) (Gilbert Strang) and 3Blue1Brown's visual series on [neural networks](https://www.3blue1brown.com/topics/neural-networks) for intuition ([module 01](01-math.md)).
- **Deep learning breadth:** fast.ai [*Practical Deep Learning*](https://course.fast.ai) (top-down, code first); Stanford [CS231n](https://cs231n.stanford.edu) (vision, excellent backprop notes); Stanford [CS224N](https://web.stanford.edu/class/cs224n/) (NLP).
- **In India:** [NPTEL](https://nptel.ac.in) hosts free IIT courses, including Mitesh Khapra's *Deep Learning* (IIT Madras), with an optional low-cost proctored certificate that some Indian employers recognize.
- **AI safety and governance:** BlueDot Impact's free [AI safety and governance courses](https://bluedot.org) (cohort-based, with reading groups), and Neel Nanda's [guide to getting started in mechanistic interpretability](https://www.neelnanda.io/mechanistic-interpretability/getting-started).

## Free books

| Book | What it is for |
|---|---|
| Deisenroth, Faisal and Ong, [*Mathematics for Machine Learning*](https://mml-book.github.io) | Linear algebra, calculus and probability for [module 01](01-math.md) |
| Simon Prince, [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/) (2023) | The clearest modern deep learning textbook, with figures and notebooks |
| François Fleuret, [*The Little Book of Deep Learning*](https://fleuret.org/public/lbdl.pdf) | A short, phone-sized summary for revision before interviews |
| Zhang et al., [*Dive into Deep Learning*](https://d2l.ai) | Runnable notebooks for every concept |
| Goodfellow, Bengio and Courville, [*Deep Learning*](https://www.deeplearningbook.org) (2016) | Classic reference for optimization and regularization theory |
| Kevin Murphy, [*Probabilistic Machine Learning*](https://probml.github.io/pml-book/) | Rigorous probabilistic reference |
| James et al., [*An Introduction to Statistical Learning*](https://www.statlearning.com) | Statistics foundations behind [module 08](08-evaluation-and-research.md) (resampling, testing) |
| Jurafsky and Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/) (3rd edition draft) | NLP foundations, including information retrieval and evaluation |
| Manning, Raghavan and Schütze, [*Introduction to Information Retrieval*](https://nlp.stanford.edu/IR-book/) | BM25, evaluation metrics, indexing: the theory behind [lab 17](../labs/17_retrieval/README.md) |
| Sutton and Barto, [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) | RL foundations |
| Nathan Lambert, [*The RLHF Book*](https://rlhfbook.com) | Post-training in depth: reward models, PPO, DPO, RL with verifiable rewards |
| Google DeepMind, [*How to Scale Your Model*](https://jax-ml.github.io/scaling-book/) | TPU/GPU scaling math: rooflines, sharding, inference |
| Hugging Face, [*The Ultra-Scale Playbook*](https://huggingface.co/spaces/nanotron/ultrascale-playbook) | Data, tensor, pipeline and context parallelism in practice |

## Paid books worth buying

- Sebastian Raschka, *Build a Large Language Model (From Scratch)* (2024): gentle companion to labs 04–05.
- Christopher Bishop and Hugh Bishop, *Deep Learning: Foundations and Concepts* (2023): rigorous reference (also readable free online from the authors' site).
- Chip Huyen, *AI Engineering* (2025): applied systems and the founder track.
- Chip Huyen, *Designing Machine Learning Systems* (2022): ML system design interviews.
- Hwu, Kirk and El Hajj, *Programming Massively Parallel Processors* (4th edition): CUDA fundamentals.

> [!NOTE]
> Indian editions of many technical books are sold at a fraction of the international price; check before importing.

## Long-form writing worth rereading

**Performance and scale**
- Horace He, [*Making Deep Learning Go Brrrr From First Principles*](https://horace.io/brrr_intro.html): compute-, memory- and overhead-bound, in one essay.
- kipply, [*Transformer Inference Arithmetic*](https://kipp.ly/transformer-inference-arithmetic/)
- Simon Boehm, [*How to Optimize a CUDA Matmul Kernel*](https://siboehm.com/articles/22/CUDA-MMM)

**Training and research practice**
- Andrej Karpathy, [*A Recipe for Training Neural Networks*](https://karpathy.github.io/2019/04/25/recipe/)
- [Lilian Weng's blog](https://lilianweng.github.io): careful surveys of RL, agents, hallucination, reward hacking.

**Transformers, visually**
- Jay Alammar, [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/)
- Harvard NLP, [*The Annotated Transformer*](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Distill](https://distill.pub) and the [Transformer Circuits Thread](https://transformer-circuits.pub): interactive interpretability articles.

**Applied systems and evals**
- Anthropic, [*Building Effective Agents*](https://www.anthropic.com/engineering/building-effective-agents), [*Contextual Retrieval*](https://www.anthropic.com/news/contextual-retrieval), [*Effective context engineering for AI agents*](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- Hamel Husain, [*Your AI Product Needs Evals*](https://hamel.dev/blog/posts/evals/)
- Eugene Yan, [*Patterns for Building LLM-based Systems and Products*](https://eugeneyan.com/writing/llm-patterns/)

## Tools to learn

| Tool | For | Module |
|---|---|---|
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Standard benchmark evals of open models; `--log_samples` saves per-item outputs for statistics | [08](08-evaluation-and-research.md) |
| [Inspect](https://inspect.aisi.org.uk) (UK AI Security Institute) | Writing your own evals, including agentic and safety evals | [08](08-evaluation-and-research.md), [09](09-interpretability-and-safety.md) |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | Hooks, caching and patching on open models | [09](09-interpretability-and-safety.md) |
| [Neuronpedia](https://www.neuronpedia.org) | Browse SAE features and attribution graphs in the browser | [09](09-interpretability-and-safety.md) |
| FAISS, or a vector database with hybrid search | Approximate nearest neighbor search at scale | [10](10-applied-llm-systems.md) |

## Communities (contribute, don't just lurk)

GPU MODE Discord · [EleutherAI](https://www.eleuther.ai) Discord · [Hugging Face forums](https://discuss.huggingface.co) · [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) · [Latent Space](https://www.latent.space) · Bengaluru meetups and AI Tinkerers.

Answer questions there after each lab, share your measured numbers (with error bars), and review others' write-ups; that is where collaborators and referrals come from.

## If you have no GPU

- Every lab's tests run on CPU, and the Triton tests use the interpreter.
- Free hosted notebooks (Google Colab, Kaggle) provide time-limited GPUs that are enough for the scale-up runs in most labs; check their current quotas.
- Interpretability: GPT-2 small runs on a laptop CPU; Neuronpedia needs only a browser.
- Evals and statistics (lab 15, module 08) need no GPU at all.

## Anti-list

Skip "master AI in 30 days" content, framework-of-the-month tutorials, and prompt-engineering certificates. If a resource doesn't make you build, derive or measure something, it's entertainment.
