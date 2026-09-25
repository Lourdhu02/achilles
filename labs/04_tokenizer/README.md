# Lab 04 — Byte-level BPE tokenizer

**Build:** GPT-style byte-level BPE: regex pre-tokenization, merge training, rank-ordered encoding,
special tokens, lossless decoding. Then use it to measure something the big labs got wrong for years.
**Time:** 6–8 h · **Reads first:** [transformers §1](../../curriculum/04-transformers.md#1-tokens-in-logits-out)
**Run:** `pytest labs/04_tokenizer`

## Why engineers who "just use the API" get this wrong

The tokenizer determines **cost** (you pay per token), **context** (limits are in tokens),
**arithmetic** (how digits group), **fairness** (Telugu text can cost several times what English
costs for the same meaning) and **security** (a user string that encodes to a special token can
hijack a chat template). It is also the one part of the model that is not learned by gradient
descent, so its mistakes are permanent.

## The algorithm

**Training.** Start from the 256 byte values. Repeat: count every adjacent pair across the corpus,
merge the most frequent pair into a new token, record the merge. The merge list *is* the tokenizer.

**Encoding.** Within each chunk, repeatedly apply the **earliest-learned** merge available, not the
most frequent one in this text and not left to right. `test_merges_apply_in_rank_order_not_frequency_order`
catches the classic bug.

**Pre-tokenization.** A regex first splits text into chunks (words with a leading space, digit
groups, punctuation runs, whitespace), and merges never cross chunk boundaries. Without it,
`"dog."`, `"dog!"` and `"dog?"` each become separate tokens and waste vocabulary. GPT-4's pattern
groups digits in threes (`12345 → 123|45`), which makes arithmetic tokenization consistent.

**Byte-level.** Every string is a byte sequence, so there is no `<unk>`. A token boundary can fall
*inside* a multi-byte UTF-8 character, so decode with `errors="replace"`, and never decode a
streamed token alone when showing it to users.

**Efficiency.** Count *unique* chunks once and weight pair counts by frequency. A 1 GB corpus has
far fewer unique words than words.

## The Telugu finding (run this)

```python
import labs._impl as L; tk = L.load("labs/04_tokenizer/test_tokenizer.py", "solution")
for p in (tk.GPT2_PATTERN, tk.GPT4_PATTERN, tk.O200K_PATTERN):
    print(tk.pretokenize("తెలుగు నమస్కారం", p))
```

GPT-2 and GPT-4 (cl100k) define a "word" as `\p{L}+`, letters only. Telugu vowel signs and viramas
are *combining marks* (`\p{M}`), so every word shatters at every vowel sign before BPE even
starts, and no merge can ever rebuild it. GPT-4o's o200k pattern adds `\p{M}` and keeps words whole.
This is one reason Indic text used to cost several times more tokens than English.
**Measure it:** train two tokenizers with the same vocab on the same Telugu+English corpus, one per
pattern, and compare bytes/token on held-out Telugu. That is a real, publishable-quality mini-study
([research ideas](../../curriculum/08-evaluation-and-research.md#research-projects-that-fit-an-8-gb-gpu)).

## Design trade-offs to be able to argue

| choice | bigger / more | smaller / fewer |
|---|---|---|
| vocab size | shorter sequences (cheaper attention and per-token compute); better multilingual coverage | fewer embedding/LM-head params; each token is seen more often in training |
| pre-tokenization | cleaner units, digit consistency | can block useful cross-boundary merges |
| byte fallback | never `<unk>` | tokens can split characters |

Llama 2 used 32k tokens, Llama 3 128k, GPT-4o ~200k. The trend follows multilingual and code
coverage, and a bigger vocab is cheap once models are large. Remember that the LM head costs
`2·d·V` FLOPs per token.

## What to implement

`get_pair_counts` → `merge` → `BPETokenizer.train` → `_encode_chunk` → `encode` (special tokens) → `decode`.

## Check yourself

1. Why must encoding apply merges in training order rather than greedily merging the most frequent pair in the input?
2. Your chat template uses `<|im_start|>`. A user pastes that string into a message. What happens with `allowed_special="all"` on user text, and what should the server do?
3. Why can a streaming UI print garbage characters mid-word, and how do you fix it?
4. "SolidGoldMagikarp" made GPT-3 behave strangely. Explain the mechanism in tokenizer terms.
5. You extend a pretrained model's vocabulary with 2,000 Telugu tokens. How do you initialize the new embeddings, and what training do you need?

<details><summary>Answers</summary>

1. The merge table defines a *deterministic* segmentation that the model was trained on. Applying merges in any other order produces different ids for the same text, which the model has never seen, so quality silently drops.
2. The user's text would encode to the real control token, letting them forge turns (prompt injection at the token level). Encode user content with specials disabled (`allowed_special="none"`) and only let your template code emit control tokens.
3. A token can end in the middle of a multi-byte UTF-8 character. Buffer bytes until they form a complete character (an incremental UTF-8 decoder) before printing.
4. Tokens that were in the tokenizer's training corpus (Reddit usernames) but almost absent from the model's training data kept near-initial, undertrained embeddings. Feeding them in produces out-of-distribution activations.
5. Initialize each new token's embedding as the mean of the embeddings of its old sub-token decomposition (or the global mean plus noise). Then do continued pretraining on Telugu-heavy data before fine-tuning. Untie or carefully handle the LM head too.
</details>

## Stretch

- Make training fast: update pair counts incrementally after each merge (only chunks containing the pair change). Train 32k merges on 100 MB and compare with `tiktoken`/`tokenizers` speed.
- Load GPT-2's released `merges.txt` into your class and reproduce `tiktoken.get_encoding("gpt2")` ids exactly. (Hint: GPT-2 remaps bytes to printable unicode; undo it.)
- Implement a unigram LM tokenizer (SentencePiece-style) and compare segmentations on Telugu.
