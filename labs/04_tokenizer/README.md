# Lab 04 — Byte-level BPE tokenizer

**Build:** GPT-style byte-level BPE: regex pre-tokenization, merge training, rank-ordered encoding,
special tokens, lossless decoding. Then use it to measure something the big labs got wrong for years.
**Time:** 6–8 h · **Reads first:** [transformers §1](../../curriculum/04-transformers.md#1-tokens-in-logits-out)<br>
**Run:** `pytest labs/04_tokenizer` (your code) · `pytest labs/04_tokenizer --impl=solution` (reference) · CPU only, seconds

## Why engineers who "just use the API" get this wrong

The tokenizer determines **cost** (you pay per token), **context** (limits are in tokens),
**arithmetic** (how digits group), **fairness** (Telugu text can cost several times what English
costs for the same meaning) and **security** (a user string that encodes to a special token can
hijack a chat template). It is also the one part of the model that is not learned by gradient
descent, so its mistakes are permanent.

## The algorithm

```
text ──regex──▶ chunks ──UTF-8──▶ byte ids (0..255) ──apply merges by rank──▶ token ids
"Hi 123" → ["Hi", " ", "123"] → [72,105] [32] [49,50,51] → ...
```

**Training.** Start from the 256 byte values. Repeat: count every adjacent pair across the corpus,
merge the most frequent pair into a new token, record the merge. The merge list *is* the tokenizer.
Worked example (`test_first_merge_of_the_classic_example`): on `"aaabdaaabac"` the first merge is
`(a, a) → 256`. Train to 259 and you get `aa → 256`, then `ab → 257` (tied at 2 with `(256, a)`,
and ties go to the smaller pair), then `(256, 257) → 258` = `"aaab"`; the text encodes as
`[258, d, 258, a, c]`.

**Encoding.** Within each chunk, repeatedly apply the **earliest-learned** merge available, not the
most frequent one in this text and not left to right. `test_merges_apply_in_rank_order_not_frequency_order`
catches the classic bug.

**Pre-tokenization.** A regex first splits text into chunks (words with a leading space, digit
groups, punctuation runs, whitespace), and merges never cross chunk boundaries. Without it,
`"dog."`, `"dog!"` and `"dog?"` each become separate tokens and waste vocabulary. GPT-4's pattern
groups digits in threes (`12345 → 123|45`), which makes arithmetic tokenization consistent. For
example, GPT-4's pattern splits `"I've 12345 apples!!"` into `I` · `'ve` · ` ` · `123` · `45` · ` apples` · `!!`.

**Byte-level.** Every string is a byte sequence, so there is no `<unk>`. A token boundary can fall
*inside* a multi-byte UTF-8 character, so decode with `errors="replace"`, and never decode a
streamed token alone when showing it to users. Telugu characters take 3 bytes each in UTF-8
(`"తెలుగు"` is 6 code points, 18 bytes), so before any merges Telugu costs 3x English per character.

**Efficiency.** Count *unique* chunks once and weight pair counts by frequency. A 1 GB corpus has
far fewer unique words than words.

## The Telugu finding (run this)

```python
import labs._impl as L; tk = L.load("labs/04_tokenizer/test_tokenizer.py", "solution")
for p in (tk.GPT2_PATTERN, tk.GPT4_PATTERN, tk.O200K_PATTERN):
    print(tk.pretokenize("తెలుగు నమస్కారం", p))
```

Output (12, 8 and 2 chunks):

```
['త', 'ె', 'ల', 'ు', 'గ', 'ు', ' నమస', '్', 'క', 'ా', 'ర', 'ం']
['త', 'ెల', 'ుగ', 'ు', ' నమస', '్క', 'ార', 'ం']
['తెలుగు', ' నమస్కారం']
```

GPT-2 and GPT-4 (cl100k) define a "word" as `\p{L}+`, letters only. Telugu vowel signs and viramas
are *combining marks* (`\p{M}`), so every word shatters at every vowel sign before BPE even
starts, and no merge can ever rebuild it. GPT-4o's o200k pattern adds `\p{M}` and keeps words whole.
This is one reason Indic text used to cost several times more tokens than English.
**Measure it:** train two tokenizers with the same vocab on the same Telugu+English corpus, one per
pattern, and compare bytes/token on held-out Telugu. That is a real, publishable-quality mini-study
([research ideas](../../curriculum/08-evaluation-and-research.md#research-projects-that-fit-an-8-gb-gpu)).

```python
tok_a = tk.BPETokenizer.train(corpus, vocab_size=8192, pattern=tk.GPT4_PATTERN)
tok_b = tk.BPETokenizer.train(corpus, vocab_size=8192, pattern=tk.O200K_PATTERN)
print(tk.bytes_per_token(tok_a, heldout_telugu), tk.bytes_per_token(tok_b, heldout_telugu))
```

Use real text for `corpus` and `heldout_telugu` (for example Telugu Wikipedia plus an English sample,
split by document, not by line). Report the English bytes/token too: a fair comparison shows what
each pattern costs the other language.

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

| function | contract | tests that check it |
|---|---|---|
| `get_pair_counts(chunks)` | count adjacent pairs, each chunk weighted by its frequency | `test_pair_counts_weighted_by_frequency` |
| `merge(ids, pair, new_id)` | replace non-overlapping occurrences, scanning left to right (`aaa` + `(a,a)` → `[256, a]`) | `test_merge_is_left_to_right_and_non_overlapping` |
| `BPETokenizer.train` | `vocab_size − 256 − len(specials)` merges; most frequent pair wins, ties go to the smallest pair; new ids from 256; special tokens get the ids after the last merge | `test_first_merge_of_the_classic_example`, `test_vocab_size_and_special_ids`, `test_training_is_deterministic`, `test_vocab_entries_are_concatenations`, `test_training_compresses` |
| `_encode_chunk(chunk)` | repeatedly merge the adjacent pair with the lowest merge id until none is mergeable | `test_merges_apply_in_rank_order_not_frequency_order`, `test_merges_never_cross_pretokenization_boundaries` |
| `encode(text, allowed_special)` | `"all"`: special strings map to their ids and are never split; `"none"`: encode them as plain text | `test_special_tokens` |
| `decode(ids)` | concatenate bytes, decode UTF-8 with `errors="replace"` | `test_roundtrip_trained`, `test_roundtrip_untrained_is_raw_bytes`, `test_decode_invalid_utf8_does_not_crash` |

The pattern tests (`test_indic_words_split_at_combining_marks_in_gpt4_but_not_o200k`,
`test_gpt4_groups_digits_in_threes`) check the provided regexes and pass once `pretokenize` works;
read them to understand what each pattern does.

## Common bugs

- **Tie-breaking by insertion order.** `Counter.most_common(1)` breaks ties by first occurrence, so
  your merges depend on corpus order. Break ties explicitly (the handout rule: smallest pair).
- **Encoding by frequency or left to right.** Encoding must replay the training order: always the
  lowest-rank mergeable pair. Greedy-by-frequency passes simple roundtrip tests and fails the rank test.
- **Overlapping merges.** `merge([a, a, a], (a, a))` must give `[256, a]`, not `[256, 256]`.
- **Merging across chunks.** Train and encode per chunk; never concatenate chunks before merging.
- **Counting every occurrence of every chunk.** Correct but slow; weight unique chunks by frequency.
- **Special tokens that shift ids.** Specials get ids *after* the last merge (`256 + len(merges)`),
  so `vocab_size = 400` with one special means 143 merges and the special at id 399.
- **`decode` raising `UnicodeDecodeError`** on a truncated multi-byte character. Use `errors="replace"`.

> [!TIP]
> Debug with bytes, not strings: print `tok.vocab[i]` for each id. Most tokenizer bugs are
> visible the moment you see which byte sequences became tokens.

> [!TIP]
> If training is slow on a larger corpus, profile before optimizing: almost all the time goes to
> recounting pairs. The stretch goal's incremental update is the standard fix.

## CPU vs GPU notes

Tokenizer training and encoding are CPU work everywhere, including at frontier labs (tokenization
runs in data-pipeline CPU jobs, not on GPUs). This lab needs no GPU. Pure-Python BPE training on
100 MB takes a long time; production trainers (Hugging Face `tokenizers`, SentencePiece) are
written in Rust or C++ and update pair counts incrementally.

## Check yourself

1. Why must encoding apply merges in training order rather than greedily merging the most frequent pair in the input?
2. Your chat template uses `<|im_start|>`. A user pastes that string into a message. What happens with `allowed_special="all"` on user text, and what should the server do?
3. Why can a streaming UI print garbage characters mid-word, and how do you fix it?
4. "SolidGoldMagikarp" made GPT-3 behave strangely. Explain the mechanism in tokenizer terms.
5. You extend a pretrained model's vocabulary with 2,000 Telugu tokens. How do you initialize the new embeddings, and what training do you need?
6. Why does GPT-4's pattern split digits into groups of at most three, and what does that do to arithmetic?
7. You double the vocabulary from 64k to 128k for a model with d = 4096. What changes in parameters, per-token FLOPs and sequence length?

<details><summary>Answers</summary>

1. The merge table defines a *deterministic* segmentation that the model was trained on. Applying merges in any other order produces different ids for the same text, which the model has never seen, so quality silently drops.
2. The user's text would encode to the real control token, letting them forge turns (prompt injection at the token level). Encode user content with specials disabled (`allowed_special="none"`) and only let your template code emit control tokens.
3. A token can end in the middle of a multi-byte UTF-8 character. Buffer bytes until they form a complete character (an incremental UTF-8 decoder) before printing.
4. Tokens that were in the tokenizer's training corpus (Reddit usernames) but almost absent from the model's training data kept near-initial, undertrained embeddings. Feeding them in produces out-of-distribution activations.
5. Initialize each new token's embedding as the mean of the embeddings of its old sub-token decomposition (or the global mean plus noise). Then do continued pretraining on Telugu-heavy data before fine-tuning. Untie or carefully handle the LM head too.
6. Without a cap, BPE learns arbitrary multi-digit tokens from frequent numbers ("2019", "1000"), so the same digit position gets tokenized differently in different numbers. Groups of at most three give a consistent, left-aligned segmentation, which makes digit-level patterns easier to learn (though left-aligned groups still misalign place values across numbers of different lengths).
7. Embedding parameters grow by 64k × 4096 = 268M (another 268M if the head is untied); LM-head FLOPs double from 2·4096·64k ≈ 0.54 GFLOP to ≈ 1.07 GFLOP per token; sequences get shorter by however much the extra merges compress your data, which you must measure (it is usually a few percent for English and much more for under-covered languages).
</details>

## Stretch

- Make training fast: update pair counts incrementally after each merge (only chunks containing the pair change). Train 32k merges on 100 MB and compare with `tiktoken`/`tokenizers` speed.
- Load GPT-2's released `merges.txt` into your class and reproduce `tiktoken.get_encoding("gpt2")` ids exactly. (Hint: GPT-2 remaps bytes to printable unicode; undo it.)
- Implement a unigram LM tokenizer (SentencePiece-style) and compare segmentations on Telugu.
- Write a streaming decoder that buffers incomplete UTF-8 sequences, and test it on Telugu text split at every possible byte boundary.
- Plot bytes/token against vocab size (1k–32k) for English, Telugu and Python code with both patterns. Where does each curve flatten?
