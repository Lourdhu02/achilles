"""Lab 04 -- a byte-level BPE tokenizer (GPT-2 / GPT-4 style) from scratch.

Text -> regex pre-tokenization into chunks -> UTF-8 bytes -> learned merges.
Byte-level means every string is representable (no <unk>), and decode(encode(s)) == s.

Handout: labs/04_tokenizer/README.md
"""

from __future__ import annotations

from collections import Counter

import regex as re

# GPT-2: letters, numbers, punctuation runs, whitespace; optional leading space on words.
GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# GPT-4 (cl100k_base): case-insensitive contractions, numbers in groups of <= 3 digits.
GPT4_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
# GPT-4o (o200k_base): words include combining marks (\p{M}), so Indic words stay whole.
O200K_PATTERN = "|".join(
    [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)

Pair = tuple[int, int]


def pretokenize(text: str, pattern: str | None) -> list[str]:
    """Split text into chunks that merges may not cross. ``pattern=None`` = one chunk."""
    if pattern is None:
        return [text] if text else []
    return re.findall(pattern, text)


def get_pair_counts(chunks: dict[tuple[int, ...], int]) -> Counter:
    """Count adjacent token pairs across chunks, weighting each chunk by its frequency."""
    # BEGIN SOLUTION
    counts: Counter = Counter()
    for ids, freq in chunks.items():
        for pair in zip(ids, ids[1:]):
            counts[pair] += freq
    return counts
    # END SOLUTION


def merge(ids: list[int] | tuple[int, ...], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` (scanning left to right) with ``new_id``."""
    # BEGIN SOLUTION
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
    # END SOLUTION


class BPETokenizer:
    def __init__(
        self,
        merges: dict[Pair, int] | None = None,
        pattern: str | None = GPT4_PATTERN,
        special_tokens: dict[str, int] | None = None,
    ):
        self.merges: dict[Pair, int] = dict(merges or {})  # pair -> new id; lower id = earlier merge
        self.pattern = pattern
        self.special_tokens: dict[str, int] = dict(special_tokens or {})
        self.vocab: dict[int, bytes] = self._build_vocab()
        self._cache: dict[str, list[int]] = {}

    def _build_vocab(self) -> dict[int, bytes]:
        vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
            vocab[idx] = vocab[a] + vocab[b]
        for token, idx in self.special_tokens.items():
            vocab[idx] = token.encode("utf-8")
        return vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ---------------------------------------------------------------- training
    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        pattern: str | None = GPT4_PATTERN,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        """Learn ``vocab_size - 256 - len(special_tokens)`` merges.

        Each round merges the most frequent adjacent pair; ties go to the smallest pair
        (compare the first id, then the second). New ids count up from 256; special
        tokens get the ids after the last merge.
        """
        special_tokens = special_tokens or []
        n_merges = vocab_size - 256 - len(special_tokens)
        if n_merges < 0:
            raise ValueError("vocab_size too small for 256 bytes + special tokens")
        merges: dict[Pair, int] = {}
        # BEGIN SOLUTION
        # HINT: count unique chunks once (Counter), then merge inside the unique chunks only.
        chunk_counts = Counter(pretokenize(text, pattern))
        chunks: dict[tuple[int, ...], int] = Counter()
        for chunk, freq in chunk_counts.items():
            chunks[tuple(chunk.encode("utf-8"))] += freq
        for i in range(n_merges):
            counts = get_pair_counts(chunks)
            if not counts:
                break
            best = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
            new_id = 256 + i
            merges[best] = new_id
            merged: dict[tuple[int, ...], int] = Counter()
            for ids, freq in chunks.items():
                merged[tuple(merge(ids, best, new_id))] += freq
            chunks = merged
        # END SOLUTION
        first_special = 256 + len(merges)
        specials = {tok: first_special + j for j, tok in enumerate(special_tokens)}
        return cls(merges, pattern, specials)

    # ---------------------------------------------------------------- encoding
    def _encode_chunk(self, chunk: str) -> list[int]:
        """Apply merges to one chunk, always choosing the earliest-learned (lowest-id) pair."""
        # BEGIN SOLUTION
        # HINT: repeat: among adjacent pairs, pick the one with the smallest merge id; stop when none is mergeable.
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids
        # END SOLUTION

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode ignoring special tokens (they are treated as plain text)."""
        out: list[int] = []
        for chunk in pretokenize(text, self.pattern):
            if chunk not in self._cache:
                self._cache[chunk] = self._encode_chunk(chunk)
            out.extend(self._cache[chunk])
        return out

    def encode(self, text: str, allowed_special: str = "all") -> list[int]:
        """Encode text. With ``allowed_special="all"``, special-token strings map to their ids
        and are never split or merged; with ``"none"`` they are encoded as ordinary text."""
        # BEGIN SOLUTION
        if allowed_special == "none" or not self.special_tokens:
            return self.encode_ordinary(text)
        splitter = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
        out: list[int] = []
        for part in re.split(splitter, text):
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
            elif part:
                out.extend(self.encode_ordinary(part))
        return out
        # END SOLUTION

    def decode(self, ids: list[int]) -> str:
        # BEGIN SOLUTION
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
        # END SOLUTION


def bytes_per_token(tokenizer: BPETokenizer, text: str) -> float:
    """Compression ratio: UTF-8 bytes per token (higher = cheaper context)."""
    return len(text.encode("utf-8")) / max(1, len(tokenizer.encode_ordinary(text)))
