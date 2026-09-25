import pytest

from labs._impl import load

tk = load(__file__)

ENGLISH = (
    "The quick brown fox jumps over the lazy dog. The dog was not amused; the fox was. "
    "Tokenizers turn text into integers, and integers into text, without losing a byte. "
)
CODE = "def add(a, b):\n    return a + b\n\nfor i in range(10):\n    print(add(i, i))\n"
TELUGU = "తెలుగు ఒక ద్రావిడ భాష. నమస్కారం! మీరు ఎలా ఉన్నారు? "
CORPUS = (ENGLISH * 8) + (CODE * 4) + (TELUGU * 4)

ROUNDTRIP = [
    "",
    "hello world",
    "  leading and trailing spaces  ",
    "tabs\tand\nnewlines\r\n\n\nmany",
    "numbers 1234567890 and 3.14159",
    "emoji 🤖🔥 and accents: café naïve",
    "తెలుగు ఒక ద్రావిడ భాష",
    "हिन्दी भाषा",
    CODE,
    "<|endoftext|> should round-trip too",
]


@pytest.fixture(scope="module")
def trained():
    return tk.BPETokenizer.train(CORPUS, vocab_size=400, special_tokens=["<|endoftext|>"])


# ------------------------------------------------------------------ primitives
def test_pair_counts_weighted_by_frequency():
    counts = tk.get_pair_counts({(1, 2, 3): 2, (2, 3): 5})
    assert counts[(1, 2)] == 2 and counts[(2, 3)] == 7


def test_merge_is_left_to_right_and_non_overlapping():
    assert tk.merge([97, 97, 97, 98], (97, 97), 256) == [256, 97, 98]
    assert tk.merge([1, 2, 1, 2, 3], (1, 2), 9) == [9, 9, 3]
    assert tk.merge([], (1, 2), 9) == []


# ---------------------------------------------------------------- training
def test_first_merge_of_the_classic_example():
    tok = tk.BPETokenizer.train("aaabdaaabac", vocab_size=257, pattern=None)
    assert tok.merges == {(97, 97): 256}
    assert tok.vocab[256] == b"aa"


def test_vocab_size_and_special_ids(trained):
    assert trained.vocab_size == 400
    assert len(trained.merges) == 399 - 256
    assert trained.special_tokens == {"<|endoftext|>": 399}
    assert sorted(trained.merges.values()) == list(range(256, 399))


def test_training_is_deterministic():
    a = tk.BPETokenizer.train(CORPUS, vocab_size=320)
    b = tk.BPETokenizer.train(CORPUS, vocab_size=320)
    assert a.merges == b.merges


def test_vocab_entries_are_concatenations(trained):
    for (x, y), idx in trained.merges.items():
        assert trained.vocab[idx] == trained.vocab[x] + trained.vocab[y]


# ---------------------------------------------------------------- encoding
@pytest.mark.parametrize("text", ROUNDTRIP)
def test_roundtrip_trained(trained, text):
    assert trained.decode(trained.encode(text)) == text


@pytest.mark.parametrize("text", ROUNDTRIP)
def test_roundtrip_untrained_is_raw_bytes(text):
    tok = tk.BPETokenizer()
    ids = tok.encode(text)
    assert ids == list(text.encode("utf-8"))
    assert tok.decode(ids) == text


def test_merges_apply_in_rank_order_not_frequency_order():
    # (a,b)->256 was learned first, so "abc" must become [256, 99] then [257].
    tok = tk.BPETokenizer(merges={(97, 98): 256, (256, 99): 257, (98, 99): 258}, pattern=None)
    assert tok.encode("abc") == [257]
    assert tok.encode("bc") == [258]


def test_merges_never_cross_pretokenization_boundaries(trained):
    text = ENGLISH + CODE
    chunked = [i for chunk in tk.pretokenize(text, trained.pattern) for i in trained.encode_ordinary(chunk)]
    assert trained.encode_ordinary(text) == chunked


def test_training_compresses(trained):
    assert tk.bytes_per_token(trained, ENGLISH) > 2.5
    assert tk.bytes_per_token(tk.BPETokenizer(), ENGLISH) == 1.0


def test_special_tokens(trained):
    ids = trained.encode("hi<|endoftext|>there")
    assert ids.count(399) == 1
    assert trained.decode(ids) == "hi<|endoftext|>there"
    plain = trained.encode("<|endoftext|>", allowed_special="none")
    assert 399 not in plain and len(plain) > 1


def test_decode_invalid_utf8_does_not_crash():
    tok = tk.BPETokenizer()
    assert tok.decode([0xE0, 0xB0]) == "�"  # a truncated Telugu character


# -------------------------------------------------------- pre-tokenization
def test_indic_words_split_at_combining_marks_in_gpt4_but_not_o200k():
    word = "తెలుగు"
    assert len(tk.pretokenize(word, tk.GPT2_PATTERN)) == 6
    assert len(tk.pretokenize(word, tk.GPT4_PATTERN)) == 4
    assert tk.pretokenize(word, tk.O200K_PATTERN) == [word]


def test_gpt4_groups_digits_in_threes():
    assert tk.pretokenize("12345", tk.GPT4_PATTERN) == ["123", "45"]
    assert tk.pretokenize(" 12345", tk.GPT2_PATTERN) == [" 12345"]
