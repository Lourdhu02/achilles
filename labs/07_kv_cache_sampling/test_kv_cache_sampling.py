import math

import pytest
import torch

from labs._impl import load

kv = load(__file__)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return kv.TinyLM(kv.Config()).eval()


# ------------------------------------------------------------------ cache
def test_mask_offsets_by_cache_position():
    m = kv.cache_attention_mask(t_new=2, pos=3)
    assert m.shape == (2, 5)
    assert m.tolist() == [[False, False, False, False, True], [False, False, False, False, False]]
    assert torch.equal(kv.cache_attention_mask(4, 0), torch.ones(4, 4, dtype=torch.bool).triu(1))


def test_cache_update_writes_and_returns_prefix():
    cfg = kv.Config()
    cache = kv.KVCache(cfg, batch=2)
    k1 = torch.randn(2, cfg.n_kv_head, 3, cfg.head_dim)
    k, v = cache.update(0, k1, k1 + 1)
    assert k.shape[-2] == 3 and torch.equal(k, k1)
    cache.advance(3)
    k2 = torch.randn(2, cfg.n_kv_head, 1, cfg.head_dim)
    k, v = cache.update(0, k2, k2)
    assert k.shape[-2] == 4
    assert torch.equal(k[..., :3, :], k1) and torch.equal(k[..., 3:, :], k2)


def test_cache_bytes_match_formula():
    cfg = kv.Config()
    cache = kv.KVCache(cfg, batch=3)
    assert cache.nbytes() == 2 * cfg.n_layer * 3 * cfg.n_kv_head * cfg.max_seq * cfg.head_dim * 4


def test_incremental_logits_match_full_forward(model):
    idx = torch.randint(0, 64, (2, 20))
    full = model(idx)
    cache = kv.KVCache(model.cfg, batch=2)
    steps = [model(idx[:, :8], cache)]  # prefill 8 tokens
    for t in range(8, 20):  # then decode one at a time
        steps.append(model(idx[:, t : t + 1], cache))
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-5, rtol=1e-5)
    assert cache.pos == 20


def test_chunked_prefill_matches_single_prefill(model):
    idx = torch.randint(0, 64, (1, 17))
    full = model(idx)
    cache = kv.KVCache(model.cfg, batch=1)
    chunks = [model(idx[:, s : s + 5], cache) for s in range(0, 17, 5)]
    torch.testing.assert_close(torch.cat(chunks, dim=1), full, atol=1e-5, rtol=1e-5)


def test_cached_generation_equals_uncached(model):
    prompt = torch.randint(0, 64, (2, 6))
    a = kv.generate(model, prompt, 15, use_cache=True, temperature=0)
    b = kv.generate(model, prompt, 15, use_cache=False, temperature=0)
    c = kv.generate(model, prompt, 15, use_cache=True, prefill_chunk=4, temperature=0)
    assert a.shape == (2, 21)
    assert torch.equal(a, b) and torch.equal(a, c)


# --------------------------------------------------------------- sampling
LOGITS = torch.log(torch.tensor([[0.5, 0.25, 0.15, 0.06, 0.04]]))


def finite(x):
    return torch.isfinite(x)[0].tolist()


def test_top_k():
    assert finite(kv.top_k_filter(LOGITS, 2)) == [True, True, False, False, False]
    assert finite(kv.top_k_filter(LOGITS, 10)) == [True] * 5


def test_top_p_keeps_the_token_that_crosses_the_threshold():
    assert finite(kv.top_p_filter(LOGITS, 0.5)) == [True, False, False, False, False]
    assert finite(kv.top_p_filter(LOGITS, 0.6)) == [True, True, False, False, False]
    assert finite(kv.top_p_filter(LOGITS, 0.9)) == [True, True, True, False, False]
    assert finite(kv.top_p_filter(LOGITS, 0.01)) == [True, False, False, False, False]


def test_top_p_works_on_unsorted_rows():
    shuffled = LOGITS[:, [3, 0, 4, 2, 1]]  # probs .06 .5 .04 .15 .25
    assert finite(kv.top_p_filter(shuffled, 0.6)) == [False, True, False, False, True]


def test_min_p():
    # threshold = 0.2 * 0.5 = 0.1 -> keeps 0.5, 0.25, 0.15
    assert finite(kv.min_p_filter(LOGITS, 0.2)) == [True, True, True, False, False]


def test_greedy_and_low_temperature():
    assert kv.sample_next(LOGITS, temperature=0).item() == 0
    g = torch.Generator().manual_seed(0)
    draws = [kv.sample_next(LOGITS, temperature=0.05, generator=g).item() for _ in range(50)]
    assert set(draws) == {0}


def test_sampling_distribution_after_top_k():
    g = torch.Generator().manual_seed(0)
    n = 20000
    draws = kv.sample_next(LOGITS.expand(n, -1), top_k=2, generator=g)
    freq = torch.bincount(draws, minlength=5).float() / n
    torch.testing.assert_close(freq[:2], torch.tensor([2 / 3, 1 / 3]), atol=0.015, rtol=0)
    assert freq[2:].sum() == 0


def test_temperature_flattens():
    g = torch.Generator().manual_seed(0)
    n = 20000
    draws = kv.sample_next(LOGITS.expand(n, -1), temperature=100.0, generator=g)
    freq = torch.bincount(draws, minlength=5).float() / n
    assert freq.min() > 0.15  # nearly uniform
    assert math.isclose(freq.sum().item(), 1.0)
