import math

import pytest
import torch
import torch.nn.functional as F

from labs._impl import load

tf = load(__file__)


def tiny(**kw):
    base = dict(vocab_size=64, block_size=32, n_layer=2, n_head=4, n_kv_head=4, d_model=64)
    base.update(kw)
    return tf.GPTConfig(**base)


# ------------------------------------------------------------------ RoPE
def test_rope_tables():
    cos, sin = tf.rope_cache(head_dim=8, max_seq=5, theta=10000.0)
    assert cos.shape == sin.shape == (5, 4)
    torch.testing.assert_close(cos[0], torch.ones(4))
    expected = torch.tensor([3 * 10000 ** (-2 * i / 8) for i in range(4)])
    torch.testing.assert_close(torch.atan2(sin[3], cos[3]), torch.atan2(expected.sin(), expected.cos()))


def test_rope_matches_complex_multiplication():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 7, 16)
    cos, sin = tf.rope_cache(16, 7)
    z = torch.view_as_complex(x.reshape(2, 3, 7, 8, 2).contiguous())
    ref = torch.view_as_real(z * torch.polar(torch.ones_like(cos), torch.atan2(sin, cos))).flatten(-2)
    torch.testing.assert_close(tf.apply_rope(x, cos, sin), ref)


def test_rope_preserves_norm_and_encodes_relative_position():
    torch.manual_seed(0)
    q, k = torch.randn(1, 1, 1, 32), torch.randn(1, 1, 1, 32)
    cos, sin = tf.rope_cache(32, 64)

    def rot(x, pos):
        return tf.apply_rope(x, cos[pos : pos + 1], sin[pos : pos + 1])

    torch.testing.assert_close(rot(q, 17).norm(), q.norm())
    # <R_m q, R_n k> depends only on m - n
    s1 = (rot(q, 10) * rot(k, 3)).sum()
    s2 = (rot(q, 40) * rot(k, 33)).sum()
    torch.testing.assert_close(s1, s2, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------- attention
def test_causal_attention_matches_sdpa():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 9, 16) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(tf.causal_attention(q, k, v), ref, atol=1e-5, rtol=1e-5)


def test_repeat_kv_groups_consecutive_heads():
    x = torch.arange(2.0).view(1, 2, 1, 1)  # two KV heads with values 0 and 1
    assert tf.repeat_kv(x, 3).flatten().tolist() == [0, 0, 0, 1, 1, 1]
    assert tf.repeat_kv(x, 1) is x


@pytest.mark.parametrize("n_kv_head", [4, 2, 1])
def test_model_is_causal(n_kv_head):
    torch.manual_seed(0)
    model = tf.GPT(tiny(n_kv_head=n_kv_head)).eval()
    idx = torch.randint(0, 64, (1, 12))
    base, _ = model(idx)
    idx2 = idx.clone()
    idx2[0, 7] = (idx2[0, 7] + 1) % 64
    changed, _ = model(idx2)
    torch.testing.assert_close(base[0, :7], changed[0, :7])
    assert not torch.allclose(base[0, 7:], changed[0, 7:])


def test_gqa_with_shared_kv_equals_mha():
    # If every query head in a group sees the same K/V weights, GQA == MHA with duplicated heads.
    torch.manual_seed(0)
    gqa = tf.GPT(tiny(n_kv_head=2))
    mha = tf.GPT(tiny(n_kv_head=4))
    state = gqa.state_dict()
    for name, tensor in list(state.items()):
        if name.endswith(("attn.wk.weight", "attn.wv.weight")):
            hd = 16
            state[name] = tensor.view(2, hd, 64).repeat_interleave(2, dim=0).reshape(4 * hd, 64)
    mha.load_state_dict(state)
    idx = torch.randint(0, 64, (2, 10))
    torch.testing.assert_close(gqa(idx)[0], mha(idx)[0], atol=1e-5, rtol=1e-5)


# ----------------------------------------------------------------- model
def test_param_count_matches_formula():
    cfg = tiny(n_kv_head=2)
    model = tf.GPT(cfg)
    d, hd, L, V, ff = cfg.d_model, cfg.head_dim, cfg.n_layer, cfg.vocab_size, cfg.d_ff
    per_layer = 2 * d * d + 2 * d * (cfg.n_kv_head * hd) + 3 * d * ff + 2 * d
    assert model.num_params() == V * d + L * per_layer + d  # tied head adds nothing
    assert cfg.d_ff == 192  # 8/3 * 64 = 170.7 -> next multiple of 64


def test_tied_embeddings_share_storage():
    model = tf.GPT(tiny())
    assert model.lm_head.weight is model.tok_emb.weight


def test_residual_projection_init_is_depth_scaled():
    torch.manual_seed(0)
    cfg = tiny(n_layer=8, d_model=256, n_head=4, n_kv_head=4)
    model = tf.GPT(cfg)
    wo = torch.cat([b.attn.wo.weight.flatten() for b in model.blocks])
    wq = torch.cat([b.attn.wq.weight.flatten() for b in model.blocks])
    assert wo.std().item() == pytest.approx(0.02 / math.sqrt(16), rel=0.05)
    assert wq.std().item() == pytest.approx(0.02, rel=0.05)


def test_initial_loss_is_near_uniform():
    # Targets must be independent of inputs: with tied embeddings an untrained model already
    # favours predicting its *own input token* (see the handout), so model(idx, idx) is ~2.9.
    torch.manual_seed(0)
    model = tf.GPT(tiny())
    data = torch.randint(0, 64, (4, 33))
    _, loss = model(data[:, :-1], data[:, 1:])
    assert abs(loss.item() - math.log(64)) < 0.1


def test_ignore_index_in_targets():
    torch.manual_seed(0)
    model = tf.GPT(tiny())
    idx = torch.randint(0, 64, (2, 8))
    targets = idx.clone()
    targets[:, :4] = -100
    logits, loss = model(idx, targets)
    ref = F.cross_entropy(logits[:, 4:].reshape(-1, 64), idx[:, 4:].reshape(-1))
    torch.testing.assert_close(loss, ref)


@pytest.mark.parametrize("qk_norm", [False, True])
def test_overfits_a_batch(qk_norm):
    torch.manual_seed(0)
    model = tf.GPT(tiny(qk_norm=qk_norm))
    data = torch.randint(0, 64, (4, 33))
    x, y = data[:, :-1], data[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1, "a correct model memorizes 4 random sequences quickly"


def test_generate_extends_and_is_greedy_with_top_k_1():
    torch.manual_seed(0)
    model = tf.GPT(tiny()).eval()
    prompt = torch.randint(0, 64, (2, 5))
    out1 = model.generate(prompt, max_new_tokens=6, top_k=1)
    out2 = model.generate(prompt, max_new_tokens=6, top_k=1)
    assert out1.shape == (2, 11)
    assert torch.equal(out1, out2)
    assert torch.equal(out1[:, :5], prompt)
