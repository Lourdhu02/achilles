import math

import pytest

from labs._impl import load

nm = load(__file__)
GiB = 2**30


# ------------------------------------------------------------------ parameters
@pytest.mark.parametrize(
    "cfg, expected",
    [
        ("GPT2_SMALL", 124_439_808),  # the famous "124M"
        ("LLAMA2_7B", 6_738_415_616),
        ("LLAMA3_8B", 8_030_261_248),
        ("MISTRAL_7B", 7_241_732_096),
    ],
)
def test_param_counts_match_released_models(cfg, expected):
    assert nm.count_params(getattr(nm, cfg)) == expected


def test_llama2_70b_is_about_69b():
    assert nm.count_params(nm.LLAMA2_70B) == pytest.approx(68.98e9, rel=1e-3)


def test_classic_block_is_12_d_squared():
    # MHA + a 4x non-gated MLP = 4d^2 + 8d^2 weights per layer: the "12 L d^2" rule.
    cfg = nm.TransformerConfig(vocab_size=1, d_model=1024, n_layers=1, n_heads=16, n_kv_heads=16, d_ff=4096, glu=False)
    assert nm.attention_weights(cfg) + nm.mlp_weights(cfg) == 12 * 1024**2


def test_gqa_shrinks_attention_weights():
    assert nm.attention_weights(nm.LLAMA3_8B) < nm.attention_weights(nm.LLAMA2_7B)


# ----------------------------------------------------------------------- FLOPs
def test_flops_per_token_gpt2():
    fwd = nm.flops_per_token(nm.GPT2_SMALL, seq_len=1024, training=False)
    assert fwd == 2 * 123_532_032 + 12 * 2 * 2 * 512 * 768
    train = nm.flops_per_token(nm.GPT2_SMALL, seq_len=1024)
    assert train == 3 * fwd
    assert train / (6 * 124.4e6) == pytest.approx(1.07, abs=0.01)  # 6N plus ~7% attention


def test_attention_term_grows_with_context():
    short = nm.flops_per_token(nm.LLAMA3_8B, seq_len=2048)
    long = nm.flops_per_token(nm.LLAMA3_8B, seq_len=131072)
    assert long / short > 1.5  # at 128k context attention rivals the weights


def test_six_n_d_and_chinchilla():
    assert nm.train_flops(70e9, 1.4e12) == pytest.approx(5.88e23)
    n, d = nm.chinchilla_optimal(5.76e23)  # Chinchilla's own budget
    assert n == pytest.approx(69.3e9, rel=0.01)
    assert d == pytest.approx(1.386e12, rel=0.01)


def test_training_days_and_mfu():
    # Llama-3-8B-like: 6 * 8e9 * 15e12 FLOPs on 1024 GPUs at 1e15 peak and 40% MFU.
    days = nm.training_days(6 * 8e9 * 15e12, n_gpus=1024, peak_flops=1e15, mfu=0.4)
    assert days == pytest.approx(20.3, rel=0.01)
    assert nm.mfu(tokens_per_second=50_000, flops_per_token_=6 * 124e6, peak_flops=1e14) == pytest.approx(0.372)


# ---------------------------------------------------------------------- memory
def test_zero_stages_match_the_zero_paper():
    # Rajbhandari et al. (2020), Fig. 1: 7.5B parameters on 64 GPUs.
    kw = dict(n_params=7.5e9, n_gpus=64)
    assert nm.training_memory_bytes(**kw, zero_stage=0)["total"] == pytest.approx(120e9)
    assert nm.training_memory_bytes(**kw, zero_stage=1)["total"] == pytest.approx(31.4e9, rel=1e-3)
    assert nm.training_memory_bytes(**kw, zero_stage=2)["total"] == pytest.approx(16.6e9, rel=3e-3)
    assert nm.training_memory_bytes(**kw, zero_stage=3)["total"] == pytest.approx(1.875e9)


def test_activation_memory_formula():
    s, b, h, a = 2048, 1, 4096, 32
    assert nm.activation_bytes_per_layer(s, b, h, a) == s * b * h * (34 + 5 * a * s / h)
    assert nm.activation_bytes_per_layer(s, b, h, a, flash=True) == 34 * s * b * h


def test_kv_cache_gqa_vs_mha():
    # Llama-2-70B at 32k context, one sequence, bf16: GQA (8 KV heads) vs MHA (64).
    gqa = nm.kv_cache_bytes(n_layers=80, n_kv_heads=8, head_dim=128, seq_len=32768)
    mha = nm.kv_cache_bytes(n_layers=80, n_kv_heads=64, head_dim=128, seq_len=32768)
    assert gqa == 10 * GiB
    assert mha == 80 * GiB
    assert nm.kv_cache_bytes(80, 8, 128, 1) == 320 * 1024  # 320 KiB per token


# ------------------------------------------------------------------- roofline
def test_matmul_intensity():
    assert nm.arithmetic_intensity_matmul(4096, 4096, 4096) == pytest.approx(4096 / 3)
    # Batch-1 decode is a matrix-vector product: ~1 FLOP per byte in bf16.
    assert nm.arithmetic_intensity_matmul(1, 4096, 4096) == pytest.approx(1.0, rel=1e-3)
    # Batching raises intensity roughly linearly until you hit the ridge.
    assert nm.arithmetic_intensity_matmul(64, 4096, 4096) == pytest.approx(62.1, rel=0.01)


def test_roofline_regimes():
    peak, bw = 100e12, 1e12
    assert nm.ridge_point(peak, bw) == 100
    assert nm.roofline_seconds(1e12, 1e9, peak, bw) == pytest.approx(0.01)  # compute-bound
    assert nm.roofline_seconds(1e9, 1e10, peak, bw) == pytest.approx(0.01)  # memory-bound


def test_decode_upper_bounds():
    bw = 448e9  # an RTX 5060's GDDR7 bandwidth
    # 8B params in 4-bit (0.5 B/param), short context: ~110 tokens/s at batch 1.
    assert nm.decode_tokens_per_second(4e9, 0, 1, bw) == pytest.approx(112.0)
    # Batching amortizes the weights until KV reads dominate.
    b1 = nm.decode_tokens_per_second(4e9, 50e6, 1, bw)
    b32 = nm.decode_tokens_per_second(4e9, 50e6, 32, bw)
    assert b32 / b1 > 20


def test_comms_and_cost():
    # 1 GB of gradients over 8 GPUs at 100 GB/s: 2 * 7/8 * 10 ms = 17.5 ms.
    assert nm.ring_allreduce_seconds(1e9, 8, 100e9) == pytest.approx(0.0175)
    assert nm.cost_per_million_tokens(2.0, 1000) == pytest.approx(0.5556, rel=1e-3)
