import torch

from labs._impl import load

qz = load(__file__)
NF4_QLORA = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
             -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
             0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
             0.7229568362236023, 1.0]


def mse(a, b):
    return ((a - b) ** 2).mean().item()


def test_int8_roundtrip_error_is_bounded():
    torch.manual_seed(0)
    w = torch.randn(64, 128)
    q, s = qz.quantize_symmetric(w, 8)
    assert q.dtype == torch.int8 and q.abs().max() <= 127
    assert (qz.dequantize(q, s) - w).abs().max() <= s / 2 + 1e-6


def test_per_channel_beats_per_tensor_with_outlier_rows():
    torch.manual_seed(0)
    w = torch.randn(64, 128)
    w[3] *= 50
    per_tensor = mse(qz.dequantize(*qz.quantize_symmetric(w, 8)), w)
    per_row = mse(qz.dequantize(*qz.quantize_symmetric(w, 8, dim=1)), w)
    assert per_row < 0.1 * per_tensor


def test_groupwise_int4_beats_per_tensor_int4():
    torch.manual_seed(0)
    w = torch.randn(32, 256) * torch.linspace(0.1, 5, 256)
    g = mse(qz.dequantize_groupwise(*qz.quantize_groupwise(w, 4, 64), 64), w)
    t = mse(qz.dequantize(*qz.quantize_symmetric(w, 4)), w)
    assert g < t


def test_int4_packing_roundtrip():
    q = torch.randint(-8, 8, (6, 10), dtype=torch.int8)
    packed = qz.pack_int4(q)
    assert packed.dtype == torch.uint8 and packed.numel() == 30
    assert torch.equal(qz.unpack_int4(packed, q.shape), q)


def test_nf4_codebook_matches_qlora():
    torch.testing.assert_close(qz.nf4_codebook(), torch.tensor(NF4_QLORA), atol=1e-4, rtol=0)


def test_nf4_beats_uniform_int4_on_gaussian_weights():
    torch.manual_seed(0)
    w = torch.randn(256, 256)
    nf4 = mse(qz.dequantize_nf4(*qz.quantize_nf4(w, 64), 64), w)
    int4 = mse(qz.dequantize_groupwise(*qz.quantize_groupwise(w, 4, 64), 64), w)
    assert nf4 < int4


def test_smoothquant_preserves_output_and_tames_outliers():
    torch.manual_seed(0)
    x = torch.randn(32, 16)
    x[:, 2] *= 40
    w = torch.randn(8, 16)
    ws, s = qz.smoothquant(w, x.abs().amax(0), 0.5)
    torch.testing.assert_close((x / s) @ ws.T, x @ w.T, atol=1e-3, rtol=1e-4)
    assert (x / s).abs().max() < 0.5 * x.abs().max()


def test_gptq_beats_round_to_nearest_on_correlated_inputs():
    torch.manual_seed(0)
    base = torch.randn(512, 8)
    x = base @ torch.randn(8, 32) + 0.1 * torch.randn(512, 32)  # strongly correlated features
    w = torch.randn(16, 32)
    qmax = 7
    scale = w.abs().amax(1, keepdim=True) / qmax
    rtn = torch.clamp(torch.round(w / scale), -qmax, qmax) * scale
    gptq = qz.gptq_quantize(w, x, bits=4)
    assert mse(x @ gptq.T, x @ w.T) < 0.5 * mse(x @ rtn.T, x @ w.T)
