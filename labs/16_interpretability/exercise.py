# Exercise stub generated from solution.py by tools/make_exercises.py.
# Replace every NotImplementedError, then run: pytest labs/16_interpretability
"""Lab 16 -- mechanistic interpretability in miniature: induction heads, activation patching,
and a sparse autoencoder that recovers features from superposition.

Handout: labs/16_interpretability/README.md
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# ------------------------------------------------------ attention-only model
class AttnOnly(nn.Module):
    """Attention-only transformer (no MLPs, no norms): the setting of Elhage et al. (2021).
    forward() records per-layer attention patterns and residual streams, and can patch the
    residual stream after a given layer."""

    def __init__(self, vocab: int = 16, d: int = 64, n_layers: int = 2, n_heads: int = 2, max_len: int = 32):
        super().__init__()
        self.n_heads, self.hd = n_heads, d // n_heads
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.qkv = nn.ModuleList(nn.Linear(d, 3 * d, bias=False) for _ in range(n_layers))
        self.out = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(n_layers))
        self.unembed = nn.Linear(d, vocab, bias=False)
        self.attn: list[Tensor] = []
        self.resid: list[Tensor] = []

    def forward(self, idx: Tensor, patch: dict[int, Tensor] | None = None) -> Tensor:
        B, T = idx.shape
        x = self.emb(idx) + self.pos.weight[:T]
        self.attn, self.resid = [], [x]
        mask = torch.ones(T, T, dtype=torch.bool, device=idx.device).triu(1)
        for layer, (qkv, out) in enumerate(zip(self.qkv, self.out)):
            q, k, v = qkv(x).view(B, T, 3, self.n_heads, self.hd).permute(2, 0, 3, 1, 4)
            a = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(self.hd)).masked_fill(mask, float("-inf")), -1)
            self.attn.append(a.detach())
            x = x + out((a @ v).transpose(1, 2).reshape(B, T, -1))
            if patch is not None and layer in patch:
                x = patch[layer]
            self.resid.append(x)
        return self.unembed(x)


def repeated_batch(batch: int, half: int, vocab: int, generator=None) -> Tensor:
    """Random tokens of length `half`, repeated once: [x_1..x_h, x_1..x_h]."""
    first = torch.randint(0, vocab, (batch, half), generator=generator)
    return torch.cat([first, first], 1)


def induction_scores(attn: Tensor, half: int) -> Tensor:
    """attn (B, H, T, T) -> per-head mean attention from each second-half position i to i - half + 1
    (the token right after the previous occurrence of the current token)."""
    raise NotImplementedError("16_interpretability: implement induction_scores")


def train_induction(steps: int = 600, half: int = 10, vocab: int = 16, seed: int = 0) -> tuple[AttnOnly, float]:
    """Train on repeated sequences; returns the model and the final loss on the second half."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = AttnOnly(vocab=vocab, max_len=2 * half)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(steps):
        seq = repeated_batch(64, half, vocab, g)
        logits = model(seq[:, :-1])
        loss = F.cross_entropy(logits[:, half:].reshape(-1, vocab), seq[:, half + 1 :].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, loss.item()


def patching_effect(model: AttnOnly, clean: Tensor, corrupt: Tensor, layer: int, answer: int, pos: int = -1) -> float:
    """Run `corrupt` with the residual stream after `layer` replaced by clean's. Returns the fraction
    of the clean-vs-corrupt logit gap on `answer` (at position `pos`) that the patch restores."""
    raise NotImplementedError("16_interpretability: implement patching_effect")


# ------------------------------------------------------- sparse autoencoder
class SparseAutoencoder(nn.Module):
    """f = ReLU((x - b_dec) W_enc + b_enc); x_hat = f W_dec + b_dec; decoder rows kept unit-norm."""

    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(d_in, d_hidden) / math.sqrt(d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(self.W_enc.detach().T.clone())
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec /= self.W_dec.norm(dim=1, keepdim=True)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("16_interpretability: implement forward")

    def loss(self, x: Tensor, l1: float) -> Tensor:
        raise NotImplementedError("16_interpretability: implement loss")


def superposition_data(n: int, n_features: int = 32, d: int = 8, p_active: float = 0.05, seed: int = 0) -> tuple[Tensor, Tensor]:
    """Sparse non-negative features embedded in d < n_features dims along random unit directions.
    Returns (x (n, d), true directions (n_features, d))."""
    g = torch.Generator().manual_seed(seed)
    dirs = F.normalize(torch.randn(n_features, d, generator=g), dim=1)
    active = torch.rand(n, n_features, generator=g) < p_active
    feats = torch.rand(n, n_features, generator=g) * active
    return feats @ dirs, dirs


def train_sae(steps: int = 3000, l1: float = 0.2, seed: int = 0) -> float:
    """Train an SAE on toy superposition data (32 features in 16 dims); return the fraction of true
    directions recovered (max cosine with some decoder row > 0.9). Sparsity strength matters: with
    l1 = 0.01 almost nothing is recovered; with 0.2, nearly everything."""
    torch.manual_seed(seed)
    x, dirs = superposition_data(20000, n_features=32, d=16, seed=seed)
    sae = SparseAutoencoder(16, 64)
    sae.b_dec.requires_grad_(False)  # the data mean mixes every feature; keep the pre-bias at 0 here
    opt = torch.optim.Adam([p for p in sae.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(steps):
        batch = x[torch.randint(0, len(x), (1024,))]
        loss = sae.loss(batch, l1=l1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.normalize_decoder()
    cos = F.normalize(dirs, dim=1) @ sae.W_dec.detach().T
    return (cos.max(1).values > 0.9).float().mean().item()
