import pytest
import torch

from labs._impl import load

mi = load(__file__)


@pytest.fixture(scope="module")
def trained():
    return mi.train_induction()


def test_induction_score_indexing():
    attn = torch.zeros(1, 1, 8, 8)
    for i in range(4, 8):
        attn[0, 0, i, i - 3] = 1.0  # exactly the induction offset for half=4
    assert mi.induction_scores(attn, half=4).item() == 1.0


def test_model_learns_in_context_copying(trained):
    _, loss = trained
    assert loss < 0.5, "second-half tokens are copyable; loss should fall far below ln(16) = 2.77"


def test_an_induction_head_forms_in_layer_2(trained):
    model, _ = trained
    seq = mi.repeated_batch(32, 10, 16, torch.Generator().manual_seed(1))
    model(seq)
    assert mi.induction_scores(model.attn[1], half=10).max() > 0.5  # chance is ~1/10


def test_patching_the_final_residual_restores_everything(trained):
    model, _ = trained
    g = torch.Generator().manual_seed(2)
    clean = mi.repeated_batch(1, 10, 16, g)
    corrupt = clean.clone()
    corrupt[0, :10] = torch.randint(0, 16, (10,), generator=g)  # break the first occurrence
    answer = clean[0, 11].item()
    assert mi.patching_effect(model, clean[:, :-1], corrupt[:, :-1], layer=1, answer=answer, pos=10) == pytest.approx(1.0, abs=1e-4)


def test_sae_recovers_features_from_superposition():
    assert mi.train_sae() > 0.7
