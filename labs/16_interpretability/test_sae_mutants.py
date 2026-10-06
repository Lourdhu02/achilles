import pytest
import torch
import math
import sys
from pathlib import Path
import os

root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root))
os.environ["LABS_IMPL"] = "solution"

from labs._impl import load
mod = load('labs/16_interpretability/solution.py')

def test_sae_mutants_missing_bias():
    """Mutation test: An SAE must subtract b_dec before encoding."""
    class MutatedSAE(mod.SparseAutoencoder):
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # Mutant: fails to subtract b_dec
            f = torch.nn.functional.relu(x @ self.W_enc + self.b_enc)
            x_hat = f @ self.W_dec + self.b_dec
            return x_hat, f

    x = torch.randn(10, 16)
    sae = MutatedSAE(16, 64)
    sae.b_dec.data.fill_(1.0)
    
    mutant_x_hat, mutant_f = sae(x)
    ref_sae = mod.SparseAutoencoder(16, 64)
    ref_sae.load_state_dict(sae.state_dict())
    ref_x_hat, ref_f = ref_sae(x)
    
    # The reference should produce different encodings since it correctly subtracts b_dec
    assert not torch.allclose(mutant_f, ref_f), "SAE mutant passed without subtracting b_dec"
