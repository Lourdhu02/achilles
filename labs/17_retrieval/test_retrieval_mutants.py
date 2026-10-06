import pytest
import math
import sys
from pathlib import Path
import os

root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root))
os.environ["LABS_IMPL"] = "solution"

from labs._impl import load
mod = load('labs/17_retrieval/solution.py')

def test_mrr_mutant_inverse_rank():
    """Mutation test: MRR must use 1/rank, not rank or 1/(rank-1)."""
    def mutated_mrr(ranked: list[int], relevant: set[int]) -> float:
        # Mutant: uses rank instead of 1/rank
        return next((float(r) for r, d in enumerate(ranked, start=1) if d in relevant), 0.0)
    
    ranked = [10, 20, 30]
    relevant = {20}
    
    mut_score = mutated_mrr(ranked, relevant)
    ref_score = mod.mrr(ranked, relevant)
    
    assert mut_score != ref_score, "MRR mutant passed using incorrect rank formula"

def test_ndcg_mutant_unsorted_ideal():
    """Mutation test: NDCG ideal distribution must be sorted descending."""
    def mutated_ndcg(ranked: list[int], gains: dict[int, float], k: int) -> float:
        # Mutant: forgets to sort the ideal gains
        dcg = sum((2 ** gains.get(d, 0) - 1) / math.log2(r + 1) for r, d in enumerate(ranked[:k], start=1))
        ideal = list(gains.values())[:k] # BUG: missing sorted(..., reverse=True)
        idcg = sum((2**g - 1) / math.log2(r + 1) for r, g in enumerate(ideal, start=1))
        return dcg / idcg if idcg > 0 else 0.0
        
    ranked = [1, 2, 3]
    gains = {1: 1.0, 2: 3.0, 3: 0.0}
    
    mut_score = mutated_ndcg(ranked, gains, 3)
    ref_score = mod.ndcg_at_k(ranked, gains, 3)
    
    assert mut_score != ref_score, "NDCG mutant passed without sorting ideal gains"
