import numpy as np
import pytest
import torch

from labs._impl import load

sd = load(__file__)

P_TABLE = {0: [0.6, 0.3, 0.1], 1: [0.1, 0.2, 0.7], 2: [0.3, 0.3, 0.4]}
Q_TABLE = {0: [0.2, 0.5, 0.3], 1: [0.4, 0.4, 0.2], 2: [0.1, 0.8, 0.1]}


def markov(table):
    return lambda prefix: np.array(table[prefix[-1]])


def test_output_distribution_is_exactly_the_target():
    rng = np.random.default_rng(0)
    p, q = markov(P_TABLE), markov(Q_TABLE)
    counts = np.zeros((3, 3))
    n = 30000
    for _ in range(n):
        seq = []
        while len(seq) < 2:
            seq += sd.speculative_step(p, q, (0,) + tuple(seq), k=2, rng=rng)
        counts[seq[0], seq[1]] += 1
    exact = np.array([[P_TABLE[0][a] * P_TABLE[a][b] for b in range(3)] for a in range(3)])
    assert np.abs(counts / n - exact).sum() / 2 < 0.015  # total-variation distance


def test_tokens_per_step_matches_theory():
    rng = np.random.default_rng(1)
    pv, qv = np.array([0.5, 0.3, 0.2]), np.array([0.3, 0.3, 0.4])
    alpha = sd.acceptance_rate(pv, qv)
    assert alpha == pytest.approx(0.8)
    lengths = [len(sd.speculative_step(lambda _: pv, lambda _: qv, (0,), 4, rng)) for _ in range(20000)]
    assert np.mean(lengths) == pytest.approx(sd.expected_tokens(alpha, 4), rel=0.02)


def test_formulas():
    assert sd.expected_tokens(0.0, 5) == 1.0
    assert sd.expected_tokens(1.0, 5) == 6.0
    assert sd.expected_speedup(0.8, 4, 0.05) == pytest.approx((1 - 0.8**5) / 0.2 / 1.2)

def test_verify_block_exactness_and_cache_rollback():
    torch.manual_seed(0)
    V = 4
    p, q = sd.CausalModel(vocab_size=V, d_model=8).double(), sd.CausalModel(vocab_size=V, d_model=8).double()
    prompt = torch.tensor([0])

    spec_probs = {}
    q_logits = q(prompt, sd.Cache())[-1]
    q_probs_0 = torch.softmax(q_logits, -1)
    
    p_logits = p(prompt, sd.Cache())[-1]
    p_probs_0 = torch.softmax(p_logits, -1)
    
    for d1 in range(V):
        q_prob_d1 = q_probs_0[d1].item()
        q_logits_1 = q(torch.cat([prompt, torch.tensor([d1])]), sd.Cache())[-1]
        q_probs_1 = torch.softmax(q_logits_1, -1)
        
        for d2 in range(V):
            q_prob_d2 = q_probs_1[d2].item()
            path_q_prob = q_prob_d1 * q_prob_d2
            
            p_logits_all = p(torch.cat([prompt, torch.tensor([d1, d2])]), sd.Cache())
            p_probs_all = torch.softmax(p_logits_all, -1)
            
            p_prob_d1 = p_probs_all[0, d1].item()
            accept_1 = min(1.0, p_prob_d1 / q_probs_0[d1].item())
            reject_1 = 1.0 - accept_1
            
            p_prob_d2 = p_probs_all[1, d2].item()
            accept_2 = min(1.0, p_prob_d2 / q_probs_1[d2].item())
            reject_2 = 1.0 - accept_2
            
            resid_1 = torch.clamp(p_probs_all[0] - q_probs_0, min=0.0)
            resid_1 = resid_1 / resid_1.sum()
            for r1 in range(V):
                prob = path_q_prob * reject_1 * resid_1[r1].item()
                if prob > 0:
                    spec_probs[(r1,)] = spec_probs.get((r1,), 0.0) + prob
            
            resid_2 = torch.clamp(p_probs_all[1] - q_probs_1, min=0.0)
            resid_2 = resid_2 / resid_2.sum()
            for r2 in range(V):
                prob = path_q_prob * accept_1 * reject_2 * resid_2[r2].item()
                if prob > 0:
                    spec_probs[(d1, r2)] = spec_probs.get((d1, r2), 0.0) + prob
            
            for b in range(V):
                prob = path_q_prob * accept_1 * accept_2 * p_probs_all[2, b].item()
                if prob > 0:
                    spec_probs[(d1, d2, b)] = spec_probs.get((d1, d2, b), 0.0) + prob

    for seq, prob in spec_probs.items():
        if len(seq) == 1:
            theory = max(0.0, p_probs_0[seq[0]].item() - q_probs_0[seq[0]].item())
        elif len(seq) == 2:
            c, d = seq
            p_c_acc = min(p_probs_0[c].item(), q_probs_0[c].item())
            p_logits_1 = p(torch.cat([prompt, torch.tensor([c])]), sd.Cache())[-1]
            p_probs_1 = torch.softmax(p_logits_1, -1)
            q_logits_1 = q(torch.cat([prompt, torch.tensor([c])]), sd.Cache())[-1]
            q_probs_1 = torch.softmax(q_logits_1, -1)
            p_d_rej = max(0.0, p_probs_1[d].item() - q_probs_1[d].item())
            theory = p_c_acc * p_d_rej
        elif len(seq) == 3:
            c, d, b = seq
            p_c_acc = min(p_probs_0[c].item(), q_probs_0[c].item())
            p_logits_1 = p(torch.cat([prompt, torch.tensor([c])]), sd.Cache())[-1]
            p_probs_1 = torch.softmax(p_logits_1, -1)
            q_logits_1 = q(torch.cat([prompt, torch.tensor([c])]), sd.Cache())[-1]
            q_probs_1 = torch.softmax(q_logits_1, -1)
            p_d_acc = min(p_probs_1[d].item(), q_probs_1[d].item())
            p_logits_2 = p(torch.cat([prompt, torch.tensor([c, d])]), sd.Cache())[-1]
            p_probs_2 = torch.softmax(p_logits_2, -1)
            p_b = p_probs_2[b].item()
            theory = p_c_acc * p_d_acc * p_b

        assert prob == pytest.approx(theory, abs=1e-12)

def test_cache_recomputation_and_target_calls():
    torch.manual_seed(0)
    p, q = sd.CausalModel(vocab_size=4, d_model=8), sd.CausalModel(vocab_size=4, d_model=8)
    class TrackedTarget(sd.CausalModel):
        calls = 0
        def forward(self, x, cache):
            self.calls += 1
            return super().forward(x, cache)
    p_tracked = TrackedTarget(vocab_size=4, d_model=8)
    p_tracked.load_state_dict(p.state_dict())
    
    prompt = torch.tensor([0, 1])
    cache_t, cache_d = sd.Cache(), sd.Cache()
    rng = torch.Generator().manual_seed(0)
    
    out, cache_t, cache_d = sd.verify_block(p_tracked, q, prompt, cache_t, cache_d, k=3, rng=rng)
    assert p_tracked.calls == 1
    
    ref_cache = sd.Cache()
    p(torch.cat([prompt, out[:-1]]), ref_cache)
    assert len(ref_cache) == len(cache_t)
    for rc, tc in zip(ref_cache, cache_t):
        torch.testing.assert_close(rc, tc, atol=1e-6, rtol=0)

def test_verify_block_edge_cases():
    torch.manual_seed(0)
    p, q = sd.CausalModel(vocab_size=4, d_model=8), sd.CausalModel(vocab_size=4, d_model=8)
    rng = torch.Generator().manual_seed(0)
    
    out, cache_t, cache_d = sd.verify_block(p, q, torch.tensor([0]), sd.Cache(), sd.Cache(), 0, rng)
    assert len(out) == 1
    
    out2, _, _ = sd.verify_block(p, p, torch.tensor([0]), sd.Cache(), sd.Cache(), 3, rng)
    assert len(out2) == 4
    
    class DisjointDraft(sd.CausalModel):
        def forward(self, x, cache):
            logits = super().forward(x, cache)
            mask = torch.full_like(logits, float('-inf'))
            mask[..., 0] = 0.0
            return mask
            
    class DisjointTarget(sd.CausalModel):
        def forward(self, x, cache):
            logits = super().forward(x, cache)
            mask = torch.full_like(logits, float('-inf'))
            mask[..., 1] = 0.0
            return mask
            
    q_disjoint = DisjointDraft(vocab_size=4, d_model=8)
    p_disjoint = DisjointTarget(vocab_size=4, d_model=8)
    out3, _, _ = sd.verify_block(p_disjoint, q_disjoint, torch.tensor([0]), sd.Cache(), sd.Cache(), 3, rng)
    assert len(out3) == 1
    assert out3[0] == 1
