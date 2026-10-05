import argparse
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABS = ROOT / "labs"

MUTANTS = {
    "08_scaling_laws": [
        (
            "best = (resid, E, float(np.exp(coef[0])), float(-coef[1]))",
            "best = (resid, E, float(-coef[1]), float(np.exp(coef[0])))",
            "swapped A and alpha"
        ),
        (
            "return E + A / np.power(N, alpha) + B / np.power(D, beta)",
            "return E + A / np.power(N, alpha) - B / np.power(D, beta)",
            "negative D term in Chinchilla loss"
        ),
        (
            "n = G * (C / 6.0) ** (beta / (alpha + beta))",
            "n = G * (C / 6.0) ** (alpha / (alpha + beta))",
            "swapped alpha/beta in exponent"
        ),
        (
            "total = 6 * n * d + 2 * n * inference_tokens",
            "total = 6 * n * d + 6 * n * inference_tokens",
            "wrong inference FLOPs per token"
        ),
        (
            "d = (B / gap) ** (1.0 / beta)",
            "d = (A / gap) ** (1.0 / beta)",
            "swapped A and B in inference_aware_optimum"
        ),
        (
            "y = np.log(L - E)",
            "y = np.log(L + E)",
            "wrong log transform"
        ),
    ],
    "09_parallelism": [
        (
            "for step in range(n - 1):  # all-gather:",
            "for step in range(0):  # all-gather:",
            "missing final all-gather"
        ),
        (
            "chunks[(r + 1) % n][c] += data",
            "chunks[r][c] += data",
            "misaligned ring all-reduce chunks"
        ),
        (
            "if stage >= 1:\n        o /= n_gpus",
            "if stage >= 1:\n        p /= n_gpus",
            "ZeRO counting optimizer states for wrong stage"
        ),
        (
            "np.concatenate([x @ w for w in w_shards], axis=-1)",
            "np.sum([x @ w for w in w_shards], axis=0)",
            "sum instead of concat in column parallel"
        ),
        (
            "sum(x @ w for x, w in zip(x_shards, w_shards))",
            "np.concatenate([x @ w for x, w in zip(x_shards, w_shards)], axis=-1)",
            "concat instead of sum in row parallel"
        ),
        (
            "return (stages - 1) / (micro_batches + stages - 1)",
            "return stages / (micro_batches + stages)",
            "wrong pipeline bubble"
        ),
    ],
    "10_lora": [
        (
            "self.base, self.r, self.scale = base, r, alpha / r",
            "self.base, self.r, self.scale = base, r, alpha",
            "LoRA scaled by alpha instead of alpha/r"
        ),
        (
            "merged.weight.copy_(self.base.weight + self.scale * self.B @ self.A)",
            "merged.weight.copy_(self.base.weight - self.scale * self.B @ self.A)",
            "merge with wrong sign (can't be undone properly)"
        ),
        (
            "self.B = nn.Parameter(torch.zeros(base.out_features, r))",
            "self.B = nn.Parameter(torch.ones(base.out_features, r))",
            "LoRA B initialized to ones"
        ),
        (
            "bias=self.base.bias is not None",
            "bias=False",
            "dropped bias in merge"
        ),
        (
            "if p.requires_grad",
            "",
            "counted all params instead of trainable"
        ),
        (
            "for p in model.parameters():\n        p.requires_grad_(False)",
            "",
            "forgot to freeze model"
        ),
    ],
    "11_dpo": [
        (
            "chosen = beta * (pi_chosen - ref_chosen)",
            "chosen = beta * pi_chosen",
            "DPO missing ref log-ratio"
        ),
        (
            "chosen = beta * (pi_chosen - ref_chosen)",
            "chosen = beta * beta * (pi_chosen - ref_chosen)",
            "beta applied twice"
        ),
        (
            "loss = -(1 - label_smoothing) * F.logsigmoid(h) - label_smoothing * F.logsigmoid(-h)",
            "loss = -F.logsigmoid(h)",
            "missing label smoothing"
        ),
        (
            "return ((h - 1 / (2 * beta)) ** 2).mean()",
            "return ((h - 1 / beta) ** 2).mean()",
            "wrong margin in IPO"
        ),
        (
            "return -F.logsigmoid(beta * (avg_chosen - avg_rejected) - gamma).mean()",
            "return -F.logsigmoid(beta * (avg_chosen - avg_rejected) + gamma).mean()",
            "wrong gamma sign in SimPO"
        ),
        (
            "return loss.mean(), chosen.detach(), rejected.detach()",
            "return loss.mean(), chosen, rejected",
            "missing detach in DPO"
        ),
    ],
    "12_grpo": [
        (
            "return centered / (rewards.std(dim=1, unbiased=False, keepdim=True) + eps) if std_norm else centered",
            "return centered",
            "GRPO advantages not normalized per group"
        ),
        (
            "torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a",
            "torch.clamp(ratio * a, 1 - clip_eps, 1 + clip_eps)",
            "clipping on the wrong ratio"
        ),
        (
            "kl = torch.exp(d) - d - 1",
            "kl = 1 + d - torch.exp(d)",
            "k3 KL estimator wrong sign"
        ),
        (
            "pi * (rewards - (pi * rewards).sum())",
            "pi * rewards",
            "missing baseline in exact_policy_gradient"
        ),
        (
            "surrogate = torch.minimum(ratio * a, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a)",
            "surrogate = ratio * a",
            "missing clipping entirely"
        ),
        (
            "((per_token * m).sum(1) / m.sum(1)).mean()",
            "(per_token * m).sum() / m.sum()",
            "wrong aggregation for GRPO"
        )
    ]
}

def run_tests(lab: str, tmpdir: str) -> bool:
    """Run tests for the lab with --impl=solution, where solution.py is the mutated one in tmpdir.
    Returns True if tests pass (mutant SURVIVED)."""
    # Pytest runs against the module. We need to overwrite the solution.py in the actual lab directory temporarily,
    # or set PYTHONPATH so it picks up tmpdir. 
    # But achilles loads impl via `labs._impl.load`, which uses `importlib.import_module(f"labs.{name}.{impl}")`.
    # Overwriting the file temporarily is safest.
    
    lab_dir = LABS / lab
    sol_path = lab_dir / "solution.py"
    backup = sol_path.read_text(encoding="utf-8")
    
    mutated = Path(tmpdir) / "solution.py"
    sol_path.write_text(mutated.read_text(encoding="utf-8"), encoding="utf-8")
    
    try:
        res = subprocess.run(
            [sys.executable, "-m", "pytest", str(lab_dir), "--impl=solution", "-q", "--tb=short"],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        # If returncode is 0, all tests passed -> SURVIVED
        return res.returncode == 0
    finally:
        sol_path.write_text(backup, encoding="utf-8")

def main():
    if len(sys.argv) > 1:
        labs_to_run = sys.argv[1:]
    else:
        labs_to_run = list(MUTANTS.keys())

    for lab in labs_to_run:
        print(f"\n=== Mutating {lab} ===")
        if lab not in MUTANTS:
            print(f"No mutants defined for {lab}")
            continue
            
        sol_path = LABS / lab / "solution.py"
        original = sol_path.read_text(encoding="utf-8")
        
        killed = 0
        total = len(MUTANTS[lab])
        
        for target, replacement, desc in MUTANTS[lab]:
            if target not in original:
                print(f"[ERROR] Target string not found for: {desc}")
                total -= 1
                continue
                
            mutated = original.replace(target, replacement, 1)
            
            with tempfile.TemporaryDirectory() as d:
                Path(d).joinpath("solution.py").write_text(mutated, encoding="utf-8")
                survived = run_tests(lab, d)
                
            if survived:
                print(f"[SURVIVED] {desc}")
            else:
                print(f"[KILLED]   {desc}")
                killed += 1
                
        print(f"Mutation score for {lab}: {killed}/{total} ({killed/total*100:.0f}%)")

if __name__ == "__main__":
    main()
