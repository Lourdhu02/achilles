# Setup

## 1. Python environment
```bash
# with uv (recommended)
uv sync
# or with pip
python -m venv .venv && source .venv/bin/activate
pip install numpy regex pytest torch
```

## 2. PyTorch for the RTX 5060 (Blackwell, compute capability 12.0 / sm_120)
- Use an NVIDIA driver from the **R570 series or newer**.
- Use a PyTorch build for **CUDA 12.8 or newer** (PyTorch ≥ 2.7). Older builds fail with *"no kernel image is available for execution on the device."*
- **Linux / WSL2:** recent PyPI wheels already target CUDA 12.8+; to be explicit, `pip install torch --index-url https://download.pytorch.org/whl/cu128`.
- **Windows native:** PyPI wheels are CPU-only, so install with the `cu128` index URL above. Triton (labs 06, `--compile`) is easiest on **WSL2**; natively, try the community `triton-windows` package.

Verify:
```python
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))  # expect (12, 0)
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16); print((x @ x).float().abs().mean())
```

## 3. Check everything
```bash
python -m pytest --impl=solution      # reference solutions pass (what CI runs)
python tools/progress.py              # your scoreboard (starts at 0%)
python tools/measure_gpu.py           # your roofline: write it in journal/
python tools/check_links.py           # docs integrity
```

## 4. Optional: data for scale-up runs
```bash
pip install huggingface_hub tiktoken
python -c "from huggingface_hub import hf_hub_download as d; d('roneneldan/TinyStories','TinyStoriesV2-GPT4-train.txt',repo_type='dataset',local_dir='data')"
```
`data/`, `runs/` and checkpoints are git-ignored. 8 GB of VRAM guide: pretrain ≤ ~150M parameters; LoRA on ≤ 1.5B in bf16; QLoRA up to 7–8B with short sequences.
