# Setup

Install the repo on whatever you have: a CPU-only laptop, an NVIDIA GPU, an Apple Silicon Mac, a free Colab or Kaggle notebook, or a GitHub Codespace.
Every lab's tests run on a CPU. A GPU only matters for the scale-up runs, and the [tier table](#what-you-can-do-on-each-tier) tells you which ones need one.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lourdhu02/achilles/blob/main/notebooks/colab_quickstart.ipynb)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Lourdhu02/achilles)

**Contents**
- [Pick your path](#pick-your-path)
- [1. Python environment](#1-python-environment)
- [2. PyTorch for your hardware](#2-pytorch-for-your-hardware)
- [3. Verify the install](#3-verify-the-install)
- [4. Cloud notebooks and Codespaces](#4-cloud-notebooks-and-codespaces)
- [What you can do on each tier](#what-you-can-do-on-each-tier)
- [5. Training settings per tier](#5-training-settings-per-tier)
- [6. Check everything](#6-check-everything)
- [7. Optional: data for scale-up runs](#7-optional-data-for-scale-up-runs)
- [Troubleshooting](#troubleshooting)

## Pick your path

| You have | Go to | Time to first passing test |
|---|---|---|
| Any laptop, no GPU (Windows, Linux or macOS) | [1](#1-python-environment), then [CPU wheels](#cpu-only-any-os) | about 10 minutes |
| NVIDIA GPU (RTX 20-series or newer, including the RTX 5060) | [1](#1-python-environment), then [CUDA wheels](#nvidia-gpu-cuda) | about 15 minutes |
| Apple Silicon Mac (M1 or newer) | [1](#1-python-environment), then [Apple Silicon](#apple-silicon-mps) | about 10 minutes |
| Only a browser | [Colab, Kaggle or Codespaces](#4-cloud-notebooks-and-codespaces) | about 5 minutes |

The full matrix of what is supported:

| OS | CPU only | NVIDIA GPU (CUDA) | Apple Silicon (MPS) |
|---|---|---|---|
| **Linux** | CPU wheels; Triton kernels run in its CPU interpreter | `cu128` wheels; Triton ships with them | not applicable |
| **Windows** | CPU wheels (plain PyPI `torch` is CPU-only here too) | `cu128` wheels; Triton via WSL2 (easiest) or the community `triton-windows` package | not applicable |
| **macOS** | Apple Silicon: default PyPI wheels. Intel Macs: PyTorch stopped publishing x86-64 macOS wheels after 2.2, and this repo needs 2.7, so use Codespaces or Colab | not available | default PyPI wheels; no Triton, so the Triton tests skip |
| **Colab / Kaggle** | CPU runtime | T4 (and on Kaggle, T4 x2 or P100); PyTorch is preinstalled | not applicable |
| **Codespaces** | CPU; the dev container installs everything | not available | not applicable |

## 1. Python environment

Use Python 3.10 to 3.12, and always a virtual environment inside the repo, never your global Python, so other projects keep their own torch.

**Linux and macOS**
```bash
git clone https://github.com/Lourdhu02/achilles.git
cd achilles
python3 -m venv .venv
source .venv/bin/activate            # run again in every new terminal
python -m pip install --upgrade pip
```

### Windows (PowerShell), step by step
Install into a virtual environment inside the repo, never into your global Python, so other projects keep their own torch.
```powershell
git clone https://github.com/Lourdhu02/achilles.git
cd achilles
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU; CPU-only laptops use .../whl/cpu instead
pip install -r requirements.txt
pip install triton-windows          # optional, NVIDIA only: Triton kernels (lab 06) and --compile
```
Run `.venv\Scripts\Activate.ps1` again in every new terminal. The prompt shows `(.venv)` when it is active.
Drivers with CUDA 13 support can try `.../whl/cu130` for newer releases. `triton-windows` is a community port: pick the release its README pairs with your torch version.

**With uv (optional).** `uv venv` creates `.venv`; then prefix the `pip` commands below with `uv` (`uv pip install torch --index-url ...`, `uv pip install -r requirements.txt`).

> [!WARNING]
> `uv sync` and a bare `pip install torch` take torch from PyPI. On Windows that build is CPU-only, and on Linux it pulls a multi-gigabyte CUDA build even on machines without a GPU. Install torch first with the index URL for your hardware, then the rest.

## 2. PyTorch for your hardware

Pick one row, run it inside the activated venv, then install the remaining dependencies.

| Hardware | OS | Install torch |
|---|---|---|
| CPU only | Linux, Windows | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| CPU only | macOS (Apple Silicon) | `pip install torch` |
| NVIDIA GPU | Linux, WSL2, Windows | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| Apple Silicon GPU (MPS) | macOS on M1 or newer | `pip install torch` (the default wheels include MPS) |

Then, for everyone:
```bash
pip install -r requirements.txt       # numpy, regex, pytest (torch is already satisfied)
```

Optional extras:
```bash
pip install triton                    # Linux CPU only: runs the lab 06 Triton tests in the interpreter
pip install -r requirements-data.txt  # tiktoken, datasets, huggingface_hub for the scale-up runs
```
CUDA builds of torch on Linux already include Triton. On macOS there is no Triton; those tests skip.

> [!NOTE]
> **Keep versions paired.** `torch` and `torchvision` must come from the same release and the same CUDA index (torchvision 0.28 needs torch 2.13, 0.26 needs 2.11, and so on). The labs don't use torchvision. If pip warns about a mismatch, install both from one index or uninstall torchvision in this repo's venv.

If pip reports no matching distribution for a CUDA variant, the current release may have moved on: the selector at [pytorch.org/get-started](https://pytorch.org/get-started/locally/) lists the variants that exist today.

### CPU only (any OS)
Everything in the labs runs: every test suite, the napkin math, and small versions of the pretraining and scaling runs (`--preset cpu`). Nothing needs a GPU until the scale-up runs.

### NVIDIA GPU (CUDA)
Any NVIDIA GPU that current PyTorch supports works. Ampere (RTX 30-series) and newer get fast bf16; older cards such as the T4 fall back to fp16, which `train.py` handles automatically.

#### PyTorch for the RTX 5060 (Blackwell, compute capability 12.0 / sm_120)
- Use an NVIDIA driver from the **R570 series or newer**.
- Use a PyTorch build for **CUDA 12.8 or newer** (PyTorch ≥ 2.7). Older builds fail with *"no kernel image is available for execution on the device."*
- **Linux / WSL2:** recent PyPI wheels already target CUDA 12.8+; to be explicit, `pip install torch --index-url https://download.pytorch.org/whl/cu128`.
- **Windows native:** PyPI wheels are CPU-only, so install with the `cu128` index URL above. Triton (labs 06, `--compile`) is easiest on **WSL2**; natively, try the community `triton-windows` package.
- **8 GB of VRAM guide:** pretrain ≤ ~150M parameters; LoRA on ≤ 1.5B in bf16; QLoRA up to 7–8B with short sequences.

### Apple Silicon (MPS)
The default macOS wheels support the Apple GPU through the MPS backend. `train.py --device auto` picks it up. A few operators have no MPS kernel yet; if you hit `NotImplementedError` for one, run with `PYTORCH_ENABLE_MPS_FALLBACK=1` to compute that operator on the CPU. `train.py` runs in fp32 on MPS (no autocast), and `--compile` is ignored there.

## 3. Verify the install

Run the line for your backend inside the activated venv.

**CPU**
```bash
python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available())"
```

**NVIDIA** (also run `nvidia-smi`, which should list your GPU and driver version)
```python
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))  # RTX 5060: expect (12, 0)
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16); print((x @ x).float().abs().mean())
```

**Apple Silicon**
```bash
python -c "import torch; print(torch.__version__, 'mps:', torch.backends.mps.is_available()); x = torch.randn(2048, 2048, device='mps'); print((x @ x).abs().mean().item())"
```

**Everyone**
```bash
python -m pytest labs/01_autograd --impl=solution -q   # 37 passed
python tools/measure_gpu.py                            # roofline numbers and your hardware tier
```
`measure_gpu.py` ends with a line such as `hardware tier: ... -> train.py --preset gpu-8gb`. Use that preset in [section 5](#5-training-settings-per-tier).

## 4. Cloud notebooks and Codespaces

### Google Colab
Click the **Open in Colab** badge at the top. Then **Runtime → Change runtime type → T4 GPU** (the free tier usually offers a T4; availability varies). The notebook clones the repo, checks the GPU, runs a lab's tests, measures the GPU and trains a small model.
- PyTorch is preinstalled with CUDA. Don't reinstall torch; `pip install -r requirements.txt` only adds what is missing.
- The T4 (compute capability 7.5) has no fast bf16, so `train.py` uses fp16 with loss scaling there.
- The runtime's disk is wiped when the session ends. Push your `exercise.py` work to your fork, or mount Google Drive, before you close the tab.

### Kaggle
Create a notebook, turn on **Internet** and pick **Accelerator → GPU T4 x2** in the notebook settings (as of September 2026 Kaggle requires a phone-verified account for both; re-verify in Kaggle's settings). Then run the same commands as the Colab notebook:
```bash
!git clone https://github.com/Lourdhu02/achilles.git
%cd achilles
!pip install -q -r requirements.txt
!python -m pytest labs/05_transformer --impl=solution -q
```
Two T4s in one machine are the cheapest way to try real multi-GPU `torch.distributed` for lab 09 (`torchrun --nproc_per_node=2`). If a P100 session prints *"no kernel image is available"*, switch the accelerator to T4.

### GitHub Codespaces
Click the **Open in GitHub Codespaces** badge (or **Code → Codespaces → Create codespace** on GitHub). The [dev container](.devcontainer/devcontainer.json) installs Python 3.11, CPU PyTorch, Triton and the test tools, then runs the reference tests, so the first terminal you open shows whether everything works. Codespaces are CPU-only. Personal accounts get a monthly allowance of free core-hours (check the current numbers on GitHub's billing page; they change), so stop the codespace when you are done.

## What you can do on each tier

Tiers, from smallest to largest. `measure_gpu.py` prints yours.

| Tier | Examples | `train.py` preset |
|---|---|---|
| **CPU** | any laptop, Codespaces, a Colab or Kaggle CPU runtime | `cpu` |
| **Apple Silicon** | M1 or newer with 16 GB or more of unified memory | `cpu` for a quick check, `gpu-8gb` for a real run |
| **NVIDIA 8–16 GB** | RTX 5060 (8 GB), RTX 3060 (12 GB), Colab/Kaggle T4 (16 GB) | `gpu-8gb` |
| **NVIDIA 24 GB+** | RTX 3090/4090, L4, A10G, A100, H100 | `gpu-24gb` |

Minimum hardware for each piece of work. "Reduced" means the same experiment at a smaller scale; "slow" means it works but takes several times longer than on the minimum tier.

| Work | CPU | Apple Silicon | NVIDIA 8–16 GB | NVIDIA 24 GB+ | Minimum |
|---|---|---|---|---|---|
| Labs 01–05, 07–17: implement and pass the tests | yes | yes | yes | yes | CPU |
| Lab 06 PyTorch FlashAttention (forward, backward) | yes | yes | yes | yes | CPU |
| Lab 06 Triton kernels: correctness tests | Linux only (interpreter) | skip (no Triton) | yes | yes | Linux CPU |
| Lab 06 Triton kernels: benchmark against SDPA | no | no | yes | yes | NVIDIA GPU |
| Lab 03 roofline (`measure_gpu.py`) | yes | yes | yes | yes | CPU |
| **S1** pretraining on TinyStories (lab 05, `train.py`) | reduced (`--preset cpu`) | yes (slow) | yes | larger model | CPU (reduced), NVIDIA 8 GB (full) |
| **S2** scaling sweep, 5 sizes × 3 budgets (lab 08) | reduced (sizes under 2M) | reduced | yes (overnight) | yes | NVIDIA 8 GB |
| Lab 09 DDP/FSDP | `torchrun` with the `gloo` backend | same as CPU | one GPU: API only | one GPU: API only | Kaggle T4 x2 for real multi-GPU |
| **S4** LoRA SFT of Qwen2.5-0.5B (lab 10), then DPO (lab 11) | no | slow (16 GB+) | yes | yes | NVIDIA 8 GB or a T4 |
| **S5** GRPO on a verifiable task (lab 12) | no | slow (16 GB+) | yes | yes | NVIDIA 8 GB or a T4 |
| Lab 13 quantization quality vs tokens/s | reduced | reduced | yes | yes | NVIDIA 8 GB |
| Lab 14 speculative decoding, 0.5B draft → 1.5B target | slow | yes | yes | yes | Apple Silicon or NVIDIA 8 GB |
| Lab 16 SAE on your S1 model | yes | yes | yes | yes | CPU |
| Lab 17 retrieval and RAG evaluation | yes | yes | yes | yes | CPU |
| Kernels: writing and profiling your own Triton/CUDA kernels | no | no | yes | yes | NVIDIA GPU on Linux or WSL2 (Colab works) |
| Serving benchmark (continuous batching, paged KV cache) | no | llama.cpp-style engines only | yes | yes | NVIDIA 8 GB on Linux or WSL2 |

> [!TIP]
> No GPU is not a reason to wait. Do the labs on CPU, run S1 with `--preset cpu`, and move the scale-up runs to a Colab or Kaggle T4 when you reach them. Serving engines list the GPU architectures they support; the T4 is compute capability 7.5, which some engines no longer cover, so check before you plan a serving benchmark on it.

## 5. Training settings per tier

`labs/05_transformer/train.py` picks the device and a model size for you. Explicit flags always win over the preset.

| Flag | Values | Default |
|---|---|---|
| `--device` | `auto` (CUDA, then MPS, then CPU), `cuda`, `mps`, `cpu` | `auto` |
| `--preset` | `cpu`, `gpu-8gb`, `gpu-24gb` | none: the `gpu-8gb` sizes |
| `--amp` | `auto` (bf16 on Ampere or newer, fp16 with loss scaling on older NVIDIA, off on CPU and MPS), `bf16`, `fp16`, `off` | `auto` |
| `--compile` | `torch.compile` on CUDA with Triton, or on CPU under Linux and macOS; ignored elsewhere with a message | off |

| Preset | Model | Context × batch (× accumulation) | Steps |
|---|---|---|---|
| `cpu` | d_model 128, 4 layers, 4 heads (about 0.9M parameters) | 128 × 32 | 1,000 |
| `gpu-8gb` | d_model 384, 6 layers, 6 heads (about 11M parameters) | 256 × 64 | 5,000 |
| `gpu-24gb` | d_model 768, 12 layers, 12 heads (about 85M parameters) | 512 × 16 (× 2) | 10,000 |

```bash
python labs/05_transformer/train.py --data data/TinyStoriesV2-GPT4-train.txt --preset cpu           # CPU laptop
python labs/05_transformer/train.py --data data/TinyStoriesV2-GPT4-train.txt --preset gpu-8gb --compile   # RTX 5060, Linux/WSL2
python labs/05_transformer/train.py --data data/TinyStoriesV2-GPT4-train.txt --preset gpu-8gb --batch-size 32   # override one value
```
If you run out of memory, halve `--batch-size` and double `--grad-accum`: the tokens per step stay the same.

## 6. Check everything
```bash
python -m pytest --impl=solution      # reference solutions pass (what CI runs)
python tools/progress.py              # your scoreboard (starts at 0%)
python tools/measure_gpu.py           # your roofline: write it in journal/
python tools/check_links.py           # docs integrity
```

## 7. Optional: data for scale-up runs
```bash
pip install -r requirements-data.txt
python -c "from huggingface_hub import hf_hub_download as d; d('roneneldan/TinyStories','TinyStoriesV2-GPT4-train.txt',repo_type='dataset',local_dir='data')"
```
`data/`, `runs/` and checkpoints are git-ignored. On a CPU or a slow connection, start with the much smaller `TinyStoriesV2-GPT4-valid.txt` from the same dataset.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `no kernel image is available for execution on the device` | Your torch build doesn't include your GPU's architecture. RTX 50-series: install from the `cu128` index and update the driver to R570 or newer. |
| `torch.cuda.is_available()` is `False` on an NVIDIA machine | You installed a CPU wheel (the PyPI default on Windows). Reinstall with the `cu128` index URL inside the venv; check `nvidia-smi` works. |
| `.venv\Scripts\Activate.ps1 cannot be loaded` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again. |
| Triton tests show as skipped | Expected on macOS and on Windows without `triton-windows`. On Linux CPU, `pip install triton` to run them in the interpreter. |
| `NotImplementedError: ... not currently implemented for the MPS device` | Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for that run. |
| `CUDA out of memory` during `train.py` | Halve `--batch-size`, double `--grad-accum`, or use a smaller preset. |
| pip tries to download several GB on a CPU-only Linux machine | It is fetching the CUDA build. Use the `.../whl/cpu` index URL. |
