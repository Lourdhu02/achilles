"""Measure your accelerator's roofline: matmul FLOP/s per dtype, memory bandwidth, and
how a decode-like matrix-vector product sits far below the compute roof.

    python tools/measure_gpu.py                  # auto: CUDA, then Apple MPS, then CPU
    python tools/measure_gpu.py --device cpu     # force a backend
    python tools/measure_gpu.py --sizes 2048 4096 8192

Write your numbers into journal/ and compare them with the spec sheet and with the
predictions you make in labs/03_napkin_math. The desktop RTX 5060's spec sheet says
448 GB/s (GDDR7, 128-bit bus); laptop parts are often clocked lower, so measure yours
rather than assume, and do the same for the bf16 tensor-core matmul rate.
"""

from __future__ import annotations

import argparse
import time

import torch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False (CPU-only torch build?). See SETUP.md.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("--device mps requested but MPS is not available (needs Apple Silicon and macOS 12.3+).")
    return torch.device(name)


def bench(fn, device: torch.device, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - start) / iters


def matmul_tflops(n: int, dtype: torch.dtype, device: torch.device) -> float:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    seconds = bench(lambda: a @ b, device)
    return 2 * n**3 / seconds / 1e12


def copy_bandwidth_gbs(n_bytes: int, device: torch.device) -> float:
    x = torch.empty(n_bytes // 4, dtype=torch.float32, device=device)
    y = torch.empty_like(x)
    seconds = bench(lambda: y.copy_(x), device)
    return 2 * x.numel() * 4 / seconds / 1e9  # one read + one write per element


def matvec_effective_gbs(n: int, dtype: torch.dtype, device: torch.device) -> tuple[float, float]:
    """Decode-like y = W x: returns (achieved TFLOP/s, effective GB/s of weight traffic)."""
    w = torch.randn(n, n, device=device, dtype=dtype)
    x = torch.randn(n, 1, device=device, dtype=dtype)
    seconds = bench(lambda: w @ x, device)
    return 2 * n * n / seconds / 1e12, w.numel() * w.element_size() / seconds / 1e9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="*", default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        cap = torch.cuda.get_device_capability(device)
        print(f"device: {props.name}  compute capability {cap[0]}.{cap[1]}  memory {props.total_memory / 2**30:.1f} GiB  SMs {props.multi_processor_count}")
        sizes = args.sizes or [1024, 2048, 4096, 8192]
        bw_bytes = 1 << 30
        dtypes = {"fp32 (no TF32)": torch.float32, "tf32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    elif device.type == "mps":
        print("device: Apple GPU (MPS) -- unified memory, so bandwidth is shared with the CPU")
        sizes = args.sizes or [1024, 2048, 4096]
        bw_bytes = 1 << 29
        dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    else:
        print("device: CPU (no CUDA found) -- numbers are for your CPU")
        sizes = args.sizes or [512, 1024, 2048]
        bw_bytes = 1 << 28
        dtypes = {"fp32": torch.float32, "bf16": torch.bfloat16}

    print(f"\n{'dtype':16}" + "".join(f"{f'n={n}':>12}" for n in sizes) + "   (dense matmul TFLOP/s)")
    best = {}
    for label, dtype in dtypes.items():
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = label == "tf32"
        try:
            row = [matmul_tflops(n, dtype, device) for n in sizes]
        except (RuntimeError, TypeError) as err:  # e.g. bf16 on older macOS
            print(f"{label:16}  skipped: {str(err).splitlines()[0][:70]}")
            continue
        best[label] = max(row)
        print(f"{label:16}" + "".join(f"{v:12.2f}" for v in row))

    bw = copy_bandwidth_gbs(bw_bytes, device)
    print(f"\nmemory bandwidth (device copy): {bw:8.1f} GB/s")

    n = sizes[-1]
    mv_dtype = torch.bfloat16 if "bf16" in best else torch.float32
    tf, gbs = matvec_effective_gbs(n, mv_dtype, device)
    print(f"{str(mv_dtype).removeprefix('torch.')} mat-vec n={n}: {tf:.3f} TFLOP/s, streaming weights at {gbs:.1f} GB/s  <- decode is memory-bound")

    key = "bf16" if "bf16" in best else next(iter(best))
    ridge = best[key] * 1e12 / (bw * 1e9)
    print(f"\nridge point ({key}): {ridge:.0f} FLOP/byte -- kernels below this intensity are memory-bound")
    print("Now predict: batch-1 decode tokens/s for a 0.5B bf16 model and an 8B 4-bit model (lab 03).")
    print(f"\nhardware tier: {hardware_tier(device)}")


def hardware_tier(device: torch.device) -> str:
    """Map this machine to a SETUP.md tier and the matching train.py preset."""
    if device.type == "cuda":
        gib = torch.cuda.get_device_properties(device).total_memory / 2**30
        if gib >= 20:
            return f"NVIDIA 24 GB+ ({gib:.0f} GiB) -> train.py --preset gpu-24gb"
        return f"NVIDIA 8-16 GB ({gib:.0f} GiB) -> train.py --preset gpu-8gb"
    if device.type == "mps":
        return "Apple Silicon -> train.py --preset gpu-8gb (or --preset cpu for a quick check)"
    return "CPU -> train.py --preset cpu (move scale-up runs to Colab or Kaggle when you reach them)"


if __name__ == "__main__":
    main()
