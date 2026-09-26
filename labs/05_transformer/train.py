"""Pretrain YOUR lab-05 GPT on a text file -- the scale-up run for an 8 GB GPU.

    # TinyStories (~2 GB text); a 10-20M-parameter byte-level model learns fluent stories in ~1 h.
    python labs/05_transformer/train.py --data data/TinyStoriesV2-GPT4-train.txt \
        --d-model 384 --n-layer 6 --n-head 6 --block-size 256 --batch-size 64 --max-steps 5000

    python labs/05_transformer/train.py --data some.txt --impl solution   # reference model

Hardware: ``--device auto`` (default) picks CUDA, then Apple MPS, then CPU. ``--preset`` sets
model, batch and schedule sizes for a hardware tier; any flag you pass explicitly wins:

    python labs/05_transformer/train.py --data data/TinyStoriesV2-GPT4-valid.txt --preset cpu
    python labs/05_transformer/train.py --data data/TinyStoriesV2-GPT4-train.txt --preset gpu-8gb --compile
    python labs/05_transformer/train.py --data data/TinyStoriesV2-GPT4-train.txt --preset gpu-24gb --batch-size 8

Precision (``--amp auto``): bf16 autocast on NVIDIA Ampere or newer, fp16 with loss scaling on
older NVIDIA GPUs (T4), fp32 on CPU and MPS. The fused AdamW kernel is used only on CUDA, and
``--compile`` is honoured only where torch.compile works (CUDA with Triton; CPU on Linux/macOS).

Tokenization: ``--tokenizer byte`` (vocab 256, no dependencies) or ``gpt2`` (tiktoken).
Token ids are cached next to the text as a .bin memmap. Everything under data/ and runs/
is git-ignored.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `labs`
from labs._impl import load  # noqa: E402


def tokenize_to_memmap(path: Path, tokenizer: str) -> tuple[np.memmap, int]:
    cache = path.with_suffix(f".{tokenizer}.bin")
    dtype = np.uint8 if tokenizer == "byte" else np.uint16
    vocab = 256 if tokenizer == "byte" else 50257
    if not cache.exists():
        print(f"tokenizing {path} -> {cache}")
        if tokenizer == "byte":
            ids = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        else:
            import tiktoken  # pip install tiktoken

            enc = tiktoken.get_encoding("gpt2")
            ids = np.array(enc.encode_ordinary(path.read_text(encoding="utf-8")), dtype=np.uint16)
        mm = np.memmap(cache, dtype=dtype, mode="w+", shape=ids.shape)
        mm[:] = ids
        mm.flush()
    return np.memmap(cache, dtype=dtype, mode="r"), vocab


# Sizes per hardware tier (see SETUP.md). Explicit command-line flags override these values.
# The argparse defaults below equal "gpu-8gb", so running without --preset behaves as it always did.
PRESETS: dict[str, dict[str, float | int]] = {
    # ~0.9M params; a few minutes of CPU time for recognisable words on TinyStories.
    "cpu": dict(d_model=128, n_layer=4, n_head=4, block_size=128, batch_size=32, grad_accum=1,
                max_steps=1000, warmup=100, lr=2e-3, min_lr=2e-4, eval_interval=200, eval_iters=10),
    # ~11M params; the S1 run from the lab handout. Fits an 8 GB card (RTX 5060) with room to spare.
    "gpu-8gb": dict(d_model=384, n_layer=6, n_head=6, block_size=256, batch_size=64, grad_accum=1,
                    max_steps=5000, warmup=200, lr=1e-3, min_lr=1e-4, eval_interval=250, eval_iters=20),
    # ~85M params (GPT-2-small-sized blocks); 16k tokens per optimizer step.
    "gpu-24gb": dict(d_model=768, n_layer=12, n_head=12, block_size=512, batch_size=16, grad_accum=2,
                     max_steps=10000, warmup=500, lr=6e-4, min_lr=6e-5, eval_interval=500, eval_iters=20),
}


def mps_available() -> bool:
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if mps_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda: torch.cuda.is_available() is False (CPU-only torch build? see SETUP.md)")
    if name == "mps" and not mps_available():
        sys.exit("--device mps: torch.backends.mps.is_available() is False (needs an Apple Silicon Mac)")
    return torch.device(name)


def resolve_amp_dtype(choice: str, device: torch.device) -> torch.dtype | None:
    """Autocast dtype, or None for plain fp32. Mixed precision is used only on CUDA."""
    if device.type != "cuda":
        if choice in ("bf16", "fp16"):
            print(f"note: --amp {choice} is CUDA-only here; {device.type} runs in fp32")
        return None
    if choice == "off":
        return None
    if choice == "auto":  # bf16 tensor cores arrived with Ampere (compute capability 8.0)
        return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[choice]


def compile_unsupported_reason(device: torch.device) -> str | None:
    if device.type == "mps":
        return "torch.compile on MPS is experimental"
    if device.type == "cuda":
        try:
            import triton  # noqa: F401
        except ImportError:
            return "torch.compile on CUDA needs Triton (Linux/WSL2, or triton-windows on Windows)"
        return None
    if sys.platform == "win32":
        return "torch.compile on a Windows CPU needs the MSVC C++ toolchain"
    return None


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def get_batch(data: np.memmap, block: int, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ix = np.random.randint(0, len(data) - block - 1, size=batch)
    x = torch.from_numpy(np.stack([data[i : i + block].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([data[i + 1 : i + 1 + block].astype(np.int64) for i in ix]))
    if device.type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def lr_at(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = min(1.0, (step - args.warmup) / max(1, args.max_steps - args.warmup))
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (args.lr - args.min_lr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                   help="auto = CUDA, then Apple MPS, then CPU")
    p.add_argument("--preset", choices=sorted(PRESETS), default=None,
                   help="model/batch/schedule sizes for a hardware tier; explicit flags override it")
    p.add_argument("--amp", choices=["auto", "bf16", "fp16", "off"], default="auto",
                   help="CUDA mixed precision: auto = bf16 on Ampere+, fp16 + loss scaling on older GPUs")
    p.add_argument("--impl", default="exercise", help="exercise (your model) or solution")
    p.add_argument("--tokenizer", choices=["byte", "gpt2"], default="byte")
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-kv-head", type=int, default=0, help="0 => same as n-head")
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--qk-norm", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--peak-tflops", type=float, default=0.0, help="measured bf16 TFLOP/s (tools/measure_gpu.py) to report MFU")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model (CUDA with Triton, or CPU on Linux/macOS; ignored elsewhere)")
    p.add_argument("--out", type=Path, default=Path("runs/gpt"))
    p.add_argument("--seed", type=int, default=0)

    # A preset replaces the defaults; anything given explicitly on the command line still wins.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", choices=sorted(PRESETS), default=None)
    preset = pre.parse_known_args(argv)[0].preset
    if preset:
        p.set_defaults(**PRESETS[preset])
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(args.amp, device)
    amp = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)  # fp16 needs loss scaling
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        name = torch.cuda.get_device_name(device)
    else:
        name = "Apple GPU (MPS)" if device.type == "mps" else "CPU"
    precision = {torch.bfloat16: "bf16 autocast", torch.float16: "fp16 autocast + loss scaling"}.get(amp_dtype, "fp32")
    print(f"device: {device.type} ({name}) | precision: {precision} | preset: {args.preset or 'none (gpu-8gb sizes)'}")
    if device.type == "cpu" and args.preset is None and args.d_model == PRESETS["gpu-8gb"]["d_model"]:
        print("tip: on a CPU, --preset cpu trains a smaller model that finishes in minutes")

    data, vocab = tokenize_to_memmap(args.data, args.tokenizer)
    split = int(0.99 * len(data))
    train_data, val_data = data[:split], data[split:]
    print(f"{len(data):,} tokens ({len(train_data):,} train / {len(val_data):,} val), vocab {vocab}")
    if len(val_data) <= args.block_size + 1:
        sys.exit(f"{args.data} is too small: the 1% validation split has {len(val_data)} tokens, "
                 f"fewer than --block-size {args.block_size} + 1. Use a bigger file or a smaller --block-size.")

    tf = load(str(Path(__file__)), args.impl)
    cfg = tf.GPTConfig(
        vocab_size=vocab, block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head,
        n_kv_head=args.n_kv_head or args.n_head, d_model=args.d_model, qk_norm=args.qk_norm,
    )
    model = tf.GPT(cfg).to(device)
    n_params = model.num_params()
    matmul_params = model.num_params(non_embedding=True) + vocab * cfg.d_model  # blocks + LM head
    flops_per_token = 6 * matmul_params + 6 * cfg.n_layer * cfg.block_size * cfg.d_model
    print(f"model: {n_params / 1e6:.2f}M params, {flops_per_token / 1e6:.1f} MFLOPs/token (training)")

    decay = [p_ for p_ in model.parameters() if p_.dim() >= 2]
    no_decay = [p_ for p_ in model.parameters() if p_.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=device.type == "cuda",
    )
    reason = compile_unsupported_reason(device) if args.compile else None
    if reason:
        print(f"--compile ignored: {reason}; running eagerly")
    step_model = torch.compile(model) if args.compile and not reason else model

    @torch.no_grad()
    def evaluate() -> dict[str, float]:
        model.eval()
        out = {}
        for name, split_data in (("train", train_data), ("val", val_data)):
            losses = []
            for _ in range(args.eval_iters):
                x, y = get_batch(split_data, args.block_size, args.batch_size, device)
                with amp:
                    losses.append(step_model(x, y)[1].item())
            out[name] = float(np.mean(losses))
        model.train()
        return out

    args.out.mkdir(parents=True, exist_ok=True)
    tokens_per_step = args.batch_size * args.block_size * args.grad_accum
    t0 = time.time()
    for step in range(args.max_steps + 1):
        if step % args.eval_interval == 0 or step == args.max_steps:
            losses = evaluate()
            print(f"step {step:6d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step}, args.out / "ckpt.pt")
            if step == args.max_steps:
                break
        for group in opt.param_groups:
            group["lr"] = lr_at(step, args)
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, args.block_size, args.batch_size, device)
            with amp:
                _, loss = step_model(x, y)
            scaler.scale(loss / args.grad_accum).backward()
        scaler.unscale_(opt)  # no-ops unless fp16 loss scaling is on
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            synchronize(device)
            dt = (time.time() - t0) / (10 if step else 1)
            t0 = time.time()
            tps = tokens_per_step / dt
            mfu = f" | MFU {100 * tps * flops_per_token / (args.peak_tflops * 1e12):.1f}%" if args.peak_tflops else ""
            print(f"step {step:6d} | loss {loss.item():.4f} | lr {lr_at(step, args):.2e} | grad {grad_norm:.2f} | {tps:,.0f} tok/s{mfu}")

    # Prompt with in-distribution text: the first tokens of the validation split.
    prompt = torch.from_numpy(val_data[:16].astype(np.int64))[None].to(device)
    model.eval()
    with amp:
        sample = model.generate(prompt, max_new_tokens=300, temperature=0.8, top_k=50)[0].tolist()
    if args.tokenizer == "byte":
        print(bytes(sample).decode("utf-8", errors="replace"))
    else:
        import tiktoken

        print(tiktoken.get_encoding("gpt2").decode(sample))


if __name__ == "__main__":
    main()
