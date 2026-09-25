"""Pretrain YOUR lab-05 GPT on a text file -- the scale-up run for an 8 GB GPU.

    # TinyStories (~2 GB text); a 10-20M-parameter byte-level model learns fluent stories in ~1 h.
    python labs/05_transformer/train.py --data data/TinyStoriesV2-GPT4-train.txt \
        --d-model 384 --n-layer 6 --n-head 6 --block-size 256 --batch-size 64 --max-steps 5000

    python labs/05_transformer/train.py --data some.txt --impl solution   # reference model

Tokenization: ``--tokenizer byte`` (vocab 256, no dependencies) or ``gpt2`` (tiktoken).
Token ids are cached next to the text as a .bin memmap. Everything under data/ and runs/
is git-ignored.
"""

from __future__ import annotations

import argparse
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
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
    p.add_argument("--compile", action="store_true", help="torch.compile the model (Linux/WSL2 + Triton)")
    p.add_argument("--out", type=Path, default=Path("runs/gpt"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    data, vocab = tokenize_to_memmap(args.data, args.tokenizer)
    split = int(0.99 * len(data))
    train_data, val_data = data[:split], data[split:]
    print(f"{len(data):,} tokens ({len(train_data):,} train / {len(val_data):,} val), vocab {vocab}")

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
    step_model = torch.compile(model) if args.compile else model

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
            (loss / args.grad_accum).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
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
