"""Train the model on the tokenized TinyStories stream.

    python train.py --size tiny      # ~12M params, fast sanity run
    python train.py --size small     # ~28M params, the main run
"""
import argparse
import math
import os
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig

SIZES = {
    "tiny":  dict(n_layer=6, n_head=6, n_kv_head=6, dim=384, hidden_dim=1024),   # ~12M
    "small": dict(n_layer=8, n_head=8, n_kv_head=4, dim=512, hidden_dim=1408),   # ~27M (GQA)
}

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=SIZES, default="small")
ap.add_argument("--data_dir", default="data")
ap.add_argument("--out_dir", default="out")
ap.add_argument("--block_size", type=int, default=512)
ap.add_argument("--batch_size", type=int, default=64)      # sequences per micro-step
ap.add_argument("--grad_accum", type=int, default=1)       # raise if you run out of VRAM
ap.add_argument("--max_iters", type=int, default=10_000)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--min_lr_frac", type=float, default=0.1)
ap.add_argument("--warmup", type=int, default=200)
ap.add_argument("--weight_decay", type=float, default=0.1)
ap.add_argument("--clip", type=float, default=1.0)
ap.add_argument("--eval_every", type=int, default=500)
ap.add_argument("--eval_batches", type=int, default=50)
ap.add_argument("--compile", action="store_true", help="torch.compile (needs Triton; skip on Windows)")
args = ap.parse_args()

torch.manual_seed(1337)
device = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
os.makedirs(args.out_dir, exist_ok=True)

tok = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))
cfg = GPTConfig(vocab_size=tok.get_vocab_size(), block_size=args.block_size, **SIZES[args.size])
model = GPT(cfg).to(device)
print(f"model: {model.num_params() / 1e6:.1f}M params")
fwd = torch.compile(model) if args.compile else model

train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(data):
    # random windows; target is the input shifted by one token
    ix = np.random.randint(0, len(data) - args.block_size - 1, (args.batch_size,))
    x = np.stack([data[i:i + args.block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + args.block_size] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device, non_blocking=True), torch.from_numpy(y).to(device, non_blocking=True)


# weight decay only on matrices (not norms / biases); tied embedding counted once
decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.95), fused=True,
)


def lr_at(it):
    if it < args.warmup:
        return args.lr * (it + 1) / args.warmup
    progress = (it - args.warmup) / max(1, args.max_iters - args.warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * cosine)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(args.eval_batches):
            x, y = get_batch(data)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = fwd(x, y)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def sample(prompt="Once upon a time", n=120):
    model.eval()
    ids = torch.tensor([tok.encode(prompt).ids], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(ids, n, eos_id=tok.token_to_id("<|endoftext|>"))
    model.train()
    return tok.decode(out[0].tolist())


def save(path):
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "args": vars(args)}, path)


tokens_per_iter = args.batch_size * args.grad_accum * args.block_size
print(f"{tokens_per_iter:,} tokens/iter, {tokens_per_iter * args.max_iters / 1e6:.0f}M tokens total")
best_val, t0 = float("inf"), time.time()
model.train()
for it in range(args.max_iters + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(it)

    if it % args.eval_every == 0:
        losses = estimate_loss()
        print(f"\n[eval] iter {it}: train {losses['train']:.4f} | val {losses['val']:.4f} "
              f"| ppl {math.exp(losses['val']):.2f}")
        print("[sample]", sample().replace("\n", " "), "\n")
        if losses["val"] < best_val:
            best_val = losses["val"]
            save(os.path.join(args.out_dir, f"{args.size}_best.pt"))
    if it == args.max_iters:
        break

    for _ in range(args.grad_accum):
        x, y = get_batch(train_data)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = fwd(x, y)
        (loss / args.grad_accum).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    opt.step()
    opt.zero_grad(set_to_none=True)

    if it % 20 == 0:
        dt = time.time() - t0
        t0 = time.time()
        print(f"iter {it:6d} | loss {loss.item():.4f} | lr {lr_at(it):.2e} | gnorm {norm:.2f} "
              f"| {dt * 1000 / (20 if it else 1):.0f} ms/iter", flush=True)

save(os.path.join(args.out_dir, f"{args.size}_final.pt"))
print("done")
