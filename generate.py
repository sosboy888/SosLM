"""Sample from a trained checkpoint:  python generate.py out/small_best.pt "Once upon a time" """
import argparse

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("prompt", nargs="?", default="Once upon a time")
ap.add_argument("--tokenizer", default="data/tokenizer.json")
ap.add_argument("--max_new_tokens", type=int, default=8000)
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--top_k", type=int, default=30)
args = ap.parse_args()

ck = torch.load(args.ckpt, map_location="cuda")
model = GPT(GPTConfig(**ck["cfg"])).cuda().eval()
model.load_state_dict(ck["model"])
tok = Tokenizer.from_file(args.tokenizer)

ids = torch.tensor([tok.encode(args.prompt).ids], device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model.generate(ids, args.max_new_tokens, args.temperature, args.top_k,
                         eos_id=tok.token_to_id("<|endoftext|>"))
print(tok.decode(out[0].tolist()))
