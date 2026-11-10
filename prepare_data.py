"""Download TinyStories, train a BPE tokenizer, and tokenize to flat uint16 .bin files.

Usage:  python prepare_data.py            (full dataset, ~2 GB download)
        python prepare_data.py --limit 50000   (quick test on 50k stories)
"""
import argparse
import os

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOS = "<|endoftext|>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N train stories")
    ap.add_argument("--tok_samples", type=int, default=100_000, help="stories used to train the tokenizer")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset("roneneldan/TinyStories")
    train, val = ds["train"], ds["validation"]
    if args.limit:
        train = train.select(range(min(args.limit, len(train))))

    # --- 1. tokenizer: byte-level BPE (can encode any text, no <unk> needed) ---
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    n = min(args.tok_samples, len(train))
    tok.train_from_iterator((train[i]["text"] for i in range(n)), trainer)
    tok.save(os.path.join(args.out_dir, "tokenizer.json"))
    eos_id = tok.token_to_id(EOS)
    print(f"tokenizer trained: vocab={tok.get_vocab_size()} eos_id={eos_id}")

    # --- 2. tokenize each split into one long stream of ids, stories separated by EOS ---
    def encode_split(split, name):
        chunks, total = [], 0
        B = 10_000
        for i in range(0, len(split), B):
            texts = split[i:i + B]["text"]
            for enc in tok.encode_batch(texts):
                chunks.append(np.array(enc.ids + [eos_id], dtype=np.uint16))
            total = sum(len(c) for c in chunks)
            print(f"\r{name}: {i + len(texts)}/{len(split)} stories, {total / 1e6:.1f}M tokens", end="")
        print()
        arr = np.concatenate(chunks)
        arr.tofile(os.path.join(args.out_dir, f"{name}.bin"))

    encode_split(train, "train")
    encode_split(val, "val")


if __name__ == "__main__":
    main()
