# sosLM

A small decoder-only language model (~26M parameters) written from scratch in PyTorch and trained on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) on a single RTX 4070 Ti Super.

The goal is to understand the transformer end to end: tokenizer, architecture, training loop, and sampling,
with no `transformers` library and no pretrained weights.

## Sample output

Generated with temperature 0.8 and top-k 50.

**Prompt:** `Once upon a time`

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she went to the park with her mommy. They played on the swings and went down the slide.
>
> After a while, Lily got thirsty. She went to the water fountain and drank some water. The water was cold and made her feel better. She saw a little boy playing with a ball and wanted to play too.
>
> Lily went over to the boy and asked if she could play too. The boy said yes and they played together. They had so much fun that they became friends. From that day on, Lily always brought her mommy's water bottle when they went to the park.

**Prompt:** `Once upon a time, a cat`

> Once upon a time, a cat named Tom was walking in the park. He saw a big tree and wanted to climb it. Tom started to climb the tree, but he was scared. He looked around and saw a bird. The bird said, "Don't be scared, Tom. I will help you climb the tree."
>
> Tom was scared, but he wanted to be brave. He said, "Okay, bird. I will climb the tree." The bird helped Tom climb the tree. They reached the top and saw a beautiful view. Tom was happy and not scared anymore.
>
> Tom and the bird became friends. They played together in the park every day. Tom learned that being brave and helping others can make you feel good. And they lived happily ever after.

The model keeps characters and plot consistent across paragraphs, handles dialogue and quotation marks, and gives each story
a small arc and a closing moral, which is the TinyStories house style.

## Architecture

A Llama-style decoder, implemented in [model.py](model.py):

| Component | Choice | Why |
|---|---|---|
| Normalisation | RMSNorm, pre-norm | Cheaper than LayerNorm and trains stably |
| Positions | Rotary embeddings (RoPE) | Encodes relative position inside attention, with no learned position table |
| MLP | SwiGLU | Gated MLP, a consistent quality gain over GELU at equal parameters |
| Attention | Causal, grouped-query (8 query heads, 4 KV heads) | Fewer KV parameters, the same idea used in modern large models |
| Attention kernel | `F.scaled_dot_product_attention` | Fused FlashAttention-style kernel |
| Embeddings | Input and output tied | Saves parameters, which matters with a small vocab and a small model |
| Biases | None | Standard in modern LLMs |

| | `tiny` | `small` (main) |
|---|---|---|
| Parameters | 12.2M | 25.7M |
| Layers / heads / KV heads | 6 / 6 / 6 | 8 / 8 / 4 |
| Model dim / MLP hidden | 384 / 1024 | 512 / 1408 |
| Context length | 512 | 512 |
| Vocab | 4096 (byte-level BPE) | 4096 (byte-level BPE) |

## Data

- **Corpus:** TinyStories, about 2.1M short synthetic stories (about 486M tokens after tokenizing, plus a 4.9M-token validation split).
- **Tokenizer:** byte-level BPE with a 4096-token vocabulary, trained on the first 100k stories
  ([prepare_data.py](prepare_data.py)). A small vocab keeps the embedding table from dominating a model this size.
- Stories are concatenated into one token stream separated by `<|endoftext|>`, stored as `uint16`, and read via
  `np.memmap`. Training samples random 512-token windows from it.

## Training

[train.py](train.py) settings for the main run:

- AdamW (β = 0.9 / 0.95), peak LR 1e-3, 200 warmup steps, cosine decay to 10%
- Weight decay 0.1 on weight matrices only (not norms)
- bf16 autocast, gradient clipping at 1.0
- Batch of 64 sequences × 512 tokens = 32,768 tokens/step, 10,000 steps (about 330M tokens, roughly 0.7 epoch)

## Results

Final evaluation at step 10,000 (`small`, 25.7M parameters):

| Metric | Value |
|---|---|
| Train loss | 1.3207 |
| Validation loss | 1.3570 |
| Validation perplexity | 3.88 |

- **Reference point:** a model guessing uniformly over the 4096-token vocabulary has loss ln(4096) ≈ 8.32 (perplexity 4096).
  Perplexity 3.88 means that at each step the model is, on average, about as uncertain as a choice between 4 tokens.
- **No overfitting:** the train/validation gap is only 0.04. The run covers about 0.7 epoch, so the model rarely sees a
  training window more than once.
- **Caveat:** perplexity is per token and depends on the tokenizer, so it is not comparable to numbers from models
  with a different vocabulary.

## Sampling

Text is generated one token at a time ([generate.py](generate.py)). Two parameters control the output:

- **Temperature** divides the model's logits before softmax. Low values (about 0.3) make output safe and repetitive,
  high values (about 1.2+) make it more varied and less coherent. The default is 0.8.
- **Top-k** keeps only the k most likely tokens at each step and discards the long tail of unlikely ones. The default is 50.

```bash
python generate.py out/small_final.pt "Once upon a time, a cat" --temperature 0.3
python generate.py out/small_final.pt "Once upon a time, a cat" --temperature 1.3 --top_k 50
```

## Usage

Requires Python 3.11, PyTorch with CUDA, and a GPU with ~16 GB (less works with a smaller `--batch_size`).

```bash
pip install -r requirements.txt

python prepare_data.py                    # download, train tokenizer, tokenize -> data/
python train.py --size small              # train -> out/small_best.pt and out/small_final.pt
python generate.py out/small_best.pt "Once upon a time"
```

Useful flags: `--size tiny` for a fast sanity run, `--batch_size 32 --grad_accum 2` if you run out of VRAM,
`--max_iters N` to shorten or lengthen the run.

## Project layout

```
model.py         transformer implementation
prepare_data.py  dataset download, tokenizer training, tokenization
train.py         training loop, LR schedule, eval, checkpointing
generate.py      sampling from a checkpoint
```

## Limitations

- TinyStories is a narrow, synthetic domain. The model writes simple children's stories and is not a general-purpose LM.
- The prose is formulaic (short sentences, repeated names, a moral at the end), which mirrors the training data.
- Sampling has no KV cache, so generation recomputes the full context each token. Fine at this size, slow for long outputs.
- Only perplexity on the held-out split is tracked. There are no downstream benchmarks, and training loss curves are not logged to a file.

## Ideas / next steps

- [ ] KV-cache for generation
- [ ] Ablations: RoPE vs. learned positions, SwiGLU vs. GELU, GQA vs. full MHA (loss curves for each)
- [ ] Log loss curves and plot them
- [ ] Scale to ~125M parameters on FineWeb-Edu
- [ ] Instruction fine-tuning (SFT)
- [x] Follow-up project: [chess language model](chessLM/) (separate repo)

## References

- Karpathy, *Let's build GPT* and *nanoGPT*
- Eldan & Li, *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* (2023)
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021)
- Shazeer, *GLU Variants Improve Transformer* (2020)
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* (2023)
