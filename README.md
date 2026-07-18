# fable-glm-vision — teach GLM-5.2 to see art

Minimal scripts to give [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) vision
by training **only a 44.8M-parameter projector** (a 2-layer MLP) between a frozen
vision encoder and the frozen language model — then teach it about **art**:
who painted an image, what style and genre it is, what a product's color and
type are.

## Credit — original method

This entire approach comes from **Harry Partridge**
([@part_harry_](https://x.com/part_harry_)),
["GLM 5.2 With Vision"](https://x.com/part_harry_/status/2077610277571637435),
which is the original source and required reading:

> "Amazingly, by training only a tiny 2 layer MLP with just 50M parameters, we
> are able to give GLM 5.2 vision capabilities equivalent to Claude 4.5 Haiku
> on MMMU-Pro (55%). Along the way, we witnessed an incredible level of
> generalization: GLM is even able to identify people who were never mentioned
> in our dataset!"
>
> "If you are interested in using GLM 5.2 with vision, let me know in the
> comments or send me a message!"
>
> — Harry Partridge, [original post](https://x.com/part_harry_/status/2077610277571637435)

This repo is an independent reproduction of that recipe, specialized to an art
curriculum. Weights: [0xSero/fable-glm-vision](https://huggingface.co/0xSero/fable-glm-vision).
Full data mix: [0xSero/glm-vision-sft-mix](https://huggingface.co/datasets/0xSero/glm-vision-sft-mix).

## How it works

```
image ──> MoonViT (Kimi K2.6 vision tower, frozen, ~450M)
      ──> VisionProjector (44.8M MLP — the ONLY thing trained)
      ──> GLM-5.2 (frozen; NF4-quantized to fit 4x80GB)
```

The projector output replaces `<|image|>` placeholder tokens inside a
`<|begin_of_image|> … <|end_of_image|>` span in GLM's input embeddings.
Loss is cross-entropy on the assistant answer tokens only.

**The grokking event**: with batch 64 / lr 5e-4 and *short* answers, loss
plateaus for hundreds of steps, then drops sharply (~step 900 in the original)
when the projector snaps into alignment with GLM's latent space. Long
descriptive answers prevent this — keep answers to a few tokens.

## Files

| file | purpose |
|---|---|
| `build_art_dataset.py` | download WikiArt + fashion-product images, emit short-QA train/val jsonl |
| `training/projector.py` | the 44.8M MLP (LayerNorm → Linear 1152→6144 → GELU → Linear 6144→6144) |
| `training/moonvit_loader.py` | load Kimi K2.6's MoonViT vision tower and wrap it as `(B,3,H,W) → (B,N,1152)` |
| `training/data.py` | jsonl dataset, GLM chat/image token layout, collator (resize → encode → project) |
| `training/train_sft.py` | the trainer: frozen GLM + frozen MoonViT, AdamW on the projector only |

## Quickstart

```bash
pip install -r requirements.txt

# 1. Sanity-check the plumbing on CPU (no GPU needed)
python -m training.projector          # 44.8M param + grad-flow check
python -m training.train_sft --smoke-test

# 2. Build the art dataset (~37k images, ~5GB download)
python build_art_dataset.py --out-dir data/sft_art --img-dir imgs/art

# 3. Train (4x80GB GPUs for the NF4-quantized 32-expert prune of GLM-5.2)
python -m training.train_sft \
    --model models/GLM-5.2-nf4 \
    --encoder encoders/moonvit \
    --data data/sft_art/train.jsonl --val data/sft_art/val.jsonl \
    --out logs/sft_art \
    --lr 5e-4 --batch 64 --microbatch 4 --epochs 3 --save-every 50
```

You need to provide:
- **GLM-5.2** weights (BF16 from the hub, or pre-quantized NF4; the trainer
  quantizes on the fly if given BF16). To fit 4x80GB we pruned 256→32 experts
  and pre-quantized to NF4 (~181GB).
- **MoonViT** weights — the Kimi K2.6 vision tower (e.g. from
  exolabs/Kimi-K2.6-vision), loaded and key-remapped by `training/moonvit_loader.py`.

Expected training signature: loss falls fast for ~50 steps, then descends a
long plateau (~0.001/step). Watch for the sharp drop — that's the grok. Warm
starting from an already-aligned projector checkpoint (`--init-projector`)
transfers alignment across datasets and skips most of the plateau.

## Data sources

- [Artificio/WikiArt_Full](https://huggingface.co/datasets/Artificio/WikiArt_Full) —
  103k paintings, 256px, artist/style/genre/date. WikiArt content is
  research/non-commercial per wikiart.org terms.
- [benitomartin/fashion-product-images-small-384x512](https://huggingface.co/datasets/benitomartin/fashion-product-images-small-384x512) —
  44k product photos with color/type/season/usage labels.

## License

Apache-2.0 (code). Model and dataset licenses are governed by their sources
(GLM-5.2, Kimi K2.6 vision tower, WikiArt, fashion-product-images).
