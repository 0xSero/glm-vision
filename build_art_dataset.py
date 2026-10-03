#!/usr/bin/env python3
"""Build an art/design SFT dataset for teaching GLM-5.2 (vision) about art.

Sources (both public on HuggingFace):
  - Artificio/WikiArt_Full : paintings with artist/style/genre/date labels
  - benitomartin/fashion-product-images-small-384x512 : product photos with
    color/type/season/usage labels

Every example is one image plus ONE short question -> ONE short answer
("Who painted this?" -> "Claude Monet"). Short on-policy answers are required
for the SFT grokking event — long descriptions prevent it (see Harry
Partridge's original write-up).

Output: <out-dir>/train.jsonl + val.jsonl in the format expected by
training/train_sft.py, images resized (<= max-pixels, dims rounded to 28px
multiples for MoonViT's 14px patches + 2x2 merge) into <img-dir>.

Usage:
    python build_art_dataset.py --out-dir data/sft_art --img-dir imgs/art \
        --n-wikiart 25000 --n-fashion 12000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PATCH = 14
MERGE = 2
ROUND = PATCH * MERGE  # 28px

# The trainer builds the real image span programmatically from image size;
# this literal placeholder in the text matches the training-time convention.
IMAGE_SPAN = "|begin_of_image|" + "|image|" * 128 + "|end_of_image|"


def preprocess_size(w: int, h: int, max_pixels: int) -> tuple[int, int]:
    """Aspect-preserving resize to <= max_pixels, dims rounded to 28px."""
    scale = (max_pixels / (w * h)) ** 0.5
    if scale < 1.0:
        w, h = int(w * scale), int(h * scale)
    w = max(ROUND, (w // ROUND) * ROUND)
    h = max(ROUND, (h // ROUND) * ROUND)
    # Raising a short side to 28px can push a wide image back over the cap.
    while w * h > max_pixels and (w > ROUND or h > ROUND):
        if w >= h and w > ROUND:
            w -= ROUND
        elif h > ROUND:
            h -= ROUND
        else:
            break
    return w, h


def save_image(img: Image.Image, img_dir: Path, prefix: str, idx: int,
               max_pixels: int) -> str:
    from PIL import Image
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = preprocess_size(*img.size, max_pixels)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.LANCZOS)
    path = img_dir / f"{prefix}_{idx:06d}.jpg"
    img.save(path, "JPEG", quality=90)
    return str(path)


def make_example(img_path: str, question: str, answer: str) -> dict:
    return {
        "image": img_path,
        "conversations": [
            {"role": "user", "content": f"{IMAGE_SPAN}\n{question}"},
            {"role": "assistant", "content": answer},
        ],
    }


def build_wikiart(args) -> list[dict]:
    from datasets import load_dataset

    print("Loading Artificio/WikiArt_Full...", flush=True)
    ds = load_dataset("Artificio/WikiArt_Full", split="train")
    drop = [c for c in ds.column_names
            if c in ("embeddings_pca512", "resnet50_non_robust_feats",
                     "resnet50_robust_feats", "description")]
    ds = ds.remove_columns(drop)
    idxs = list(range(len(ds)))
    random.shuffle(idxs)

    rows, used = [], 0
    for i in idxs:
        if used >= args.n_wikiart:
            break
        r = ds[i]
        artist = (r.get("artist") or "").strip()
        style = (r.get("style") or "").strip()
        genre = (r.get("genre") or "").strip()
        date = (r.get("date") or "").strip()
        img = r.get("image")
        if img is None:
            continue
        qa = []
        if artist and artist.lower() not in ("unknown", "unknown artist"):
            qa.append((random.choice([
                "Who painted this?",
                "Who is the artist of this painting?",
                "Name the artist who created this work.",
            ]), artist))
        if style:
            qa.append((random.choice([
                "What art style is this painting?",
                "What artistic movement does this painting belong to?",
            ]), style))
        if genre:
            qa.append((random.choice([
                "What genre is this painting?",
                "What type of painting is this?",
            ]), genre))
        if date and date.isdigit() and len(date) == 4:
            qa.append(("When was this painted?", date))
        if not qa:
            continue
        random.shuffle(qa)
        try:
            path = save_image(img, args.img_dir, "wikiart", i, args.max_pixels)
        except Exception:
            continue
        for q, a in qa[:2]:  # up to 2 QA pairs per image
            rows.append(make_example(path, q, a))
        used += 1
        if used % 5000 == 0:
            print(f"  {used:,} wikiart images", flush=True)
    print(f"  wikiart: {len(rows):,} examples from {used:,} images", flush=True)
    return rows


def build_fashion(args) -> list[dict]:
    from datasets import load_dataset

    print("Loading fashion-product-images-small-384x512...", flush=True)
    ds = load_dataset("benitomartin/fashion-product-images-small-384x512",
                      split="train")
    idxs = list(range(len(ds)))
    random.shuffle(idxs)

    rows, used = [], 0
    for i in idxs:
        if used >= args.n_fashion:
            break
        r = ds[i]
        img = r.get("image")
        if img is None:
            continue
        fields = {k: (r.get(k) or "").strip() if isinstance(r.get(k), str) else ""
                  for k in ("baseColour", "articleType", "season", "usage",
                            "masterCategory")}
        qa = []
        if fields["baseColour"]:
            qa.append((random.choice([
                "What is the main color of this item?",
                "What color is this product?",
            ]), fields["baseColour"]))
        if fields["articleType"]:
            qa.append((random.choice([
                "What type of item is this?",
                "What product is shown in this image?",
            ]), fields["articleType"]))
        if fields["season"]:
            qa.append(("What season is this item designed for?", fields["season"]))
        if fields["usage"]:
            qa.append(("What occasion is this item for?", fields["usage"]))
        if fields["masterCategory"]:
            qa.append(("What product category does this belong to?",
                       fields["masterCategory"]))
        if not qa:
            continue
        random.shuffle(qa)
        try:
            path = save_image(img, args.img_dir, "fashion", i, args.max_pixels)
        except Exception:
            continue
        for q, a in qa[:2]:
            rows.append(make_example(path, q, a))
        used += 1
        if used % 5000 == 0:
            print(f"  {used:,} fashion images", flush=True)
    print(f"  fashion: {len(rows):,} examples from {used:,} images", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("data/sft_art"))
    ap.add_argument("--img-dir", type=Path, default=Path("imgs/art"))
    ap.add_argument("--n-wikiart", type=int, default=25_000)
    ap.add_argument("--n-fashion", type=int, default=12_000)
    ap.add_argument("--max-pixels", type=int, default=300_000)
    ap.add_argument("--train-frac", type=float, default=0.97)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    args.img_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_wikiart(args) + build_fashion(args)
    random.shuffle(rows)
    n_train = int(len(rows) * args.train_frac)
    with open(args.out_dir / "train.jsonl", "w") as f:
        for r in rows[:n_train]:
            f.write(json.dumps(r) + "\n")
    with open(args.out_dir / "val.jsonl", "w") as f:
        for r in rows[n_train:]:
            f.write(json.dumps(r) + "\n")
    print(f"DONE: {n_train:,} train / {len(rows) - n_train:,} val -> {args.out_dir}")


if __name__ == "__main__":
    main()
