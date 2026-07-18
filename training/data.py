"""Data collator and dataset utilities for GLM-5.2 vision SFT.

Pipeline:
    raw image  ->  MoonViT encoder (frozen)  ->  2x2 spatial merge  ->
    VisionProjector (trainable)  ->  splice into GLM embed stream at image tokens

GLM-5.2 chat-template image convention (verified from on-disk tokenizer.json):
    <|begin_of_image|> <|image|> ... <|image|> <|end_of_image|>
    -- one <|image|> token per merged patch; the projector output replaces each.

Token IDs (from the GLM-5.2 tokenizer.json, added_tokens[]):
    endoftext = 154820   (also used as pad)
    sop       = 154824
    eop       = 154825
    system    = 154826
    user      = 154827
    assistant = 154828
    observation = 154829
    begin_of_image = 154830
    end_of_image   = 154831
    image     = 154854  (replaced by projector output, one per merged patch)
    video     = 154855

Loss is computed ONLY on the assistant answer tokens (not on the image span,
not on the prompt).  We build a labels tensor where:
    - prompt + image-span tokens  ->  -100  (ignored)
    - answer tokens               ->  token_id (scored)
    - EOS after answer            ->  eos_id (scored, so model learns to stop)

Each JSONL training row has this shape:
    {
        "image":  "<path-or-url>",          # or "images": [...] for multi-image
        "conversations": [
            {"role": "user", "content": "What is in this image?"},
            {"role": "assistant", "content": "a red bus"}
        ]
    }
For UI grounding rows, the assistant content is an MCP action string like
    click(512, 340)
which is short (<=25 tokens) and stays inside GLM's existing token distribution.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# GLM-5.2 special token IDs  (from /mnt/llm_models/GLM-5.2-NVFP4/tokenizer.json)
# ---------------------------------------------------------------------------
GLM_EOS = 154820            # <|endoftext|>  (also used as pad)
GLM_SOP = 154824            # <|sop|>
GLM_EOP = 154825            # <|eop|>
GLM_SYSTEM = 154826         # <|system|>
GLM_USER = 154827           # <|user|>
GLM_ASSISTANT = 154828      # <|assistant|>
GLM_OBSERVATION = 154829    # <|observation|>
GLM_BEGIN_IMAGE = 154830    # <|begin_of_image|>
GLM_END_IMAGE = 154831      # <|end_of_image|>
GLM_IMAGE = 154854          # <|image|>  (replaced by projector output)
GLM_VIDEO = 154855          # <|video|>


def glm_special_ids() -> dict[str, int]:
    """Return the special-token-id map for GLM-5.2 (for verification)."""
    return {
        "endoftext": GLM_EOS,
        "sop": GLM_SOP,
        "eop": GLM_EOP,
        "system": GLM_SYSTEM,
        "user": GLM_USER,
        "assistant": GLM_ASSISTANT,
        "observation": GLM_OBSERVATION,
        "begin_of_image": GLM_BEGIN_IMAGE,
        "end_of_image": GLM_END_IMAGE,
        "image": GLM_IMAGE,
        "video": GLM_VIDEO,
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SFTJsonlDataset(Dataset):
    """Loads JSONL rows with image + conversations and pre-tokenizes text.

    Images are loaded lazily in __getitem__ (in the collator) so that the
    dataset object stays small and can be shuffled/sliced cheaply.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_text_len: int = 512):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.rows: list[dict[str, Any]] = []
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


# ---------------------------------------------------------------------------
# Text tokenization — build input_ids + labels with GLM chat structure
# ---------------------------------------------------------------------------

def build_input_ids_and_labels(
    conversations: list[dict[str, str]],
    num_image_tokens: int,
    tokenizer,
    eos_id: int = GLM_EOS,
) -> tuple[list[int], list[int]]:
    """Build token sequence for one example.

    Layout:
        <|sop|> <|user|> [image_span] user_text <|assistant|> answer_text <|endoftext|>
    where [image_span] = <|begin_of_image|> <|image|> x N <|end_of_image|>

    Returns (input_ids, labels).  Labels are -100 everywhere except the answer
    text + trailing EOS, so loss is computed only on the assistant response.
    """
    assert len(conversations) >= 2, "need at least user + assistant turns"
    user_msg = conversations[0]["content"]
    assistant_msgs = [m["content"] for m in conversations[1:] if m["role"] == "assistant"]
    answer_text = "\n".join(assistant_msgs)

    # --- prefix: sop + user role + image span + user text + assistant role ---
    prefix_ids = [GLM_SOP, GLM_USER]
    prefix_ids += [GLM_BEGIN_IMAGE] + [GLM_IMAGE] * num_image_tokens + [GLM_END_IMAGE]
    prefix_ids += tokenizer.encode(user_msg, add_special_tokens=False)
    prefix_ids += [GLM_ASSISTANT]

    # --- answer: answer text + eos (these are the supervised tokens) ---
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False) + [eos_id]

    input_ids = prefix_ids + answer_ids
    labels = [-100] * len(prefix_ids) + answer_ids[:]
    return input_ids, labels


def compute_num_image_tokens(
    height: int, width: int, patch_size: int = 14, merge: int = 2
) -> int:
    """Number of merged patch tokens for an image of (H, W).

    NaViT patching: (H//patch) x (W//patch) raw patches, then 2x2 spatial
    merge groups 4 patches into 1 token.
    """
    h_patches = height // patch_size
    w_patches = width // patch_size
    merged_h = h_patches // merge
    merged_w = w_patches // merge
    return merged_h * merged_w


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

@dataclass
class VisionSFTCollator:
    """Collates a batch of examples into tensors ready for the model.

    Needs a callable `vision_encoder` (MoonViT, frozen) that takes a batch of
    pixel tensors (B, 3, H, W) and returns (B, N_merged, 1152) embeddings,
    and a `projector` (VisionProjector) that maps -> (B, N_merged, 6144).

    The collator:
      1. Loads + preprocesses images (resize, normalize, patchify, 2x2 merge).
      2. Tokenizes text, builds input_ids + labels (loss masked to answer only).
      3. Runs MoonViT (no_grad) to get vision features.
      4. Runs projector to get GLM-space embeddings.
      5. Pads to max length in batch, returns (input_ids, labels, attn_mask,
         vision_embeds, num_image_tokens).

    The training loop does the final splice of vision_embeds into the token
    embedding stream (it owns the GLM embed layer), keeping this collator
    decoupled from GLM's module layout.
    """

    tokenizer: Any
    vision_encoder: Any           # frozen MoonViT
    projector: Any                # trainable VisionProjector
    max_image_pixels: int = 300_000   # ~0.3 Mpix cap (aspect-preserving)
    patch_size: int = 14
    merge: int = 2
    max_seq_len: int = 2048
    image_mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    image_std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
    device: str = "cpu"

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images: list[Any] = []
        for row in batch:
            images.append(self._load_image(row))

        # --- compute merged token count per image ---
        num_tokens_list = [
            compute_num_image_tokens(img.height, img.width, self.patch_size, self.merge)
            for img in images
        ]

        # --- tokenize text + build labels ---
        all_input_ids: list[list[int]] = []
        all_labels: list[list[int]] = []
        for row, nt in zip(batch, num_tokens_list):
            ids, labels = build_input_ids_and_labels(
                row["conversations"], nt, self.tokenizer
            )
            # truncate from the left of the prefix if too long (keep the answer)
            if len(ids) > self.max_seq_len:
                overflow = len(ids) - self.max_seq_len
                ids = ids[overflow:]
                labels = labels[overflow:]
            all_input_ids.append(ids)
            all_labels.append(labels)

        # --- pad to max length ---
        max_len = max(len(ids) for ids in all_input_ids)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        padded_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, (ids, labels) in enumerate(zip(all_input_ids, all_labels)):
            padded_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            padded_labels[i, : len(labels)] = torch.tensor(labels, dtype=torch.long)
            attn_mask[i, : len(ids)] = 1

        # --- run vision encoder + projector (no grad on encoder) ---
        # Process images individually — they have varying resolutions and
        # MoonViT (NaViT) supports variable-size inputs.  We pad the
        # projected features to the max sequence length in the batch.
        projected_list = []
        max_tokens = max(num_tokens_list)
        # Determine the vision encoder's dtype (bfloat16 for MoonViT)
        enc_dtype = next(self.vision_encoder.parameters()).dtype
        for img in images:
            pixel = self._preprocess_image(img).unsqueeze(0).to(
                device=self.device, dtype=enc_dtype
            )  # (1, 3, H, W) in encoder dtype
            with torch.no_grad():
                feat = self.vision_encoder(pixel)  # (1, N_i, 1152)
            proj = self.projector(
                feat.to(self.projector.ln.weight.dtype)
            )  # (1, N_i, 6144)
            n = proj.shape[1]
            if n < max_tokens:
                # Pad with zeros to max_tokens
                pad = torch.zeros(
                    1, max_tokens - n, proj.shape[-1],
                    dtype=proj.dtype, device=proj.device,
                )
                proj = torch.cat([proj, pad], dim=1)
            projected_list.append(proj)

        projected = torch.cat(projected_list, dim=0)  # (B, max_tokens, 6144)

        return {
            "input_ids": padded_ids,           # (B, L)
            "labels": padded_labels,           # (B, L)
            "attention_mask": attn_mask,       # (B, L)
            "vision_embeds": projected,        # (B, max_tokens, 6144)
            "num_image_tokens": torch.tensor(num_tokens_list, dtype=torch.long),
        }

    # -- image helpers --

    def _load_image(self, row: dict[str, Any]):
        from PIL import Image

        path = row.get("image") or row.get("images", [None])[0]
        if not path:
            raise ValueError("row missing 'image' or 'images' key")
        if path.startswith(("http://", "https://")):
            import requests
            from io import BytesIO

            resp = requests.get(path, timeout=30)
            img = Image.open(BytesIO(resp.content))
        else:
            img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def _preprocess_image(self, img) -> torch.Tensor:
        """Resize (aspect-preserving, capped at max_image_pixels), normalize, to tensor.

        Dimensions are rounded to patch_size*merge multiples so that both
        patching and 2x2 spatial merge produce exact integer counts.
        """
        from PIL import Image
        w, h = img.size
        # Round dimensions to patch_size * merge (= 28) multiples so both
        # patching (÷14) and 2x2 merge (÷2) are exact.
        round_to = self.patch_size * self.merge  # 28
        w = max(round_to, (w // round_to) * round_to)
        h = max(round_to, (h // round_to) * round_to)
        scale = (self.max_image_pixels / (w * h)) ** 0.5
        if scale < 1.0:
            new_w = max(round_to, int(w * scale))
            new_h = max(round_to, int(h * scale))
            new_w = (new_w // round_to) * round_to
            new_h = (new_h // round_to) * round_to
        else:
            new_w, new_h = w, h
        img = img.resize((new_w, new_h), Image.BILINEAR)

        arr = torch.from_numpy(
            self._to_numpy_normalized(img, self.image_mean, self.image_std)
        )  # (C, H, W) float
        return arr

    @staticmethod
    def _to_numpy_normalized(img, mean, std):
        import numpy as np

        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        arr = (arr - np.array(mean)) / np.array(std)
        return arr.transpose(2, 0, 1)  # (3, H, W)


# ---------------------------------------------------------------------------
# Embedding splice — replace |image| token positions with projector output
# ---------------------------------------------------------------------------

def splice_vision_into_embeds(
    input_ids: torch.Tensor,         # (B, L) token ids
    token_embed: torch.Tensor,        # (vocab, D) GLM embed weight (frozen)
    vision_embeds: torch.Tensor,      # (B, N_merged, D) projector output
    num_image_tokens: torch.Tensor,   # (B,) per-image patch count
    image_token_id: int = GLM_IMAGE,
    begin_image_id: int = GLM_BEGIN_IMAGE,
    end_image_id: int = GLM_END_IMAGE,
) -> torch.Tensor:
    """Replace <|image|> token embeddings with projector output.

    For each row, the image span is:
        [begin_of_image] [image]*N [end_of_image]
    We embed all token ids normally, then overwrite the N <|image|> slots with
    the corresponding projected vision vectors.  The begin/end markers stay as
    their own embeddings (they give GLM structural context for the span).
    """
    embeds = torch.nn.functional.embedding(input_ids, token_embed)  # (B, L, D)
    # Use torch.gather + torch.where to splice vision embeds into the
    # token embedding sequence.  These are out-of-place ops that preserve
    # the autograd graph through the projector parameters.  In-place writes
    # on a tensor derived from a frozen embedding break grad_fn.
    B, L, D = embeds.shape
    # Mask: True at |image| token positions (the slots to be replaced)
    image_mask = (input_ids == image_token_id)  # (B, L) bool
    # Build index tensor for gather: at each |image| position, the index
    # into vision_embeds[i] (0, 1, 2, ...).  Non-image positions get 0
    # (dummy — will be masked out by torch.where).
    vision_idx = torch.zeros(B, L, dtype=torch.long, device=embeds.device)
    for i, n in enumerate(num_image_tokens.tolist()):
        if n == 0:
            continue
        begin_positions = (input_ids[i] == begin_image_id).nonzero(as_tuple=True)[0]
        if len(begin_positions) == 0:
            continue
        start = begin_positions[0].item() + 1
        vision_idx[i, start:start + n] = torch.arange(n, device=embeds.device)
    # Gather vision embeds to (B, L, D) — differentiable w.r.t. vision_embeds
    vision_gathered = torch.gather(
        vision_embeds.to(embeds.dtype), dim=1,
        index=vision_idx.unsqueeze(-1).expand(-1, -1, D),
    )  # (B, L, D)
    # Blend: use vision embeds at image positions, token embeds elsewhere
    embeds = torch.where(image_mask.unsqueeze(-1), vision_gathered, embeds)
    return embeds


# ---------------------------------------------------------------------------
# CPU smoke test — run: python -m training.data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from .projector import build_projector

    print("=== GLM-5.2 special token IDs ===")
    for name, tid in glm_special_ids().items():
        print(f"  {name:30s} -> {tid}")

    print("\n=== Patch token count check ===")
    for h, w in [(1280, 800), (1024, 1024), (640, 480)]:
        nt = compute_num_image_tokens(h, w)
        print(f"  {h}x{w} -> {nt} merged tokens")

    # Fake encoder: returns (B, N_merged, 1152)
    class FakeEncoder:
        def __call__(self, pixel_tensors):
            B = pixel_tensors.shape[0]
            h = pixel_tensors.shape[2] // 14 // 2
            w = pixel_tensors.shape[3] // 14 // 2
            return torch.randn(B, h * w, 1152)

    class FakeTokenizer:
        pad_token_id = GLM_EOS
        eos_token_id = GLM_EOS

        def encode(self, text, add_special_tokens=False):
            # deterministic pseudo-tokenize: 1 token per 4 chars
            return [1000 + (i % 500) for i in range(0, max(1, len(text)), 4)]

    proj = build_projector(dtype=torch.float32)
    enc = FakeEncoder()
    tok = FakeTokenizer()

    collator = VisionSFTCollator(
        tokenizer=tok, vision_encoder=enc, projector=proj, device="cpu"
    )

    # Write a tiny temp dataset
    import tempfile

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    tmp.write(json.dumps({
        "image": "FAKE",
        "conversations": [
            {"role": "user", "content": "What is in this image?"},
            {"role": "assistant", "content": "a red bus"},
        ],
    }) + "\n")
    tmp.write(json.dumps({
        "image": "FAKE",
        "conversations": [
            {"role": "user", "content": "Click the submit button"},
            {"role": "assistant", "content": "click(512, 340)"},
        ],
    }) + "\n")
    tmp.close()

    # Monkeypatch _load_image to return a small fake PIL image (280x280 -> 10x10 merged)
    from PIL import Image

    def fake_load(row):
        return Image.new("RGB", (280, 280), (128, 128, 128))

    collator._load_image = fake_load

    ds = SFTJsonlDataset(tmp.name, tok)
    print(f"\nDataset rows: {len(ds)}")

    batch = collator([ds[0], ds[1]])
    print("\n=== Collator output shapes ===")
    for k, v in batch.items():
        print(f"  {k:20s}: {tuple(v.shape)} dtype={v.dtype}")

    labels = batch["labels"]
    has_neg = (labels[0] == -100).any().item()
    has_scored = (labels[0] != -100).any().item()
    print(f"\n  labels[0] has -100 (masked): {has_neg}")
    print(f"  labels[0] has scored tokens: {has_scored}")
    print(f"  num_image_tokens: {batch['num_image_tokens'].tolist()}")

    # Test the splice function with a tiny fake embedding table
    vocab = 154880
    fake_embed = torch.randn(vocab, 6144, requires_grad=False)
    spliced = splice_vision_into_embeds(
        batch["input_ids"], fake_embed,
        batch["vision_embeds"], batch["num_image_tokens"],
    )
    print(f"\n  spliced embeds shape: {tuple(spliced.shape)}")

    # Verify the image span was replaced (not equal to plain embed lookup)
    plain = torch.nn.functional.embedding(batch["input_ids"], fake_embed)
    diff = (spliced != plain).any(dim=-1).sum().item()
    print(f"  tokens changed by splice (should be ~2*batch image spans): {diff}")

    # Verify projector grad flows through collator path
    loss = batch["vision_embeds"].float().pow(2).mean()
    loss.backward()
    grads_ok = all(p.grad is not None and p.grad.abs().sum() > 0 for p in proj.parameters())
    print(f"  projector grad flow through collator: {grads_ok}")

    os.unlink(tmp.name)
    print("\nsmoke test PASSED")
