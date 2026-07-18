"""Phase-1 SFT trainer: projector-only training on a frozen GLM-5.2 backbone.

Architecture:
    image -> MoonViT (frozen) -> VisionProjector (TRAINABLE) -> GLM-5.2 (frozen, NF4)

Only the 44.8M projector receives gradients.  GLM-5.2's 504B parameters are
frozen and (optionally) loaded in 4-bit NF4 via bitsandbytes.  The projector
maps MoonViT's 1152-dim features into GLM's 6144-dim embedding space, and the
projected vectors replace the |image| placeholder tokens in the input embedding
stream.

Loss is cross-entropy on assistant answer tokens only (masked by the collator).

Usage:
    python -m training.train_sft \
        --model models/GLM-5.2-nf4 \
        --encoder encoders/moonvit \
        --data data/sft/train.jsonl \
        --val data/sft/val.jsonl \
        --out logs/sft_phase1 \
        --lr 5e-4 --batch 64 --microbatch 4 --epochs 2

CPU smoke test (no GPU, tiny fake model):
    python -m training.train_sft --smoke-test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import (
    GLM_EOS,
    GLM_IMAGE,
    GLM_BEGIN_IMAGE,
    GLM_END_IMAGE,
    SFTJsonlDataset,
    VisionSFTCollator,
    compute_num_image_tokens,
    splice_vision_into_embeds,
)
from .projector import build_projector, load_projector


def log(*args):
    print("[sft]", *args, flush=True)


# ---------------------------------------------------------------------------
# Model wrapper: frozen GLM + trainable projector
# ---------------------------------------------------------------------------

class GlmVisionModel(nn.Module):
    """Wraps a frozen GLM-5.2 backbone with a trainable vision projector.

    The forward pass:
      1. Embed all input_ids via GLM's token embedding (frozen).
      2. Splice projected vision tokens in place of |image| positions.
      3. Run GLM transformer + LM head (frozen) to get logits.

    Only the projector has requires_grad=True; everything else is frozen.
    """

    def __init__(self, backbone: nn.Module, projector: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        # Freeze the entire backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        # Extract the embedding layer for the splice (and freeze it too,
        # in case it's not a direct child of backbone)
        self.embed_tokens = backbone.model.embed_tokens
        for p in self.embed_tokens.parameters():
            p.requires_grad_(False)
        # LM head (some GLM variants tie weights; handle both)
        self.lm_head = getattr(backbone, "lm_head", None)
        if self.lm_head is not None:
            for p in self.lm_head.parameters():
                p.requires_grad_(False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vision_embeds: torch.Tensor,
        num_image_tokens: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Embed all tokens (frozen lookup)
        token_embed = self.embed_tokens.weight  # (vocab, D)
        inputs_embeds = F.embedding(input_ids, token_embed)  # (B, L, D)

        # 2. Splice vision embeddings at |image| positions
        inputs_embeds = splice_vision_into_embeds(
            input_ids, token_embed, vision_embeds, num_image_tokens,
            image_token_id=GLM_IMAGE,
            begin_image_id=GLM_BEGIN_IMAGE,
            end_image_id=GLM_END_IMAGE,
        )

        # 3. Forward through backbone (frozen)
        # GLM-5.2's forward takes inputs_embeds when input_ids is None
        outputs = self.backbone(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return outputs.logits  # (B, L, vocab)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import prepare_model_for_kbit_training
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # --- Load tokenizer ---
    log("loading tokenizer:", args.model)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = GLM_EOS

    # --- Load backbone (frozen, NF4) ---
    from transformers import BitsAndBytesConfig, AutoConfig
    import os as _os
    _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    backbone_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    num_layers = backbone_config.num_hidden_layers  # 78

    # Check if model is already pre-quantized (has bnb quantized tensors)
    # If so, load directly without quantization config — avoids BF16 loading
    # spike that causes OOM on GPUs with limited VRAM.
    is_prequantized = _os.path.exists(
        _os.path.join(args.model, "model.safetensors.index.json")
    ) and any(
        "quant" in f or "4bit" in f
        for f in _os.listdir(args.model)
        if f.endswith(".safetensors")
    )

    # Manual device map: split layers evenly across GPUs.
    device_map = {}
    device_map["model.embed_tokens"] = 0
    layer_idx = 0
    layer_counts = [18, 20, 20, 20]  # adjusted for 4 GPUs
    for gpu in range(ngpu):
        count = layer_counts[gpu] if gpu < len(layer_counts) else num_layers // ngpu
        for _ in range(count):
            if layer_idx < num_layers:
                device_map[f"model.layers.{layer_idx}"] = gpu
                layer_idx += 1
    while layer_idx < num_layers:
        device_map[f"model.layers.{layer_idx}"] = ngpu - 1
        layer_idx += 1
    device_map["model.norm"] = ngpu - 1
    device_map["lm_head"] = ngpu - 1
    log(f"manual device_map: {ngpu} GPUs, layer_counts={layer_counts[:ngpu]}")

    if is_prequantized:
        log("loading pre-quantized GLM-5.2 backbone (NF4)...")
        backbone = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        log("loading GLM-5.2 backbone (BF16 -> NF4 on-the-fly)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        backbone = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    backbone.config.use_cache = False
    backbone.gradient_checkpointing_enable()
    backbone.enable_input_require_grads()

    # --- Load MoonViT encoder (frozen, wrapped for (B,3,H,W) -> (B,N,1152)) ---
    from .moonvit_loader import load_moonvit_wrapped
    log("loading MoonViT encoder...")
    vision_encoder = load_moonvit_wrapped(args.encoder, device=device, dtype=torch.bfloat16)

    # --- Build projector (trainable) ---
    # Place projector on GPU 0 (same as embed_tokens — vision features enter there)
    projector = build_projector(dtype=torch.bfloat16).to(0 if torch.cuda.is_available() else "cpu")
    log(f"projector params: {sum(p.numel() for p in projector.parameters()):,}")

    # --- Optional warm-start from a saved projector checkpoint ---
    # Lets us switch datasets mid-campaign without throwing away already-learned
    # vision-to-GLM alignment. The optimizer state is NOT restored — only weights.
    if getattr(args, "init_projector", None):
        if os.path.exists(args.init_projector):
            load_projector(projector, args.init_projector)
            log(f"loaded projector weights from {args.init_projector}")
        else:
            log(f"WARN: --init-projector path not found: {args.init_projector} — starting fresh")

    # --- Wrap into combined model ---
    # Do NOT call .to(device) on the wrapper — the backbone is already
    # distributed across GPUs via device_map.  Calling .to() would try to
    # move the entire 194GB backbone to a single GPU and OOM.
    model = GlmVisionModel(backbone, projector)

    # --- Optimizer: projector only ---
    trainable_params = [p for p in projector.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)

    # --- Data (wait for dataset build to finish if still in progress) ---
    for _ in range(720):  # up to 2 hours
        if os.path.exists(args.data):
            break
        log(f"waiting for dataset {args.data}...")
        time.sleep(10)
    assert os.path.exists(args.data), f"dataset not found: {args.data}"
    train_ds = SFTJsonlDataset(args.data, tok, max_text_len=512)
    val_ds = SFTJsonlDataset(args.val, tok, max_text_len=512) if args.val else None
    log(f"train: {len(train_ds)} examples, val: {len(val_ds) if val_ds else 0}")

    collator = VisionSFTCollator(
        tokenizer=tok,
        vision_encoder=vision_encoder,
        projector=projector,
        device=device,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.microbatch, shuffle=True,
        collate_fn=collator, num_workers=0, drop_last=True,
    )

    # --- Training loop ---
    step = 0
    micro = 0
    acc_loss = 0.0
    tokens_seen = 0
    t0 = time.time()
    grad_accum = args.batch // args.microbatch

    model.train()
    opt.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        log(f"--- epoch {epoch+1}/{args.epochs} ---")
        for batch in train_loader:
            # Move to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            vision_embeds = batch["vision_embeds"].to(device)
            num_img = batch["num_image_tokens"].to(device)

            logits = model(input_ids, attn_mask, vision_embeds, num_img)
            # Shift: predict next token
            shift_logits = logits[:, :-1, :].float()
            shift_labels = labels[:, 1:]
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
            (loss / grad_accum).backward()
            acc_loss += float(loss.detach())
            tokens_seen += input_ids.shape[1]
            micro += 1

            if micro % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0:
                    elapsed = time.time() - t0
                    log(
                        f"step {step}  loss {acc_loss / micro:.4f}  "
                        f"{elapsed / step:.1f}s/step  {tokens_seen / elapsed:.0f} tok/s"
                    )
                    # Grok detection: watch for sharp loss drop ~step 900
                    if args.watch_grok and step >= 800:
                        _check_grok(step, acc_loss / micro)

                if args.save_every and step % args.save_every == 0:
                    _save_projector(projector, args.out, step)
                    log(f"  saved projector @ step {step}")

        # End-of-epoch validation
        if val_ds is not None:
            val_loss = _validate(model, val_ds, collator, device)
            log(f"  val loss: {val_loss:.4f}")

    _save_projector(projector, args.out, step)
    elapsed = time.time() - t0
    log(f"DONE  final_loss={acc_loss / max(1, micro):.4f}  "
        f"{tokens_seen / max(1.0, elapsed):.0f} tok/s  {elapsed / 60:.1f} min")


def _validate(model, val_ds, collator, device, max_batches=50):
    """Run validation and return average loss."""
    model.eval()
    # num_workers=0: the collator runs the vision encoder on GPU, which
    # cannot be re-initialized inside forked DataLoader subprocesses.
    loader = DataLoader(
        val_ds, batch_size=4, shuffle=False, collate_fn=collator, num_workers=0
    )
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            vision_embeds = batch["vision_embeds"].to(device)
            num_img = batch["num_image_tokens"].to(device)
            logits = model(input_ids, attn_mask, vision_embeds, num_img)
            shift_logits = logits[:, :-1, :].float()
            shift_labels = labels[:, 1:]
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
            total_loss += float(loss)
            n += 1
    model.train()
    return total_loss / max(1, n)


_grok_prev_loss = None
_grok_alerted = False


def _check_grok(step: int, loss: float):
    """Detect the sharp loss drop characteristic of the alignment grok."""
    global _grok_prev_loss, _grok_alerted
    if _grok_prev_loss is not None and not _grok_alerted:
        drop = _grok_prev_loss - loss
        if drop > 0.5:  # >0.5 nats drop in one log interval = grok signal
            log("*** GROK DETECTED *** loss dropped "
                f"{_grok_prev_loss:.4f} -> {loss:.4f} at step {step}")
            _grok_alerted = True
    _grok_prev_loss = loss


def _save_projector(projector, out_dir: str, step: int):
    os.makedirs(out_dir, exist_ok=True)
    from safetensors.torch import save_file

    state = {k: v.to(torch.bfloat16).cpu() for k, v in projector.state_dict().items()}
    save_file(state, os.path.join(out_dir, "projector.safetensors"))
    with open(os.path.join(out_dir, "latest_step.txt"), "w") as f:
        f.write(f"{step}\n")


def _load_moonvit(encoder_path: str, device: str) -> nn.Module:
    """Load MoonViT vision tower from the exolabs/Kimi-K2.6-vision safetensors.

    Returns a callable that takes (B, 3, H, W) and returns (B, N_merged, 1152).
    Uses the standalone MoonViT implementation from moonshotai/Kimi-VL-A3B
    architecture (loaded via transformers trust_remote_code if available, or a
    minimal reimplementation).
    """
    from safetensors.torch import load_file as load_safetensors

    # Try loading via transformers Kimi-VL architecture
    try:
        from transformers import AutoModel
        enc = AutoModel.from_pretrained(
            encoder_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)
        return enc.to(device)
    except Exception as e:
        log(f"  could not load MoonViT via AutoModel ({e}); using stub")
        return _MoonViTStub(encoder_path, device)


class _MoonViTStub(nn.Module):
    """Minimal MoonViT loader for when the full model class is unavailable.

    Loads the raw vision_tower weights from the safetensors file and runs a
    simplified NaViT forward.  This is a fallback; prefer the real class.
    """

    def __init__(self, path: str, device: str):
        super().__init__()
        weights = load_safetensors(os.path.join(path, "kimi_k26_vision.safetensors"))
        # Keep only vision_tower.* tensors (skip mm_projector which targets 7168)
        vt = {k.replace("vision_tower.", ""): v for k, v in weights.items()
              if k.startswith("vision_tower.")}
        self.patch_size = 14
        self.merge = 2
        self.hidden = 1152
        # Store for reference; real forward requires the full NaViT module.
        # In practice, the AutoModel path above should succeed.
        self._stub = True
        self.to(device)

    def forward(self, pixel_tensors: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pixel_tensors.shape
        n = compute_num_image_tokens(H, W, self.patch_size, self.merge)
        # Stub: return random features (real training uses the actual encoder)
        return torch.randn(B, n, self.hidden, device=pixel_tensors.device,
                           dtype=pixel_tensors.dtype) * 0.01


# ---------------------------------------------------------------------------
# CPU smoke test
# ---------------------------------------------------------------------------

def smoke_test():
    """Test the model wrapper + collator with a tiny fake backbone on CPU."""
    log("=== CPU smoke test ===")

    # Tiny fake backbone that mimics GLM's interface.  Uses a 1-layer attention
    # so that answer-token logits depend on the image tokens (grad flows back
    # to the projector through attention).  All weights frozen.
    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            vocab = 154880
            D = 6144
            self.model = type("M", (), {"embed_tokens": nn.Embedding(vocab, D)})()
            # A simple linear "mixing" layer so downstream logits depend on
            # upstream (image) positions (simulates causal attention's context).
            self.mixer = nn.Linear(D, D, bias=False)
            self.lm_head = nn.Linear(D, vocab, bias=False)
            self.config = type("C", (), {"use_cache": False})()

        def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                    use_cache=False):
            # Mix: each position's representation depends on all positions
            # (cheap proxy for attention; enough for grad-flow testing).
            mixed = self.mixer(inputs_embeds.mean(dim=1, keepdim=True).expand_as(inputs_embeds))
            logits = self.lm_head(inputs_embeds + mixed * 0.1)
            return type("O", (), {"logits": logits})()

    class FakeTokenizer:
        pad_token_id = GLM_EOS
        eos_token_id = GLM_EOS

        def encode(self, text, add_special_tokens=False):
            return [1000 + (i % 500) for i in range(0, max(1, len(text)), 4)]

    class FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            # dummy parameter so the collator can probe the encoder dtype
            self.dtype_probe = nn.Parameter(torch.zeros(1), requires_grad=False)

        def forward(self, pixel_tensors):
            B = pixel_tensors.shape[0]
            h = pixel_tensors.shape[2] // 14 // 2
            w = pixel_tensors.shape[3] // 14 // 2
            return torch.randn(B, h * w, 1152) * 0.01

    backbone = FakeBackbone()
    proj = build_projector(dtype=torch.float32)
    model = GlmVisionModel(backbone, proj)

    # Verify only projector has grad
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    log(f"trainable params: {trainable}")
    assert all("projector" in n for n in trainable), "non-projector params are trainable!"
    assert len(trainable) == 6, f"expected 6 projector tensors, got {len(trainable)}"

    # Build a fake batch
    collator = VisionSFTCollator(
        tokenizer=FakeTokenizer(), vision_encoder=FakeEncoder(),
        projector=proj, device="cpu",
    )

    # Monkeypatch image loading
    from PIL import Image
    collator._load_image = lambda row: Image.new("RGB", (280, 280), (128, 128, 128))

    batch = collator([{
        "conversations": [
            {"role": "user", "content": "What is this?"},
            {"role": "assistant", "content": "a cat"},
        ]
    }])
    log(f"batch shapes: ids={tuple(batch['input_ids'].shape)} "
        f"embeds={tuple(batch['vision_embeds'].shape)}")

    # Forward pass
    logits = model(
        batch["input_ids"], batch["attention_mask"],
        batch["vision_embeds"], batch["num_image_tokens"],
    )
    log(f"logits: {tuple(logits.shape)}")
    assert logits.shape[0] == 1
    assert logits.shape[2] == 154880

    # Loss + backward
    shift_logits = logits[:, :-1, :].float()
    shift_labels = batch["labels"][:, 1:]
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )
    loss.backward()

    # Verify grad flows to projector, NOT to backbone
    proj_grads = [p.grad is not None and p.grad.abs().sum() > 0 for p in proj.parameters()]
    backbone_grads = [p.grad is not None for p in backbone.parameters()]
    log(f"projector grads: {proj_grads}")
    log(f"backbone grads (should be None): {backbone_grads}")
    assert all(proj_grads), "projector missing gradients"
    assert not any(backbone_grads), "backbone should have no gradients"

    log("smoke test PASSED")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="GLM-5.2 vision SFT (projector-only)")
    ap.add_argument("--model", default="models/GLM-5.2-nf4",
                    help="path to a local GLM-5.2 checkpoint (NF4-prequantized or BF16)")
    ap.add_argument("--encoder", default="encoders/moonvit",
                    help="path to the MoonViT (Kimi K2.6 vision tower) weights")
    ap.add_argument("--data", default="data/sft/train.jsonl")
    ap.add_argument("--val", default="data/sft/val.jsonl")
    ap.add_argument("--out", default="logs/sft_phase1")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=64, help="global batch size")
    ap.add_argument("--microbatch", type=int, default=4, help="per-GPU batch")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--init-projector", default=None,
                    help="path to projector.safetensors to warm-start from (weights only)")
    ap.add_argument("--watch-grok", action="store_true", default=True)
    ap.add_argument("--smoke-test", action="store_true", help="CPU smoke test, no GPU")
    args = ap.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    train(args)


if __name__ == "__main__":
    main()
