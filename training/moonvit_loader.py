"""Load the exolabs/Kimi-K2.6-vision MoonViT weights into a HF Kimi_K25VisionModel.

The exolabs checkpoint uses different key names than the transformers
Kimi_K25VisionModel. This module remaps them:

  encoder.blocks.N.wqkv  (3456 = 3x1152)  ->  layers.N.attn.{q,k,v}_proj  (split)
  encoder.blocks.N.wo                      ->  layers.N.attn.proj
  encoder.blocks.N.norm0                    ->  layers.N.norm1
  encoder.blocks.N.norm1                    ->  layers.N.norm2
  encoder.blocks.N.mlp.fc0                  ->  layers.N.mlp.fc1
  encoder.blocks.N.mlp.fc1                  ->  layers.N.mlp.fc2
  encoder.final_layernorm                   ->  final_layernorm
  patch_embed.pos_emb.weight                ->  patch_embed.pos_emb.position_embeddings
  patch_embed.proj.*                        ->  patch_embed.proj.*

Usage:
    from training.moonvit_loader import load_moonvit
    encoder = load_moonvit("encoders/moonvit", device="cuda")
    # encoder: Kimi_K25VisionModel, takes pixel tensors -> (B, N_merged, 1152)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from safetensors.torch import load_file


def _remap_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap exolabs vision_tower.* keys to Kimi_K25VisionModel keys."""
    remapped: dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        # Strip "vision_tower." prefix
        if key.startswith("vision_tower."):
            k = key[len("vision_tower."):]
        else:
            k = key

        # encoder.blocks.N.*  ->  layers.N.*
        if k.startswith("encoder.blocks."):
            parts = k.split(".")
            # parts = ["encoder", "blocks", N, attr...]
            layer_idx = parts[2]
            rest = ".".join(parts[3:])  # everything after the layer index
            if rest == "wqkv.weight":
                # Fused QKV: (3*hidden, hidden) -> split into q,k,v
                h = val.shape[1]
                q, k_w, v = val.chunk(3, dim=0)
                remapped[f"layers.{layer_idx}.attn.q_proj.weight"] = q
                remapped[f"layers.{layer_idx}.attn.k_proj.weight"] = k_w
                remapped[f"layers.{layer_idx}.attn.v_proj.weight"] = v
            elif rest == "wqkv.bias":
                q, k_b, v = val.chunk(3, dim=0)
                remapped[f"layers.{layer_idx}.attn.q_proj.bias"] = q
                remapped[f"layers.{layer_idx}.attn.k_proj.bias"] = k_b
                remapped[f"layers.{layer_idx}.attn.v_proj.bias"] = v
            elif rest == "wo.weight":
                remapped[f"layers.{layer_idx}.attn.proj.weight"] = val
            elif rest == "wo.bias":
                remapped[f"layers.{layer_idx}.attn.proj.bias"] = val
            elif rest == "norm0.weight":
                remapped[f"layers.{layer_idx}.norm1.weight"] = val
            elif rest == "norm0.bias":
                remapped[f"layers.{layer_idx}.norm1.bias"] = val
            elif rest == "norm1.weight":
                remapped[f"layers.{layer_idx}.norm2.weight"] = val
            elif rest == "norm1.bias":
                remapped[f"layers.{layer_idx}.norm2.bias"] = val
            elif rest == "mlp.fc0.weight":
                remapped[f"layers.{layer_idx}.mlp.fc1.weight"] = val
            elif rest == "mlp.fc0.bias":
                remapped[f"layers.{layer_idx}.mlp.fc1.bias"] = val
            elif rest == "mlp.fc1.weight":
                remapped[f"layers.{layer_idx}.mlp.fc2.weight"] = val
            elif rest == "mlp.fc1.bias":
                remapped[f"layers.{layer_idx}.mlp.fc2.bias"] = val
            else:
                remapped[f"layers.{layer_idx}.{rest}"] = val
        elif k.startswith("encoder.final_layernorm."):
            suffix = k.split("encoder.final_layernorm.")[1]
            remapped[f"final_layernorm.{suffix}"] = val
        elif k == "patch_embed.pos_emb.weight":
            remapped["patch_embed.pos_emb.position_embeddings"] = val
        elif k.startswith("patch_embed.proj."):
            suffix = k.split("patch_embed.proj.")[1]
            remapped[f"patch_embed.proj.{suffix}"] = val
        else:
            remapped[k] = val
    return remapped


def load_moonvit(
    encoder_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Load MoonViT from the exolabs safetensors into a Kimi_K25VisionModel.

    Args:
        encoder_path: directory containing kimi_k26_vision.safetensors + config.json
        device: target device
        dtype: model dtype (bf16 by default)

    Returns a frozen Kimi_K25VisionModel that takes (B, 3, H, W) pixel tensors
    and returns (B, N_merged, 1152) features.
    """
    from transformers.models.kimi_k25 import Kimi_K25VisionConfig, Kimi_K25VisionModel

    # Build config from the exolabs vision_config
    import json, os

    with open(os.path.join(encoder_path, "config.json")) as f:
        cfg = json.load(f)

    vc = cfg["vision_config"]
    config = Kimi_K25VisionConfig(
        patch_size=vc["patch_size"],
        pos_emb_height=vc["init_pos_emb_height"],
        pos_emb_width=vc["init_pos_emb_width"],
        pos_emb_time=vc["init_pos_emb_time"],
        num_attention_heads=vc["vt_num_attention_heads"],
        num_hidden_layers=vc["vt_num_hidden_layers"],
        hidden_size=vc["vt_hidden_size"],
        intermediate_size=vc["vt_intermediate_size"],
        hidden_act="quick_gelu",  # MoonViT uses quick_gelu
        merge_kernel_size=vc["merge_kernel_size"],
    )

    model = Kimi_K25VisionModel(config)
    model = model.to(dtype)

    # Load and remap weights
    safetensors_path = os.path.join(encoder_path, "kimi_k26_vision.safetensors")
    raw = load_file(safetensors_path)
    vt = {k: v for k, v in raw.items() if k.startswith("vision_tower.")}
    remapped = _remap_keys(vt)

    # Load with strict=False to see what matches
    result = model.load_state_dict(remapped, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith("rotary_emb")]
    unexpected = result.unexpected_keys
    if missing:
        print(f"[moonvit] WARNING: {len(missing)} missing keys (first 5: {missing[:5]})")
    if unexpected:
        print(f"[moonvit] WARNING: {len(unexpected)} unexpected keys (first 5: {unexpected[:5]})")

    # Freeze
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    model.to(device)
    return model


class MoonViTWrapper(nn.Module):
    """Wraps Kimi_K25VisionModel to expose a simple (B, 3, H, W) -> (B, N_merged, 1152) interface.

    The raw Kimi_K25VisionModel expects pixel_values as (total_patches, 3, P, P)
    and grid_thw as (num_images, 3), returning pooler_output of shape
    (total_merged, kernel_h*kernel_w, hidden).  This wrapper:
      1. Patchifies (B, 3, H, W) -> (B*H/P*W/P, 3, P, P) [assumes same H,W for batch]
      2. Calls the model with grid_thw
      3. Averages the kernel_h*kernel_w dimension -> (total_merged, hidden)
      4. Reshapes to (B, N_merged_per_img, hidden)
    """

    def __init__(self, model: nn.Module, patch_size: int = 14, merge: int = 2):
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        self.merge = merge
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, pixel_tensors: torch.Tensor) -> torch.Tensor:
        """pixel_tensors: (B, 3, H, W) -> (B, N_merged, 1152)."""
        B, C, H, W = pixel_tensors.shape
        P = self.patch_size
        h_patches = H // P
        w_patches = W // P
        patches_per_img = h_patches * w_patches

        # Patchify: (B, 3, H, W) -> (B, 3, h, w, P, P) -> (B*h*w, 3, P, P)
        x = pixel_tensors
        x = x.view(B, C, h_patches, P, w_patches, P)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B * patches_per_img, C, P, P)

        # grid_thw: (B, 3) = (T=1, h_patches, w_patches) per image
        grid_thw = torch.tensor(
            [[1, h_patches, w_patches]] * B, device=x.device, dtype=torch.long
        )

        out = self.model(x, grid_thw=grid_thw)
        pooled = out.pooler_output  # (total_merged, kernel_h*kernel_w, hidden)

        # Average the 2x2 group -> (total_merged, hidden)
        merged = pooled.mean(dim=1)

        # Reshape to (B, N_merged_per_img, hidden)
        n_merged = (h_patches // self.merge) * (w_patches // self.merge)
        merged = merged.view(B, n_merged, -1)
        return merged


def load_moonvit_wrapped(
    encoder_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Load MoonViT and wrap it for simple (B,3,H,W) -> (B,N,1152) interface."""
    model = load_moonvit(encoder_path, device=device, dtype=dtype)
    return MoonViTWrapper(model, patch_size=14, merge=2)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "encoders/moonvit"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32 if device == "cpu" else torch.bfloat16
    enc = load_moonvit_wrapped(path, device=device, dtype=dt)
    print(f"Loaded MoonViT: {sum(p.numel() for p in enc.parameters()):,} params")

    # Quick forward test: 2 images of 280x280 -> 10*10=100 merged tokens each
    x = torch.randn(2, 3, 280, 280, device=device, dtype=dt)
    with torch.no_grad():
        out = enc(x)
    print(f"Output shape: {tuple(out.shape)} (expected (2, 100, 1152))")
    assert out.shape == (2, 100, 1152), f"got {out.shape}"
    print("MoonViT loader OK")
