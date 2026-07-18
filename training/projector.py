"""Vision projector: maps frozen MoonViT features into GLM-5.2's embedding space.

Architecture (from RESEARCH.md — matches the article's "~50M" 2-layer MLP):
    LayerNorm(1152) -> Linear(1152, 6144) -> GELU -> Linear(6144, 6144)

Input:  MoonViT patch tokens, shape (B, N_patches, 1152), BF16.
Output: GLM embedding-space tokens, shape (B, N_patches, 6144), BF16.

These output vectors replace <|media_pad|> placeholder tokens in GLM's input
embedding stream. Only this module is trained; GLM-5.2 stays frozen (NF4).

Parameter count: 1152*6144 + 6144 + 6144*6144 + 6144 + 1152*2  ~= 44.8M
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn


class VisionProjector(nn.Module):
    """2-layer MLP projector: MoonViT (1152) -> GLM-5.2 hidden (6144).

    The 2x2 spatial merge of MoonViT is parameter-free mean-pooling (keeps dim at
    1152), done upstream in the data collator — so the projector always sees
    (B, N_merged_patches, 1152).
    """

    def __init__(
        self,
        encoder_dim: int = 1152,
        glm_hidden: int = 6144,
        act: str = "gelu",
        ln_eps: float = 1e-5,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.glm_hidden = glm_hidden

        self.ln = nn.LayerNorm(encoder_dim, eps=ln_eps, dtype=dtype)
        self.fc1 = nn.Linear(encoder_dim, glm_hidden, bias=True, dtype=dtype)
        self.act = nn.GELU() if act == "gelu" else getattr(nn, act)()
        self.fc2 = nn.Linear(glm_hidden, glm_hidden, bias=True, dtype=dtype)

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        """vision_tokens: (B, N, 1152) -> (B, N, 6144)."""
        x = self.ln(vision_tokens)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_projector(dtype: torch.dtype = torch.bfloat16) -> VisionProjector:
    return VisionProjector(encoder_dim=1152, glm_hidden=6144, dtype=dtype)


def save_projector(projector: VisionProjector, path: str) -> None:
    """Save projector weights to a safetensors file."""
    from safetensors.torch import save_file
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    state = {k: v.contiguous() for k, v in projector.state_dict().items()}
    save_file(state, path)


def load_projector(projector: VisionProjector, path: str) -> VisionProjector:
    """Load projector weights from a safetensors file."""
    from safetensors.torch import load_file
    state = load_file(path)
    projector.load_state_dict(state)
    return projector


# ---------------------------------------------------------------------------
# CPU smoke test — run: python -m training.projector
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    proj = build_projector(dtype=torch.float32)  # float32 for CPU grad check

    n_params = proj.num_params()
    print(f"projector params: {n_params:,}  ({n_params / 1e6:.1f}M)")
    assert 40e6 < n_params < 50e6, f"param count {n_params} outside ~44.8M range"

    # Fake MoonViT output: batch 2, 1024 merged patches, 1152-dim
    x = torch.randn(2, 1024, 1152, requires_grad=False)
    y = proj(x)
    print(f"input  {tuple(x.shape)} -> output {tuple(y.shape)}")
    assert y.shape == (2, 1024, 6144), "wrong output shape"

    # Grad flows to projector params (not to the frozen vision tokens)
    loss = y.float().pow(2).mean()
    loss.backward()
    grads = [p.grad is not None and p.grad.abs().sum().item() for p in proj.parameters()]
    assert all(g > 0 for g in grads), "no grad flow to projector params"
    print(f"grad flow: OK  (all {len(grads)} param tensors have nonzero grad)")
    print(f"loss: {loss.item():.4f}")
    print("smoke test PASSED")
