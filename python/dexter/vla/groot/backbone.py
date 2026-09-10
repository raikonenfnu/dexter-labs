"""Qwen3-VL backbone for GR00T N1.7, on dexter ops.

Two properties of this backbone are worth exploiting and are easy to miss:

* **It is truncated.** N1.7 reads hidden states from an intermediate LLM layer
  (`select_layer`), so the layers above it never contribute to an action. The
  checkpoint stores 16; inference needs 13. Skipping the rest is free.
* **Grouped-query attention.** 16 query heads share 8 key/value heads, so the
  KV projections are half width. Expanding K/V to full head count before
  attention is the simple route and is what happens here; a KV-aware kernel
  would avoid the copy, which matters once this runs in a loop.

Rotary here is the *half-offset* convention (Qwen3, and HF models generally),
not the adjacent pairing Wan uses.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dexter import ops
from dexter.models.layers import AffineLayerNorm, Linear, RMSNorm
from dexter.vla.groot.config import LanguageConfig, VisionConfig


def rope_tables(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device,
                                        dtype=torch.float32) / head_dim))
    angles = torch.arange(seq_len, device=device, dtype=torch.float32)[:, None] * inv[None, :]
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


class VisionBlock(nn.Module):
    """Pre-norm ViT block; attention is unmasked over the patch sequence."""

    def __init__(self, cfg: VisionConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.heads, self.head_dim = cfg.heads, cfg.head_dim
        self.norm1 = AffineLayerNorm(cfg.dim, cfg.eps, dtype)
        self.norm2 = AffineLayerNorm(cfg.dim, cfg.eps, dtype)
        self.qkv = Linear(cfg.dim, 3 * cfg.dim, dtype=dtype)
        self.proj = Linear(cfg.dim, cfg.dim, dtype=dtype)
        self.fc1 = Linear(cfg.dim, cfg.mlp_dim, dtype=dtype)
        self.fc2 = Linear(cfg.mlp_dim, cfg.dim, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        attended = ops.blockwise_causal_attention(q, k, v)   # no mask: patches see all
        x = x + self.proj(attended.reshape(b, n, -1))
        return x + self.fc2(nn.functional.gelu(self.fc1(self.norm2(x)), approximate="tanh"))


class VisionTower(nn.Module):
    """Patchify -> ViT -> spatial merge into the language width."""

    def __init__(self, cfg: VisionConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.cfg = cfg
        t, h, w = cfg.patch
        self.patch_embed = Linear(cfg.in_channels * t * h * w, cfg.dim, dtype=dtype)
        self.blocks = nn.ModuleList(VisionBlock(cfg, dtype) for _ in range(cfg.depth))
        self.merger_norm = AffineLayerNorm(cfg.dim, cfg.eps, dtype)
        self.merger_fc1 = Linear(cfg.merge_dim, cfg.merge_dim, dtype=dtype)
        self.merger_fc2 = Linear(cfg.merge_dim, cfg.out_dim, dtype=dtype)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """``patches`` is ``[B, N, in_channels * prod(patch)]``."""
        x = self.patch_embed(patches)
        for block in self.blocks:
            x = block(x)
        # The merger folds each spatial_merge**2 group of patches into one token.
        merge = self.cfg.spatial_merge**2
        x = self.merger_norm(x).reshape(x.shape[0], -1, self.cfg.merge_dim)
        return self.merger_fc2(nn.functional.gelu(self.merger_fc1(x), approximate="tanh"))


class LanguageBlock(nn.Module):
    """Qwen3 decoder block: GQA with per-head QK norm, then SwiGLU."""

    def __init__(self, cfg: LanguageConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_layernorm = RMSNorm(cfg.dim, cfg.eps, dtype)
        self.post_attention_layernorm = RMSNorm(cfg.dim, cfg.eps, dtype)
        self.q_proj = Linear(cfg.dim, cfg.heads * cfg.head_dim, bias=False, dtype=dtype)
        self.k_proj = Linear(cfg.dim, cfg.kv_heads * cfg.head_dim, bias=False, dtype=dtype)
        self.v_proj = Linear(cfg.dim, cfg.kv_heads * cfg.head_dim, bias=False, dtype=dtype)
        self.o_proj = Linear(cfg.heads * cfg.head_dim, cfg.dim, bias=False, dtype=dtype)
        # Qwen3 normalises each head individually, so these are head_dim wide.
        self.q_norm = RMSNorm(cfg.head_dim, cfg.eps, dtype)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.eps, dtype)
        self.gate_proj = Linear(cfg.dim, cfg.mlp_dim, bias=False, dtype=dtype)
        self.up_proj = Linear(cfg.dim, cfg.mlp_dim, bias=False, dtype=dtype)
        self.down_proj = Linear(cfg.mlp_dim, cfg.dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        b, n, _ = x.shape
        h = self.input_layernorm(x)
        q = self.q_norm(self.q_proj(h).view(b, n, cfg.heads, cfg.head_dim))
        k = self.k_norm(self.k_proj(h).view(b, n, cfg.kv_heads, cfg.head_dim))
        v = self.v_proj(h).view(b, n, cfg.kv_heads, cfg.head_dim)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        # Expand the shared KV heads to match the query heads.
        repeat = cfg.heads // cfg.kv_heads
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)

        attended = ops.blockwise_causal_attention(q, k, v, causal=True)
        x = x + self.o_proj(attended.reshape(b, n, -1))

        h = self.post_attention_layernorm(x)
        return x + self.down_proj(nn.functional.silu(self.gate_proj(h)) * self.up_proj(h))


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Half-offset rotary (Qwen3), applied to ``[B, L, H, D]``."""
    half = x.shape[-1] // 2
    c = cos[None, : x.shape[1], None, :].to(x.dtype)
    s = sin[None, : x.shape[1], None, :].to(x.dtype)
    real, imag = x.split(half, dim=-1)
    return torch.cat((real * c - imag * s, real * s + imag * c), dim=-1)
