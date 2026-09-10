"""GR00T N1.7's diffusion-transformer action head, on dexter ops.

Shape of the head, read from the checkpoint:

* 4 self-attention blocks over the 2048-wide VLM features (`vl_self_attention`),
  which condition once per observation.
* 32 DiT blocks of width 1536. Each carries a single AdaLN (`norm1.linear`
  emits shift and scale) and one attention whose **queries come from the action
  tokens while keys and values come from the 2048-wide VLM features** -- it is
  cross-attention despite the `attn1` name.
* Per-embodiment encoders and decoder, the same category-specific pattern
  DreamZero uses, so they reuse `models.layers`.

The action space is padded to 132 dims across 32 embodiments; a given robot
uses a prefix of that.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from dexter import ops
from dexter.models.layers import (
    AffineLayerNorm, CategorySpecificLinear, CategorySpecificMLP, Linear,
)
from dexter.vla.groot.config import ActionHeadConfig


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat((torch.cos(args), torch.sin(args)), dim=-1)


class VLSelfAttentionBlock(nn.Module):
    """Plain pre-norm transformer block over the VLM features."""

    def __init__(self, dim: int, mlp_dim: int, heads: int, eps: float,
                 dtype: torch.dtype) -> None:
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.norm1 = AffineLayerNorm(dim, eps, dtype)
        self.norm3 = AffineLayerNorm(dim, eps, dtype)
        self.to_q = Linear(dim, dim, dtype=dtype)
        self.to_k = Linear(dim, dim, dtype=dtype)
        self.to_v = Linear(dim, dim, dtype=dtype)
        self.to_out = Linear(dim, dim, dtype=dtype)
        self.ff_in = Linear(dim, mlp_dim, dtype=dtype)
        self.ff_out = Linear(mlp_dim, dim, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.norm1(x)
        shape = (b, n, self.heads, self.head_dim)
        attended = ops.blockwise_causal_attention(
            self.to_q(h).view(shape), self.to_k(h).view(shape), self.to_v(h).view(shape))
        x = x + self.to_out(attended.reshape(b, n, -1))
        h = self.norm3(x)
        return x + self.ff_out(nn.functional.gelu(self.ff_in(h), approximate="tanh"))


class DiTBlock(nn.Module):
    """AdaLN over the action tokens, then cross-attention into the VLM features."""

    def __init__(self, cfg: ActionHeadConfig, heads: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.eps = cfg.eps
        self.heads, self.head_dim = heads, cfg.dim // heads
        # One linear emits both modulation terms; dexter's fused adaln consumes
        # them directly, so the block never materialises a normalised copy.
        self.norm1_linear = Linear(cfg.dim, 2 * cfg.dim, dtype=dtype)
        self.to_q = Linear(cfg.dim, cfg.dim, dtype=dtype)
        self.to_k = Linear(cfg.vlm_dim, cfg.dim, dtype=dtype)
        self.to_v = Linear(cfg.vlm_dim, cfg.dim, dtype=dtype)
        self.to_out = Linear(cfg.dim, cfg.dim, dtype=dtype)
        self.ff_in = Linear(cfg.dim, 4 * cfg.dim, dtype=dtype)
        self.ff_out = Linear(4 * cfg.dim, cfg.dim, dtype=dtype)

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                conditioning: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        shift, scale = self.norm1_linear(
            nn.functional.silu(conditioning)).unsqueeze(1).chunk(2, dim=-1)
        h = ops.adaln_modulate(x, scale, shift, self.eps)

        q = self.to_q(h).view(b, n, self.heads, self.head_dim)
        ctx = context.shape[1]
        k = self.to_k(context).view(b, ctx, self.heads, self.head_dim)
        v = self.to_v(context).view(b, ctx, self.heads, self.head_dim)
        attended = ops.blockwise_causal_attention(q, k, v)
        x = x + self.to_out(attended.reshape(b, n, -1))
        return x + self.ff_out(nn.functional.gelu(self.ff_in(x), approximate="tanh"))


class ActionHead(nn.Module):
    """Flow-matching action head conditioned on the backbone's features."""

    def __init__(self, cfg: ActionHeadConfig, depth: int = 32, vl_depth: int = 4,
                 heads: int = 32, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.cfg = cfg
        n = cfg.num_embodiments
        self.vl_blocks = nn.ModuleList(
            VLSelfAttentionBlock(cfg.vlm_dim, 4 * cfg.vlm_dim, heads, cfg.eps, dtype)
            for _ in range(vl_depth))
        self.blocks = nn.ModuleList(DiTBlock(cfg, heads, dtype) for _ in range(depth))

        self.timestep_in = Linear(cfg.dim, cfg.dim, dtype=dtype)
        self.timestep_out = Linear(cfg.dim, cfg.dim, dtype=dtype)

        self.state_encoder = CategorySpecificMLP(n, cfg.state_dim, cfg.hidden, cfg.dim, dtype)
        self.action_encoder_w1 = CategorySpecificLinear(n, cfg.action_dim, cfg.dim, dtype)
        self.action_encoder_w2 = CategorySpecificLinear(n, 2 * cfg.dim, cfg.dim, dtype)
        self.action_encoder_w3 = CategorySpecificLinear(n, cfg.dim, cfg.dim, dtype)
        self.action_decoder = CategorySpecificMLP(n, cfg.hidden, cfg.hidden, cfg.action_dim, dtype)

        self.proj_out_1 = Linear(cfg.dim, 2 * cfg.dim, dtype=dtype)
        self.proj_out_2 = Linear(cfg.dim, cfg.hidden, dtype=dtype)

    def encode_actions(self, actions: torch.Tensor, timestep: torch.Tensor,
                       embodiment: torch.Tensor) -> torch.Tensor:
        a = self.action_encoder_w1(actions, embodiment)
        tau = sinusoidal_embedding(timestep, self.cfg.dim).to(a.dtype)
        tau = tau[:, None, :].expand_as(a)
        x = nn.functional.silu(self.action_encoder_w2(torch.cat((a, tau), dim=-1), embodiment))
        return self.action_encoder_w3(x, embodiment)
