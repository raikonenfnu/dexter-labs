"""Reference QK-norm + rotary."""

from __future__ import annotations

import torch

from dexter.registry import Priority, register


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = x.float()
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)
    return (out * weight.float()).to(x.dtype)


@register("rope", "qk_norm_rope", name="torch_qk_norm_rope", priority=Priority.REFERENCE)
def qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = _rms(q, q_weight, eps)
    k = _rms(k, k_weight, eps)
    # cos/sin are [L, D//2] -> broadcast over batch and heads as [1, L, 1, D]
    cos = torch.cat((cos, cos), dim=-1)[None, :, None, :].to(q.dtype)
    sin = torch.cat((sin, sin), dim=-1)[None, :, None, :].to(q.dtype)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
