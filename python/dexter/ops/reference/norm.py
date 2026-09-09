"""Reference norms and residual fusions."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dexter.registry import Priority, register


@register("norm", "layer_norm", name="torch_layer_norm", priority=Priority.REFERENCE)
def layer_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), eps=eps)


@register("norm", "adaln", name="torch_adaln", priority=Priority.REFERENCE)
def adaln_modulate(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float
) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), eps=eps) * (1 + scale) + shift


@register("norm", "gated_residual", name="torch_gated_residual", priority=Priority.REFERENCE)
def gated_residual(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return x + y * gate


@register("norm", "rms_norm", name="torch_rms_norm", priority=Priority.REFERENCE)
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight.float()).to(dtype)
