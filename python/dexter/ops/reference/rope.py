"""Reference QK-norm + rotary."""

from __future__ import annotations

import torch

from dexter.registry import Priority, register


def _rope_adjacent(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent element pairs -- Wan's convention, not split-half.

    ``x`` is ``[B, L, H, D]``; ``cos``/``sin`` are ``[L, D//2]`` and broadcast
    over batch and heads.
    """
    b, seq_len, heads, head_dim = x.shape
    pairs = x.float().reshape(b, seq_len, heads, head_dim // 2, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    c = cos.float()[None, :, None, :]
    s = sin.float()[None, :, None, :]
    out = torch.stack((real * c - imag * s, real * s + imag * c), dim=-1)
    return out.reshape(b, seq_len, heads, head_dim).to(x.dtype)


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Wan's QK norm: RMS over the whole ``dim``, across all heads together.

    ``x`` arrives as ``[B, L, H, D]`` but the norm upstream is applied to the
    ``[B, L, H*D]`` projection before it is split into heads, so the
    denominator and the learned weight both span ``H*D``.
    """
    b, seq_len, heads, head_dim = x.shape
    flat = x.reshape(b, seq_len, heads * head_dim).float()
    flat = flat * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + eps)
    flat = flat * weight.float().reshape(1, 1, -1)
    return flat.reshape(b, seq_len, heads, head_dim).to(x.dtype)


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
    return _rope_adjacent(q, cos, sin), _rope_adjacent(k, cos, sin)
