"""Public op surface.

Every function here is a thin dispatcher: it validates nothing beyond what the
selected kernel needs, asks the registry for the best implementation on this
platform, and calls it. Model code imports only from this module, so a new
architecture is a new directory under ``ops/`` and no model edits.

The op set is deliberately small and shaped by the DreamZero DiT block, which
is what a World Action Model spends ~95% of its step in:

    x = gated_residual(x, self_attn(adaln(x, s1, b1)), g1)
    x = x + cross_attn(layer_norm(x))
    x = gated_residual(x, ffn(adaln(x, s2, b2)), g2)

Fusing ``adaln`` and ``gated_residual`` matters here for a reason specific to
this class of model: the DiT runs at ~125 query tokens per denoising step, so
every elementwise pass is a full round trip to LPDDR for a tensor that never
had a chance to stay resident. On gfx1151 those passes are pure bandwidth.
"""

from __future__ import annotations

import torch

from dexter.registry import select

__all__ = [
    "adaln_modulate",
    "gated_residual",
    "layer_norm",
    "rms_norm",
    "qk_norm_rope",
    "linear",
    "ffn_gelu",
    "blockwise_causal_attention",
]


def adaln_modulate(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """``layer_norm(x) * (1 + scale) + shift`` in one pass.

    ``x`` is ``[B, L, C]``; ``scale``/``shift`` are ``[B, L, C]`` or ``[B, 1, C]``.
    """
    return select("norm", "adaln", x, scale, shift)(x, scale, shift, eps)


def gated_residual(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``x + y * gate``, fused so the residual is read and written once."""
    return select("norm", "gated_residual", x, y, gate)(x, y, gate)


def layer_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """LayerNorm with no affine parameters (Wan's ``WanLayerNorm``)."""
    return select("norm", "layer_norm", x)(x, eps)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return select("norm", "rms_norm", x, weight)(x, weight, eps)


def qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm then RoPE on q and k, fused.

    ``q``/``k`` are ``[B, L, H, D]``; ``cos``/``sin`` are ``[L, D // 2]``.
    Wan applies QK-norm and rotary back to back, and neither is worth its own
    round trip at these sequence lengths.
    """
    return select("rope", "qk_norm_rope", q, k)(q, k, q_weight, k_weight, cos, sin, eps)


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    scales: torch.Tensor | None = None,
    zeros: torch.Tensor | None = None,
    group_size: int = 128,
    bits: int = 4,
) -> torch.Tensor:
    """``x @ weight.T + bias``, dense or weight-only quantised.

    When ``scales`` is given, ``weight`` is int4 (two nibbles per byte) or int8
    and is dequantised inside the kernel, so only the quantised bytes cross the
    memory bus. That is the whole point on a 256 GB/s part: the DiT is weight
    bound at every batch size a robot control loop will ever use.
    """
    if scales is None:
        return select("gemm", "dense", x, weight)(x, weight, bias)
    # Both widths are stored as uint8, so the dtype cannot tell them apart --
    # the caller states the width.
    if bits not in (4, 8):
        raise ValueError(f"weight-only quantisation supports 4 or 8 bits, got {bits}")
    mode = f"w{bits}a16"
    return select("gemm", mode, x, weight, scales)(x, weight, scales, zeros, bias, group_size)


def ffn_gelu(
    x: torch.Tensor,
    up: torch.Tensor,
    up_bias: torch.Tensor | None,
    down: torch.Tensor,
    down_bias: torch.Tensor | None,
) -> torch.Tensor:
    """Wan's FFN: ``Linear -> GELU(tanh) -> Linear``, with GELU in the epilogue."""
    return select("gemm", "ffn_gelu", x, up, down)(x, up, up_bias, down, down_bias)


def blockwise_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    kv_cache: torch.Tensor | None = None,
    cache_len: int = 0,
    block_boundaries: torch.Tensor | None = None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Attention over ``[video tokens | action register]`` with block causality.

    Queries in block ``i`` attend to every key in blocks ``<= i`` plus the
    cached history of length ``cache_len``. ``block_boundaries`` holds the
    start index of each block; passing ``None`` means a single block, which is
    what a single closed-loop step looks like once history is in the cache.
    """
    return select("attention", "blockwise_causal", q, k, v)(
        q, k, v, kv_cache, cache_len, block_boundaries, softmax_scale
    )


# Importing the implementation packages is what populates the registry.
from dexter.ops import reference  # noqa: E402,F401  (registers REFERENCE kernels)
from dexter.ops import gfx1151  # noqa: E402,F401  (registers RDNA3.5 kernels)
