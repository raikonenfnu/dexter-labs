"""Weight-only quantisation to the layout in :mod:`dexter.ops.reference.gemm`.

Why weight-only, and why this is the first thing the engine does on RDNA3.5:

A DreamZero step runs the DiT once per denoising step over ~125 query tokens.
At that shape the arithmetic intensity is roughly one MAC per weight *byte*
loaded, so the step time is ``weight_bytes / achievable_bandwidth`` and nothing
else. On a 256 GB/s Strix Halo part a bf16 5B backbone cannot go faster than
~40 ms per denoising step no matter how good the math kernels are. int4 moves
that floor to ~10 ms. Activations stay bf16 because they are not the problem
and quantising them would cost accuracy for no bandwidth.

This is also where the MI300X blog post does *not* port: its FP8 GEMM win comes
from a native FP8 matrix path that RDNA3.5's WMMA does not have. The saving is
recovered by quantising the *storage* and keeping the *math* in bf16.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["QuantizedWeight", "quantize", "DEFAULT_GROUP_SIZE", "BF16_MAGIC_BIAS"]

DEFAULT_GROUP_SIZE = 128

# The int4 kernel materialises a nibble as bf16 by OR-ing it into the mantissa
# of 0x4300, which yields 128 + n rather than n. Rather than subtract 128 per
# element on the device, the bias is added to the zero point once, here.
BF16_MAGIC_BIAS = 128.0


@dataclass
class QuantizedWeight:
    """A packed weight plus everything needed to dequantise it in-kernel."""

    qweight: torch.Tensor  # [K//2, N] uint8 for 4-bit, [K, N] uint8 for 8-bit
    scales: torch.Tensor  # [K//G, N]
    zeros: torch.Tensor | None  # [K//G, N], None if symmetric
    bias: torch.Tensor | None
    group_size: int
    bits: int
    shape: tuple[int, int]  # (K, N) of the logical weight

    @property
    def nbytes(self) -> int:
        total = self.qweight.numel() * self.qweight.element_size()
        total += self.scales.numel() * self.scales.element_size()
        if self.zeros is not None:
            total += self.zeros.numel() * self.zeros.element_size()
        return total

    def to(self, device: torch.device | str) -> "QuantizedWeight":
        move = lambda t: None if t is None else t.to(device)  # noqa: E731
        return QuantizedWeight(
            move(self.qweight), move(self.scales), move(self.zeros), move(self.bias),
            self.group_size, self.bits, self.shape,
        )


def _pack_nibbles(q: torch.Tensor) -> torch.Tensor:
    """``[K, N]`` uint8 in 0..15 -> ``[K//2, N]``, split-half.

    Byte ``[i, n]`` carries ``k = i`` in the low nibble and ``k = i + K//2`` in
    the high nibble -- the two halves of K, not adjacent k. That keeps the
    activation loads in the GEMM contiguous: unpacking one byte tile yields two
    weight tiles whose k-ranges are each contiguous, so ``x`` is read as two
    plain slices instead of a stride-2 gather.
    """
    half = q.shape[0] // 2
    return (q[:half] | (q[half:] << 4)).contiguous()


def quantize(
    weight: torch.Tensor,
    *,
    bits: int = 4,
    group_size: int = DEFAULT_GROUP_SIZE,
    bias: torch.Tensor | None = None,
    symmetric: bool = False,
) -> QuantizedWeight:
    """Quantise an ``nn.Linear``-style ``[N, K]`` weight into the packed layout.

    Asymmetric (the default) fits a per-group min/max, which costs one extra
    tensor of zero points and is worth it: DiT weights are not centred, and a
    symmetric grid throws away roughly a bit of the four on a skewed group.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2D weight, got shape {tuple(weight.shape)}")
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")

    w = weight.t().contiguous().float()  # [K, N] -- K-major, as the kernel wants
    k, n = w.shape
    if k % group_size:
        raise ValueError(f"K={k} must be a multiple of group_size={group_size}")

    groups = w.view(k // group_size, group_size, n)
    qmax = float((1 << bits) - 1)

    if symmetric:
        # Grid centred on the midpoint; only a scale is stored.
        amax = groups.abs().amax(dim=1).clamp(min=1e-8)
        scales = 2 * amax / qmax
        zeros = None
        offset = float(1 << (bits - 1))
    else:
        lo = groups.amin(dim=1)
        hi = groups.amax(dim=1)
        scales = ((hi - lo) / qmax).clamp(min=1e-8)
        zeros = (-lo / scales).round().clamp(0, qmax)
        offset = zeros.unsqueeze(1)

    q = (groups / scales.unsqueeze(1) + offset).round().clamp(0, qmax)
    q = q.view(k, n).to(torch.uint8)

    if bits == 4 and zeros is not None:
        # Pre-bias the zero point for the kernel's bit-pattern dequant. The
        # symmetric path keeps zeros as None and the kernel folds the bias into
        # its constant, so no tensor is materialised just to carry a constant.
        zeros = zeros + BF16_MAGIC_BIAS

    dtype = weight.dtype if weight.dtype.is_floating_point else torch.bfloat16
    return QuantizedWeight(
        qweight=_pack_nibbles(q) if bits == 4 else q.contiguous(),
        scales=scales.to(dtype).contiguous(),
        zeros=None if zeros is None else zeros.to(dtype).contiguous(),
        bias=bias,
        group_size=group_size,
        bits=bits,
        shape=(k, n),
    )
