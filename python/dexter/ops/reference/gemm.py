"""Reference GEMMs, including the weight-only quantised formats.

Quantisation layout, shared by the reference and the gfx1151 kernels. Packed
weights are stored **K-major** (``[K, N]``-shaped), not in ``nn.Linear``'s
``[N, K]``, because that is the orientation the GEMM's B operand wants: a
kernel walking the K reduction reads consecutive ``n`` from consecutive bytes.

    w4a16   qweight [K // 2, N] uint8 -- byte ``[i, n]`` holds k = i in the low
            nibble and k = i + K//2 in the high nibble ("split-half"). Values
            are unsigned, 0..15. Splitting by half rather than by adjacent k
            means each unpacked nibble plane covers a contiguous k-range, so
            the GEMM reads ``x`` as two slices instead of a strided gather.
    w8a16   qweight [K, N] uint8, values 0..255.

    scales  [K // group_size, N], in the activation dtype
    zeros   [K // group_size, N], same dtype, or None for a symmetric
            quantiser (zero point fixed at the midpoint of the range).

    dequant: w[k, n] = (qweight[k, n] - zero[k // G, n]) * scale[k // G, n]
    output:  y = x @ w  (+ bias)

For 4-bit, a stored zero point carries an extra ``+128`` (``BF16_MAGIC_BIAS``).
The kernel builds a nibble's bf16 value by bit pattern, which lands on
``128 + n``; biasing the zero point once on the host cancels that for free,
instead of a subtract per element on the device.

Groups run along K, the reduction axis, so one scale load serves ``group_size``
weights and stays in registers for the whole k-tile.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dexter.registry import Priority, register


@register("gemm", "dense", name="torch_mm", priority=Priority.REFERENCE)
def dense(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    return F.linear(x, weight, bias)


def unpack_int4(qweight: torch.Tensor) -> torch.Tensor:
    """``[K//2, N]`` uint8 -> ``[K, N]`` uint8 with values in 0..15."""
    low = qweight & 0x0F
    high = (qweight >> 4) & 0x0F
    return torch.cat((low, high), dim=0)


def dequantize(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    group_size: int,
    bits: int,
) -> torch.Tensor:
    """Reconstruct the ``[K, N]`` full-precision weight.

    Reference only. A real kernel never materialises this -- not materialising
    it is the entire bandwidth saving.
    """
    from dexter.quant import BF16_MAGIC_BIAS

    q = unpack_int4(qweight) if bits == 4 else qweight
    k, n = q.shape
    q = q.to(scales.dtype).view(k // group_size, group_size, n)
    if zeros is not None:
        # Stored 4-bit zero points are magic-biased; undo it to get the real one.
        zero = zeros.unsqueeze(1) - (BF16_MAGIC_BIAS if bits == 4 else 0.0)
    else:
        zero = float(1 << (bits - 1))
    return ((q - zero) * scales.unsqueeze(1)).view(k, n)


def _quantized_matmul(x, qweight, scales, zeros, bias, group_size, bits):
    w = dequantize(qweight, scales, zeros, group_size, bits).to(x.dtype)
    out = x @ w
    return out + bias if bias is not None else out


@register("gemm", "w4a16", name="torch_w4a16", priority=Priority.REFERENCE)
def w4a16(x, qweight, scales, zeros, bias, group_size):
    return _quantized_matmul(x, qweight, scales, zeros, bias, group_size, bits=4)


@register("gemm", "w8a16", name="torch_w8a16", priority=Priority.REFERENCE)
def w8a16(x, qweight, scales, zeros, bias, group_size):
    return _quantized_matmul(x, qweight, scales, zeros, bias, group_size, bits=8)


@register("gemm", "ffn_gelu", name="torch_ffn_gelu", priority=Priority.REFERENCE)
def ffn_gelu(
    x: torch.Tensor,
    up: torch.Tensor,
    up_bias: torch.Tensor | None,
    down: torch.Tensor,
    down_bias: torch.Tensor | None,
) -> torch.Tensor:
    return F.linear(F.gelu(F.linear(x, up, up_bias), approximate="tanh"), down, down_bias)
