"""Weight-only quantised GEMM for RDNA3.5.

``y = x @ dequant(qweight)`` with x in bf16 and the weight in packed int4 or
int8. The dequantisation happens in registers between the global load and the
WMMA, so the only weight bytes that ever cross the memory bus are the
quantised ones. On a 231 GB/s part that is the entire optimisation: the math
is bf16 either way, and bf16 WMMA is not the bottleneck at DiT decode shapes.

Two decisions are worth stating because they are easy to get backwards:

**One M-block, not many.** A DiT denoising step has M ~ 125 query tokens. If
``BLOCK_M`` were the 16 of a single WMMA tile, eight M-blocks would each stream
the *same* weight tile from memory and the kernel would move 8x the bytes it
needs to. ``BLOCK_M`` is therefore sized to cover the whole of a small M in one
block, and the weights are read exactly once.

**BLOCK_K is the quantisation group.** With ``BLOCK_K == group_size`` each k
tile has exactly one scale and one zero per column. They are loaded once as
vectors and stay in registers for the tile, instead of being re-derived per
element.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from dexter.ops.gfx1151._common import RDNA3
from dexter.registry import Priority, register

# Cap on how wide a single M block gets. On most parts you would push this up
# so the weights are read exactly once, but gfx1151 has a 32 MB Infinity Cache
# (MALL) in front of LPDDR: a per-layer weight tensor is tens of MB, so the
# second M-block's re-read is largely a cache hit rather than a DRAM trip.
# Measured on Strix Halo, BLOCK_M 64 beats 128 by ~2x at M=83 -- the taller
# tile costs more in masked-out work and register pressure than the extra pass
# costs in bandwidth.
MAX_BLOCK_M = 64

# RDNA3.5 gives a workgroup 64 KB of LDS. Triton stages operand tiles through
# it, so the tile shape is not free -- pick it and the pipeline depth together
# or the launch fails at load time with OutOfResources.
LDS_BYTES = 64 * 1024


@triton.jit
def _w4a16_kernel(
    x_ptr, q_ptr, scale_ptr, zero_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_qk, stride_qn,
    stride_sg, stride_sn,
    GROUP_SIZE,
    HAS_ZERO: tl.constexpr, HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """int4, split-half packed.

    Byte row ``i`` of ``q_ptr`` holds k = i in its low nibble and k = i + K//2
    in its high nibble. One trip round the loop therefore consumes ``BLOCK_K``
    bytes and produces two ``BLOCK_K``-deep slices of the reduction, one from
    each half of K -- both contiguous in ``x``.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    half_k = K // 2
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, half_k, BLOCK_K):
        k_lo = k0 + offs_k
        k_hi = half_k + k0 + offs_k

        packed = tl.load(
            q_ptr + k_lo[:, None] * stride_qk + offs_n[None, :] * stride_qn,
            mask=(k_lo[:, None] < half_k) & mask_n[None, :],
            other=0,
        )

        # One scale/zero pair per column for this tile, in each half of K.
        g_lo = k0 // GROUP_SIZE
        g_hi = (half_k + k0) // GROUP_SIZE
        s_lo = tl.load(scale_ptr + g_lo * stride_sg + offs_n * stride_sn, mask=mask_n, other=0.0)
        s_hi = tl.load(scale_ptr + g_hi * stride_sg + offs_n * stride_sn, mask=mask_n, other=0.0)
        if HAS_ZERO:
            z_lo = tl.load(zero_ptr + g_lo * stride_sg + offs_n * stride_sn, mask=mask_n, other=0.0)
            z_hi = tl.load(zero_ptr + g_hi * stride_sg + offs_n * stride_sn, mask=mask_n, other=0.0)
        else:
            z_lo = tl.full((BLOCK_N,), 8.0 + 128.0, tl.bfloat16)
            z_hi = z_lo

        # Build the bf16 value by bit pattern instead of converting.
        # bf16 is s(1) e(8) m(7); the word 0x4300 is exponent 134, i.e. 2^7 =
        # 128 with an empty mantissa. Integers 128..255 are exactly
        # representable at that exponent, so OR-ing a nibble into the mantissa
        # yields exactly 128 + n and an integer->float convert becomes an OR.
        # The +128 bias is folded into the zero point on the host (see
        # BF16_MAGIC_BIAS), so nothing here has to subtract it back out.
        w_lo = ((tl.cast(packed & 0x0F, tl.uint16) | 0x4300).to(tl.bfloat16, bitcast=True) - z_lo[None, :]) * s_lo[None, :]
        w_hi = ((tl.cast((packed >> 4) & 0x0F, tl.uint16) | 0x4300).to(tl.bfloat16, bitcast=True) - z_hi[None, :]) * s_hi[None, :]

        x_lo = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_lo[None, :] * stride_xk,
            mask=mask_m[:, None] & (k_lo[None, :] < half_k), other=0.0,
        )
        x_hi = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_hi[None, :] * stride_xk,
            mask=mask_m[:, None] & (k_hi[None, :] < K), other=0.0,
        )

        acc = tl.dot(x_lo, w_lo.to(x_lo.dtype), acc)
        acc = tl.dot(x_hi, w_hi.to(x_hi.dtype), acc)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]

    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _w8a16_kernel(
    x_ptr, q_ptr, scale_ptr, zero_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_qk, stride_qn,
    stride_sg, stride_sn,
    GROUP_SIZE,
    HAS_ZERO: tl.constexpr, HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """int8. Same structure as int4 without the nibble split."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        mask_k = k < K

        q = tl.load(
            q_ptr + k[:, None] * stride_qk + offs_n[None, :] * stride_qn,
            mask=mask_k[:, None] & mask_n[None, :], other=0,
        )
        g = k0 // GROUP_SIZE
        s = tl.load(scale_ptr + g * stride_sg + offs_n * stride_sn, mask=mask_n, other=0.0)
        if HAS_ZERO:
            z = tl.load(zero_ptr + g * stride_sg + offs_n * stride_sn, mask=mask_n, other=0.0)
        else:
            z = tl.full((BLOCK_N,), 128.0, tl.float32)

        w = (q.to(tl.float32) - z[None, :].to(tl.float32)) * s[None, :].to(tl.float32)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0,
        )
        acc = tl.dot(x, w.to(x.dtype), acc)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]

    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _block_m(m: int) -> int:
    """Smallest power-of-two block that keeps M in one block, up to the cap."""
    return min(max(triton.next_power_of_2(m), 16), MAX_BLOCK_M)


def _pick_tile(m: int, n: int, group_size: int, act_tiles: int) -> tuple[int, int, int, int]:
    """Choose ``(BLOCK_M, BLOCK_N, BLOCK_K, num_stages)`` inside the LDS budget.

    ``BLOCK_M`` is fixed first, because keeping M in a single block is what
    makes the kernel read the weights once; everything else gives way to it.
    ``BLOCK_K`` then shrinks until the staged tiles fit. It must divide
    ``group_size`` so a k-tile still maps to exactly one quantisation group.

    ``act_tiles`` is how many activation tiles are live per k-step: 2 for the
    split-half int4 kernel, 1 for int8.
    """
    block_m = _block_m(m)
    block_n = 64 if n <= 8192 else 128

    # Pipeline depth is chosen before k-tile depth. A deeper k-tile only cuts
    # loop overhead, while a second stage overlaps the next tile's loads with
    # this tile's dequantise-and-dot -- and that inner loop is dominated by the
    # dequantise, not the loads. Searching block_k first picks the deepest tile
    # that fits, which then leaves no LDS for a second stage; measured on
    # gfx1151 that choice costs ~2.4x.
    for stages in (2, 1):
        for block_k in (group_size, 64, 32, 16):
            if block_k > group_size or group_size % block_k:
                continue
            act = act_tiles * block_m * block_k * 2  # bf16 activations
            weights = block_k * block_n              # uint8, packed
            if (act + weights) * stages <= LDS_BYTES:
                return block_m, block_n, block_k, stages

    raise RuntimeError(f"no tile fits {LDS_BYTES} B of LDS for M={m}, N={n}, G={group_size}")


def _launch(kernel, x, qweight, scales, zeros, bias, group_size, k_dim, act_tiles):
    *lead, k_in = x.shape
    if k_in != k_dim:
        raise ValueError(f"x has K={k_in} but the packed weight has K={k_dim}")
    n = qweight.shape[1]

    x2 = x.reshape(-1, k_in)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)

    block_m, block_n, block_k, stages = _pick_tile(m, n, group_size, act_tiles)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))

    kernel[grid](
        x2, qweight, scales, zeros, bias, out,
        m, n, k_dim,
        x2.stride(0), x2.stride(1),
        qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        group_size,
        HAS_ZERO=zeros is not None, HAS_BIAS=bias is not None,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=4, num_stages=stages,
    )
    return out.view(*lead, n)


@register("gemm", "w4a16", name="triton_w4a16_gfx1151",
          capability=RDNA3, priority=Priority.PERFORMANT)
def w4a16(x, qweight, scales, zeros, bias, group_size):
    return _launch(_w4a16_kernel, x, qweight, scales, zeros, bias, group_size,
                   k_dim=qweight.shape[0] * 2, act_tiles=2)


@register("gemm", "w8a16", name="triton_w8a16_gfx1151",
          capability=RDNA3, priority=Priority.PERFORMANT)
def w8a16(x, qweight, scales, zeros, bias, group_size):
    return _launch(_w8a16_kernel, x, qweight, scales, zeros, bias, group_size,
                   k_dim=qweight.shape[0], act_tiles=1)
