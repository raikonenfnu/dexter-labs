"""Wan's FFN with GELU folded into the first GEMM's epilogue.

``Linear(C -> 4C) -> GELU(tanh) -> Linear(4C -> C)``. Unfused, the intermediate
``[B, L, ffn_dim]`` tensor is written once by the GEMM, read by GELU, written
again, then read by the second GEMM. At ffn_dim 14336 that intermediate is
larger than the block activation itself, so folding GELU into the epilogue
removes the single widest elementwise round trip in the block.

The two GEMMs stay separate. Fusing them would need the whole ``ffn_dim``
reduction resident, and 14336 bf16 columns do not fit in 64 KB of LDS.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from dexter.ops.gfx1151._common import RDNA3
from dexter.registry import Priority, register

# sqrt(2/pi). A Triton kernel may only close over globals built by
# tl.constexpr(...) -- a constexpr *annotation* is not the same thing.
_SQRT_2_OVER_PI = tl.constexpr(0.7978845608028654)


@triton.jit
def _gelu_tanh(x):
    """GELU, tanh approximation -- matches ``F.gelu(approximate='tanh')``."""
    inner = _SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)
    # tanh(z) via the numerically stable exp form; Triton has no tanh intrinsic
    # that lowers well here, and this keeps everything in the f32 accumulator.
    e = tl.exp(2.0 * inner)
    return 0.5 * x * (1.0 + (e - 1.0) / (e + 1.0))


@triton.jit
def _gemm_gelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk, stride_wn, stride_wk,
    HAS_BIAS: tl.constexpr, APPLY_GELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """``out = act(x @ w.T + b)`` for an ``nn.Linear``-shaped ``[N, K]`` weight."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < K
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        # w is [N, K]; transpose into [K, N] tiles for the dot.
        w = tl.load(w_ptr + offs_n[None, :] * stride_wn + k[:, None] * stride_wk,
                    mask=mask_n[None, :] & mask_k[:, None], other=0.0)
        acc = tl.dot(x, w, acc)

    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]
    if APPLY_GELU:
        acc = _gelu_tanh(acc)

    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :],
             acc.to(out_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


def _linear(x2: torch.Tensor, weight: torch.Tensor, bias, apply_gelu: bool) -> torch.Tensor:
    m, k = x2.shape
    n = weight.shape[0]
    out = torch.empty((m, n), device=x2.device, dtype=x2.dtype)

    block_m = min(max(triton.next_power_of_2(m), 16), 128)
    block_n = 64 if n <= 4096 else 128
    block_k = 64

    _gemm_gelu_kernel[(triton.cdiv(m, block_m), triton.cdiv(n, block_n))](
        x2, weight, bias, out,
        m, n, k,
        x2.stride(0), x2.stride(1), weight.stride(0), weight.stride(1),
        HAS_BIAS=bias is not None, APPLY_GELU=apply_gelu,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=4, num_stages=2,
    )
    return out


@register("gemm", "dense", name="hipblaslt_dense_gfx1151",
          capability=RDNA3, priority=Priority.PERFORMANT)
def hipblaslt_dense(x, weight, bias):
    """Dense bf16 GEMM via hipBLASLt.

    Measured on gfx1151 across M = 1..913 at DiT shapes, hipBLASLt beat the
    hand-written Triton GEMM below by a consistent ~1.4x. Assembly-tuned vendor
    BLAS is simply the right answer for a plain dense GEMM here, and pretending
    otherwise would cost the engine 40% on its single hottest op. The Triton
    path stays registered one band lower as the portable fallback and as the
    body the fused epilogues are built from.
    """
    return torch.nn.functional.linear(x, weight, bias)


@register("gemm", "ffn_gelu", name="hipblaslt_ffn_gelu_gfx1151",
          capability=RDNA3, priority=Priority.PERFORMANT)
def hipblaslt_ffn_gelu(x, up, up_bias, down, down_bias):
    """Same finding as ``hipblaslt_dense``: at Wan's ffn_dim the two GEMMs
    dominate so thoroughly that saving the GELU round trip does not pay for a
    slower GEMM. Fusing wins on the narrow elementwise ops, not on these."""
    hidden = torch.nn.functional.gelu(
        torch.nn.functional.linear(x, up, up_bias), approximate="tanh"
    )
    return torch.nn.functional.linear(hidden, down, down_bias)


@register("gemm", "dense", name="triton_dense_gfx1151",
          capability=RDNA3, priority=Priority.PORTABLE)
def dense(x, weight, bias):
    *lead, k = x.shape
    out = _linear(x.reshape(-1, k).contiguous(), weight, bias, apply_gelu=False)
    return out.view(*lead, weight.shape[0])


@register("gemm", "ffn_gelu", name="triton_ffn_gelu_gfx1151",
          capability=RDNA3, priority=Priority.PORTABLE)
def ffn_gelu(x, up, up_bias, down, down_bias):
    *lead, k = x.shape
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    hidden = _linear(x2, up, up_bias, apply_gelu=True)
    out = _linear(hidden, down, down_bias, apply_gelu=False)
    return out.view(*lead, down.shape[0])
