"""QK RMSNorm and rotary, fused.

Wan normalises q and k then immediately rotates them. Separately those are four
passes over the projections; here they are one. Each program owns a single
``(token, head)`` pair, so the RMS reduction is over ``head_dim`` -- 128 values,
comfortably one wave's worth at wave32.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from dexter.ops.gfx1151._common import RDNA3
from dexter.registry import Priority, register


@triton.jit
def _qk_norm_rope_kernel(
    q_ptr, k_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr,
    q_out_ptr, k_out_ptr,
    n_heads, seq_len, head_dim, eps,
    HALF: tl.constexpr, BLOCK: tl.constexpr,
):
    """One program per (batch*token, head).

    RoPE pairs element ``i`` with ``i + head_dim/2``, so the kernel loads the
    two halves separately rather than loading the row and shuffling it.
    """
    pid = tl.program_id(0)
    token = (pid // n_heads) % seq_len  # position within the sequence, not the batch

    half = tl.arange(0, BLOCK)
    mask = half < HALF

    base = pid * head_dim
    q_lo = tl.load(q_ptr + base + half, mask=mask, other=0.0).to(tl.float32)
    q_hi = tl.load(q_ptr + base + HALF + half, mask=mask, other=0.0).to(tl.float32)
    k_lo = tl.load(k_ptr + base + half, mask=mask, other=0.0).to(tl.float32)
    k_hi = tl.load(k_ptr + base + HALF + half, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm over the whole head_dim, i.e. both halves together.
    q_inv = tl.rsqrt((tl.sum(q_lo * q_lo, 0) + tl.sum(q_hi * q_hi, 0)) / head_dim + eps)
    k_inv = tl.rsqrt((tl.sum(k_lo * k_lo, 0) + tl.sum(k_hi * k_hi, 0)) / head_dim + eps)

    qw_lo = tl.load(qw_ptr + half, mask=mask, other=0.0).to(tl.float32)
    qw_hi = tl.load(qw_ptr + HALF + half, mask=mask, other=0.0).to(tl.float32)
    kw_lo = tl.load(kw_ptr + half, mask=mask, other=0.0).to(tl.float32)
    kw_hi = tl.load(kw_ptr + HALF + half, mask=mask, other=0.0).to(tl.float32)

    q_lo, q_hi = q_lo * q_inv * qw_lo, q_hi * q_inv * qw_hi
    k_lo, k_hi = k_lo * k_inv * kw_lo, k_hi * k_inv * kw_hi

    cos = tl.load(cos_ptr + token * HALF + half, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + token * HALF + half, mask=mask, other=0.0).to(tl.float32)

    tl.store(q_out_ptr + base + half, (q_lo * cos - q_hi * sin).to(q_out_ptr.dtype.element_ty), mask=mask)
    tl.store(q_out_ptr + base + HALF + half, (q_hi * cos + q_lo * sin).to(q_out_ptr.dtype.element_ty), mask=mask)
    tl.store(k_out_ptr + base + half, (k_lo * cos - k_hi * sin).to(k_out_ptr.dtype.element_ty), mask=mask)
    tl.store(k_out_ptr + base + HALF + half, (k_hi * cos + k_lo * sin).to(k_out_ptr.dtype.element_ty), mask=mask)


@register("rope", "qk_norm_rope", name="triton_qk_norm_rope_gfx1151",
          capability=RDNA3, priority=Priority.PERFORMANT)
def qk_norm_rope(q, k, q_weight, k_weight, cos, sin, eps):
    b, seq_len, heads, head_dim = q.shape
    q, k = q.contiguous(), k.contiguous()
    q_out, k_out = torch.empty_like(q), torch.empty_like(k)

    half = head_dim // 2
    _qk_norm_rope_kernel[(b * seq_len * heads,)](
        q, k, q_weight, k_weight, cos.contiguous(), sin.contiguous(),
        q_out, k_out,
        heads, seq_len, head_dim, eps,
        HALF=half, BLOCK=triton.next_power_of_2(half),
        num_warps=2,
    )
    return q_out, k_out
