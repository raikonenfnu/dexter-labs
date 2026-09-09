"""QK norm and rotary, fused.

Wan normalises q and k then immediately rotates them, so separately these are
four passes over the projections; here they are one.

The subtlety is *what* the norm reduces over. ``WanRMSNorm(dim)`` is applied to
the ``[B, L, dim]`` projection **before** it is reshaped into heads, so the RMS
denominator spans every head, not each head separately, and the learned weight
is ``dim`` long rather than ``head_dim``. Getting that wrong still runs and
still trains -- it just quietly computes a different model, which is why the
reference in ``ops/reference`` does the same reduction and the tests compare
against it.

One program therefore owns a whole token: a ``[heads, head_dim/2]`` tile, with
the reduction over the full plane and the rotation applied per head.
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
    seq_len, dim, eps,
    HEADS: tl.constexpr, HALF: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """One program per (batch, token).

    Wan pairs *adjacent* elements -- ``x.reshape(..., head_dim/2, 2)`` -- so
    element ``2j`` rotates against ``2j+1``. This is not the split-half
    convention (``i`` against ``i + head_dim/2``) that most LLM RoPE uses, and
    the two are not interchangeable: the wrong one still runs, still produces
    finite output, and quietly destroys spatial structure. The two halves of a
    pair are therefore loaded as stride-2 tiles.
    """
    pid = tl.program_id(0)
    token = pid % seq_len

    heads = tl.arange(0, BLOCK_H)[:, None]
    half = tl.arange(0, BLOCK_D)[None, :]
    mask = (heads < HEADS) & (half < HALF)

    head_dim = 2 * HALF
    # Adjacent pairing: element 2j and 2j+1 of each head.
    lo_off = heads * head_dim + 2 * half
    hi_off = lo_off + 1
    base = pid * dim

    q_lo = tl.load(q_ptr + base + lo_off, mask=mask, other=0.0).to(tl.float32)
    q_hi = tl.load(q_ptr + base + hi_off, mask=mask, other=0.0).to(tl.float32)
    k_lo = tl.load(k_ptr + base + lo_off, mask=mask, other=0.0).to(tl.float32)
    k_hi = tl.load(k_ptr + base + hi_off, mask=mask, other=0.0).to(tl.float32)

    # RMS over the whole [heads, head_dim] plane -- all heads together.
    q_inv = tl.rsqrt((tl.sum(q_lo * q_lo) + tl.sum(q_hi * q_hi)) / dim + eps)
    k_inv = tl.rsqrt((tl.sum(k_lo * k_lo) + tl.sum(k_hi * k_hi)) / dim + eps)

    qw_lo = tl.load(qw_ptr + lo_off, mask=mask, other=0.0).to(tl.float32)
    qw_hi = tl.load(qw_ptr + hi_off, mask=mask, other=0.0).to(tl.float32)
    kw_lo = tl.load(kw_ptr + lo_off, mask=mask, other=0.0).to(tl.float32)
    kw_hi = tl.load(kw_ptr + hi_off, mask=mask, other=0.0).to(tl.float32)

    q_lo, q_hi = q_lo * q_inv * qw_lo, q_hi * q_inv * qw_hi
    k_lo, k_hi = k_lo * k_inv * kw_lo, k_hi * k_inv * kw_hi

    # Rotary is per head: the same cos/sin row serves every head.
    cos = tl.load(cos_ptr + token * HALF + half, mask=half < HALF, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + token * HALF + half, mask=half < HALF, other=0.0).to(tl.float32)

    tl.store(q_out_ptr + base + lo_off, (q_lo * cos - q_hi * sin).to(q_out_ptr.dtype.element_ty), mask=mask)
    tl.store(q_out_ptr + base + hi_off, (q_hi * cos + q_lo * sin).to(q_out_ptr.dtype.element_ty), mask=mask)
    tl.store(k_out_ptr + base + lo_off, (k_lo * cos - k_hi * sin).to(k_out_ptr.dtype.element_ty), mask=mask)
    tl.store(k_out_ptr + base + hi_off, (k_hi * cos + k_lo * sin).to(k_out_ptr.dtype.element_ty), mask=mask)


@register("rope", "qk_norm_rope", name="triton_qk_norm_rope_gfx1151",
          capability=RDNA3, priority=Priority.PERFORMANT)
def qk_norm_rope(q, k, q_weight, k_weight, cos, sin, eps):
    b, seq_len, heads, head_dim = q.shape
    q, k = q.contiguous(), k.contiguous()
    q_out, k_out = torch.empty_like(q), torch.empty_like(k)

    half = head_dim // 2
    _qk_norm_rope_kernel[(b * seq_len,)](
        q, k, q_weight, k_weight, cos.contiguous(), sin.contiguous(),
        q_out, k_out,
        seq_len, heads * head_dim, eps,
        HEADS=heads, HALF=half,
        BLOCK_H=triton.next_power_of_2(heads), BLOCK_D=triton.next_power_of_2(half),
        num_warps=4,
    )
    return q_out, k_out
