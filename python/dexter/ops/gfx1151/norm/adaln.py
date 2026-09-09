"""Fused adaptive layer norm and gated residual.

A Wan DiT block touches its ``[B, L, C]`` activation six times per layer in
elementwise work: two LayerNorms, two modulations, two gated residual adds.
Unfused that is twelve trips over the activation. These two kernels cut it to
four -- one read and one write per fusion -- which on a bandwidth-bound part is
the whole win. The math is unchanged; only the traffic is.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from dexter.ops.gfx1151._common import RDNA3, block_size, row_warps
from dexter.registry import Priority, register


@triton.jit
def _adaln_kernel(
    x_ptr, scale_ptr, shift_ptr, out_ptr,
    x_row_stride, mod_row_stride,
    n_cols, eps,
    BLOCK: tl.constexpr,
):
    """One program per ``[B, L]`` row: normalise, then scale and shift.

    ``mod_row_stride`` is 0 when scale/shift are broadcast over L, which is the
    common case -- Wan's modulation is per-frame, not per-token.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols
    centred = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centred * centred, axis=0) / n_cols
    normed = centred * tl.rsqrt(var + eps)

    mod_off = row * mod_row_stride + cols
    scale = tl.load(scale_ptr + mod_off, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + mod_off, mask=mask, other=0.0).to(tl.float32)

    out = normed * (1.0 + scale) + shift
    tl.store(out_ptr + row * x_row_stride + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _gated_residual_kernel(
    x_ptr, y_ptr, gate_ptr, out_ptr,
    row_stride, gate_row_stride,
    n_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0)
    y = tl.load(y_ptr + row * row_stride + cols, mask=mask, other=0.0)
    gate = tl.load(gate_ptr + row * gate_row_stride + cols, mask=mask, other=0.0)

    out = x.to(tl.float32) + y.to(tl.float32) * gate.to(tl.float32)
    tl.store(out_ptr + row * row_stride + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _layer_norm_kernel(x_ptr, out_ptr, row_stride, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    centred = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centred * centred, axis=0) / n_cols
    out = centred * tl.rsqrt(var + eps)
    tl.store(out_ptr + row * row_stride + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rms_norm_kernel(x_ptr, w_ptr, out_ptr, row_stride, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    tl.store(out_ptr + row * row_stride + cols, (x * inv * w).to(out_ptr.dtype.element_ty), mask=mask)


def _broadcast_row_stride(mod: torch.Tensor, rows: int, n_cols: int) -> tuple[torch.Tensor, int]:
    """Flatten a modulation tensor to rows, using stride 0 when it broadcasts."""
    flat = mod.reshape(-1, n_cols)
    if flat.shape[0] == rows:
        return flat.contiguous(), n_cols
    if flat.shape[0] == 1:
        return flat.contiguous(), 0
    # [B, 1, C] against [B, L, C]: repeat over L is a stride-0 read per batch,
    # which this row-major launch cannot express, so materialise it.
    return mod.expand(-1, rows // flat.shape[0], -1).reshape(rows, n_cols).contiguous(), n_cols


@register("norm", "adaln", name="triton_adaln_gfx1151",
          capability=RDNA3, priority=Priority.PERFORMANT)
def adaln_modulate(x, scale, shift, eps):
    n_cols = x.shape[-1]
    flat = x.reshape(-1, n_cols)
    rows = flat.shape[0]
    out = torch.empty_like(flat)

    scale_f, scale_stride = _broadcast_row_stride(scale, rows, n_cols)
    shift_f, shift_stride = _broadcast_row_stride(shift, rows, n_cols)
    if scale_stride != shift_stride:  # keep one stride argument honest
        scale_f, scale_stride = scale_f.expand(rows, n_cols).contiguous(), n_cols
        shift_f, shift_stride = shift_f.expand(rows, n_cols).contiguous(), n_cols

    _adaln_kernel[(rows,)](
        flat, scale_f, shift_f, out,
        flat.stride(0), scale_stride,
        n_cols, eps,
        BLOCK=block_size(n_cols), num_warps=row_warps(n_cols),
    )
    return out.view_as(x)


@register("norm", "gated_residual", name="triton_gated_residual_gfx1151",
          capability=RDNA3, priority=Priority.PERFORMANT)
def gated_residual(x, y, gate):
    n_cols = x.shape[-1]
    flat_x = x.reshape(-1, n_cols)
    flat_y = y.reshape(-1, n_cols)
    rows = flat_x.shape[0]
    out = torch.empty_like(flat_x)

    gate_f, gate_stride = _broadcast_row_stride(gate, rows, n_cols)

    _gated_residual_kernel[(rows,)](
        flat_x, flat_y, gate_f, out,
        flat_x.stride(0), gate_stride,
        n_cols,
        BLOCK=block_size(n_cols), num_warps=row_warps(n_cols),
    )
    return out.view_as(x)


@register("norm", "layer_norm", name="triton_layer_norm_gfx1151",
          capability=RDNA3, priority=Priority.PERFORMANT)
def layer_norm(x, eps):
    n_cols = x.shape[-1]
    flat = x.reshape(-1, n_cols)
    out = torch.empty_like(flat)
    _layer_norm_kernel[(flat.shape[0],)](
        flat, out, flat.stride(0), n_cols, eps,
        BLOCK=block_size(n_cols), num_warps=row_warps(n_cols),
    )
    return out.view_as(x)


@register("norm", "rms_norm", name="triton_rms_norm_gfx1151",
          capability=RDNA3, priority=Priority.PERFORMANT)
def rms_norm(x, weight, eps):
    n_cols = x.shape[-1]
    flat = x.reshape(-1, n_cols)
    out = torch.empty_like(flat)
    _rms_norm_kernel[(flat.shape[0],)](
        flat, weight, out, flat.stride(0), n_cols, eps,
        BLOCK=block_size(n_cols), num_warps=row_warps(n_cols),
    )
    return out.view_as(x)
