"""Blockwise-causal attention with a KV cache, flash-style, for RDNA3.5.

DreamZero's sequence per step is ``[video tokens of this block | action
register]`` and every query may see the whole cached history plus keys up to
the end of its own block. Two properties make this cheaper than it looks:

* The mask is **monotone in the query index**. Query ``i``'s visible key set is
  always a prefix ``[0, limit[i]]``. Block causality, plain causality and full
  attention are all just different ``limit`` vectors, so the kernel carries one
  code path and the host decides the shape of the mask.
* ``limit`` is non-decreasing, so a k-tile past the *last* query in the current
  q-tile is invisible to every query in it and the loop simply stops. That is
  the same early-exit a triangular causal kernel gets, without hardcoding
  triangularity.

Softmax is the standard online (flash) formulation, so a step never
materialises the ``[q_len, kv_len]`` score matrix -- which at a 5-second
history is a few hundred MB that this part has no bandwidth to spare for.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from dexter.ops.gfx1151._common import RDNA3
from dexter.registry import Priority, register


@triton.jit
def _attention_kernel(
    q_ptr, k_ptr, v_ptr, limit_ptr, out_ptr,
    stride_qb, stride_qs, stride_qh,
    stride_kb, stride_ks, stride_kh,
    q_len, kv_len, n_heads, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // n_heads
    head = bh % n_heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < q_len

    q = tl.load(
        q_ptr + batch * stride_qb + offs_m[:, None] * stride_qs + head * stride_qh + offs_d[None, :],
        mask=mask_m[:, None], other=0.0,
    )

    # Highest key index any query in this tile may attend to. limit is
    # non-decreasing, so the last valid query bounds the whole tile.
    limit = tl.load(limit_ptr + offs_m, mask=mask_m, other=-1)
    tile_limit = tl.max(limit, 0)

    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    running_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)

    for n0 in range(0, kv_len, BLOCK_N):
        if n0 <= tile_limit:
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < kv_len

            k = tl.load(
                k_ptr + batch * stride_kb + offs_n[:, None] * stride_ks + head * stride_kh + offs_d[None, :],
                mask=mask_n[:, None], other=0.0,
            )
            v = tl.load(
                v_ptr + batch * stride_kb + offs_n[:, None] * stride_ks + head * stride_kh + offs_d[None, :],
                mask=mask_n[:, None], other=0.0,
            )

            scores = tl.dot(q, tl.trans(k)) * scale
            visible = (offs_n[None, :] <= limit[:, None]) & mask_n[None, :] & mask_m[:, None]
            scores = tl.where(visible, scores, float("-inf"))

            # Online softmax rescale.
            tile_max = tl.max(scores, 1)
            new_max = tl.maximum(running_max, tile_max)
            # A fully masked tile leaves new_max at -inf; clamp so exp stays finite.
            safe_max = tl.where(new_max == float("-inf"), 0.0, new_max)

            # The rescale factor for what is already accumulated. Before the
            # first visible tile there is nothing accumulated and running_max is
            # -inf, which must give a correction of *zero*, not exp(0 - max):
            # with strongly negative scores that expression overflows to +inf
            # and the 0 * inf in the accumulator update becomes NaN. Real
            # weights reach scores near -5e4 by layer 33, so this is reachable,
            # not theoretical.
            correction = tl.where(
                running_max == float("-inf"), 0.0, tl.exp(running_max - safe_max)
            )
            p = tl.exp(scores - safe_max[:, None])
            p = tl.where(visible, p, 0.0)

            acc = acc * correction[:, None] + tl.dot(p.to(v.dtype), v)
            running_sum = running_sum * correction + tl.sum(p, 1)
            running_max = new_max

    out = acc / tl.where(running_sum == 0.0, 1.0, running_sum)[:, None]
    tl.store(
        out_ptr + batch * stride_qb + offs_m[:, None] * stride_qs + head * stride_qh + offs_d[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )


def visibility_limits(
    q_len: int,
    cache_len: int,
    block_boundaries: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Per-query index of the last visible key.

    Cached history is always visible, so every limit is at least
    ``cache_len - 1``. Inside the current call, a query in the block starting at
    ``starts[i]`` sees up to the end of that block.
    """
    if block_boundaries is None:
        return torch.full((q_len,), cache_len + q_len - 1, dtype=torch.int32, device=device)

    # Computed on device: reading block_boundaries back to the host would both
    # stall the pipeline every step and make the step uncapturable as a graph.
    starts = block_boundaries.to(device=device, dtype=torch.int64)
    ends = torch.cat((starts[1:], torch.full((1,), q_len, dtype=torch.int64, device=device)))
    queries = torch.arange(q_len, device=device, dtype=torch.int64)
    block_of = torch.searchsorted(starts, queries, right=True) - 1
    return (cache_len + ends[block_of] - 1).to(torch.int32)


@register("attention", "blockwise_causal", name="triton_blockwise_causal_gfx1151",
          capability=RDNA3, priority=Priority.PERFORMANT)
def blockwise_causal_attention(q, k, v, kv_cache, cache_len, block_boundaries, softmax_scale):
    b, q_len, heads, head_dim = q.shape
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)

    if kv_cache is not None and cache_len > 0:
        k = torch.cat((kv_cache[0, :, :cache_len], k), dim=1)
        v = torch.cat((kv_cache[1, :, :cache_len], v), dim=1)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    kv_len = k.shape[1]
    out = torch.empty_like(q)
    limits = visibility_limits(q_len, cache_len, block_boundaries, q.device)

    block_m = min(max(triton.next_power_of_2(q_len), 16), 64)
    block_n = 64

    _attention_kernel[(triton.cdiv(q_len, block_m), b * heads)](
        q, k, v, limits, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        q_len, kv_len, heads, scale,
        BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=head_dim,
        num_warps=4, num_stages=2,
    )
    return out
