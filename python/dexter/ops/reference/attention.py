"""Reference blockwise-causal attention with a KV cache.

The mask is the part worth reading. A DreamZero step sees one sequence

    [ video tokens of this block | action register (actions + state) ]

and every query may attend to (a) everything already in the KV cache, which is
strictly older blocks, and (b) keys in its own block or an earlier block of the
current call. Within a block attention is dense, not triangular -- the block is
one timestep, so its tokens are mutually visible.
"""

from __future__ import annotations

import math

import torch

from dexter.registry import Priority, register


def block_causal_mask(
    q_len: int,
    cache_len: int,
    block_boundaries: torch.Tensor | None,
    device: torch.device,
    causal: bool = False,
) -> torch.Tensor:
    """``[q_len, cache_len + q_len]`` bool mask; True means "may attend"."""
    mask = torch.zeros(q_len, cache_len + q_len, dtype=torch.bool, device=device)
    mask[:, :cache_len] = True  # history is always visible

    if causal:
        rows = torch.arange(q_len, device=device)[:, None]
        cols = torch.arange(q_len, device=device)[None, :]
        mask[:, cache_len:] = cols <= rows
        return mask

    if block_boundaries is None:
        mask[:, cache_len:] = True
        return mask

    starts = block_boundaries.tolist() + [q_len]
    for i in range(len(starts) - 1):
        lo, hi = starts[i], starts[i + 1]
        mask[lo:hi, cache_len : cache_len + hi] = True
    return mask


@register("attention", "blockwise_causal", name="torch_blockwise_causal", priority=Priority.REFERENCE)
def blockwise_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor | None,
    cache_len: int,
    block_boundaries: torch.Tensor | None,
    softmax_scale: float | None,
    causal: bool = False,
) -> torch.Tensor:
    """q, k, v are ``[B, L, H, D]``. ``kv_cache`` is ``[2, B, S, H, D]``."""
    b, q_len, h, d = q.shape
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)

    if kv_cache is not None and cache_len > 0:
        k = torch.cat((kv_cache[0, :, :cache_len], k), dim=1)
        v = torch.cat((kv_cache[1, :, :cache_len], v), dim=1)

    # [B, H, L, D] for the matmuls.
    qt, kt, vt = (t.transpose(1, 2) for t in (q, k, v))
    logits = torch.matmul(qt.float(), kt.float().transpose(-1, -2)) * scale

    mask = block_causal_mask(q_len, cache_len, block_boundaries, q.device, causal)
    logits = logits.masked_fill(~mask, float("-inf"))

    out = torch.matmul(torch.softmax(logits, dim=-1), vt.float())
    return out.transpose(1, 2).to(q.dtype)
