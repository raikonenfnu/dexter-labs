"""Per-kernel microbenchmarks: each fused op against its unfused reference."""

from __future__ import annotations

import torch

from dexter import ops
from dexter.bench.harness import measure, measure_kernel
from dexter.quant import quantize
from dexter.registry import _REGISTRY, select


def _reference(family: str, mode: str):
    return next(s.impl for s in _REGISTRY[f"{family}/{mode}"] if s.priority == 0)


def run(dim: int = 3072, ffn_dim: int = 14336, seq_len: int = 83,
        heads: int = 24, head_dim: int = 128, dtype=torch.bfloat16) -> list[dict]:
    dev = "cuda"
    rows: list[dict] = []

    def compare(name, fused, ref, moved_bytes, small=True):
        timer = measure_kernel if small else measure
        f = timer(fused)
        r = timer(ref)
        rows.append({
            "op": name, "fused_ms": f, "ref_ms": r, "speedup": r / f,
            "gbps": moved_bytes / (f * 1e-3) / 1e9,
        })

    x = torch.randn(1, seq_len, dim, device=dev, dtype=dtype)
    y = torch.randn_like(x)
    scale = torch.randn(1, 1, dim, device=dev, dtype=dtype)
    shift = torch.randn(1, 1, dim, device=dev, dtype=dtype)
    act_bytes = x.numel() * x.element_size()

    compare("adaln", lambda: ops.adaln_modulate(x, scale, shift),
            lambda: _reference("norm", "adaln")(x, scale, shift, 1e-6), 2 * act_bytes)
    compare("gated_residual", lambda: ops.gated_residual(x, y, scale),
            lambda: _reference("norm", "gated_residual")(x, y, scale), 3 * act_bytes)

    q = torch.randn(1, seq_len, heads, head_dim, device=dev, dtype=dtype)
    k = torch.randn_like(q)
    qw = torch.ones(heads * head_dim, device=dev, dtype=dtype)
    pos = torch.arange(seq_len, device=dev, dtype=torch.float32)[:, None]
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=dev, dtype=torch.float32) / head_dim))[None, :]
    cos, sin = torch.cos(pos * inv).to(dtype), torch.sin(pos * inv).to(dtype)
    compare("qk_norm_rope", lambda: ops.qk_norm_rope(q, k, qw, qw, cos, sin),
            lambda: _reference("rope", "qk_norm_rope")(q, k, qw, qw, cos, sin, 1e-6),
            4 * q.numel() * q.element_size())

    xf = x.reshape(-1, dim)
    up = torch.randn(ffn_dim, dim, device=dev, dtype=dtype) * 0.02
    down = torch.randn(dim, ffn_dim, device=dev, dtype=dtype) * 0.02
    compare("ffn_gelu", lambda: ops.ffn_gelu(xf, up, None, down, None),
            lambda: _reference("gemm", "ffn_gelu")(xf, up, None, down, None),
            (up.numel() + down.numel()) * up.element_size(), small=False)

    # Quantised GEMM against the dense GEMM it replaces -- the comparison that
    # matters is bytes moved, so report both.
    w = torch.randn(ffn_dim, dim, device=dev, dtype=dtype) * 0.02
    dense_bytes = w.numel() * w.element_size()
    for bits in (8, 4):
        qw_packed = quantize(w, bits=bits).to(dev)
        mode = "w4a16" if bits == 4 else "w8a16"
        impl = select("gemm", mode)
        fused = lambda impl=impl, p=qw_packed: impl(xf, p.qweight, p.scales, p.zeros, None, p.group_size)
        ms = measure(fused)
        dense_ms = measure(lambda: torch.nn.functional.linear(xf, w))
        rows.append({
            "op": f"gemm w{bits}a16", "fused_ms": ms, "ref_ms": dense_ms,
            "speedup": dense_ms / ms, "gbps": qw_packed.nbytes / (ms * 1e-3) / 1e9,
        })
        del qw_packed

    return rows
