"""``dexter-bench`` -- roofline, kernels, and end-to-end closed-loop latency."""

from __future__ import annotations

import argparse

import torch

from dexter.bench import kernels
from dexter.bench.harness import bandwidth
from dexter.models.dreamzero.config import PRESETS
from dexter.platform import current_platform
from dexter.registry import explain


def _cell(value, fmt: str) -> str:
    return format(value, fmt) if fmt else str(value)


def _table(rows: list[dict], columns: list[tuple[str, str, str]]) -> None:
    cells = [[_cell(r[key], fmt) for key, _, fmt in columns] for r in rows]
    widths = [max(len(header), *(len(row[i]) for row in cells))
              for i, (_, header, _) in enumerate(columns)]
    print("  ".join(h.ljust(w) for (_, h, _), w in zip(columns, widths)))
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))


def cmd_platform(args) -> None:
    p = current_platform()
    print(p.describe())
    print(f"  features: {', '.join(sorted(p.features))}")
    print(f"\n  measured read bandwidth: {bandwidth():.0f} GB/s "
          f"(datasheet peak {p.memory_bandwidth:.0f} GB/s)")
    print()
    for family, mode in [("gemm", "w4a16"), ("norm", "adaln"),
                         ("attention", "blockwise_causal"), ("rope", "qk_norm_rope")]:
        print(explain(family, mode))
        print()


def peak_tflops() -> float:
    """Achievable dense bf16 throughput, from a large square matmul."""
    from dexter.bench.harness import measure

    n = 4096
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    ms = measure(lambda: a @ b, warmup=5, iters=15)
    del a, b
    torch.cuda.empty_cache()
    return 2 * n ** 3 / (ms * 1e-3) / 1e12


def cmd_roofline(args) -> None:
    gbps = bandwidth()
    tflops = peak_tflops()
    balance = tflops * 1e12 / (gbps * 1e9)
    print(f"measured: {gbps:.0f} GB/s read, {tflops:.1f} TFLOP/s bf16")
    print(f"machine balance: {balance:.0f} FLOP/byte\n")

    rows = []
    for cfg in PRESETS.values():
        for bits in (16, 8, 4):
            mem_ms = cfg.weight_bytes(bits) / (gbps * 1e9) * 1e3
            cmp_ms = cfg.flops_per_denoise_step() / (tflops * 1e12) * 1e3
            step_ms = max(mem_ms, cmp_ms)
            rows.append({
                "model": cfg.name.split("-")[-1],
                "bits": bits,
                "weights_gb": cfg.weight_bytes(bits) / 1e9,
                "intensity": cfg.arithmetic_intensity(bits),
                "bound": "memory" if mem_ms >= cmp_ms else "compute",
                "denoise_ms": step_ms,
                "control_ms": step_ms * cfg.num_inference_steps,
                "hz": 1e3 / (step_ms * cfg.num_inference_steps),
            })
    _table(rows, [
        ("model", "model", "s"), ("bits", "bits", "d"),
        ("weights_gb", "weights GB", ".2f"), ("intensity", "FLOP/byte", ".0f"),
        ("bound", "bound by", "s"), ("denoise_ms", "ms/denoise", ".1f"),
        ("control_ms", "ms/control", ".1f"), ("hz", "Hz", ".1f"),
    ])
    print("\nFloors, not predictions: max(weight traffic / bandwidth, FLOPs / peak).")
    print("Note where the bound flips. Weight-only quantisation buys time only")
    print("while a shape is memory bound; past the balance point it buys capacity.")


def cmd_kernels(args) -> None:
    cfg = PRESETS[args.model]
    rows = kernels.run(dim=cfg.dim, ffn_dim=cfg.ffn_dim, seq_len=cfg.seq_len,
                       heads=cfg.num_heads, head_dim=cfg.head_dim)
    _table(rows, [
        ("op", "op", "s"), ("fused_ms", "dexter ms", ".3f"),
        ("ref_ms", "torch ms", ".3f"), ("speedup", "speedup x", ".2f"),
        ("gbps", "GB/s", ".0f"),
    ])


def cmd_e2e(args) -> None:
    from dexter.bench import e2e

    cfg = PRESETS[args.model]
    print(cfg.describe())
    print(f"(measuring {args.layers or cfg.num_layers} layers)\n")

    rows = []
    for bits in args.bits:
        for graph in ([False, True] if args.graph else [False]):
            rows.append(e2e.run(cfg, bits=None if bits == 16 else bits,
                                use_graph=graph, layers=args.layers))
            print(".", end="", flush=True)
    print("\n")
    _table(rows, [
        ("precision", "precision", "s"), ("graph", "graph", ""),
        ("weight_gb", "weights GB", ".2f"), ("step_ms", "step ms", ".1f"),
        ("hz", "Hz", ".1f"), ("device_ms", "device ms", ".1f"),
        ("host_ms", "host ms", ".1f"),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(prog="dexter-bench")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("platform", help="device capabilities and kernel selection")
    sub.add_parser("roofline", help="bandwidth floors per model and precision")

    k = sub.add_parser("kernels", help="fused kernels vs torch references")
    k.add_argument("--model", default="5b", choices=sorted(PRESETS))

    e = sub.add_parser("e2e", help="closed-loop step latency")
    e.add_argument("--model", default="5b", choices=sorted(PRESETS))
    e.add_argument("--bits", type=int, nargs="+", default=[16, 8, 4])
    e.add_argument("--layers", type=int, default=None,
                   help="override layer count to fit a smaller memory budget")
    e.add_argument("--graph", action="store_true", help="also measure with HIP graph capture")

    args = parser.parse_args()
    torch.manual_seed(0)
    {"platform": cmd_platform, "roofline": cmd_roofline,
     "kernels": cmd_kernels, "e2e": cmd_e2e}[args.command](args)


if __name__ == "__main__":
    main()
