"""``dexter-demo`` -- run the closed-loop robot demo."""

from __future__ import annotations

import argparse
import dataclasses

import torch

from dexter.demo.robot import CONTROL_HZ, OPEN_LOOP_HORIZON, ReachTask, run_episode
from dexter.models.dreamzero import CausalWanDiT
from dexter.models.dreamzero.config import PRESETS
from dexter.models.dreamzero.policy import DreamZeroPolicy
from dexter.platform import current_platform


def main() -> None:
    ap = argparse.ArgumentParser(prog="dexter-demo")
    ap.add_argument("--model", default="5b", choices=sorted(PRESETS))
    ap.add_argument("--layers", type=int, default=None,
                    help="override layer count (default: the full model)")
    ap.add_argument("--bits", type=int, default=None, choices=[4, 8],
                    help="weight-only quantisation (default: bf16)")
    ap.add_argument("--ticks", type=int, default=120)
    ap.add_argument("--control-hz", type=float, default=CONTROL_HZ)
    ap.add_argument("--horizon", type=int, default=OPEN_LOOP_HORIZON,
                    help="actions executed per chunk before re-querying")
    ap.add_argument("--prefetch", type=int, nargs="+", default=[4],
                    help="pipelined: actions before exhaustion to start inference; "
                         "several values sweep the knob on one loaded model")
    ap.add_argument("--scheduler", nargs="+", default=["blocking", "pipelined"],
                    choices=["blocking", "pipelined"])
    args = ap.parse_args()

    cfg = PRESETS[args.model]
    if args.layers is not None:
        cfg = dataclasses.replace(cfg, num_layers=args.layers)

    print(current_platform().describe())
    print(cfg.describe())
    print(f"control {args.control_hz:.0f} Hz, chunk {cfg.num_action_per_block}, "
          f"open-loop horizon {args.horizon}, {args.ticks} ticks "
          f"({args.ticks / args.control_hz:.1f}s of robot time)\n")

    model = CausalWanDiT(cfg)
    if args.bits:
        model.quantize_(bits=args.bits)
    policy = DreamZeroPolicy(model, batch=1, cache_blocks=64)
    policy.set_instruction(torch.randn(1, cfg.text_len, cfg.text_dim,
                                       device="cuda", dtype=model.dtype))

    print(f"{'scheduler':<10} {'rate':>9}  {'calls':>9}  {'stalled':>19}  "
          f"{'worst':>10}  {'deadline misses':>19}")
    results: list = []
    for scheduler in args.scheduler:
        # Prefetch only means anything to the pipelined scheduler.
        for prefetch in (args.prefetch if scheduler == "pipelined" else [0]):
            stats = run_episode(
                policy, scheduler=scheduler, ticks=args.ticks,
                control_hz=args.control_hz, open_loop_horizon=args.horizon,
                prefetch=prefetch, task=ReachTask(),
            )
            label = f"{scheduler}" if scheduler == "blocking" else f"{scheduler}/pf{prefetch}"
            print(f"{label:<14}" + stats.summary()[len(stats.scheduler):].lstrip().rjust(0)
                  if False else f"{label:<14}{stats.summary()[10:]}")
            results.append((label, prefetch, stats))

    blocking = next((s for label, _, s in results if label == "blocking"), None)
    best = min((r for r in results if r[0] != "blocking"),
               key=lambda r: r[2].total_stall_s, default=None)
    if blocking is not None and best is not None:
        label, prefetch, stats = best
        budget = prefetch / args.control_hz * 1e3
        worst = max(stats.stall_ms) if stats.stall_ms else 0.0
        print(f"\nBest overlap ({label}) recovered "
              f"{blocking.total_stall_s - stats.total_stall_s:.2f}s of "
              f"{blocking.wall_s:.1f}s, taking the arm from "
              f"{blocking.achieved_hz:.1f} Hz to {stats.achieved_hz:.1f} Hz "
              f"against a {args.control_hz:.0f} Hz target.")
        print(f"A prefetch of {prefetch} actions buys {budget:.0f} ms of overlap; "
              f"worst observed stall was {worst:.0f} ms.")


if __name__ == "__main__":
    main()
