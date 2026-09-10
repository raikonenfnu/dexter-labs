"""``dexter-droid-eval`` -- score π0.5-DROID against real DROID actions."""

from __future__ import annotations

import argparse

import torch

from dexter.demo.droid_eval import DroidSlice, evaluate, load_policy_with_stats, load_stats
from dexter.platform import current_platform


def main() -> None:
    ap = argparse.ArgumentParser(prog="dexter-droid-eval")
    ap.add_argument("--checkpoint", default="/home/stanley/nod/checkpoints/pi05_droid")
    ap.add_argument("--dataset", default="/home/stanley/nod/datasets/droid_1.0.1_slice")
    ap.add_argument("--norm-stats",
                    default="/home/stanley/nod/checkpoints/pi05_droid/openpi_norm_stats.json",
                    help="openpi training statistics (NOT the LeRobot dataset stats)")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=1,
                    help="average this many sampled chunks per observation")
    args = ap.parse_args()

    print(current_platform().describe())
    stats = load_stats(args.norm_stats)
    print(f"norm stats: {args.norm_stats.split('/')[-1]} -> {sorted(stats)} "
          f"(q01/q99: {'q01' in stats['action']})")

    policy, pre, post = load_policy_with_stats(args.checkpoint, stats)
    print(f"pi05_droid: {sum(p.numel() for p in policy.parameters())/1e9:.2f}B params, "
          f"chunk {policy.config.chunk_size}\n")

    ds = DroidSlice(args.dataset)
    print(f"DROID slice: {len(ds.episodes)} episodes, {len(ds.frames)} frames @ {ds.fps} Hz")
    print(f"evaluating {args.samples} prediction points ...\n")

    result = evaluate(policy, pre, post, ds, samples=args.samples,
                      seed=args.seed, draws=args.draws)
    print(result.report())
    print(f"\npeak GPU: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
