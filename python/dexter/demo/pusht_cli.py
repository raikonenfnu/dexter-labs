"""``dexter-pusht`` -- run a trained policy on the PushT task."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from dexter.demo.pusht import load_policy, make_env, run_episode, save_video
from dexter.platform import current_platform


def main() -> None:
    ap = argparse.ArgumentParser(prog="dexter-pusht")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out", default="pusht.mp4", help="video of the best episode")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(current_platform().describe())
    policy, stats = load_policy(args.device)
    params = sum(p.numel() for p in policy.parameters())
    print(f"lerobot/diffusion_pusht: {params/1e6:.1f}M params on {args.device}\n")

    env = make_env(args.max_steps)
    results = []
    for episode in range(args.episodes):
        start = time.perf_counter()
        result = run_episode(policy, env, stats, seed=episode)
        wall = time.perf_counter() - start
        rate = result.steps / wall
        print(f"  episode {episode}: best coverage {result.best_coverage:5.3f}  "
              f"{'SOLVED' if result.success else 'unsolved'}  "
              f"{result.steps:3d} steps in {wall:5.1f}s ({rate:5.1f} Hz)")
        results.append(result)

    best = max(results, key=lambda r: r.best_coverage)
    save_video(best.frames, args.out)
    coverage = [r.best_coverage for r in results]
    successes = sum(r.success for r in results)
    print(f"\n{successes}/{len(results)} episodes solved "
          f"(coverage > 0.95); median best coverage "
          f"{statistics.median(coverage):.3f}, worst {min(coverage):.3f}")
    print(f"wrote {args.out} ({len(best.frames)} frames of the best episode)")


if __name__ == "__main__":
    main()
