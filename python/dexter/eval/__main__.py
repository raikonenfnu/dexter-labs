"""``dexter-eval`` -- score a VLA against demonstrated actions."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from dexter.data import LeRobotSlice
from dexter.eval import score_chunks
from dexter.platform import current_platform
from dexter.vla import available, load

# How a dataset's columns become the policy's action space. DROID's consolidated
# `action` column is joint *positions*; pi0.5 predicts joint *velocities*.
ACTION_COLUMNS = {
    "pi05_droid": ("action.joint_velocity", "action.gripper_position"),
}
STATE_COLUMN = {"pi05_droid": "observation.state"}
STREAMS = {
    "pi05_droid": ("observation.images.exterior_1_left", "observation.images.wrist_left"),
}


def truth_actions(rows, columns) -> np.ndarray:
    parts = [np.stack([np.atleast_1d(np.asarray(v, dtype=np.float32)) for v in rows[c]])
             for c in columns]
    return np.concatenate(parts, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(prog="dexter-eval")
    ap.add_argument("--policy", default="pi05_droid", choices=available())
    ap.add_argument("--checkpoint", default="/home/stanley/nod/checkpoints/pi05_droid")
    ap.add_argument("--dataset", default="/home/stanley/nod/datasets/droid_1.0.1_slice")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-motion", type=float, default=0.05,
                    help="skip points where the operator was idle")
    args = ap.parse_args()

    print(current_platform().describe())
    policy = load(args.policy, checkpoint=args.checkpoint)
    print(f"{policy.name}: chunk {policy.chunk_size} @ {policy.control_hz:.0f} Hz "
          f"({policy.budget_ms:.0f} ms budget)")

    data = LeRobotSlice(args.dataset, STREAMS[args.policy])
    print(f"dataset: {len(data.episodes)} decodable episodes of "
          f"{data.frames.episode_index.nunique()} in the shard\n")

    columns = ACTION_COLUMNS[args.policy]
    state_column = STATE_COLUMN[args.policy]

    def usable(sample) -> bool:
        if not sample.task.strip():
            return False
        return np.abs(truth_actions(sample.frames, columns)[:, :7]).mean() >= args.min_motion

    preds, truths, times = [], [], []
    from dexter.vla.base import Observation

    for sample in data.iter_samples(policy.chunk_size, count=args.samples,
                                    seed=args.seed, accept=usable):
        obs = Observation(
            images=sample.images,
            state=np.asarray(sample.frames.iloc[0][state_column], dtype=np.float32),
            task=sample.task,
        )
        start = time.perf_counter()
        preds.append(policy.act(obs))
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)
        truths.append(truth_actions(sample.frames, columns))

    print(score_chunks(preds, truths).report())
    print(f"  latency       {np.median(times):.0f} ms/chunk vs {policy.budget_ms:.0f} ms budget"
          f"  ({policy.budget_ms / np.median(times):.2f}x headroom)")


if __name__ == "__main__":
    main()
