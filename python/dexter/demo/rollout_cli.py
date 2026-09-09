"""``dexter-rollout`` -- dream forward from a real robot frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dexter.demo.rollout import read_first_frame, run_rollout, save_video
from dexter.models.dreamzero.checkpoint import load_dreamzero
from dexter.models.dreamzero.config import PRESETS
from dexter.models.dreamzero.policy import DreamZeroPolicy
from dexter.perception.encode import ObservationEncoder
from dexter.platform import current_platform


def main() -> None:
    ap = argparse.ArgumentParser(prog="dexter-rollout")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--video", required=True, help="video whose first frame is the observation")
    ap.add_argument("--prompt", default="pick up the object and place it on the plate")
    ap.add_argument("--model", default="14b", choices=sorted(PRESETS))
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", default="rollout.mp4")
    args = ap.parse_args()

    cfg = PRESETS[args.model]
    print(current_platform().describe())
    print(cfg.describe(), "\n")

    encoder = ObservationEncoder(args.checkpoint)

    # Language first, alone: umt5-xxl is 11.4 GB and the DiT is 9 GB, so they
    # do not coexist on a 32 GB part. Encode once, then let the tower go.
    print(f'encoding instruction: "{args.prompt}"')
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    text = encoder.encode_text(args.prompt, tokenizer)
    encoder.release("text")
    print(f"  text {tuple(text.shape)}, text tower released\n")

    frame = read_first_frame(args.video)
    print(f"observation: {args.video}  {tuple(frame.shape)}")

    print(f"loading DiT as int{args.bits} ...")
    model = load_dreamzero(args.checkpoint, cfg, bits=args.bits)
    print(f"  {model.weight_bytes() / 1e9:.2f} GB resident\n")

    policy = DreamZeroPolicy(model, batch=1, cache_blocks=args.steps + 1)
    clip, _ = encoder.encode_observation(frame, num_frames=5)
    policy.set_instruction(text, clip)

    print(f"dreaming {args.steps} blocks forward ...")
    rollout = run_rollout(policy, encoder, frame, steps=args.steps)

    save_video(rollout, args.out)
    traj = rollout.trajectory
    print(f"\nwrote {args.out}: {len(rollout.frames)} frames "
          f"({np.mean(rollout.step_ms) / 1e3:.1f} s per block)")
    print(f"action trajectory {traj.shape}")
    np.save(Path(args.out).with_suffix(".actions.npy"), traj)
    print(f"  joint deltas per step (first 7 dims), |mean| = "
          f"{np.abs(traj[:, :7]).mean():.4f}, max = {np.abs(traj[:, :7]).max():.4f}")
    print(f"  gripper channel range [{traj[:, 7].min():.3f}, {traj[:, 7].max():.3f}]")
    print(f"  saved {Path(args.out).with_suffix('.actions.npy')}")


if __name__ == "__main__":
    main()
