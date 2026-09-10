"""``dexter-pi05`` -- can a SOTA VLA keep up with DROID's control rate?"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from dexter.demo.pi05 import CAMERA_KEYS, droid_observation, load_policy, measure
from dexter.platform import current_platform

# Real DROID footage shipped in the DreamZero repo: two exterior views and a
# wrist view, which is exactly the triple this checkpoint expects.
DEFAULT_VIDEOS = {
    CAMERA_KEYS[0]: "exterior_image_1_left.mp4",
    CAMERA_KEYS[1]: "wrist_image_left.mp4",
    CAMERA_KEYS[2]: "exterior_image_2_left.mp4",
}


def read_frames(video_dir: str) -> dict[str, np.ndarray]:
    import imageio.v3 as iio
    from pathlib import Path

    frames = {}
    for key, name in DEFAULT_VIDEOS.items():
        path = Path(video_dir) / name
        frames[key] = iio.imread(str(path), plugin="pyav")[0]
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(prog="dexter-pi05")
    ap.add_argument("--checkpoint", default="/home/stanley/nod/checkpoints/pi05_droid")
    ap.add_argument("--video-dir", required=True, help="directory of DROID .mp4 views")
    ap.add_argument("--prompt", default="pick up the black bowl and place it on the plate")
    ap.add_argument("--chunks", type=int, default=5)
    args = ap.parse_args()

    print(current_platform().describe())
    print("loading pi05_droid (PyTorch, no JAX) ...")
    policy, pre, post = load_policy(args.checkpoint)
    params = sum(p.numel() for p in policy.parameters())
    resident = sum(p.numel() * p.element_size() for p in policy.parameters())
    print(f"  {params/1e9:.2f}B params, {resident/1e9:.2f} GB resident, "
          f"chunk_size {policy.config.chunk_size}\n")

    frames = read_frames(args.video_dir)
    for key, img in frames.items():
        print(f"  {key.split('.')[-1]:22s} {img.shape}")

    # DROID state is 7 joints + gripper; zeros stand in for a real arm.
    dtype = next(policy.parameters()).dtype
    batch = droid_observation(frames, np.zeros(8, dtype=np.float32), "cuda", dtype)
    print(f'\nprompt: "{args.prompt}"')

    report, actions = measure(policy, pre, post, batch, args.prompt, chunks=args.chunks)
    a = actions.float().cpu().numpy()
    print(f"\naction chunk {a.shape}  finite={np.isfinite(a).all()}  "
          f"|mean| {np.abs(a[..., :7]).mean():.4f}")
    print("\n" + report.summary())
    print(f"  per-chunk ms: {[round(t) for t in report.chunk_ms]}")
    print(f"  effective policy rate: {1e3/report.median_ms:.2f} Hz "
          f"(one chunk = {report.chunk_size} actions = "
          f"{report.chunk_size/report.control_hz:.2f} s of motion)")
    print(f"  peak GPU: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
