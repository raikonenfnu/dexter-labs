"""Evaluate π0.5-DROID against ground-truth actions from the DROID dataset.

This answers a different question from the latency demo: not "does it fit in the
control budget" but "does it predict what the human teleoperator actually did".

Two things make the number interpretable rather than decorative:

**The right normalisation.** π0.5 normalises state and action with *quantile*
statistics (q01/q99), which the released checkpoint does not carry -- they come
from the dataset. Getting this wrong does not raise; it produces a policy that
looks incompetent, which is exactly what happened with the PushT demo before
the statistics were found.

**A trivial baseline.** DROID's action space is joint *position targets* at
15 Hz, so the next action is nearly the current joint state. "Repeat the current
state" therefore scores extremely well, and any error number without it beside
it is meaningless. The policy has to beat that, not just be small.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# DROID camera streams -> the names π0.5 expects. DROID has no right wrist
# camera, so that slot is zero-filled and masked out, per openpi's DroidInputs.
CAMERA_MAP = {
    "observation.images.exterior_1_left": "observation.images.base_0_rgb",
    "observation.images.wrist_left": "observation.images.left_wrist_0_rgb",
}
# Deliberately absent from the batch -- see the note in evaluate().
ABSENT_CAMERA = "observation.images.right_wrist_0_rgb"


@dataclass
class EvalResult:
    predicted: np.ndarray   # [N, horizon, 8]
    truth: np.ndarray       # [N, horizon, 8]
    state: np.ndarray       # [N, 8] joint state at prediction time

    def _mae(self, pred: np.ndarray) -> np.ndarray:
        return np.abs(pred - self.truth).mean(axis=(0, 1))

    @property
    def mae(self) -> np.ndarray:
        return self._mae(self.predicted)

    @property
    def baseline_mae(self) -> np.ndarray:
        """Error of commanding zero velocity -- i.e. not moving at all.

        For a velocity action space this is the trivial baseline the policy has
        to beat. (For position targets the equivalent would be holding the
        current state; that comparison does not apply here.)
        """
        zero = np.zeros_like(self.truth)
        zero[..., 7] = self.state[:, None, 7]  # gripper stays where it is
        return self._mae(zero)

    @property
    def correlation(self) -> float:
        """Pearson r between predicted and true joint velocities, pooled.

        This is the metric that separates "not tracking" from "tracking but
        differing in detail". MAE alone cannot: DROID teleop velocity is noisy
        and often near zero, so commanding zero scores well on MAE while having
        exactly zero correlation with what the operator did.
        """
        a = self.predicted[..., :7].ravel()
        b = self.truth[..., :7].ravel()
        if a.std() < 1e-9 or b.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def moving_subset(self, threshold: float = 0.15) -> "EvalResult | None":
        """Restrict to samples where the operator was actually moving.

        Do-nothing is only a strong baseline while the arm is near-stationary;
        the interesting comparison is on frames with real motion.
        """
        speed = np.abs(self.truth[..., :7]).mean(axis=(1, 2))
        keep = speed > threshold
        if keep.sum() < 2:
            return None
        return EvalResult(self.predicted[keep], self.truth[keep], self.state[keep])

    @property
    def per_sample_correlation(self) -> np.ndarray:
        """Correlation with the demonstrated chunk, one value per sample.

        Reported as a distribution rather than a pooled scalar because π0.5 is
        *generative*: it samples one plausible action chunk by flow matching,
        and a plausible chunk is not obliged to be the one this particular
        operator chose. Pooling hides that -- a policy that tracks well most of
        the time and picks a different valid strategy occasionally looks
        mediocre on average and obviously working in the distribution.
        """
        out = []
        for pred, true in zip(self.predicted, self.truth):
            a, b = pred[:, :7].ravel(), true[:, :7].ravel()
            out.append(np.nan if a.std() < 1e-9 or b.std() < 1e-9
                       else np.corrcoef(a, b)[0, 1])
        return np.asarray(out, dtype=float)

    def report(self) -> str:
        joints, grip = slice(0, 7), 7
        lines = [
            f"samples {len(self.truth)}  horizon {self.truth.shape[1]}",
            f"{'':22s}{'policy':>10s}{'do-nothing':>12s}",
            f"  joints MAE (rad/s) {self.mae[joints].mean():10.4f}"
            f"{self.baseline_mae[joints].mean():12.4f}",
            f"  gripper MAE        {self.mae[grip]:10.4f}"
            f"{self.baseline_mae[grip]:12.4f}",
        ]
        ratio = self.baseline_mae[joints].mean() / max(self.mae[joints].mean(), 1e-9)
        lines.append(f"  policy vs do-nothing on joints: {ratio:.2f}x "
                     f"({'better' if ratio > 1 else 'WORSE'})")
        per = self.per_sample_correlation
        per = per[~np.isnan(per)]
        lines += [
            f"  correlation(pred, true) pooled: {self.correlation:+.3f}",
            f"  per-sample correlation: median {np.median(per):+.3f}  "
            f"quartiles {np.percentile(per, 25):+.3f}/{np.percentile(per, 75):+.3f}",
            f"    tracking (r>0.5): {(per > 0.5).mean()*100:4.0f}% of samples   "
            f"anti/uncorrelated (r<0): {(per < 0).mean()*100:4.0f}%",
        ]
        lines.append("  per-joint MAE: " + " ".join(f"{v:.4f}" for v in self.mae[joints]))

        moving = self.moving_subset()
        if moving is not None:
            r = moving.baseline_mae[joints].mean() / max(moving.mae[joints].mean(), 1e-9)
            lines += [
                "",
                f"  on the {len(moving.truth)} samples with real motion:",
                f"    joints MAE {moving.mae[joints].mean():.4f}  "
                f"do-nothing {moving.baseline_mae[joints].mean():.4f}  "
                f"({r:.2f}x {'better' if r > 1 else 'worse'})",
                f"    correlation {moving.correlation:+.3f}",
            ]
        return "\n".join(lines)


def load_stats(path: str | Path) -> dict:
    """Normalisation statistics from openpi's *training* assets.

    These must be the statistics the checkpoint was trained with, not whatever
    a LeRobot port of DROID happens to ship. The two disagree about what an
    action even is: openpi's ``actions`` are centred near zero with std ~0.2
    (joint **velocities**), while LeRobot's ``action`` column mirrors the state
    (joint **positions**). Normalising with the wrong one produces a policy that
    scores worse than doing nothing, which is exactly what it did.
    """
    raw = json.loads(Path(path).read_text())
    norm = raw.get("norm_stats", raw)
    rename = {"state": "observation.state", "actions": "action"}
    out = {}
    for src, dst in rename.items():
        out[dst] = {k: torch.tensor(v, dtype=torch.float32)
                    for k, v in norm[src].items() if k in ("mean", "std", "q01", "q99")}
    return out


class DroidSlice:
    """Random access into a downloaded slice of the LeRobot DROID dataset."""

    def __init__(self, root: str | Path) -> None:
        import pandas as pd

        self.root = Path(root)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.fps = self.info["fps"]
        self.frames = pd.read_parquet(self.root / "data/chunk-000/file-000.parquet")
        episodes = pd.read_parquet(self.root / "meta/episodes/chunk-000/file-000.parquet")
        # Only episodes whose video actually landed in this slice.
        present = set(self.frames.episode_index.unique())
        self.episodes = episodes[episodes.episode_index.isin(present)].reset_index(drop=True)
        self._readers: dict[str, object] = {}

    def _frame_at(self, stream: str, timestamp: float) -> np.ndarray:
        """Decode one frame from a concatenated episode video by timestamp."""
        import av

        path = self.root / f"videos/{stream}/chunk-000/file-000.mp4"
        container = self._readers.get(stream)
        if container is None:
            container = av.open(str(path))
            self._readers[stream] = container

        target = int(timestamp / float(container.streams.video[0].time_base))
        container.seek(target, stream=container.streams.video[0], any_frame=False)
        for frame in container.decode(video=0):
            if frame.pts is not None and frame.pts >= target:
                return frame.to_ndarray(format="rgb24")
        raise RuntimeError(f"no frame at {timestamp}s in {stream}")

    def sample(self, episode_index: int, offset: int, horizon: int):
        """Return ``(images, state, actions)`` for one prediction point."""
        ep = self.episodes[self.episodes.episode_index == episode_index].iloc[0]
        rows = self.frames[self.frames.episode_index == episode_index]
        rows = rows.iloc[offset : offset + horizon]
        if len(rows) < horizon:
            return None

        images = {}
        for stream, target in CAMERA_MAP.items():
            base = float(ep[f"videos/{stream}/from_timestamp"])
            images[target] = self._frame_at(stream, base + offset / self.fps)

        state = np.asarray(rows.iloc[0]["observation.state"], dtype=np.float32)
        # DROID's action space for pi0.5 is joint *velocity* (7) plus gripper
        # *position* (1) -- not the dataset's consolidated `action` column,
        # which holds position targets.
        actions = np.stack([
            np.concatenate([
                np.asarray(v, dtype=np.float32),
                np.atleast_1d(np.asarray(g, dtype=np.float32)),
            ])
            for v, g in zip(rows["action.joint_velocity"], rows["action.gripper_position"])
        ])
        task = rows.iloc[0].get("language_instruction") or ""
        return images, state, actions, str(task)


def load_policy_with_stats(checkpoint: str, stats: dict, device: str = "cuda",
                           dtype: torch.dtype = torch.bfloat16):
    """Load π0.5 with processors built from *dataset* statistics.

    The checkpoint's saved pipelines carry empty ``features``, so their
    normaliser and unnormaliser are no-ops. Rebuilding them through
    ``make_pi05_pre_post_processors`` with the DROID statistics is what puts the
    state into the [-1, 1] quantile space the model was trained on, and brings
    the predicted actions back out into radians.
    """
    from dexter.demo._lerobot_compat import install
    from dexter.demo.pi05 import compat_checkpoint

    install()
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

    ckpt = compat_checkpoint(checkpoint, dtype=str(dtype).replace("torch.", ""))
    policy = PI05Policy.from_pretrained(ckpt).to(device=device).eval()
    policy.config.device = device
    pre, post = make_pi05_pre_post_processors(policy.config, dataset_stats=stats)
    return policy, pre, post


@torch.inference_mode()
def evaluate(policy, pre, post, slice_: "DroidSlice", *, samples: int = 20,
             seed: int = 0, device: str = "cuda", draws: int = 1) -> EvalResult:
    """Predict an action chunk at ``samples`` points and score against truth."""
    horizon = policy.config.chunk_size
    rng = np.random.default_rng(seed)
    episodes = slice_.episodes.episode_index.to_numpy()

    preds, truths, states = [], [], []
    attempts = 0
    while len(preds) < samples and attempts < samples * 8:
        attempts += 1
        ep = int(rng.choice(episodes))
        rows = slice_.frames[slice_.frames.episode_index == ep]
        if len(rows) < horizon + 2:
            continue
        offset = int(rng.integers(0, len(rows) - horizon))
        got = slice_.sample(ep, offset, horizon)
        if got is None:
            continue
        images, state, actions, task = got
        if not task.strip():
            continue

        batch = {}
        for key, img in images.items():
            # Native resolution, [0, 1], channels-first. The model resizes with
            # *padding* to 224x224 itself; pre-resizing with plain interpolation
            # distorts 180x320 to square and then gets padded again.
            batch[key] = (torch.from_numpy(img).float().permute(2, 0, 1) / 255.0).to(device)
        # DROID has no right wrist camera. Leaving the key *out* makes lerobot
        # pad it with -1 and set its attention mask to zero, which is what
        # openpi does (image_mask=False). Passing a zeros image instead would
        # be unmasked, so the model would attend to a black frame as if real.
        # openpi's statistics are 32-dim (DROID's 8 dims plus zero padding), so
        # the state must be padded to 32 *before* normalisation rather than
        # inside the model.
        padded = torch.zeros(policy.config.max_state_dim, dtype=torch.float32)
        padded[: len(state)] = torch.from_numpy(state)
        batch["observation.state"] = padded.to(device)

        policy.reset()
        processed = pre({**batch, "task": task})
        if draws > 1:
            # π0.5 samples a chunk by flow matching from fresh noise, so one
            # draw is one plausible strategy. Averaging draws estimates the
            # policy's mean behaviour rather than a single sample from it.
            raw = torch.stack([policy.predict_action_chunk(processed)
                               for _ in range(draws)]).mean(0)
        else:
            raw = policy.predict_action_chunk(processed)
        # Unnormalise all 32 dims (the statistics are 32-wide), then keep
        # DROID's first 8 -- the order openpi's DroidOutputs uses.
        chunk = post(raw[:, :horizon])
        preds.append(chunk[0, :, :8].float().cpu().numpy())
        truths.append(actions)
        states.append(state)

    if not preds:
        raise RuntimeError("no usable samples (episodes may lack language instructions)")
    return EvalResult(np.stack(preds), np.stack(truths), np.stack(states))
