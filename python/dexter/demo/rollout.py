"""World-model rollout: a real observation in, imagined future and actions out.

DreamZero is a *world* action model — it predicts the video its own actions
would produce, jointly with the actions. That makes a meaningful demo possible
on hardware far too slow to drive a real arm: instead of executing actions at
15 Hz, let the model dream forward from a real robot frame and decode what it
expects to see. The video is the model's answer to "what happens if I do this",
and the action chunk is the "this".

The loop is the upstream closed-loop contract with the environment replaced by
the model's own prediction:

    observe -> encode -> denoise (video + actions jointly)
            -> decode predicted frames -> feed the last one back as the next
               observation

which is exactly autoregressive rollout of a world model, and is what the
KV cache across blocks is for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dexter.models.dreamzero.policy import DreamZeroPolicy, StepTrace
from dexter.perception.encode import (
    VIDEO_HEIGHT, VIDEO_WIDTH, ObservationEncoder, patchify,
)


@dataclass
class Rollout:
    """What one rollout produced."""

    frames: list[np.ndarray] = field(default_factory=list)  # HxWx3 uint8
    actions: list[np.ndarray] = field(default_factory=list)  # [chunk, action_dim]
    step_ms: list[float] = field(default_factory=list)

    @property
    def trajectory(self) -> np.ndarray:
        """All action chunks stacked, ``[steps * chunk, action_dim]``."""
        return np.concatenate(self.actions, axis=0) if self.actions else np.empty((0, 0))


def _to_uint8(frame: torch.Tensor) -> np.ndarray:
    """``[3, H, W]`` in ``[-1, 1]`` -> ``HxWx3`` uint8."""
    x = (frame.float().clamp(-1, 1) + 1) * 127.5
    return x.permute(1, 2, 0).round().byte().cpu().numpy()


def read_first_frame(path: str | Path) -> torch.Tensor:
    """First frame of a video as ``[3, H, W]`` in ``[-1, 1]``."""
    import imageio.v3 as iio

    frames = iio.imread(str(path), plugin="pyav")
    return torch.from_numpy(frames[0]).permute(2, 0, 1).float() / 127.5 - 1.0


@torch.inference_mode()
def run_rollout(
    policy: DreamZeroPolicy,
    encoder: ObservationEncoder,
    frame: torch.Tensor,
    *,
    steps: int = 4,
    num_frames: int = 5,
    seed: int = 0,
) -> Rollout:
    """Dream ``steps`` blocks forward from ``frame``.

    ``num_frames`` is the pixel horizon per block. The VAE compresses time 4x,
    so 5 pixel frames is 2 latent frames, which is the ``num_frame_per_block``
    the released config was trained with.
    """
    cfg = policy.cfg
    out = Rollout()
    generator = torch.Generator(device=policy.device).manual_seed(seed)

    current = frame
    out.frames.append(_to_uint8(
        F.interpolate(current[None], size=(VIDEO_HEIGHT, VIDEO_WIDTH),
                      mode="bicubic", align_corners=False)[0]
    ))

    for step in range(steps):
        clip, y = encoder.encode_observation(current, num_frames=num_frames)
        conditioning = patchify(y)
        if conditioning.shape[1] != cfg.video_tokens:
            raise ValueError(
                f"conditioning has {conditioning.shape[1]} tokens but the model "
                f"expects {cfg.video_tokens}; check the video resolution"
            )

        # The instruction is already bound; refresh only the image branch, which
        # is what actually changed between steps.
        policy.ctx = policy.model.project_context(policy.text_embedding, clip)

        if step == 0:
            # Fill the cache with the real observation before asking for
            # anything: one clean latent frame at timestep 0.
            #
            # That latent is already inside `y` -- its first frame, past the 4
            # mask channels -- so re-encoding here would both waste a VAE pass
            # and risk using a differently-scaled frame than the conditioning.
            observed = y[:, 4:, :1]
            first = torch.cat(
                (patchify(observed), conditioning[:, :cfg.frame_seqlen]), dim=-1)
            policy.prime(first)

        # State is the robot's proprioception. Without a real arm attached there
        # is nothing to report, so it stays zero and the model is driven by
        # vision and language alone.
        state = torch.zeros(1, cfg.num_state_per_block, cfg.max_state_dim,
                            device=policy.device, dtype=policy.dtype)

        trace = StepTrace()
        actions, video_latent = policy.step(
            conditioning, state, generator=generator, trace=trace, return_video=True
        )
        out.step_ms.append(trace.total)
        out.actions.append(actions[0].float().cpu().numpy())

        # policy.step already returns a latent, not tokens.
        predicted = encoder.decode_latents(video_latent)[0]  # [3, T, H, W]
        for t in range(predicted.shape[1]):
            out.frames.append(_to_uint8(predicted[:, t]))

        current = predicted[:, -1]  # autoregress on the model's own prediction

    return out


def save_video(rollout: Rollout, path: str | Path, fps: int = 8) -> None:
    import imageio.v3 as iio

    iio.imwrite(str(path), np.stack(rollout.frames), fps=fps, codec="libx264")
