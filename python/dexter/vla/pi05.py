"""π0.5-DROID, via LeRobot's PyTorch port.

DROID's contract, which openpi's `DroidInputs` defines and this mirrors:

* images  `exterior_1_left -> base_0_rgb`, `wrist_left -> left_wrist_0_rgb`.
  There is no right wrist camera; that key is **omitted** so the model pads it
  with −1 and masks it out. Passing zeros instead leaves it unmasked and the
  model attends to a black frame as though it were real.
* state   7 joint positions + 1 gripper, normalised with openpi's *training*
  statistics — not a LeRobot port's dataset statistics, which describe a
  different action space entirely.
* actions 7 joint velocities + 1 gripper position.

The order of operations is load-bearing: openpi normalises, then tokenises the
state into the prompt, then pads to 32. Padding first puts 24 filler bins into
every prompt.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from dexter.vla.base import Observation
from dexter.vla.registry import register

# DROID camera streams -> the names the checkpoint expects.
CAMERAS = {
    "observation.images.exterior_1_left": "observation.images.base_0_rgb",
    "observation.images.wrist_left": "observation.images.left_wrist_0_rgb",
}
DROID_DIM = 8       # 7 joints + gripper
CONTROL_HZ = 15.0


def load_norm_stats(path: str | Path, width: int = DROID_DIM) -> dict:
    """openpi's training statistics, trimmed to the robot's real width."""
    raw = json.loads(Path(path).read_text())
    norm = raw.get("norm_stats", raw)
    return {
        dst: {k: torch.tensor(v, dtype=torch.float32)[:width]
              for k, v in norm[src].items() if k in ("mean", "std", "q01", "q99")}
        for src, dst in (("state", "observation.state"), ("actions", "action"))
    }


class Pi05Droid:
    """π0.5 specialised to DROID."""

    name = "pi05_droid"
    control_hz = CONTROL_HZ

    def __init__(self, policy, pre, post) -> None:
        self._policy, self._pre, self._post = policy, pre, post
        self.chunk_size = policy.config.chunk_size
        self.device = next(policy.parameters()).device

    @property
    def budget_ms(self) -> float:
        return self.chunk_size / self.control_hz * 1e3

    def _batch(self, observation: Observation) -> dict:
        batch = {
            CAMERAS[stream]: (torch.from_numpy(img).float().permute(2, 0, 1) / 255.0).to(self.device)
            for stream, img in observation.images.items() if stream in CAMERAS
        }
        batch["observation.state"] = torch.from_numpy(
            np.asarray(observation.state, dtype=np.float32)).to(self.device)
        return {**batch, "task": observation.task}

    @torch.inference_mode()
    def act(self, observation: Observation) -> np.ndarray:
        self._policy.reset()
        processed = self._pre(self._batch(observation))
        raw = self._policy.predict_action_chunk(processed)
        chunk = self._post(raw[:, : self.chunk_size, :DROID_DIM])
        return chunk[0].float().cpu().numpy()


@register("pi05_droid")
def load_pi05_droid(checkpoint: str, norm_stats: str | None = None,
                    device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> Pi05Droid:
    from dexter.vla.compat import compat_checkpoint, install_lerobot_shims

    install_lerobot_shims()

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

    stats = load_norm_stats(norm_stats or Path(checkpoint) / "openpi_norm_stats.json")
    ckpt = compat_checkpoint(checkpoint, dtype=str(dtype).replace("torch.", ""))
    policy = PI05Policy.from_pretrained(ckpt).to(device=device).eval()
    policy.config.device = device

    # The processors work at the robot's width; the model pads to
    # max_state_dim itself. Leaving 32 here would both normalise against
    # padded statistics and put 32 numbers in the prompt.
    cfg = copy.deepcopy(policy.config)
    for features in (cfg.input_features, cfg.output_features):
        for key, feature in features.items():
            if key in stats:
                features[key] = type(feature)(type=feature.type, shape=(DROID_DIM,))
    cfg.max_state_dim = DROID_DIM
    pre, post = make_pi05_pre_post_processors(cfg, dataset_stats=stats)
    return Pi05Droid(policy, pre, post)
