"""π0.5-DROID: a SOTA VLA, timed against DROID's control rate.

The question this answers is narrow: does a current vision-language-action model
keep up with a 15 Hz robot on gfx1151? π0.5 is the right candidate because it is
DROID-native (same Franka, same rate, same action space, no fine-tuning) and
because it decodes an action *chunk* by flow matching rather than emitting
action tokens autoregressively.

That distinction is the whole ballgame on a bandwidth-poor part. An
autoregressive VLA pays one full pass over its weights per token: a 3B model in
bf16 is ~6 GB, which is ~26 ms per token at this machine's measured 233 GB/s,
so a few dozen action tokens blows a 1 s budget on memory traffic alone. Flow
matching pays one prefill over the vision-language prefix, then a handful of
passes over a small action expert -- and the prefill is compute-bound, where
this machine has far more headroom.

``chunk_size`` is 15, so at 15 Hz one inference buys exactly 1.0 s of motion.
That, not the 66.7 ms tick, is the deadline to beat.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import numpy as np
import torch

CONTROL_HZ = 15.0
# DROID's three views, in the names the checkpoint expects.
CAMERA_KEYS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)
IMAGE_SIZE = 224


@dataclass
class LatencyReport:
    chunk_ms: list[float]
    chunk_size: int
    control_hz: float = CONTROL_HZ

    @property
    def median_ms(self) -> float:
        return statistics.median(self.chunk_ms)

    @property
    def budget_ms(self) -> float:
        """Motion one chunk buys -- the real deadline."""
        return self.chunk_size / self.control_hz * 1e3

    @property
    def keeps_up(self) -> bool:
        return self.median_ms < self.budget_ms

    @property
    def headroom(self) -> float:
        return self.budget_ms / self.median_ms

    def summary(self) -> str:
        verdict = "KEEPS UP" if self.keeps_up else "TOO SLOW"
        return (
            f"median {self.median_ms:7.1f} ms/chunk   "
            f"budget {self.budget_ms:6.1f} ms ({self.chunk_size} actions @ "
            f"{self.control_hz:.0f} Hz)   {self.headroom:4.2f}x   {verdict}"
        )


# Fields the published checkpoint carries that this lerobot's PI05Config does
# not accept. Dropping a config field is only safe if it cannot change what the
# model computes, so each is checked against its value rather than assumed:
#
#   use_peft=false, freeze_vision_encoder=false, train_expert_only=false
#       training-time switches, inert at inference.
#   use_relative_actions=false
#       the one that WOULD matter -- it selects a different action space. It is
#       off, so the absolute-action path this lerobot implements is correct.
#   relative_exclude_joints=["gripper"]
#       only read when use_relative_actions is true.
#   action_feature_names=null, rtc_config=null
#       unset.
#
# If a future checkpoint sets use_relative_actions=true, this shim must fail
# rather than strip it, hence the explicit guard.
_INERT_UNKNOWN_FIELDS = (
    "use_peft", "use_relative_actions", "relative_exclude_joints",
    "action_feature_names", "rtc_config", "freeze_vision_encoder",
    "train_expert_only",
)


def compat_checkpoint(path: str, work_dir: str | None = None,
                      dtype: str = "bfloat16") -> str:
    """Return a checkpoint dir this lerobot can load, without copying weights.

    The 16 GB safetensors is symlinked; only config.json is rewritten.
    """
    import json
    import os
    from pathlib import Path

    src = Path(path)
    config = json.loads((src / "config.json").read_text())
    unknown = [k for k in _INERT_UNKNOWN_FIELDS if k in config]
    if not unknown:
        return str(src)

    if config.get("use_relative_actions"):
        raise RuntimeError(
            "checkpoint sets use_relative_actions=true, which this lerobot does "
            "not implement; loading it would silently use the wrong action space"
        )

    dst = Path(work_dir or (src.parent / f"{src.name}_compat"))
    dst.mkdir(parents=True, exist_ok=True)
    rewritten = {"config.json", "policy_preprocessor.json", "policy_postprocessor.json"}
    for item in src.iterdir():
        if item.name in rewritten:
            continue
        link = dst / item.name
        if not link.exists():
            os.symlink(item.resolve(), link)
    cleaned = {k: v for k, v in config.items() if k not in unknown}
    # PI05 keeps its own internals (noise, timesteps, the action expert's
    # projections) consistent with config.dtype. Casting the module afterwards
    # instead leaves float32 noise meeting bf16 weights, so set it here.
    cleaned["dtype"] = dtype
    (dst / "config.json").write_text(json.dumps(cleaned, indent=2))
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        _strip_unknown_steps(src / name, dst / name)
    return str(dst)


# Processor steps this lerobot has no implementation for. Same rule as the
# config fields: only droppable while provably inert.
_DROPPABLE_STEPS = {"relative_actions_processor", "absolute_actions_processor"}


def _strip_unknown_steps(src: Path, dst: Path) -> None:
    """Copy a processor pipeline, dropping steps this lerobot cannot build.

    A dropped step must be disabled in the checkpoint. Silently removing an
    *active* step would change what the policy computes while still running,
    so an enabled one raises instead.
    """
    import json

    pipeline = json.loads(src.read_text())
    kept = []
    for step in pipeline.get("steps", []):
        name = step.get("registry_name")
        if name in _DROPPABLE_STEPS:
            if (step.get("config") or {}).get("enabled"):
                raise RuntimeError(
                    f"checkpoint enables '{name}', which this lerobot does not "
                    f"implement; dropping it would change the policy"
                )
            continue
        kept.append(step)
    pipeline["steps"] = kept
    dst.write_text(json.dumps(pipeline, indent=2))


def load_policy(path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Return ``(policy, preprocessor, postprocessor)``.

    The processors are not optional decoration: the pre-processor normalises the
    observation, discretises the state and runs the PaliGemma tokenizer that
    produces ``observation.language.tokens``, and the post-processor unnormalises
    the action chunk. Hand-rolling that is how the PushT demo ended up scoring
    zero -- so the checkpoint's own saved pipelines are used.
    """
    from dexter.demo._lerobot_compat import install

    install()
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.utils.constants import (
        POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    ckpt = compat_checkpoint(path, dtype=str(dtype).replace("torch.", ""))
    policy = PI05Policy.from_pretrained(ckpt).to(device=device).eval()
    # The checkpoint was saved with device="cpu"; its device step would leave
    # the tokenised language on the host while the weights sit on the GPU.
    pre = PolicyProcessorPipeline.from_pretrained(
        ckpt, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        overrides={"device_processor": {"device": device}})
    # The post-processor unnormalises an action tensor, so it needs the
    # tensor<->transition adapters the factory would otherwise supply.
    from lerobot.processor.converters import (
        policy_action_to_transition, transition_to_policy_action,
    )

    post = PolicyProcessorPipeline.from_pretrained(
        ckpt, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action)
    return policy, pre, post


def droid_observation(frames: dict[str, np.ndarray], joint_state: np.ndarray,
                      device: str, dtype: torch.dtype) -> dict:
    """Build one batch in the checkpoint's expected layout.

    Images are ``[B, 3, 224, 224]`` in ``[-1, 1]``; state is padded to 32 dims,
    which is how DROID's 8-DoF (7 joints + gripper) reaches a model trained
    across embodiments of differing width.
    """
    import torch.nn.functional as F

    batch = {}
    for key in CAMERA_KEYS:
        img = torch.from_numpy(frames[key]).float().permute(2, 0, 1)[None]
        img = F.interpolate(img, size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode="bilinear", align_corners=False)
        # The preprocessor normalises; hand it [0, 1] and no batch dimension,
        # which is the layout its AddBatchDimension step expects.
        batch[key] = (img[0] / 255.0).to(device=device, dtype=torch.float32)

    state = torch.zeros(32, dtype=torch.float32, device=device)
    state[: len(joint_state)] = torch.from_numpy(joint_state)
    batch["observation.state"] = state
    return batch


@torch.inference_mode()
def measure(policy, pre, post, observation: dict, prompt: str, *,
            warmup: int = 2, chunks: int = 5) -> tuple[LatencyReport, torch.Tensor]:
    """Time whole action chunks, which is what the robot actually waits on.

    Preprocessing is inside the timed region on purpose: normalisation and
    tokenisation happen once per control step in a real deployment, so they are
    part of the latency the robot experiences.
    """
    chunk_size = policy.config.chunk_size

    def one_chunk():
        # reset() clears the action queue, so select_action is forced to run the
        # model rather than pop an already-computed action.
        policy.reset()
        batch = pre({**observation, "task": prompt})
        return post(policy.select_action(batch))

    for _ in range(warmup):
        actions = one_chunk()
    torch.cuda.synchronize()

    times = []
    for _ in range(chunks):
        start = time.perf_counter()
        actions = one_chunk()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)
    return LatencyReport(times, chunk_size), actions
