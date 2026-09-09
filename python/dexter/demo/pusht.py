"""A robot that actually completes a task.

DreamZero is a 14B video world model: on this hardware one control block takes
~24 s, so it can predict what a robot *would* do but cannot drive one. To see a
policy close a loop and finish something, this runs a small trained policy on a
real simulated task.

The task is PushT: a 2-DoF end effector shoves a T-shaped block onto a target
outline. Reward is the fraction of the target the block covers, and an episode
succeeds at 95% coverage. The policy is LeRobot's pretrained
``lerobot/diffusion_pusht`` -- a diffusion policy over action chunks, which is
the same family of idea as DreamZero's action head, three orders of magnitude
smaller.

What this shares with the rest of the engine is the shape of the problem:
observe, denoise an action chunk, execute part of it, re-plan. What it does not
share is the backbone, so it runs through LeRobot rather than dexter ops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class EpisodeResult:
    """Outcome of one episode.

    ``coverage`` is the env's own ``info["coverage"]``, not its ``reward``:
    reward is ``clip(coverage / 0.95, 0, 1)``, so it saturates at 1.00 for
    anything at or above threshold and cannot distinguish "just solved" from
    "perfectly solved". Success is the env's ``info["is_success"]``.
    """

    coverage: float = 0.0        # final true coverage
    best_coverage: float = 0.0   # best true coverage reached
    success: bool = False
    steps: int = 0
    frames: list = field(default_factory=list)


# The published checkpoint stores normalisation statistics under names this
# lerobot version no longer reads, so `from_pretrained` drops them with only a
# warning and the policy runs unnormalised -- which looks like a policy that
# simply cannot do the task (0.00 coverage) rather than like a loading bug.
# They are recovered from the checkpoint and passed in explicitly.
_STAT_KEYS = {
    "observation.image": {"mean": "normalize_inputs.buffer_observation_image.mean",
                          "std": "normalize_inputs.buffer_observation_image.std"},
    "observation.state": {"min": "normalize_inputs.buffer_observation_state.min",
                          "max": "normalize_inputs.buffer_observation_state.max"},
    "action": {"min": "normalize_targets.buffer_action.min",
               "max": "normalize_targets.buffer_action.max"},
}


def load_dataset_stats(repo: str) -> dict:
    """Pull the normalisation statistics out of the published checkpoint."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    tensors = load_file(hf_hub_download(repo, "model.safetensors"))
    stats = {}
    for feature, keys in _STAT_KEYS.items():
        if all(k in tensors for k in keys.values()):
            stats[feature] = {name: tensors[key].clone() for name, key in keys.items()}
    if not stats:
        raise RuntimeError(f"{repo}: no normalisation statistics found in the checkpoint")
    return stats


def load_policy(device: str = "cuda", repo: str = "lerobot/diffusion_pusht"):
    """Return ``(policy, stats)``.

    This lerobot version no longer normalises inside the policy -- the
    statistics live in the checkpoint but nothing reads them, and the
    docstring promising a ``dataset_stats`` argument is stale. So the caller
    normalises, which is also the version-proof option: the arithmetic below
    is fixed by the checkpoint, not by whichever lerobot is installed.
    """
    from dexter.demo._lerobot_compat import install

    install()
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    policy = DiffusionPolicy.from_pretrained(repo)
    stats = {k: {n: t.to(device) for n, t in v.items()}
             for k, v in load_dataset_stats(repo).items()}
    return policy.to(device).eval(), stats


def _min_max_to_unit(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """LeRobot's MIN_MAX normalisation: map [lo, hi] onto [-1, 1]."""
    return (x - lo) / (hi - lo) * 2.0 - 1.0


def _unit_to_min_max(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) / 2.0 * (hi - lo) + lo


@torch.inference_mode()
def run_episode(policy, env, stats, *, seed: int = 0, record: bool = True) -> EpisodeResult:
    """One episode, driven by the policy until the env terminates."""
    policy.reset()
    obs, _ = env.reset(seed=seed)
    device = next(policy.parameters()).device
    result = EpisodeResult()

    done = False
    while not done:
        # Channel-first image in [0, 1], mean/std normalised; agent position
        # in raw pixel coordinates, min/max normalised to [-1, 1].
        image = torch.from_numpy(obs["pixels"]).float().permute(2, 0, 1)[None].to(device) / 255.0
        image = (image - stats["observation.image"]["mean"]) / stats["observation.image"]["std"]
        state = torch.from_numpy(obs["agent_pos"]).float()[None].to(device)
        state = _min_max_to_unit(state, stats["observation.state"]["min"],
                                 stats["observation.state"]["max"])

        normed = policy.select_action({"observation.image": image,
                                       "observation.state": state})
        action = _unit_to_min_max(normed, stats["action"]["min"],
                                  stats["action"]["max"])[0].cpu().numpy()

        obs, _, terminated, truncated, info = env.step(action)
        if record:
            result.frames.append(env.render())
        result.steps += 1
        result.coverage = float(info["coverage"])
        result.best_coverage = max(result.best_coverage, result.coverage)
        result.success = result.success or bool(info["is_success"])
        done = terminated or truncated

    return result


def make_env(max_steps: int = 300):
    import gym_pusht  # noqa: F401  (registers the environment)
    import gymnasium as gym

    return gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                    render_mode="rgb_array", max_episode_steps=max_steps)


def save_video(frames: list, path: str | Path, fps: int = 30) -> None:
    import imageio.v3 as iio

    iio.imwrite(str(path), np.stack(frames), fps=fps, codec="libx264")
