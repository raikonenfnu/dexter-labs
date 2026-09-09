"""A closed-loop robot demo driven by the DreamZero policy.

This follows the contract in ``eval_utils/run_sim_eval.py`` upstream, because
that is the part of a VLA deployment the engine actually has to satisfy:

* the policy returns a chunk of ``num_action_per_block`` (32) actions;
* only ``open_loop_horizon`` (8) of them are executed;
* then the policy is queried again with a fresh observation.

Upstream that query is a **blocking** websocket call, so the robot stops moving
for the whole inference. At DROID's 15 Hz that is the dominant deployment
problem on this hardware, and it is why the vLLM-Omni post's 564 -> 398 ms
mattered: the number being cut is dead time the arm spends frozen mid-motion.

Two schedulers are implemented so the difference is measurable rather than
asserted:

``blocking``   faithful to upstream -- infer inline, arm waits.
``pipelined``  start inference ``prefetch`` actions before the chunk runs out,
               on a worker thread, so it overlaps execution of the actions
               already in hand. The robot only stalls if inference is slower
               than the buffer it was given to hide behind.

The arm is a kinematic stand-in, not a dynamics simulator, and with
uninitialised weights the actions are not meaningful robot commands. What is
real, and what this measures, is the *timing*: control rate held, stall
duration, and deadline misses.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from dexter.models.dreamzero.policy import DreamZeroPolicy

# DROID's action space and control rate, from the upstream client.
ACTION_DIM = 8  # 7 joints + 1 gripper
CONTROL_HZ = 15.0
OPEN_LOOP_HORIZON = 8


@dataclass
class ReachTask:
    """A 7-DoF arm moving toward a target joint configuration.

    Deliberately trivial: the demo is about when actions arrive, not what they
    are. Progress is reported so a run that *did* have trained weights would
    show it, and so a stalled loop is visible as flat progress over wall time.
    """

    target: np.ndarray = field(default_factory=lambda: np.array(
        [0.4, -0.6, 0.3, -1.2, 0.2, 0.8, -0.4], dtype=np.float32))
    joints: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    gripper: float = 0.0
    max_joint_step: float = 0.05  # rad per control tick, a plausible velocity clamp

    def apply(self, action: np.ndarray) -> None:
        delta = np.clip(action[:7], -self.max_joint_step, self.max_joint_step)
        self.joints = self.joints + delta
        self.gripper = 1.0 if action[7] > 0.5 else 0.0  # upstream binarises this

    @property
    def error(self) -> float:
        return float(np.linalg.norm(self.target - self.joints))

    def observation(self, cfg, device, dtype):
        """Encoded observation for the policy.

        A real deployment runs three 180x320 camera views through the Wan VAE
        to get this latent. The VAE is out of scope for this engine (see the
        README), so the demo hands the DiT a latent of the right shape directly
        and keeps the joint and gripper readings as the state vector, which is
        what the action register actually consumes.
        """
        latent = torch.randn(1, cfg.video_tokens, cfg.condition_dim or cfg.patch_dim,
                             device=device, dtype=dtype)
        state = torch.zeros(1, cfg.num_state_per_block, cfg.max_state_dim,
                            device=device, dtype=dtype)
        state[0, 0, :7] = torch.from_numpy(self.joints).to(device=device, dtype=dtype)
        state[0, 0, 7] = self.gripper
        return latent, state


@dataclass
class RunStats:
    """Timing of one episode. Every field is wall time the robot experienced."""

    scheduler: str
    ticks: int = 0
    policy_calls: int = 0
    wall_s: float = 0.0
    stall_ms: list[float] = field(default_factory=list)
    tick_ms: list[float] = field(default_factory=list)
    final_error: float = 0.0

    @property
    def achieved_hz(self) -> float:
        return self.ticks / self.wall_s if self.wall_s else 0.0

    @property
    def total_stall_s(self) -> float:
        return sum(self.stall_ms) / 1e3

    @property
    def stall_fraction(self) -> float:
        return self.total_stall_s / self.wall_s if self.wall_s else 0.0

    @property
    def deadline_misses(self) -> int:
        """Ticks that overran the control period -- the robot fell behind."""
        period_ms = 1e3 / CONTROL_HZ
        return sum(1 for t in self.tick_ms if t > period_ms * 1.05)

    def summary(self) -> str:
        worst = max(self.stall_ms) if self.stall_ms else 0.0
        return (
            f"{self.scheduler:<10} {self.achieved_hz:6.1f} Hz  "
            f"{self.policy_calls:3d} calls  "
            f"stall {self.total_stall_s:5.2f}s ({self.stall_fraction * 100:4.1f}%)  "
            f"worst {worst:6.1f} ms  "
            f"misses {self.deadline_misses:3d}/{self.ticks}"
        )


class _Inference:
    """Runs the policy, either inline or on a worker thread."""

    def __init__(self, policy: DreamZeroPolicy, task: ReachTask) -> None:
        self.policy = policy
        self.task = task
        self.cfg = policy.cfg
        self._result: queue.Queue = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None

    def _run(self) -> np.ndarray:
        latent, state = self.task.observation(self.cfg, self.policy.device, self.policy.dtype)
        actions = self.policy.step(latent, state)
        return actions[0].float().cpu().numpy()

    def call_inline(self) -> np.ndarray:
        return self._run()

    def start(self) -> None:
        """Kick off inference in the background. At most one is ever in flight."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=lambda: self._result.put(self._run()), daemon=True)
        self._thread.start()

    def collect(self) -> np.ndarray:
        """Wait for the in-flight inference. Returns immediately if it finished."""
        if self._thread is None:
            return self.call_inline()
        actions = self._result.get()
        self._thread.join()
        self._thread = None
        return actions

    def drain(self) -> None:
        """Retire any in-flight inference before the episode ends.

        Without this, a worker can still be inside a HIP call when the
        interpreter starts tearing down, which surfaces as a bare
        "terminate called without an active exception" after the results print.
        """
        if self._thread is not None:
            self.collect()


def _to_robot_actions(chunk: np.ndarray) -> np.ndarray:
    """Take the first ACTION_DIM dims of the policy's action space."""
    return chunk[:, :ACTION_DIM]


def run_episode(
    policy: DreamZeroPolicy,
    *,
    scheduler: str = "blocking",
    ticks: int = 120,
    control_hz: float = CONTROL_HZ,
    open_loop_horizon: int = OPEN_LOOP_HORIZON,
    prefetch: int = 4,
    task: ReachTask | None = None,
) -> RunStats:
    """Drive the arm for ``ticks`` control steps and report the timing.

    ``prefetch`` (pipelined only) is how many actions before the chunk runs out
    to start the next inference. It is the knob that decides whether the robot
    ever stalls: the overlap available is ``prefetch / control_hz`` seconds.
    """
    if scheduler not in ("blocking", "pipelined"):
        raise ValueError(f"unknown scheduler {scheduler!r}")

    task = task or ReachTask()
    engine = _Inference(policy, task)
    stats = RunStats(scheduler=scheduler)
    period = 1.0 / control_hz

    policy.reset()
    chunk: np.ndarray | None = None
    consumed = 0

    episode_start = time.perf_counter()
    for _ in range(ticks):
        tick_start = time.perf_counter()

        # Refill the action buffer when the open-loop horizon is exhausted.
        if chunk is None or consumed >= open_loop_horizon:
            stall_start = time.perf_counter()
            chunk = _to_robot_actions(engine.collect() if scheduler == "pipelined"
                                     else engine.call_inline())
            stats.stall_ms.append((time.perf_counter() - stall_start) * 1e3)
            stats.policy_calls += 1
            consumed = 0

        # Start the next inference early enough that it lands before we run dry.
        if scheduler == "pipelined" and consumed == max(open_loop_horizon - prefetch, 0):
            engine.start()

        task.apply(chunk[consumed])
        consumed += 1
        stats.ticks += 1

        # Hold the control rate. Falling behind here is a real deadline miss,
        # not a scheduling artefact -- there is no time left to sleep away.
        elapsed = time.perf_counter() - tick_start
        if elapsed < period:
            time.sleep(period - elapsed)
        stats.tick_ms.append((time.perf_counter() - tick_start) * 1e3)

    stats.wall_s = time.perf_counter() - episode_start
    engine.drain()
    stats.final_error = task.error
    return stats
