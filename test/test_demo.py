"""The closed-loop scheduler must obey the upstream chunking contract."""
import dataclasses

import numpy as np
import pytest
import torch

from dexter.demo.robot import ACTION_DIM, ReachTask, run_episode
from dexter.models.dreamzero import CausalWanDiT, WAN22_TI2V_5B
from dexter.models.dreamzero.policy import DreamZeroPolicy

TINY = dataclasses.replace(
    WAN22_TI2V_5B, name="tiny", num_layers=1, dim=768, ffn_dim=1536, num_heads=6
)


@pytest.fixture(scope="module")
def policy():
    model = CausalWanDiT(TINY)
    pol = DreamZeroPolicy(model, batch=1, cache_blocks=64)
    pol.set_instruction(torch.randn(1, TINY.text_len, TINY.text_dim,
                                    device="cuda", dtype=model.dtype))
    return pol


@pytest.mark.parametrize("scheduler", ["blocking", "pipelined"])
def test_one_policy_call_per_open_loop_horizon(policy, scheduler):
    """The policy is queried every ``horizon`` ticks, not every tick.

    This is the whole point of action chunking: a 15 Hz robot driven by a
    2.8 Hz policy only works because one inference covers many ticks.
    """
    ticks, horizon = 24, 8
    stats = run_episode(policy, scheduler=scheduler, ticks=ticks,
                        control_hz=200.0, open_loop_horizon=horizon)
    assert stats.ticks == ticks
    assert stats.policy_calls == ticks // horizon


def test_horizon_cannot_exceed_the_chunk(policy):
    """Executing more actions than the chunk holds would index past its end."""
    with pytest.raises(IndexError):
        run_episode(policy, scheduler="blocking", ticks=40, control_hz=200.0,
                    open_loop_horizon=TINY.num_action_per_block + 1)


def test_pipelining_never_makes_the_arm_stall_more(policy):
    ticks, horizon = 32, 8
    kwargs = dict(ticks=ticks, control_hz=60.0, open_loop_horizon=horizon)
    blocking = run_episode(policy, scheduler="blocking", **kwargs)
    pipelined = run_episode(policy, scheduler="pipelined", prefetch=horizon, **kwargs)
    assert pipelined.total_stall_s <= blocking.total_stall_s
    # Cold start is unavoidable: the first chunk has nothing to overlap with.
    assert len(pipelined.stall_ms) == len(blocking.stall_ms)


def test_actions_move_the_arm_and_binarise_the_gripper():
    task = ReachTask()
    start = task.joints.copy()
    task.apply(np.array([0.5] * 7 + [0.9], dtype=np.float32))
    # Joint deltas are velocity-clamped, and the gripper is a binary command.
    assert np.allclose(task.joints - start, task.max_joint_step)
    assert task.gripper == 1.0
    task.apply(np.zeros(ACTION_DIM, dtype=np.float32))
    assert task.gripper == 0.0


def test_reported_rate_reflects_wall_time(policy):
    stats = run_episode(policy, scheduler="blocking", ticks=20, control_hz=50.0,
                        open_loop_horizon=8)
    assert stats.achieved_hz == pytest.approx(stats.ticks / stats.wall_s)
    assert stats.achieved_hz <= 50.0 * 1.05


def test_min_max_normalisation_round_trips():
    """The PushT demo normalises outside the policy, so the arithmetic is ours.

    LeRobot's MIN_MAX maps [lo, hi] onto [-1, 1]. Getting this wrong does not
    raise -- it produces a policy that appears simply unable to do the task.
    """
    from dexter.demo.pusht import _min_max_to_unit, _unit_to_min_max

    lo = torch.tensor([12.0, 25.0])
    hi = torch.tensor([511.0, 511.0])
    x = torch.tensor([[12.0, 268.0], [511.0, 25.0]])
    unit = _min_max_to_unit(x, lo, hi)
    assert unit.min() >= -1.0 - 1e-6 and unit.max() <= 1.0 + 1e-6
    assert torch.allclose(_unit_to_min_max(unit, lo, hi), x, atol=1e-4)
