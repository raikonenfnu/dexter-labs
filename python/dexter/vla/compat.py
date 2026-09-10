"""Compatibility shims for checkpoints newer than the installed lerobot.

Every shim here follows one rule: a difference may only be papered over once it
has been *checked to be inert*. Anything that could change what the model
computes raises instead. That is why each helper inspects values rather than
just dropping unknown keys -- a silently mismatched action space produces
plausible numbers and no error at all.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

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


# ---------------------------------------------------------------------------
# lerobot package-import shim
# ---------------------------------------------------------------------------
# lerobot.policies imports every policy eagerly, and the GR00T config in the
# pinned checkout declares field(init=False) members with no default after
# inherited defaulted ones, which Python 3.12 dataclasses reject. Stub that
# subpackage rather than editing lerobot.
import importlib.util
import sys
import types

def install_lerobot_shims() -> None:
    if "lerobot.policies.groot" in sys.modules:
        return
    for name in ("lerobot.policies.groot",
                 "lerobot.policies.groot.configuration_groot",
                 "lerobot.policies.groot.modeling_groot",
                 "lerobot.policies.groot.groot_n1"):
        module = types.ModuleType(name)
        module.__spec__ = importlib.util.spec_from_loader(name, loader=None)
        module.__path__ = []  # allow it to act as a package
        sys.modules[name] = module

    # The names lerobot.policies.__init__ pulls out of the stubbed modules.
    class _Missing:
        def __init__(self, *a, **k):
            raise RuntimeError("GR00T policy is stubbed out in this environment")

    sys.modules["lerobot.policies.groot.configuration_groot"].GrootConfig = _Missing
    sys.modules["lerobot.policies.groot.modeling_groot"].GrootPolicy = _Missing
