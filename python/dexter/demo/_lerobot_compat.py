"""Import shim for lerobot.

``lerobot.policies.__init__`` eagerly imports every policy, and the GR00T
config in the currently checked-out lerobot declares ``field(init=False)``
members with no default after inherited defaulted ones, which Python 3.12's
dataclasses reject outright. That has nothing to do with the diffusion policy
used here, so this stubs the GR00T subpackage before the import runs rather
than editing lerobot itself.

Remove this once lerobot's GR00T config gains defaults.
"""

from __future__ import annotations

import importlib.util
import sys
import types


def install() -> None:
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
