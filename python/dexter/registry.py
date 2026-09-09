"""Kernel registry and selection.

One op ("family/mode", e.g. ``gemm/w4a16``) has many implementations. Each
declares what hardware it needs and how specialised it is; selection keeps the
candidates the platform satisfies and takes the most specialised one.

The rules are deliberately boring, because kernel dispatch that surprises you
is worse than kernel dispatch that is slow:

* A kernel that does not declare a capability it needs is a bug, not a
  fallback. Gates are checked before priority is ever consulted.
* Every family has a ``REFERENCE`` torch implementation. It is never selected
  over a real kernel, and it is what correctness tests compare against.
* ``DEXTER_KERNEL=<family/mode>:<name>`` pins one choice, so a regression can
  be bisected without editing code.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Iterator

from dexter.platform import Capability, PlatformInfo, current_platform

logger = logging.getLogger(__name__)

__all__ = ["Priority", "KernelSpec", "register", "select", "candidates", "explain"]


class Priority(IntEnum):
    """How specialised an implementation is. Higher wins.

    REFERENCE   readable torch; the numeric ground truth, never auto-selected
                over a real kernel.
    PORTABLE    generic Triton that runs on any supported vendor.
    PERFORMANT  optimised for an architecture family (e.g. all of RDNA3+).
    SPECIALISED tuned for one target and gated on shape as well as arch.
    """

    REFERENCE = 0
    PORTABLE = 4
    PERFORMANT = 8
    SPECIALISED = 12


@dataclass(frozen=True)
class KernelSpec:
    name: str
    family: str  # "gemm", "norm", "attention", "rope"
    mode: str  # "w4a16", "adaln", "blockwise_causal", ...
    impl: Callable
    capability: Capability = field(default_factory=Capability)
    priority: Priority | int = Priority.PERFORMANT
    # Optional predicate on the call's shapes; lets a kernel decline work it is
    # not tuned for without being unregistered everywhere.
    accepts: Callable[..., bool] | None = None

    @property
    def key(self) -> str:
        return f"{self.family}/{self.mode}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<KernelSpec {self.key}:{self.name} pri={int(self.priority)}>"


_REGISTRY: dict[str, list[KernelSpec]] = defaultdict(list)


def register(
    family: str,
    mode: str,
    *,
    name: str,
    capability: Capability | None = None,
    priority: Priority | int = Priority.PERFORMANT,
    accepts: Callable[..., bool] | None = None,
) -> Callable[[Callable], Callable]:
    """Decorator registering ``fn`` as an implementation of ``family/mode``."""

    def decorator(fn: Callable) -> Callable:
        spec = KernelSpec(
            name=name,
            family=family,
            mode=mode,
            impl=fn,
            capability=capability or Capability(),
            priority=priority,
            accepts=accepts,
        )
        bucket = _REGISTRY[spec.key]
        if any(existing.name == name for existing in bucket):
            raise ValueError(f"duplicate kernel {spec.key}:{name}")
        bucket.append(spec)
        # Sort once at registration; selection is then a linear scan of a
        # already-ordered list and stays allocation-free on the hot path.
        bucket.sort(key=lambda s: int(s.priority), reverse=True)
        fn.dexter_spec = spec  # type: ignore[attr-defined]
        return fn

    return decorator


def candidates(family: str, mode: str, platform: PlatformInfo | None = None) -> Iterator[KernelSpec]:
    """Registered kernels for ``family/mode`` this platform can run, best first."""
    platform = platform or current_platform()
    for spec in _REGISTRY[f"{family}/{mode}"]:
        if spec.capability.satisfied_by(platform):
            yield spec


def _pinned(key: str) -> str | None:
    """Read ``DEXTER_KERNEL=gemm/w4a16:name,norm/adaln:other``."""
    env = os.environ.get("DEXTER_KERNEL", "")
    for entry in env.split(","):
        entry = entry.strip()
        if not entry:
            continue
        pin_key, _, pin_name = entry.partition(":")
        if pin_key == key:
            return pin_name
    return None


def select(family: str, mode: str, *args, **kwargs) -> Callable:
    """Best implementation of ``family/mode`` for this platform and call shape.

    ``args``/``kwargs`` are only inspected by a spec's ``accepts`` predicate;
    they are not forwarded.
    """
    key = f"{family}/{mode}"
    pin = _pinned(key)
    viable = list(candidates(family, mode))

    if pin is not None:
        for spec in viable:
            if spec.name == pin:
                return spec.impl
        raise LookupError(f"DEXTER_KERNEL pinned {key}:{pin}, which is not available here")

    for spec in viable:
        if spec.accepts is not None and not spec.accepts(*args, **kwargs):
            continue
        return spec.impl

    raise LookupError(
        f"no kernel for {key} on {current_platform().describe()}. "
        f"Registered: {[s.name for s in _REGISTRY[key]] or 'none'}"
    )


def explain(family: str, mode: str, *args, **kwargs) -> str:
    """Human-readable account of why a kernel was picked. For `dexter-bench`."""
    key = f"{family}/{mode}"
    platform = current_platform()
    lines = [f"{key} on {platform.describe()}"]
    chosen = None
    for spec in _REGISTRY[key]:
        if not spec.capability.satisfied_by(platform):
            missing = sorted(spec.capability.features - platform.features)
            reason = f"needs {missing}" if missing else "arch gate"
            lines.append(f"  - {spec.name:<28} skipped ({reason})")
        elif spec.accepts is not None and not spec.accepts(*args, **kwargs):
            lines.append(f"  - {spec.name:<28} skipped (shape not accepted)")
        elif chosen is None:
            chosen = spec
            lines.append(f"  * {spec.name:<28} SELECTED (pri={int(spec.priority)})")
        else:
            lines.append(f"  - {spec.name:<28} viable (pri={int(spec.priority)})")
    return "\n".join(lines)
