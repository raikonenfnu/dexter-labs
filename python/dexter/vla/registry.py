"""Name -> loader, so callers never import a policy module directly."""

from __future__ import annotations

from typing import Callable

_LOADERS: dict[str, Callable] = {}


def register(name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        _LOADERS[name] = fn
        return fn
    return decorator


def available() -> list[str]:
    _import_all()
    return sorted(_LOADERS)


def load(name: str, **kwargs):
    _import_all()
    if name not in _LOADERS:
        raise KeyError(f"unknown policy {name!r}; available: {sorted(_LOADERS)}")
    return _LOADERS[name](**kwargs)


def _import_all() -> None:
    from dexter.vla import pi05  # noqa: F401  (registers "pi05_droid")
