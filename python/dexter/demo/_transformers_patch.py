"""Apply openpi's transformers patch, reversibly.

LeRobot's π0.5 refuses to build unless a patched `transformers` is installed:
it looks for `transformers.models.siglip.check`, which upstream does not ship.
Both openpi and lerobot distribute the same fix -- openpi as files to copy over
the installed package, lerobot as a git branch of transformers.

Copying files into site-packages is the openpi-documented route and needs no
network, but it edits a shared dependency, so every original is backed up next
to itself as ``*.orig`` and `revert()` puts them back. The patch touches
siglip, paligemma and gemma only; nothing else in this environment uses those.
"""

from __future__ import annotations

import shutil
from pathlib import Path

PATCH_SRC = Path.home() / "nod/openpi/src/openpi/models_pytorch/transformers_replace"
REQUIRED_VERSION = "4.53.2"


def _transformers_dir() -> Path:
    import transformers

    return Path(transformers.__file__).parent


def status() -> dict:
    import transformers

    dst = _transformers_dir()
    files = sorted(PATCH_SRC.rglob("*.py")) if PATCH_SRC.exists() else []
    return {
        "version": transformers.__version__,
        "version_ok": transformers.__version__ == REQUIRED_VERSION,
        "patch_available": bool(files),
        "applied": (dst / "models/siglip/check.py").exists(),
        "files": [str(f.relative_to(PATCH_SRC)) for f in files],
    }


def apply() -> list[str]:
    """Copy the patch in, backing up any file it replaces. Returns files written."""
    import transformers

    if transformers.__version__ != REQUIRED_VERSION:
        raise RuntimeError(
            f"patch targets transformers {REQUIRED_VERSION}, found "
            f"{transformers.__version__}; copying it onto a different version "
            f"would be worse than not patching"
        )
    if not PATCH_SRC.exists():
        raise FileNotFoundError(f"openpi patch files not found at {PATCH_SRC}")

    dst_root = _transformers_dir()
    written = []
    for src in sorted(PATCH_SRC.rglob("*.py")):
        dst = dst_root / src.relative_to(PATCH_SRC)
        dst.parent.mkdir(parents=True, exist_ok=True)
        backup = dst.with_suffix(dst.suffix + ".orig")
        if dst.exists() and not backup.exists():
            shutil.copy2(dst, backup)
        shutil.copy2(src, dst)
        written.append(str(dst.relative_to(dst_root)))
    return written


def revert() -> list[str]:
    """Restore every backed-up original; remove files the patch added."""
    dst_root = _transformers_dir()
    restored = []
    for src in sorted(PATCH_SRC.rglob("*.py")):
        dst = dst_root / src.relative_to(PATCH_SRC)
        backup = dst.with_suffix(dst.suffix + ".orig")
        if backup.exists():
            shutil.move(str(backup), str(dst))
            restored.append(str(dst.relative_to(dst_root)))
        elif dst.exists():
            dst.unlink()
            restored.append(f"{dst.relative_to(dst_root)} (removed, was added)")
    return restored
