"""Platform detection and capability description.

The engine picks kernels from what the device can actually do, never from a
device name. Everything a kernel needs to know is a *feature string* on
:class:`PlatformInfo`; adding a new architecture means adding one row to
``_AMD_ARCHS`` and (optionally) a directory of kernels under ``ops/``.

RDNA3.5 (``gfx1151``, Strix Halo) is the reference target. It differs from the
CDNA parts most inference engines assume in three ways that change kernel
design, not just tuning constants:

* **wave32, WMMA instead of MFMA.** The matrix unit is
  ``v_wmma_*_16x16x16`` over a 32-lane wave, so the natural MMA tile is
  16x16x16 and there is no 32x32x8 accumulator to amortise over.
* **No FP8 matmul.** WMMA covers f16/bf16/int8/int4 only. The MI300X FP8
  recipe does not port; the equivalent memory saving comes from *weight-only*
  int8/int4 with a bf16 WMMA math path.
* **Unified LPDDR5X memory.** Host and device share one ~256 GB/s pool. There
  is no PCIe hop to hide, but there is also no HBM: weight traffic, not FLOPs,
  sets the floor for every DiT step.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field

__all__ = [
    "ArchVersion",
    "PlatformInfo",
    "Capability",
    "current_platform",
    "override_platform",
]


@dataclass(frozen=True, order=True)
class ArchVersion:
    """Hardware generation, ordered so kernels can gate on ``>=``."""

    major: int
    minor: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


@dataclass(frozen=True)
class AmdArch:
    """Static facts about one AMD GPU architecture."""

    version: ArchVersion
    family: str
    wavefront: int
    lds_bytes: int
    cus_per_group: int  # CUs per unit torch counts as a "multiprocessor"
    features: frozenset[str]


# Feature vocabulary used by kernel gates. Keep names stable; kernels match on them.
#   matrix:{f16,bf16,int8,int4,fp8}  -- dtypes the matrix unit accepts natively
#   mma:{wmma,mfma}                  -- which matrix instruction family
#   memory:{unified,async_copy,tma}  -- data movement the kernel may assume
#   dual_issue                       -- VOPD; favours wide elementwise epilogues
_AMD_ARCHS: dict[str, AmdArch] = {
    "gfx1151": AmdArch(  # RDNA3.5 APU (Strix Halo)
        version=ArchVersion(11, 5),
        family="RDNA3.5",
        wavefront=32,
        lds_bytes=64 * 1024,
        # RDNA pairs CUs into workgroup processors, and that is what torch
        # reports as multi_processor_count -- 20 WGPs on a 40-CU Strix Halo.
        cus_per_group=2,
        features=frozenset(
            {
                "matrix:f16",
                "matrix:bf16",
                "matrix:int8",
                "matrix:int4",
                "mma:wmma",
                "memory:unified",
                "dual_issue",
            }
        ),
    ),
    "gfx1150": AmdArch(  # RDNA3.5 APU (Strix Point) -- same ISA, fewer CUs
        version=ArchVersion(11, 5),
        family="RDNA3.5",
        wavefront=32,
        lds_bytes=64 * 1024,
        # RDNA pairs CUs into workgroup processors, and that is what torch
        # reports as multi_processor_count -- 20 WGPs on a 40-CU Strix Halo.
        cus_per_group=2,
        features=frozenset(
            {
                "matrix:f16",
                "matrix:bf16",
                "matrix:int8",
                "matrix:int4",
                "mma:wmma",
                "memory:unified",
                "dual_issue",
            }
        ),
    ),
    "gfx1100": AmdArch(  # RDNA3 dGPU (RX 7900) -- same WMMA, discrete VRAM
        version=ArchVersion(11, 0),
        family="RDNA3",
        wavefront=32,
        lds_bytes=64 * 1024,
        # RDNA pairs CUs into workgroup processors, and that is what torch
        # reports as multi_processor_count -- 20 WGPs on a 40-CU Strix Halo.
        cus_per_group=2,
        features=frozenset(
            {
                "matrix:f16",
                "matrix:bf16",
                "matrix:int8",
                "matrix:int4",
                "mma:wmma",
                "dual_issue",
            }
        ),
    ),
}


@dataclass(frozen=True)
class PlatformInfo:
    """Everything kernel selection is allowed to depend on."""

    vendor: str  # "amd" | "nvidia" | "cpu"
    arch: str  # "gfx1151", "sm90", ...
    arch_version: ArchVersion
    family: str  # "RDNA3.5", "CDNA4", "Hopper", ...
    device_name: str
    compute_units: int
    wavefront: int
    lds_bytes: int
    total_memory: int  # bytes
    memory_bandwidth: float  # GB/s, measured or datasheet
    features: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_amd(self) -> bool:
        return self.vendor == "amd"

    @property
    def is_rdna3_plus(self) -> bool:
        return self.is_amd and self.arch_version >= ArchVersion(11, 0)

    @property
    def has_unified_memory(self) -> bool:
        return "memory:unified" in self.features

    def supports(self, *features: str) -> bool:
        return all(f in self.features for f in features)

    def describe(self) -> str:
        gb = self.total_memory / 1e9
        return (
            f"{self.device_name} [{self.arch} / {self.family}] "
            f"{self.compute_units} CU, wave{self.wavefront}, "
            f"{gb:.1f} GB @ {self.memory_bandwidth:.0f} GB/s"
        )


@dataclass(frozen=True)
class Capability:
    """A kernel's requirements on the platform."""

    vendors: frozenset[str] | None = None
    min_arch: ArchVersion | None = None
    max_arch: ArchVersion | None = None
    features: frozenset[str] = field(default_factory=frozenset)

    def satisfied_by(self, platform: PlatformInfo) -> bool:
        if self.vendors is not None and platform.vendor not in self.vendors:
            return False
        if self.min_arch is not None and platform.arch_version < self.min_arch:
            return False
        if self.max_arch is not None and platform.arch_version > self.max_arch:
            return False
        return self.features.issubset(platform.features)


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

# Datasheet peak bandwidth in GB/s. Used only for roofline reporting; the
# benchmark harness measures the achievable number and prefers that.
_PEAK_BANDWIDTH: dict[str, float] = {
    "gfx1151": 256.0,  # 256-bit LPDDR5X-8000, shared with the CPU
    "gfx1150": 128.0,  # 128-bit LPDDR5X-7500
    "gfx1100": 960.0,  # 384-bit GDDR6
}


def _detect_amd(props, device_name: str, device_count: int) -> PlatformInfo:
    arch = props.gcnArchName.split(":")[0]  # strip ":xnack-" style target features
    known = _AMD_ARCHS.get(arch)
    if known is None:
        raise RuntimeError(
            f"dexter does not know AMD architecture {arch!r}. "
            f"Add a row to platform._AMD_ARCHS; supported: {', '.join(sorted(_AMD_ARCHS))}."
        )
    return PlatformInfo(
        vendor="amd",
        arch=arch,
        arch_version=known.version,
        family=known.family,
        device_name=device_name,
        compute_units=props.multi_processor_count * known.cus_per_group,
        wavefront=known.wavefront,
        lds_bytes=known.lds_bytes,
        total_memory=props.total_memory,
        memory_bandwidth=_PEAK_BANDWIDTH.get(arch, 0.0),
        features=known.features,
    )


def _detect_cpu() -> PlatformInfo:
    """Fallback so the reference kernels and tests run without a GPU."""
    return PlatformInfo(
        vendor="cpu",
        arch="cpu",
        arch_version=ArchVersion(0, 0),
        family="CPU",
        device_name="cpu",
        compute_units=os.cpu_count() or 1,
        wavefront=1,
        lds_bytes=0,
        total_memory=0,
        memory_bandwidth=0.0,
        features=frozenset(),
    )


@functools.lru_cache(maxsize=1)
def _detect() -> PlatformInfo:
    import torch

    if not torch.cuda.is_available():
        return _detect_cpu()

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if getattr(torch.version, "hip", None):
        return _detect_amd(props, props.name, torch.cuda.device_count())

    return PlatformInfo(
        vendor="nvidia",
        arch=f"sm{props.major}{props.minor}",
        arch_version=ArchVersion(props.major, props.minor),
        family="NVIDIA",
        device_name=props.name,
        compute_units=props.multi_processor_count,  # SMs map 1:1
        wavefront=32,
        lds_bytes=props.shared_memory_per_block,
        total_memory=props.total_memory,
        memory_bandwidth=0.0,
        features=frozenset({"matrix:f16", "matrix:bf16", "mma:mma"}),
    )


_override: PlatformInfo | None = None


def current_platform() -> PlatformInfo:
    """Detected platform, cached for the life of the process."""
    return _override if _override is not None else _detect()


def override_platform(platform: PlatformInfo | None) -> None:
    """Force a platform, for tests and for cross-arch kernel-gate checks."""
    global _override
    _override = platform
