"""Shared gates and launch heuristics for the RDNA3.5 kernels."""

from __future__ import annotations

import triton

from dexter.platform import ArchVersion, Capability

# Every kernel in this package targets RDNA3+ WMMA in wave32. Kernels that are
# additionally APU-specific add "memory:unified" to their own capability.
RDNA3 = Capability(
    vendors=frozenset({"amd"}),
    min_arch=ArchVersion(11, 0),
    features=frozenset({"mma:wmma", "matrix:bf16"}),
)

WAVEFRONT = 32


def row_warps(num_elements: int) -> int:
    """Waves per row-wise program.

    A row-per-program kernel wants enough lanes to keep the memory pipe full
    without spilling the row out of registers. At wave32, 8 waves is 256 lanes,
    which covers a 3072- or 5120-wide DiT row at 12-20 elements per lane.
    """
    if num_elements <= 1024:
        return 2
    if num_elements <= 4096:
        return 8
    return 16


def block_size(num_elements: int) -> int:
    return triton.next_power_of_2(num_elements)
