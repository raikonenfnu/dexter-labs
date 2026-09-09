"""Measurement helpers.

Everything reported by ``dexter-bench`` goes through here so the timing
methodology is stated once: warm up until Triton has compiled and the clocks
have settled, then take the *median* of the samples. Median rather than mean
because this is a shared-power APU -- the CPU and GPU trade a single package
budget, so the slow tail is thermal, not algorithmic, and averaging it in
tells you about the chassis rather than the kernel.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import torch


def measure(fn: Callable[[], object], *, warmup: int = 5, iters: int = 20) -> float:
    """Median wall time of ``fn`` in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def measure_kernel(fn: Callable[[], object], *, warmup: int = 20,
                   reps: int = 200, trials: int = 7) -> float:
    """Per-call time of a *small* kernel, in milliseconds.

    Distinct from :func:`measure` because at these sizes the thing being
    measured is smaller than the thing measuring it. A DiT elementwise kernel
    runs in ~20 microseconds; a Python call plus a HIP launch plus a
    synchronise is comparable, so timing one call per sync measures the harness
    and reports it as the kernel. Timing ``reps`` back-to-back calls behind a
    single sync amortises that away, and the *minimum* over trials is taken
    rather than the median: the floor is the kernel, everything above it is
    contention from a CPU sharing this package's power budget.

    Measured both ways, the same adaln kernel reads as 0.59x or 1.60x against
    its torch reference. Only the second number is about the kernel.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(trials):
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) / reps * 1e3)
    return min(samples)


def bandwidth(size_mb: int = 1024, dtype: torch.dtype = torch.bfloat16) -> float:
    """Achievable read bandwidth in GB/s, from a large streaming reduction.

    Read-only rather than copy: weight streaming is what the DiT does, and on
    LPDDR a read-modify-write mix measures lower than a pure read.
    """
    n = size_mb * 1024 * 1024 // dtype.itemsize
    buf = torch.randn(n, device="cuda", dtype=dtype)
    ms = measure(lambda: buf.sum(), warmup=3, iters=10)
    gbps = buf.numel() * dtype.itemsize / (ms * 1e-3) / 1e9
    del buf
    torch.cuda.empty_cache()
    return gbps
