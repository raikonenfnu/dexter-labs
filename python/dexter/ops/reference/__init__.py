"""Torch reference implementations.

These define what every op *means*. They are registered at
``Priority.REFERENCE`` so a real kernel always wins, and they are the tensor a
correctness test compares against. Keep them obvious rather than fast.
"""

from dexter.ops.reference import attention, gemm, norm, rope  # noqa: F401
