"""The contract every VLA in this engine satisfies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class Observation:
    """What a policy is given at one control step.

    Images stay as native-resolution uint8 HxWx3: every VLA here resizes
    internally (with padding, to preserve aspect ratio), so resizing at the
    call site would distort the frame before the model's own resize.
    """

    images: dict[str, np.ndarray]
    state: np.ndarray
    task: str = ""
    extra: dict = field(default_factory=dict)


class ActionChunkPolicy(Protocol):
    """A policy that emits a chunk of future actions from one observation."""

    name: str
    chunk_size: int          # actions produced per call
    control_hz: float        # rate the chunk is executed at

    def act(self, observation: Observation) -> np.ndarray:
        """Return ``[chunk_size, action_dim]`` in the robot's action units."""

    @property
    def budget_ms(self) -> float:
        """Wall time one chunk buys, i.e. the deadline for producing it."""
        return self.chunk_size / self.control_hz * 1e3
