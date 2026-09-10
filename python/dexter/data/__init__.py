"""Dataset readers.

One reader per storage format, not per robot. `LeRobotSlice` knows how LeRobot
v3 datasets are laid out (parquet frames, chunked videos, episode metadata);
what those columns *mean* for a given robot lives in the policy adapter.
"""

from dexter.data.lerobot import LeRobotSlice, Sample

__all__ = ["LeRobotSlice", "Sample"]
