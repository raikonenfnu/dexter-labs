"""DreamZero -- a World Action Model, served for RDNA3.5.

DreamZero (NVIDIA GEAR) is a Wan video DiT with an *action register* spliced
into the token sequence. One closed-loop step denoises a chunk of robot actions
and the video frames that would follow from them, jointly.
"""

from dexter.models.dreamzero.config import WAMConfig, WAN21_I2V_14B, WAN22_TI2V_5B
from dexter.models.dreamzero.model import CausalWanDiT, KVCache
from dexter.models.dreamzero.policy import DreamZeroPolicy, StepTrace

__all__ = [
    "WAMConfig", "WAN21_I2V_14B", "WAN22_TI2V_5B",
    "CausalWanDiT", "KVCache", "DreamZeroPolicy", "StepTrace",
]
