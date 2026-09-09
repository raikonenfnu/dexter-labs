"""Perception front-end: pixels and language into the DiT's conditioning.

These are *vendored* from NVIDIA's DreamZero release (Apache 2.0) with only the
edits needed to run here, rather than re-expressed on dexter ops like the DiT
was. That is a deliberate split, not laziness: the VAE is 0.25 GB and the CLIP
tower 1.26 GB, and each runs once per control step, against a 9 GB DiT that
runs four times. Rewriting them would optimise ~2% of the step while adding a
second place for the architecture to be subtly wrong -- and being subtly wrong
about a checkpoint's architecture is the failure mode that has cost the most
time in this project already.

Upstream: https://github.com/dreamzero0/dreamzero
  groot/vla/model/dreamzero/modules/wan_video_vae.py
  groot/vla/model/dreamzero/modules/wan_video_image_encoder.py
"""

from dexter.perception.encode import ObservationEncoder

__all__ = ["ObservationEncoder"]
