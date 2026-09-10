"""Vision-language-action policies, behind one interface.

Adding a policy should be a new module here plus one registry entry — not a new
copy of the loading, normalisation and scoring plumbing. Each module owns
exactly two things: how its checkpoint loads, and how an observation maps onto
the tensors it expects.
"""

from dexter.vla.base import ActionChunkPolicy, Observation
from dexter.vla.registry import available, load

__all__ = ["ActionChunkPolicy", "Observation", "available", "load"]
