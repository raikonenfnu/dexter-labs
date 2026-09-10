"""GR00T N1.7 architecture, recovered from the checkpoint rather than a config."""
from pathlib import Path

import pytest

from dexter.vla.groot.config import GrootConfig
from dexter.vla.groot.weights import build_mapping, checkpoint_shapes, is_unused

CHECKPOINT = Path("/home/stanley/nod/checkpoints/GR00T-N1.7-3B")
needs_checkpoint = pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors.index.json").exists(),
    reason="GR00T-N1.7-3B not downloaded",
)


def test_config_matches_the_shapes_it_was_derived_from():
    cfg = GrootConfig()
    assert cfg.vision.head_dim * cfg.vision.heads == cfg.vision.dim
    assert cfg.vision.merge_dim == cfg.vision.dim * 4          # 2x2 spatial merge
    assert cfg.language.heads % cfg.language.kv_heads == 0     # GQA must divide
    assert cfg.language.select_layer < cfg.language.depth


@needs_checkpoint
def test_mapping_accounts_for_every_checkpoint_tensor():
    """Nothing unclaimed, nothing missing.

    An architecture that is subtly wrong shows up here as an unclaimed tensor,
    long before it shows up as bad actions -- which is the only cheap way to
    catch it without a reference implementation to diff against.
    """
    shapes = checkpoint_shapes(CHECKPOINT)
    mapping = build_mapping(vision_depth=24, language_depth=16,
                            dit_depth=32, vl_depth=4, deepstack=3)
    claimed = set(mapping.values())
    unused = {k for k in shapes if is_unused(k)}

    assert not [v for v in claimed if v not in shapes], "mapping names a tensor that does not exist"
    assert not [k for k in shapes if k not in claimed and k not in unused], "checkpoint tensor unclaimed"
    assert claimed | unused >= set(shapes)


@needs_checkpoint
def test_unused_subtrees_are_worth_skipping():
    """lm_head and the layers above select_layer never run; that is real memory."""
    shapes = checkpoint_shapes(CHECKPOINT)
    skipped = sum(
        __import__("math").prod(shape) for key, shape in shapes.items() if is_unused(key)
    )
    assert skipped > 300e6, "expected the vocabulary head alone to exceed 300M params"
