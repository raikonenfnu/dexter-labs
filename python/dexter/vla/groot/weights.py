"""Map GR00T N1.7 checkpoint tensors onto dexter modules.

The mapping is written out in full and then *checked for exhaustiveness*: every
tensor in the checkpoint must be either claimed by a module or listed as
deliberately unused. That check is what catches an architecture that is subtly
wrong -- on DreamZero it turned a day of guessing into a single assertion.

Two subtrees are deliberately unused at inference and are worth naming, because
skipping them is free performance:

``backbone.model.lm_head``            311M parameters, 622 MB in bf16. It turns
    hidden states into vocabulary logits, and the action head reads hidden
    states. Never executed.
``language_model.layers.13..15``      N1.7 reads an intermediate layer
    (`select_layer`), so the layers above it cannot influence an action.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

BACKBONE = "backbone.model.model."
HEAD = "action_head."

# Tensors present in the checkpoint that inference never needs.
UNUSED_PATTERNS = (
    r"^backbone\.model\.lm_head\.",                       # logits head
    r"^backbone\.model\.model\.language_model\.layers\.(1[3-9]|[2-9]\d)\.",  # above select_layer
)


def checkpoint_shapes(root: str | Path) -> dict[str, list[int]]:
    """Tensor shapes from the safetensors headers, without loading any data."""
    root = Path(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    shapes: dict[str, list[int]] = {}
    for shard in sorted(set(index["weight_map"].values())):
        with open(root / shard, "rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            for key, meta in json.loads(handle.read(length)).items():
                if key != "__metadata__":
                    shapes[key] = meta["shape"]
    return shapes


def is_unused(key: str) -> bool:
    return any(re.match(p, key) for p in UNUSED_PATTERNS)


def build_mapping(vision_depth: int, language_depth: int, dit_depth: int,
                  vl_depth: int, deepstack: int) -> dict[str, str]:
    """dexter parameter name -> checkpoint key."""
    m: dict[str, str] = {
        # -- vision tower ------------------------------------------------
        "vision.patch_embed.weight": f"{BACKBONE}visual.patch_embed.proj.weight",
        "vision.patch_embed.bias": f"{BACKBONE}visual.patch_embed.proj.bias",
        "vision.pos_embed.weight": f"{BACKBONE}visual.pos_embed.weight",
        "vision.merger_norm.weight": f"{BACKBONE}visual.merger.norm.weight",
        "vision.merger_norm.bias": f"{BACKBONE}visual.merger.norm.bias",
        "vision.merger_fc1.weight": f"{BACKBONE}visual.merger.linear_fc1.weight",
        "vision.merger_fc1.bias": f"{BACKBONE}visual.merger.linear_fc1.bias",
        "vision.merger_fc2.weight": f"{BACKBONE}visual.merger.linear_fc2.weight",
        "vision.merger_fc2.bias": f"{BACKBONE}visual.merger.linear_fc2.bias",
        # -- language ----------------------------------------------------
        "language.embed_tokens.weight": f"{BACKBONE}language_model.embed_tokens.weight",
        "language.norm.weight": f"{BACKBONE}language_model.norm.weight",
        # -- action head -------------------------------------------------
        "head.vlln.weight": f"{HEAD}vlln.weight",
        "head.vlln.bias": f"{HEAD}vlln.bias",
        "head.position_embedding.weight": f"{HEAD}position_embedding.weight",
        "head.timestep_in.weight": f"{HEAD}model.timestep_encoder.timestep_embedder.linear_1.weight",
        "head.timestep_in.bias": f"{HEAD}model.timestep_encoder.timestep_embedder.linear_1.bias",
        "head.timestep_out.weight": f"{HEAD}model.timestep_encoder.timestep_embedder.linear_2.weight",
        "head.timestep_out.bias": f"{HEAD}model.timestep_encoder.timestep_embedder.linear_2.bias",
        "head.proj_out_1.weight": f"{HEAD}model.proj_out_1.weight",
        "head.proj_out_1.bias": f"{HEAD}model.proj_out_1.bias",
        "head.proj_out_2.weight": f"{HEAD}model.proj_out_2.weight",
        "head.proj_out_2.bias": f"{HEAD}model.proj_out_2.bias",
    }
    for part in ("W1", "W2", "W3"):
        for suffix in ("W", "b"):
            m[f"head.action_encoder_{part.lower()}.{suffix}"] = f"{HEAD}action_encoder.{part}.{suffix}"
    for module in ("state_encoder", "action_decoder"):
        for layer in ("layer1", "layer2"):
            for suffix in ("W", "b"):
                m[f"head.{module}.{layer}.{suffix}"] = f"{HEAD}{module}.{layer}.{suffix}"

    for i in range(vision_depth):
        src = f"{BACKBONE}visual.blocks.{i}."
        dst = f"vision.blocks.{i}."
        for a, b in (("qkv", "attn.qkv"), ("proj", "attn.proj"),
                     ("fc1", "mlp.linear_fc1"), ("fc2", "mlp.linear_fc2")):
            for suffix in ("weight", "bias"):
                m[f"{dst}{a}.{suffix}"] = f"{src}{b}.{suffix}"
        for norm in ("norm1", "norm2"):
            for suffix in ("weight", "bias"):
                m[f"{dst}{norm}.{suffix}"] = f"{src}{norm}.{suffix}"

    for i in range(deepstack):
        src = f"{BACKBONE}visual.deepstack_merger_list.{i}."
        dst = f"vision.deepstack.{i}."
        for a, b in (("norm", "norm"), ("fc1", "linear_fc1"), ("fc2", "linear_fc2")):
            for suffix in ("weight", "bias"):
                m[f"{dst}{a}.{suffix}"] = f"{src}{b}.{suffix}"

    for i in range(language_depth):
        src = f"{BACKBONE}language_model.layers.{i}."
        dst = f"language.blocks.{i}."
        for name in ("q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"):
            key = f"{src}mlp.{name}.weight" if name.endswith(("gate_proj", "up_proj", "down_proj")) \
                else f"{src}self_attn.{name}.weight"
            m[f"{dst}{name}.weight"] = key
        for name in ("q_norm", "k_norm"):
            m[f"{dst}{name}.weight"] = f"{src}self_attn.{name}.weight"
        m[f"{dst}input_layernorm.weight"] = f"{src}input_layernorm.weight"
        m[f"{dst}post_attention_layernorm.weight"] = f"{src}post_attention_layernorm.weight"

    for i in range(vl_depth):
        src = f"{HEAD}vl_self_attention.transformer_blocks.{i}."
        dst = f"head.vl_blocks.{i}."
        for a, b in (("to_q", "attn1.to_q"), ("to_k", "attn1.to_k"), ("to_v", "attn1.to_v"),
                     ("to_out", "attn1.to_out.0"), ("ff_in", "ff.net.0.proj"), ("ff_out", "ff.net.2")):
            for suffix in ("weight", "bias"):
                m[f"{dst}{a}.{suffix}"] = f"{src}{b}.{suffix}"
        for norm in ("norm1", "norm3"):
            for suffix in ("weight", "bias"):
                m[f"{dst}{norm}.{suffix}"] = f"{src}{norm}.{suffix}"

    for i in range(dit_depth):
        src = f"{HEAD}model.transformer_blocks.{i}."
        dst = f"head.blocks.{i}."
        for a, b in (("to_q", "attn1.to_q"), ("to_k", "attn1.to_k"), ("to_v", "attn1.to_v"),
                     ("to_out", "attn1.to_out.0"), ("ff_in", "ff.net.0.proj"), ("ff_out", "ff.net.2"),
                     ("norm1_linear", "norm1.linear")):
            for suffix in ("weight", "bias"):
                m[f"{dst}{a}.{suffix}"] = f"{src}{b}.{suffix}"
    return m
