"""Load a released DreamZero checkpoint into the dexter DiT.

Two things make this more than a name mapping.

**It streams.** The DROID checkpoint's DiT is ~27 GB of bf16 in a 32 GB pool
that also holds the OS. Building the model densely and then quantising would
peak above what the machine has. So the module tree is built on the ``meta``
device -- shapes, no storage -- and each parameter is materialised, packed, and
the dense copy dropped before moving to the next. Peak residency is the packed
model plus one layer.

**Names.** Upstream's tree is ``action_head.model.*`` (``CausalWanModel`` inside
``WANPolicyHead``). The mapping below is the whole translation; anything the
checkpoint has that is not claimed here is reported rather than ignored, so a
silently half-loaded model is not a possible outcome.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn as nn

from dexter.models.dreamzero.config import WAMConfig
from dexter.models.dreamzero.model import CausalWanDiT
from dexter.models.layers import Linear

SRC = "action_head.model."

# dexter parameter name -> checkpoint suffix (under SRC).
# Blocks are handled separately since they repeat per layer.
_TOP_LEVEL = {
    "patch_embed.weight": "patch_embedding.weight",
    "patch_embed.bias": "patch_embedding.bias",
    "time_embed_in.weight": "time_embedding.0.weight",
    "time_embed_in.bias": "time_embedding.0.bias",
    "time_embed_out.weight": "time_embedding.2.weight",
    "time_embed_out.bias": "time_embedding.2.bias",
    "time_projection.weight": "time_projection.1.weight",
    "time_projection.bias": "time_projection.1.bias",
    "text_embed_in.weight": "text_embedding.0.weight",
    "text_embed_in.bias": "text_embedding.0.bias",
    "text_embed_out.weight": "text_embedding.2.weight",
    "text_embed_out.bias": "text_embedding.2.bias",
    # img_emb.proj is [LayerNorm, Linear, GELU, Linear, LayerNorm].
    "img_norm_in.weight": "img_emb.proj.0.weight",
    "img_norm_in.bias": "img_emb.proj.0.bias",
    "img_proj_in.weight": "img_emb.proj.1.weight",
    "img_proj_in.bias": "img_emb.proj.1.bias",
    "img_proj_out.weight": "img_emb.proj.3.weight",
    "img_proj_out.bias": "img_emb.proj.3.bias",
    "img_norm_out.weight": "img_emb.proj.4.weight",
    "img_norm_out.bias": "img_emb.proj.4.bias",
    "head.head.weight": "head.head.weight",
    "head.head.bias": "head.head.bias",
    "head.modulation": "head.modulation",
}

# Within one block, dexter suffix -> checkpoint suffix.
_BLOCK = {
    "modulation": "modulation",
    "norm3.weight": "norm3.weight",
    "norm3.bias": "norm3.bias",
    "ffn_up.weight": "ffn.0.weight",
    "ffn_up.bias": "ffn.0.bias",
    "ffn_down.weight": "ffn.2.weight",
    "ffn_down.bias": "ffn.2.bias",
    "self_attn.q_norm.weight": "self_attn.norm_q.weight",
    "self_attn.k_norm.weight": "self_attn.norm_k.weight",
    "cross_attn.q_norm.weight": "cross_attn.norm_q.weight",
    "cross_attn.k_norm.weight": "cross_attn.norm_k.weight",
    "cross_attn.k_img_norm.weight": "cross_attn.norm_k_img.weight",
}
for _attn in ("self_attn", "cross_attn"):
    for _proj in ("q", "k", "v", "o"):
        for _p in ("weight", "bias"):
            _BLOCK[f"{_attn}.{_proj}.{_p}"] = f"{_attn}.{_proj}.{_p}"
for _proj in ("k_img", "v_img"):
    for _p in ("weight", "bias"):
        _BLOCK[f"cross_attn.{_proj}.{_p}"] = f"cross_attn.{_proj}.{_p}"

# The action register's category-specific encoders map one to one.
for _mod, _parts in (("action_encoder", ("W1", "W2", "W3")),
                     ("state_encoder", ("layer1", "layer2")),
                     ("action_decoder", ("layer1", "layer2"))):
    for _part in _parts:
        for _p in ("W", "b"):
            _TOP_LEVEL[f"{_mod}.{_part}.{_p}"] = f"{_mod}.{_part}.{_p}"


def build_name_map(num_layers: int) -> dict[str, str]:
    """Full dexter-parameter -> checkpoint-key mapping."""
    mapping = {k: SRC + v for k, v in _TOP_LEVEL.items()}
    for layer in range(num_layers):
        for dst, src in _BLOCK.items():
            mapping[f"blocks.{layer}.{dst}"] = f"{SRC}blocks.{layer}.{src}"
    return mapping


class _ShardReader:
    """Reader over a sharded safetensors checkpoint, one shard mapped at a time.

    ``safe_open`` hands back tensors backed by the file mapping rather than
    copies, so every tensor read keeps its pages resident for as long as the
    handle lives. Holding all eight DiT shards open therefore grows RSS by the
    full dense size of everything read -- about 0.8 GB per transformer layer,
    measured -- even though each weight was already copied to the GPU and
    packed. On a unified-memory part that RSS competes with the GPU for the
    same 32 GB and the load wedges before it finishes.

    Keeping exactly one handle open bounds resident file pages to a single
    shard. Callers are expected to request keys grouped by shard (see
    ``plan_by_shard``); ``reopens`` counts violations so a bad access order
    shows up as a number rather than as mysterious slowness.
    """

    def __init__(self, root: Path) -> None:
        from safetensors import safe_open

        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self.shard_order: dict[str, int] = {
            name: i for i, name in enumerate(sorted(set(self.weight_map.values())))
        }
        self.reopens = 0
        self._open = safe_open
        self._root = root
        self._shard: str | None = None
        self._handle = None

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def shard_of(self, key: str) -> int:
        return self.shard_order[self.weight_map[key]]

    def get(self, key: str) -> torch.Tensor:
        shard = self.weight_map[key]
        if shard != self._shard:
            self.release()
            path = self._root / shard
            if not path.exists():
                raise FileNotFoundError(
                    f"{key} lives in {shard}, which is not downloaded. "
                    f"The DiT spans shards 3-10 of this checkpoint."
                )
            self._handle = self._open(str(path), framework="pt")
            self._shard = shard
            self.reopens += 1
        return self._handle.get_tensor(key)

    def release(self) -> None:
        """Drop the current mapping, releasing its resident pages."""
        self._handle = None
        self._shard = None

    def dit_keys(self) -> set[str]:
        return {k for k in self.weight_map if k.startswith(SRC)}


def _reshape_for(name: str, tensor: torch.Tensor, target: torch.Size) -> torch.Tensor:
    """Adapt a checkpoint tensor to dexter's parameter shape.

    The only real case is ``patch_embedding``: upstream stores it as a Conv3d
    of stride ``patch_size``, ``[dim, in_dim, t, h, w]``. That convolution is
    exactly a linear map over ``in_dim * t * h * w`` inputs, and the memory
    order already matches, so the reshape is free and loses nothing.
    """
    if tensor.shape == target:
        return tensor
    if tensor.numel() == target.numel():
        return tensor.reshape(target)
    raise ValueError(f"{name}: checkpoint has {tuple(tensor.shape)}, model wants {tuple(target)}")


def _iter_modules_with_params(model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if any(p is not None for p in module._parameters.values()):
            yield name, module


def load_dreamzero(
    path: str | Path,
    cfg: WAMConfig,
    *,
    bits: int | None = None,
    group_size: int = 128,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    strict: bool = True,
) -> CausalWanDiT:
    """Materialise a DreamZero DiT from a released checkpoint.

    ``bits`` packs each transformer Linear as it lands, so the dense weight for
    a layer exists only long enough to be quantised. That is what makes a 27 GB
    bf16 DiT loadable as ~7 GB of int4 on a 32 GB machine -- quantising after a
    dense load would need both at once and does not fit.
    """
    root = Path(path)
    reader = _ShardReader(root)
    mapping = build_name_map(cfg.num_layers)

    # Shapes only, no storage: 27 GB of parameters that are never allocated.
    model = CausalWanDiT(cfg, dtype=dtype, device="meta")

    missing: list[str] = []
    consumed: set[str] = set()

    # Visit modules in checkpoint order, so each shard is mapped exactly once.
    plan = []
    for module_name, module in _iter_modules_with_params(model):
        keys = {}
        for param_name, param in module._parameters.items():
            if param is None:
                continue
            full = f"{module_name}.{param_name}" if module_name else param_name
            key = mapping.get(full)
            if key is None or not reader.has(key):
                missing.append(full)
                continue
            keys[param_name] = (full, key)
        if keys:
            first = min(reader.shard_of(k) for _, k in keys.values())
            plan.append((first, module_name, module, keys))
    plan.sort(key=lambda item: item[0])

    for index, (_, module_name, module, keys) in enumerate(plan):
        for param_name, (full, key) in keys.items():
            target = module._parameters[param_name].shape
            tensor = _reshape_for(full, reader.get(key), target)
            module._parameters[param_name] = nn.Parameter(
                tensor.to(device=device, dtype=dtype), requires_grad=False
            )
            del tensor
            consumed.add(key)

        # Pack this Linear now, while only its own dense weight is resident.
        if bits is not None and isinstance(module, Linear) and module_name.startswith("blocks."):
            module.quantize_(bits=bits, group_size=group_size)

        # Quantisation allocates fp32 temporaries the size of a dense weight.
        # Left alone the caching allocator keeps growing its pool, and on a
        # unified-memory part that pool is the same RAM the mapping needs.
        if index % 64 == 0:
            torch.cuda.empty_cache()

    reader.release()

    # Buffers (rope tables, block boundaries) were built on meta too.
    model.rebuild_buffers(device)

    if reader.reopens > len(reader.shard_order):
        import warnings
        warnings.warn(
            f"re-mapped shards {reader.reopens} times for "
            f"{len(reader.shard_order)} shards; load order is not shard-grouped",
            RuntimeWarning, stacklevel=2,
        )

    unclaimed = reader.dit_keys() - consumed
    if strict and (missing or unclaimed):
        raise RuntimeError(
            f"checkpoint did not map cleanly: {len(missing)} model parameters unfilled "
            f"(e.g. {missing[:3]}), {len(unclaimed)} checkpoint tensors unused "
            f"(e.g. {sorted(unclaimed)[:3]})"
        )
    torch.cuda.empty_cache()
    return model
