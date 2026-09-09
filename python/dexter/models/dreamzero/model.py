"""The DreamZero DiT, expressed on dexter ops.

This is the compute graph of ``CausalWanModel`` from the upstream repo
(``groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py``),
re-expressed so every elementwise stage goes through a fused op. The layer
shapes, the modulation structure and the sequence layout are the upstream
ones; what changed is how many times each tensor crosses the memory bus.

Sequence layout for one closed-loop step:

    [ video tokens (frame_seqlen * frames_per_block) | actions | state ]
      <------------ noisy latent of this block ----->  <-- action register -->

Both halves run through the same transformer. The video slice is decoded back
to latent noise; the register slice is decoded to action noise. Blockwise
causality keeps a query from seeing anything newer than its own block, and the
KV cache holds every block before it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from dexter import ops
from dexter.models.dreamzero.config import WAMConfig
from dexter.models.layers import (
    AffineLayerNorm, CategorySpecificLinear, CategorySpecificMLP, Linear, RMSNorm,
)


class KVCache:
    """Per-layer self-attention cache over the episode's earlier blocks.

    Allocated once for the whole episode. A control loop runs at a fixed rate
    for a bounded horizon, so the capacity is known up front and the cache
    never reallocates mid-episode -- an allocation inside the loop would show
    up directly in the tail latency the robot feels.
    """

    def __init__(self, layers: int, batch: int, capacity: int, heads: int, head_dim: int,
                 device: torch.device, dtype: torch.dtype) -> None:
        self.cache = [
            torch.zeros(2, batch, capacity, heads, head_dim, device=device, dtype=dtype)
            for _ in range(layers)
        ]
        self.capacity = capacity
        self.length = 0

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        n = k.shape[1]
        if self.length + n > self.capacity:
            raise RuntimeError(f"KV cache full: {self.length} + {n} > {self.capacity}")
        self.cache[layer][0, :, self.length : self.length + n] = k
        self.cache[layer][1, :, self.length : self.length + n] = v

    def commit(self, n: int) -> None:
        """Advance the write cursor once every layer has appended its block."""
        self.length += n

    def reset(self) -> None:
        self.length = 0


@dataclass
class CrossAttnCache:
    """Conditioning keys and values, projected once per episode.

    One dict per layer, holding text ``k``/``v`` and, for i2v, the CLIP image
    ``k_img``/``v_img`` as well.
    """

    layers: list[dict]


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard diffusion timestep embedding, ``[B] -> [B, dim]``."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat((torch.cos(args), torch.sin(args)), dim=-1)


def rope_tables(seq_len: int, head_dim: int, device, dtype, base: float = 10000.0):
    """``(cos, sin)`` of shape ``[seq_len, head_dim // 2]``."""
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = pos[:, None] * inv[None, :]
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


class SelfAttention(nn.Module):
    """Blockwise-causal self attention with QK-norm and rotary."""

    def __init__(self, cfg: WAMConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.heads, self.head_dim = cfg.num_heads, cfg.head_dim
        self.q = Linear(cfg.dim, cfg.dim, dtype=dtype)
        self.k = Linear(cfg.dim, cfg.dim, dtype=dtype)
        self.v = Linear(cfg.dim, cfg.dim, dtype=dtype)
        self.o = Linear(cfg.dim, cfg.dim, dtype=dtype)
        # WanRMSNorm spans the full dim, not head_dim: the norm runs on the
        # projection before it is split into heads.
        self.q_norm = RMSNorm(cfg.dim, cfg.eps, dtype)
        self.k_norm = RMSNorm(cfg.dim, cfg.eps, dtype)
        self.eps = cfg.eps

    def forward(self, x, cos, sin, kv_cache: KVCache | None, layer: int,
                block_boundaries: torch.Tensor | None):
        b, seq_len, _ = x.shape
        shape = (b, seq_len, self.heads, self.head_dim)
        q = self.q(x).view(shape)
        k = self.k(x).view(shape)
        v = self.v(x).view(shape)

        q, k = ops.qk_norm_rope(q, k, self.q_norm.weight, self.k_norm.weight, cos, sin, self.eps)

        cache_len = kv_cache.length if kv_cache is not None else 0
        if kv_cache is not None:
            kv_cache.append(layer, k, v)

        out = ops.blockwise_causal_attention(
            q, k, v,
            kv_cache=kv_cache.cache[layer] if kv_cache is not None else None,
            cache_len=cache_len,
            block_boundaries=block_boundaries,
        )
        return self.o(out.reshape(b, seq_len, -1))


class CrossAttention(nn.Module):
    """Attention onto the conditioning tokens, cached per episode.

    Wan's i2v variant attends to *two* conditioning streams with a shared
    query: the umt5 text embedding and the CLIP image features of the first
    frame. Their outputs are summed before the output projection. Both streams
    are constant for an episode, so all four projections run once in
    ``project_context`` and never again inside the control loop.

    Note the context arriving here is already at ``dim``: the DiT projects text
    through ``text_embedding`` and CLIP through ``img_emb`` before any block
    sees it, so ``k``/``v`` are ``dim -> dim``, not ``text_dim -> dim``.
    """

    def __init__(self, cfg: WAMConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.heads, self.head_dim = cfg.num_heads, cfg.head_dim
        self.image_branch = cfg.model_type == "i2v"
        self.q = Linear(cfg.dim, cfg.dim, dtype=dtype)
        self.k = Linear(cfg.dim, cfg.dim, dtype=dtype)
        self.v = Linear(cfg.dim, cfg.dim, dtype=dtype)
        self.o = Linear(cfg.dim, cfg.dim, dtype=dtype)
        self.q_norm = RMSNorm(cfg.dim, cfg.eps, dtype)
        self.k_norm = RMSNorm(cfg.dim, cfg.eps, dtype)
        if self.image_branch:
            self.k_img = Linear(cfg.dim, cfg.dim, dtype=dtype)
            self.v_img = Linear(cfg.dim, cfg.dim, dtype=dtype)
            self.k_img_norm = RMSNorm(cfg.dim, cfg.eps, dtype)
        self.eps = cfg.eps

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        return t.view(t.shape[0], t.shape[1], self.heads, self.head_dim)

    def _norm_heads(self, t: torch.Tensor, norm: RMSNorm) -> torch.Tensor:
        """Norm across the full dim first, then split -- upstream's order."""
        return self._heads(norm(t))

    def project_context(self, text: torch.Tensor, image: torch.Tensor | None):
        """Run once per episode; the result is what ``forward`` consumes."""
        kv = {
            "k": self._norm_heads(self.k(text), self.k_norm),
            "v": self._heads(self.v(text)),
        }
        if self.image_branch and image is not None:
            kv["k_img"] = self._norm_heads(self.k_img(image), self.k_img_norm)
            kv["v_img"] = self._heads(self.v_img(image))
        return kv

    def forward(self, x, kv: dict):
        b, seq_len, _ = x.shape
        q = self._norm_heads(self.q(x), self.q_norm)
        # Conditioning is fully visible: no mask, no cache offset.
        out = ops.blockwise_causal_attention(q, kv["k"], kv["v"])
        if "k_img" in kv:
            out = out + ops.blockwise_causal_attention(q, kv["k_img"], kv["v_img"])
        return self.o(out.reshape(b, seq_len, -1))


class DiTBlock(nn.Module):
    """One Wan DiT block.

    The six modulation tensors come from ``modulation + timestep_embedding``
    and split into (shift1, scale1, gate1, shift2, scale2, gate2). Each pair
    feeds a fused ``adaln_modulate``; each gate feeds a fused
    ``gated_residual``. That is six fused ops where an unfused block would run
    twelve passes over ``x``.
    """

    def __init__(self, cfg: WAMConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.eps = cfg.eps
        self.self_attn = SelfAttention(cfg, dtype)
        self.cross_attn = CrossAttention(cfg, dtype)
        self.norm3 = AffineLayerNorm(cfg.dim, cfg.eps, dtype)
        self.ffn_up = Linear(cfg.dim, cfg.ffn_dim, dtype=dtype)
        self.ffn_down = Linear(cfg.ffn_dim, cfg.dim, dtype=dtype)
        self.modulation = nn.Parameter(torch.randn(1, 6, cfg.dim, dtype=dtype) / cfg.dim ** 0.5)

    def forward(self, x, e, cos, sin, ctx, kv_cache, layer, block_boundaries):
        # e is [B, 6, C]; each part broadcasts over the sequence.
        m = (self.modulation + e).unsqueeze(2)  # [B, 6, 1, C]
        shift1, scale1, gate1, shift2, scale2, gate2 = m.unbind(dim=1)

        y = self.self_attn(
            ops.adaln_modulate(x, scale1, shift1, self.eps),
            cos, sin, kv_cache, layer, block_boundaries,
        )
        x = ops.gated_residual(x, y, gate1)

        x = x + self.cross_attn(self.norm3(x), ctx)

        y = self._ffn(ops.adaln_modulate(x, scale2, shift2, self.eps))
        return ops.gated_residual(x, y, gate2)

    def _ffn(self, x):
        """Fused when both projections are dense; the quantised path keeps the
        two GEMMs separate because the epilogue lives in the dequant kernel."""
        if self.ffn_up.packed is None and self.ffn_down.packed is None:
            return ops.ffn_gelu(x, self.ffn_up.weight, self.ffn_up.bias,
                                self.ffn_down.weight, self.ffn_down.bias)
        return self.ffn_down(nn.functional.gelu(self.ffn_up(x), approximate="tanh"))


def action_time_embedding(timesteps: torch.Tensor, tokens: int, dim: int) -> torch.Tensor:
    """Sinusoidal encoding of the action flow time, ``[B] -> [B, tokens, dim]``.

    Upstream builds this over a ``[B, T]`` timestep grid; every action in a
    chunk shares one flow time, so the grid is a broadcast of the scalar.
    """
    half = dim // 2
    exponent = -torch.arange(half, device=timesteps.device, dtype=torch.float32) * (
        math.log(10000.0) / half
    )
    t = timesteps.float()[:, None, None].expand(-1, tokens, 1)
    freqs = t * exponent.exp()[None, None, :]
    return torch.cat((torch.sin(freqs), torch.cos(freqs)), dim=-1)


class MultiEmbodimentActionEncoder(nn.Module):
    """Noisy actions + flow timestep -> action-register tokens.

    ``W1`` lifts the action, the sinusoidal flow time is concatenated, ``W2``
    mixes the pair through a swish, and ``W3`` projects out. All three are
    category-specific so one checkpoint serves several robots.
    """

    def __init__(self, cfg: WAMConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.dim = cfg.dim
        n = cfg.num_embodiments
        self.W1 = CategorySpecificLinear(n, cfg.action_dim, cfg.dim, dtype)
        self.W2 = CategorySpecificLinear(n, 2 * cfg.dim, cfg.dim, dtype)
        self.W3 = CategorySpecificLinear(n, cfg.dim, cfg.dim, dtype)

    def forward(self, actions, timesteps, cat_ids):
        a = self.W1(actions, cat_ids)
        tau = action_time_embedding(timesteps, actions.shape[1], self.dim).to(a.dtype)
        x = torch.cat((a, tau), dim=-1)
        x = torch.nn.functional.silu(self.W2(x, cat_ids))  # swish
        return self.W3(x, cat_ids)


class CausalHead(nn.Module):
    """Video output head: its own two-way modulation, then a projection."""

    def __init__(self, cfg: WAMConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.eps = cfg.eps
        self.head = Linear(cfg.dim, cfg.head_out_dim, dtype=dtype)
        self.modulation = nn.Parameter(torch.zeros(1, 2, cfg.dim, dtype=dtype))

    def forward(self, x, e):
        shift, scale = (self.modulation + e).unsqueeze(2).unbind(dim=1)
        return self.head(ops.adaln_modulate(x, scale, shift, self.eps))


class CausalWanDiT(nn.Module):
    """The DreamZero backbone: video latent and action register, jointly denoised."""

    def __init__(self, cfg: WAMConfig, dtype: torch.dtype = torch.bfloat16,
                 device: torch.device | str = "cuda") -> None:
        super().__init__()
        self.cfg, self.dtype = cfg, dtype

        # Build directly on the target device. Constructing on the host first
        # and moving would peak at twice the model's size, and on a unified
        # memory part that is the same pool the weights have to live in.
        with torch.device(device):
            self._build(cfg, dtype)
        self.to(dtype=dtype)
        self.rebuild_buffers(device)

    def rebuild_buffers(self, device: torch.device | str) -> None:
        """(Re)create the derived buffers on ``device``.

        Separate from ``_build`` because a streaming checkpoint load builds the
        module tree on ``meta`` -- rope tables and block boundaries are computed,
        not loaded, so they have to be materialised once the real device is known.
        """
        cfg, dtype = self.cfg, self.dtype
        cos, sin = rope_tables(cfg.seq_len * 8, cfg.head_dim, device, dtype)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        # Video tokens form one block, the action register a second. Built once:
        # allocating it inside forward would stall the step and, more sharply,
        # make the step impossible to capture into a graph.
        self.register_buffer(
            "block_boundaries",
            torch.tensor([0, cfg.video_tokens], device=device, dtype=torch.int64),
            persistent=False,
        )

    def _build(self, cfg: WAMConfig, dtype: torch.dtype) -> None:
        # Patchify. The checkpoint stores this as a Conv3d of stride patch_size,
        # which is the same linear map over in_dim * prod(patch_size) inputs.
        self.patch_embed = Linear(cfg.patch_dim, cfg.dim, dtype=dtype)

        # Diffusion timestep: sinusoid(freq_dim) -> MLP -> 6 modulation vectors.
        self.time_embed_in = Linear(cfg.freq_dim, cfg.dim, dtype=dtype)
        self.time_embed_out = Linear(cfg.dim, cfg.dim, dtype=dtype)
        self.time_projection = Linear(cfg.dim, 6 * cfg.dim, dtype=dtype)

        # Conditioning projections, both run once per episode.
        self.text_embed_in = Linear(cfg.text_dim, cfg.dim, dtype=dtype)
        self.text_embed_out = Linear(cfg.dim, cfg.dim, dtype=dtype)
        if cfg.model_type == "i2v":
            self.img_norm_in = AffineLayerNorm(cfg.clip_dim, cfg.eps, dtype)
            self.img_proj_in = Linear(cfg.clip_dim, cfg.clip_dim, dtype=dtype)
            self.img_proj_out = Linear(cfg.clip_dim, cfg.dim, dtype=dtype)
            self.img_norm_out = AffineLayerNorm(cfg.dim, cfg.eps, dtype)

        self.blocks = nn.ModuleList(DiTBlock(cfg, dtype) for _ in range(cfg.num_layers))
        self.head = CausalHead(cfg, dtype)

        # Action register: encoders in, decoder out. All category-specific.
        self.action_encoder = MultiEmbodimentActionEncoder(cfg, dtype)
        self.state_encoder = CategorySpecificMLP(
            cfg.num_embodiments, cfg.max_state_dim, cfg.state_hidden, cfg.dim, dtype)
        self.action_decoder = CategorySpecificMLP(
            cfg.num_embodiments, cfg.dim, cfg.state_hidden, cfg.action_dim, dtype)

    @torch.no_grad()
    def quantize_(self, bits: int = 4, group_size: int = 128) -> "CausalWanDiT":
        """Repack every linear in the transformer stack.

        The embeddings and output heads stay dense: together they are well under
        1% of the parameters, so quantising them buys no bandwidth and only
        risks accuracy at the two points where the model meets the robot.
        """
        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, Linear):
                    module.quantize_(bits=bits, group_size=group_size)
        torch.cuda.empty_cache()
        return self

    def weight_bytes(self) -> int:
        return sum(m.weight_bytes for m in self.modules() if isinstance(m, Linear))

    def embed_text(self, text: torch.Tensor) -> torch.Tensor:
        """umt5 embedding -> dim, through the DiT's own text MLP."""
        return self.text_embed_out(
            torch.nn.functional.gelu(self.text_embed_in(text), approximate="tanh")
        )

    def embed_image(self, clip: torch.Tensor) -> torch.Tensor:
        """CLIP features of the conditioning frame -> dim."""
        x = self.img_proj_in(self.img_norm_in(clip))
        x = self.img_proj_out(torch.nn.functional.gelu(x, approximate="tanh"))
        return self.img_norm_out(x)

    def project_context(self, text: torch.Tensor,
                        clip: torch.Tensor | None = None) -> CrossAttnCache:
        """Project the conditioning through every block's cross-attention.

        Done once per instruction. Inside the control loop the language and the
        conditioning frame are constant, so repeating these projections every
        denoising step would be pure waste -- exactly the kind of per-step work
        the upstream blocking server pays for.
        """
        context = self.embed_text(text)
        image = self.embed_image(clip) if (clip is not None and hasattr(self, "img_proj_in")) else None
        return CrossAttnCache([b.cross_attn.project_context(context, image) for b in self.blocks])

    def forward(
        self,
        latent: torch.Tensor,  # [B, video_tokens, patch_dim] noisy, patchified
        actions: torch.Tensor,  # [B, num_action_per_block, action_dim] noisy actions
        state: torch.Tensor,  # [B, num_state_per_block, max_state_dim]
        timestep: torch.Tensor,  # [B] flow-matching time
        ctx: CrossAttnCache,
        kv_cache: KVCache | None = None,
        embodiment: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(video_noise, action_noise)``."""
        cfg = self.cfg
        if embodiment is None:
            embodiment = torch.zeros(latent.shape[0], dtype=torch.long, device=latent.device)

        # [ video | actions | state ] -- one sequence through one transformer.
        x = torch.cat((
            self.patch_embed(latent),
            self.action_encoder(actions, timestep, embodiment),
            self.state_encoder(state, embodiment),
        ), dim=1)

        t = torch.nn.functional.silu(
            self.time_embed_in(sinusoidal_embedding(timestep, cfg.freq_dim).to(x.dtype))
        )
        t = self.time_embed_out(t)
        e = self.time_projection(torch.nn.functional.silu(t)).view(-1, 6, cfg.dim)

        seq_len = x.shape[1]
        cos, sin = self.rope_cos[:seq_len], self.rope_sin[:seq_len]

        for layer, block in enumerate(self.blocks):
            x = block(x, e, cos, sin, ctx.layers[layer], kv_cache, layer, self.block_boundaries)

        if kv_cache is not None:
            kv_cache.commit(seq_len)

        # The head takes the raw time embedding; its own [1, 2, dim] modulation
        # broadcasts against [B, 1, dim] to give the shift/scale pair.
        video_noise = self.head(x[:, : cfg.video_tokens], t.unsqueeze(1))
        action_slice = x[:, cfg.video_tokens : cfg.video_tokens + cfg.num_action_per_block]
        action_noise = self.action_decoder(action_slice, embodiment)
        return video_noise, action_noise
