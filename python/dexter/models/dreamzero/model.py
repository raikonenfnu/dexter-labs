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
from dexter.models.layers import AffineLayerNorm, Linear, RMSNorm


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
    """Text keys and values, projected once per episode and reused every step."""

    k: list[torch.Tensor]
    v: list[torch.Tensor]


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
        self.q_norm = RMSNorm(cfg.head_dim, cfg.eps, dtype)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.eps, dtype)
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
    """Attention onto the umt5 text embedding. Text k/v are cached per episode."""

    def __init__(self, cfg: WAMConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.heads, self.head_dim = cfg.num_heads, cfg.head_dim
        self.q = Linear(cfg.dim, cfg.dim, dtype=dtype)
        self.k = Linear(cfg.text_dim, cfg.dim, dtype=dtype)
        self.v = Linear(cfg.text_dim, cfg.dim, dtype=dtype)
        self.o = Linear(cfg.dim, cfg.dim, dtype=dtype)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.eps, dtype)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.eps, dtype)
        self.eps = cfg.eps

    def project_context(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run once per episode; the result is what ``forward`` consumes."""
        b, text_len, _ = context.shape
        shape = (b, text_len, self.heads, self.head_dim)
        k = self.k_norm(self.k(context).view(shape))
        return k, self.v(context).view(shape)

    def forward(self, x, k, v):
        b, seq_len, _ = x.shape
        q = self.q_norm(self.q(x).view(b, seq_len, self.heads, self.head_dim))
        # Text is fully visible: no mask, no cache offset.
        out = ops.blockwise_causal_attention(q, k, v)
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

    def forward(self, x, e, cos, sin, ctx_k, ctx_v, kv_cache, layer, block_boundaries):
        # e is [B, 6, C]; each part broadcasts over the sequence.
        m = (self.modulation + e).unsqueeze(2)  # [B, 6, 1, C]
        shift1, scale1, gate1, shift2, scale2, gate2 = m.unbind(dim=1)

        y = self.self_attn(
            ops.adaln_modulate(x, scale1, shift1, self.eps),
            cos, sin, kv_cache, layer, block_boundaries,
        )
        x = ops.gated_residual(x, y, gate1)

        x = x + self.cross_attn(self.norm3(x), ctx_k, ctx_v)

        y = self._ffn(ops.adaln_modulate(x, scale2, shift2, self.eps))
        return ops.gated_residual(x, y, gate2)

    def _ffn(self, x):
        """Fused when both projections are dense; the quantised path keeps the
        two GEMMs separate because the epilogue lives in the dequant kernel."""
        if self.ffn_up.packed is None and self.ffn_down.packed is None:
            return ops.ffn_gelu(x, self.ffn_up.weight, self.ffn_up.bias,
                                self.ffn_down.weight, self.ffn_down.bias)
        return self.ffn_down(nn.functional.gelu(self.ffn_up(x), approximate="tanh"))


class ActionEncoder(nn.Module):
    """Noisy actions + state + flow timestep -> action-register tokens."""

    def __init__(self, cfg: WAMConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_in = Linear(cfg.action_dim, cfg.dim, dtype=dtype)
        self.state_in = Linear(cfg.max_state_dim, cfg.dim, dtype=dtype)
        self.time_in = Linear(cfg.dim, cfg.dim, dtype=dtype)

    def forward(self, actions, state, timestep):
        # actions [B, A, action_dim], state [B, S, max_state_dim], timestep [B]
        t = self.time_in(sinusoidal_embedding(timestep, self.cfg.dim).to(actions.dtype))
        action_tokens = self.action_in(actions) + t[:, None, :]
        return torch.cat((action_tokens, self.state_in(state)), dim=1)


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
        self.patch_embed = Linear(cfg.in_dim, cfg.dim, dtype=dtype)
        self.time_embed = Linear(cfg.dim, cfg.dim, dtype=dtype)
        self.time_projection = Linear(cfg.dim, 6 * cfg.dim, dtype=dtype)
        self.blocks = nn.ModuleList(DiTBlock(cfg, dtype) for _ in range(cfg.num_layers))
        self.head_norm = AffineLayerNorm(cfg.dim, cfg.eps, dtype)
        self.video_out = Linear(cfg.dim, cfg.out_dim, dtype=dtype)
        self.action_out = Linear(cfg.dim, cfg.action_dim, dtype=dtype)
        self.action_encoder = ActionEncoder(cfg, dtype)

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

    def project_context(self, context: torch.Tensor) -> CrossAttnCache:
        """Project the text embedding through every block's cross-attention.

        Done once per language instruction. Inside the control loop the text is
        constant, so these projections are pure waste if repeated -- which is
        exactly the kind of per-step host work the upstream server pays.
        """
        ks, vs = [], []
        for block in self.blocks:
            k, v = block.cross_attn.project_context(context)
            ks.append(k)
            vs.append(v)
        return CrossAttnCache(ks, vs)

    def forward(
        self,
        latent: torch.Tensor,  # [B, video_tokens, in_dim] noisy video latent
        actions: torch.Tensor,  # [B, num_action_per_block, action_dim] noisy actions
        state: torch.Tensor,  # [B, num_state_per_block, max_state_dim]
        timestep: torch.Tensor,  # [B] flow-matching time
        ctx: CrossAttnCache,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(video_noise, action_noise)``."""
        cfg = self.cfg
        video = self.patch_embed(latent)
        register = self.action_encoder(actions, state, timestep)
        x = torch.cat((video, register), dim=1)

        t = self.time_embed(sinusoidal_embedding(timestep, cfg.dim).to(x.dtype))
        e = self.time_projection(t).view(-1, 6, cfg.dim)

        seq_len = x.shape[1]
        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]

        for layer, block in enumerate(self.blocks):
            x = block(x, e, cos, sin, ctx.k[layer], ctx.v[layer], kv_cache, layer,
                      self.block_boundaries)

        if kv_cache is not None:
            kv_cache.commit(seq_len)

        x = self.head_norm(x)
        video_noise = self.video_out(x[:, : cfg.video_tokens])
        action_noise = self.action_out(x[:, cfg.video_tokens : cfg.video_tokens + cfg.num_action_per_block])
        return video_noise, action_noise
