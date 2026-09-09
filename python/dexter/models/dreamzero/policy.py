"""Closed-loop policy: observation in, action chunk out.

One control step is a short flow-matching denoise loop over the action
register, with the video latent denoised alongside it. Upstream runs four
steps, so the loop is short enough that per-step *host* cost is a first-order
term rather than a rounding error -- which is the same thing the vLLM-Omni
write-up found on MI300X, and the reason :mod:`dexter.runtime` exists.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from dexter.models.dreamzero.config import WAMConfig
from dexter.models.dreamzero.model import CausalWanDiT, CrossAttnCache, KVCache


@dataclass
class StepTrace:
    """Where one closed-loop step went. All times in milliseconds."""

    total: float = 0.0
    denoise: list[float] = field(default_factory=list)
    host: float = 0.0

    @property
    def device(self) -> float:
        return sum(self.denoise)

    def __str__(self) -> str:
        per = " ".join(f"{d:.1f}" for d in self.denoise)
        return (f"step {self.total:6.1f} ms = device {self.device:6.1f} "
                f"(steps: {per}) + host {self.host:5.1f}")


def flow_match_timesteps(num_steps: int, device, shift: float = 5.0) -> torch.Tensor:
    """Rectified-flow sigmas from 1 (noise) to 0 (data).

    ``shift`` is Wan's resolution-dependent timestep shift: it spends more of a
    short schedule at high noise, which is where a 4-step sampler needs it.
    """
    t = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=torch.float32)
    return shift * t / (1.0 + (shift - 1.0) * t)


class DreamZeroPolicy:
    """A World Action Model driving a control loop.

    The text embedding and the cross-attention projections are computed once
    per instruction; the KV cache carries video history across steps. Inside a
    step nothing is allocated and nothing synchronises, so the host can run
    ahead of the device.
    """

    def __init__(
        self,
        model: CausalWanDiT,
        *,
        batch: int = 1,
        cache_blocks: int = 16,
        runner=None,
    ) -> None:
        self.model = model
        self.cfg: WAMConfig = model.cfg
        self.batch = batch
        self.runner = runner  # a dexter.runtime.GraphRunner, or None for eager

        device, dtype = next(model.parameters()).device, model.dtype
        self.device, self.dtype = device, dtype
        self.kv_cache = KVCache(
            layers=self.cfg.num_layers, batch=batch,
            capacity=self.cfg.seq_len * cache_blocks,
            heads=self.cfg.num_heads, head_dim=self.cfg.head_dim,
            device=device, dtype=dtype,
        )
        self.sigmas = flow_match_timesteps(self.cfg.num_inference_steps, device)
        self.ctx: CrossAttnCache | None = None

    def set_instruction(self, text_embedding: torch.Tensor,
                        clip_features: torch.Tensor | None = None) -> None:
        """Bind a language instruction and, for i2v, the conditioning frame.

        Call once per instruction, not once per step: both projections are
        constant for the episode and cost 40 layers of GEMM to redo.
        """
        self.ctx = self.model.project_context(text_embedding, clip_features)

    def reset(self) -> None:
        self.kv_cache.reset()

    @torch.inference_mode()
    def step(
        self,
        latent: torch.Tensor,  # [B, video_tokens, condition_dim] encoded observation
        state: torch.Tensor,  # [B, num_state_per_block, max_state_dim]
        *,
        generator: torch.Generator | None = None,
        trace: StepTrace | None = None,
    ) -> torch.Tensor:
        """Denoise one action chunk. Returns ``[B, num_action_per_block, action_dim]``."""
        if self.ctx is None:
            raise RuntimeError("call set_instruction() before stepping")
        cfg = self.cfg
        wall_start = time.perf_counter()

        actions = torch.randn(
            (self.batch, cfg.num_action_per_block, cfg.action_dim),
            device=self.device, dtype=self.dtype, generator=generator,
        )
        # Only the denoised channels carry noise; `latent` is the conditioning
        # half and is held fixed for the whole step.
        video = torch.randn(
            (self.batch, latent.shape[1], cfg.head_out_dim),
            device=self.device, dtype=self.dtype, generator=generator,
        )

        # Only the last denoising step writes the KV cache: the earlier ones
        # evaluate the same block at different noise levels, so caching any but
        # the final pass would poison the history with intermediate states.
        last = cfg.num_inference_steps - 1
        for i in range(cfg.num_inference_steps):
            sigma, next_sigma = self.sigmas[i], self.sigmas[i + 1]
            timestep = sigma.expand(self.batch) * 1000.0

            device_ms = None
            if trace is not None:
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()

            model_input = torch.cat((video, latent), dim=-1) if cfg.condition_dim else video
            video_noise, action_noise = self._forward(
                model_input, actions, state, timestep,
                kv_cache=self.kv_cache if i == last else None,
            )

            # Rectified-flow Euler: the model predicts velocity, so a step is a
            # straight line in the direction it points.
            dt = next_sigma - sigma
            actions = actions + dt * action_noise
            video = video + dt * video_noise

            if trace is not None:
                end_evt.record()
                torch.cuda.synchronize()
                device_ms = start_evt.elapsed_time(end_evt)
                trace.denoise.append(device_ms)

        if trace is None:
            torch.cuda.synchronize()
        total = (time.perf_counter() - wall_start) * 1e3
        if trace is not None:
            trace.total = total
            trace.host = max(total - trace.device, 0.0)
        return actions

    def _forward(self, video, actions, state, timestep, kv_cache):
        if self.runner is not None and kv_cache is None:
            return self.runner(video, actions, state, timestep)
        return self.model(video, actions, state, timestep, self.ctx, kv_cache)
