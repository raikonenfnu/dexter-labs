"""End-to-end closed-loop step latency.

This mirrors the methodology of the vLLM-Omni DreamZero write-up -- measure the
*closed-loop step*, the thing the robot waits on, not a kernel in isolation --
and applies its three optimisation classes as they translate to RDNA3.5:

  1. host-side scheduling      -> HIP graph capture of the denoising step
  2. FP8 GEMM on MI300X        -> weight-only int8/int4 (no FP8 matrix path here)
  3. fused VAE conv3d          -> not implemented; see the README
"""

from __future__ import annotations

import dataclasses

import torch

from dexter.bench.harness import measure
from dexter.models.dreamzero import CausalWanDiT, DreamZeroPolicy, WAMConfig
from dexter.models.dreamzero.policy import StepTrace
from dexter.runtime import capture_denoise_step


def _inputs(cfg: WAMConfig, device, dtype):
    latent = torch.randn(1, cfg.video_tokens, cfg.condition_dim or cfg.patch_dim,
                         device=device, dtype=dtype)
    state = torch.randn(1, cfg.num_state_per_block, cfg.max_state_dim, device=device, dtype=dtype)
    text = torch.randn(1, cfg.text_len, cfg.text_dim, device=device, dtype=dtype)
    clip = torch.randn(1, 257, cfg.clip_dim, device=device, dtype=dtype)
    return latent, state, text, clip


def run(cfg: WAMConfig, *, bits: int | None, use_graph: bool,
        layers: int | None = None, dtype=torch.bfloat16) -> dict:
    """Build, optionally quantise, and time one configuration."""
    if layers is not None:
        cfg = dataclasses.replace(cfg, num_layers=layers)

    model = CausalWanDiT(cfg, dtype=dtype)
    if bits is not None:
        model.quantize_(bits=bits)

    latent, state, text, clip = _inputs(cfg, "cuda", dtype)
    policy = DreamZeroPolicy(model, batch=1, cache_blocks=4)
    policy.set_instruction(text, clip)

    if use_graph:
        actions = torch.randn(1, cfg.num_action_per_block, cfg.action_dim, device="cuda", dtype=dtype)
        timestep = torch.zeros(1, device="cuda", dtype=torch.float32)
        policy.runner = capture_denoise_step(
            model, policy.ctx, (latent, actions, state, timestep)
        )

    def one_step():
        policy.reset()
        policy.step(latent, state)

    ms = measure(one_step, warmup=3, iters=10)

    trace = StepTrace()
    policy.reset()
    policy.step(latent, state, trace=trace)

    result = {
        "config": cfg.name,
        "layers": cfg.num_layers,
        "precision": "bf16" if bits is None else f"int{bits}",
        "graph": use_graph,
        "weight_gb": model.weight_bytes() / 1e9,
        "step_ms": ms,
        "hz": 1e3 / ms,
        "device_ms": trace.device,
        "host_ms": trace.host,
    }

    del policy, model, latent, state, text, clip
    torch.cuda.empty_cache()
    return result
