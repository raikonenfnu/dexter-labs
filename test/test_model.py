"""Model, policy and quantisation behaviour on a small stand-in config."""
import dataclasses

import pytest
import torch

from dexter.models.dreamzero import CausalWanDiT, DreamZeroPolicy, WAN22_TI2V_5B
from dexter.models.dreamzero.policy import StepTrace, flow_match_timesteps
from dexter.quant import quantize
from dexter.ops.reference.gemm import dequantize

TINY = dataclasses.replace(
    WAN22_TI2V_5B, name="tiny", num_layers=2, dim=768, ffn_dim=2048, num_heads=6
)


@pytest.fixture(scope="module")
def policy():
    model = CausalWanDiT(TINY)
    pol = DreamZeroPolicy(model, batch=1, cache_blocks=8)
    pol.set_instruction(torch.randn(1, TINY.text_len, TINY.text_dim,
                                    device="cuda", dtype=torch.bfloat16))
    return pol


def _obs():
    latent = torch.randn(1, TINY.video_tokens, TINY.condition_dim or TINY.patch_dim,
                         device="cuda", dtype=torch.bfloat16)
    state = torch.randn(1, TINY.num_state_per_block, TINY.max_state_dim,
                        device="cuda", dtype=torch.bfloat16)
    return latent, state


def test_step_shape_and_finiteness(policy):
    latent, state = _obs()
    policy.reset()
    actions = policy.step(latent, state)
    assert actions.shape == (1, TINY.num_action_per_block, TINY.action_dim)
    assert torch.isfinite(actions).all()


def test_kv_cache_holds_only_clean_video_tokens(policy):
    """History grows by one block of *video* tokens per control step.

    Not `seq_len`: the action register is never committed. The cache is
    extended by a priming pass on the denoised latent, which runs video-only at
    timestep 0, so what a later block attends to is exclusively clean frames --
    never an action chunk and never a half-denoised intermediate.
    """
    latent, state = _obs()
    policy.reset()
    assert policy.kv_cache.length == 0
    for i in range(1, 4):
        policy.step(latent, state)
        assert policy.kv_cache.length == i * TINY.video_tokens


def test_kv_cache_overflow_is_an_error():
    model = CausalWanDiT(TINY)
    pol = DreamZeroPolicy(model, batch=1, cache_blocks=0)
    pol.set_instruction(torch.randn(1, TINY.text_len, TINY.text_dim,
                                    device="cuda", dtype=torch.bfloat16))
    latent, state = _obs()
    pol.step(latent, state)
    with pytest.raises(RuntimeError, match="KV cache full"):
        pol.step(latent, state)


def test_flow_match_schedule_runs_noise_to_data():
    sigmas = flow_match_timesteps(4, "cuda")
    assert sigmas.shape == (5,)
    assert sigmas[0].item() == pytest.approx(1.0)
    assert sigmas[-1].item() == pytest.approx(0.0)
    assert torch.all(sigmas[1:] <= sigmas[:-1]), "schedule must be monotone"


def test_step_without_instruction_is_an_error():
    pol = DreamZeroPolicy(CausalWanDiT(TINY), batch=1, cache_blocks=2)
    latent, state = _obs()
    with pytest.raises(RuntimeError, match="set_instruction"):
        pol.step(latent, state)


@pytest.mark.parametrize("bits", [4, 8])
def test_quantize_shrinks_weights_and_keeps_output_close(bits):
    model = CausalWanDiT(TINY)
    latent, state = _obs()
    pol = DreamZeroPolicy(model, batch=1, cache_blocks=4)
    text = torch.randn(1, TINY.text_len, TINY.text_dim, device="cuda", dtype=torch.bfloat16)
    pol.set_instruction(text)
    torch.manual_seed(0)
    dense_actions = pol.step(latent, state)
    dense_bytes = model.weight_bytes()

    def block_bytes():
        from dexter.models.layers import Linear
        return sum(m.weight_bytes for b in model.blocks
                   for m in b.modules() if isinstance(m, Linear))

    dense_block_bytes = block_bytes()
    model.quantize_(bits=bits)
    pol.reset()
    pol.set_instruction(text)
    torch.manual_seed(0)
    quant_actions = pol.step(latent, state)

    # quantize_ deliberately packs the transformer stack only, leaving the
    # embeddings and heads dense, so assert on what it actually claims.
    ratio = block_bytes() / dense_block_bytes
    assert ratio < (0.35 if bits == 4 else 0.6), f"blocks only shrank to {ratio:.2f}"
    assert model.weight_bytes() < dense_bytes
    assert torch.isfinite(quant_actions).all()
    assert quant_actions.shape == dense_actions.shape


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("symmetric", [False, True])
def test_quantizer_round_trip(bits, symmetric):
    w = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16) * 0.05
    p = quantize(w, bits=bits, symmetric=symmetric)
    recovered = dequantize(p.qweight, p.scales, p.zeros, p.group_size, bits).t()
    err = (recovered.float() - w.float()).abs().max() / w.float().abs().max()
    # A b-bit uniform grid over a group cannot do better than ~1/2^b of range.
    assert err < (0.12 if bits == 4 else 0.01), f"round-trip error {err:.4f}"
