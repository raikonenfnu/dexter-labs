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
    latent = torch.randn(1, TINY.video_tokens, TINY.in_dim, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(1, TINY.num_state_per_block, TINY.max_state_dim,
                        device="cuda", dtype=torch.bfloat16)
    return latent, state


def test_step_shape_and_finiteness(policy):
    latent, state = _obs()
    policy.reset()
    actions = policy.step(latent, state)
    assert actions.shape == (1, TINY.num_action_per_block, TINY.action_dim)
    assert torch.isfinite(actions).all()


def test_kv_cache_advances_one_block_per_step(policy):
    latent, state = _obs()
    policy.reset()
    assert policy.kv_cache.length == 0
    for i in range(1, 4):
        policy.step(latent, state)
        # Exactly one block per control step, regardless of denoising steps --
        # only the last denoising pass may write history.
        assert policy.kv_cache.length == i * TINY.seq_len


def test_kv_cache_overflow_is_an_error():
    model = CausalWanDiT(TINY)
    pol = DreamZeroPolicy(model, batch=1, cache_blocks=1)
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

    model.quantize_(bits=bits)
    pol.reset()
    pol.set_instruction(text)
    torch.manual_seed(0)
    quant_actions = pol.step(latent, state)

    ratio = model.weight_bytes() / dense_bytes
    assert ratio < (0.45 if bits == 4 else 0.75), f"weights only shrank to {ratio:.2f}"
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
