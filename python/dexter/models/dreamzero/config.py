"""DreamZero backbone shapes, taken from the upstream Hydra configs.

Numbers match ``groot/vla/configs/model/dreamzero/action_head/`` in
github.com/dreamzero0/dreamzero:
``wan_flow_matching_action_tf.yaml`` (Wan2.1-I2V-14B, the released
DreamZero-DROID checkpoint) and ``..._wan22.yaml`` (Wan2.2-TI2V-5B).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["WAMConfig", "WAN21_I2V_14B", "WAN22_TI2V_5B", "PRESETS"]


@dataclass(frozen=True)
class WAMConfig:
    """Shape of one World Action Model."""

    name: str
    model_type: str  # "i2v" | "ti2v"

    # DiT
    dim: int
    ffn_dim: int
    num_heads: int
    num_layers: int
    in_dim: int  # latent channels in
    out_dim: int  # latent channels out
    eps: float = 1e-6
    freq_dim: int = 256  # width of the sinusoidal timestep basis
    patch_size: tuple[int, int, int] = (1, 2, 2)  # (t, h, w) patchification

    # Text conditioning (umt5-xxl), cross-attended and cached per episode.
    text_dim: int = 4096
    text_len: int = 512

    # Sequence layout
    frame_seqlen: int = 880  # video tokens per frame after patch embedding
    # Patch grid behind frame_seqlen. 3D rotary needs the (height, width) split,
    # which the token count alone does not determine.
    latent_height: int = 22
    latent_width: int = 40
    num_frame_per_block: int = 1
    num_action_per_block: int = 32
    num_state_per_block: int = 1

    # Action head
    action_dim: int = 32
    max_state_dim: int = 64
    state_hidden: int = 1024  # hidden width of the state/action MLPs
    num_embodiments: int = 1  # category-specific weights, one per embodiment

    # i2v image conditioning: CLIP features cross-attended alongside text.
    clip_dim: int = 1280

    # Flow-matching schedule
    num_inference_steps: int = 4

    @property
    def head_dim(self) -> int:
        if self.dim % self.num_heads:
            raise ValueError(f"dim {self.dim} is not divisible by num_heads {self.num_heads}")
        return self.dim // self.num_heads

    def __post_init__(self) -> None:
        expected = self.latent_height * self.latent_width
        if expected != self.frame_seqlen:
            raise ValueError(
                f"{self.name}: latent grid {self.latent_height}x{self.latent_width} "
                f"= {expected} tokens, but frame_seqlen is {self.frame_seqlen}"
            )

    @property
    def patch_grid(self) -> tuple[int, int, int]:
        """``(frames, height, width)`` of patch tokens in one block."""
        return (self.num_frame_per_block, self.latent_height, self.latent_width)

    @property
    def patch_dim(self) -> int:
        """Input width of ``patch_embedding``.

        Wan patchifies with a Conv3d of stride ``patch_size``, which is exactly
        a linear map over ``in_dim * prod(patch_size)`` inputs. Storing it as
        that linear keeps the GEMM path uniform, and the checkpoint's conv
        weight reshapes into it without reordering.
        """
        t, h, w = self.patch_size
        return self.in_dim * t * h * w

    @property
    def head_out_dim(self) -> int:
        """Output width of the video head, before unpatchifying.

        Also the width of the *noisy* part of the DiT input: the model denoises
        ``out_dim`` channels and predicts exactly those.
        """
        t, h, w = self.patch_size
        return self.out_dim * t * h * w

    @property
    def condition_dim(self) -> int:
        """Width of the conditioning channels appended to the noisy latent.

        Wan's i2v input is ``[noisy latent ; conditioning]`` along channels --
        ``in_dim`` 36 is 16 denoised channels plus 20 carrying the first-frame
        latent and its mask -- while ``out_dim`` is only the 16. So a denoising
        step updates a ``head_out_dim``-wide tensor and re-concatenates the
        conditioning, which is constant for the block. Zero for t2v/ti2v
        backbones, where the latent is the whole input.
        """
        return self.patch_dim - self.head_out_dim

    @property
    def video_tokens(self) -> int:
        """Video tokens the DiT sees in one step."""
        return self.frame_seqlen * self.num_frame_per_block

    @property
    def register_tokens(self) -> int:
        """Action register: one action chunk plus the state token(s)."""
        return self.num_action_per_block + self.num_state_per_block

    @property
    def seq_len(self) -> int:
        """Query length of a single denoising step."""
        return self.video_tokens + self.register_tokens

    def parameters_per_layer(self) -> int:
        d, f, t = self.dim, self.ffn_dim, self.text_dim
        self_attn = 4 * d * d  # q, k, v, o
        cross_attn = 2 * d * d + 2 * t * d  # q, o from x; k, v from text
        ffn = 2 * d * f
        return self_attn + cross_attn + ffn

    def dit_parameters(self) -> int:
        return self.num_layers * self.parameters_per_layer()

    def weight_bytes(self, bits: int = 16, group_size: int = 128) -> int:
        """Bytes of DiT weight the memory system must deliver per denoising step.

        This is the quantity that sets the step time on a bandwidth-bound part,
        so it includes the quantisation scales and zero points -- they are real
        traffic, roughly ``2 * 2 / group_size`` bytes per weight.
        """
        params = self.dit_parameters()
        if bits == 16:
            return params * 2
        overhead = 2 * 2 / group_size  # bf16 scale + bf16 zero per group
        return int(params * (bits / 8 + overhead))

    def flops_per_denoise_step(self) -> int:
        """Multiply-accumulates x2 for one pass of ``seq_len`` tokens.

        Attention's own QK/PV work is left out: at 83 tokens against a few
        thousand cached keys it is under 2% of this, and including it would
        imply a precision the rest of the estimate does not have.
        """
        return 2 * self.dit_parameters() * self.seq_len

    def arithmetic_intensity(self, bits: int = 16, group_size: int = 128) -> float:
        """FLOP per byte of weight traffic for one denoising step.

        Compare against the machine balance (peak FLOP/s over achievable
        bandwidth) to see which side of the roofline a configuration sits on.
        Weight-only quantisation raises this ratio, so it moves a memory-bound
        shape toward compute -- and stops paying once it arrives.
        """
        return self.flops_per_denoise_step() / self.weight_bytes(bits, group_size)

    def describe(self) -> str:
        return (
            f"{self.name}: {self.dit_parameters()/1e9:.1f}B DiT params, "
            f"{self.num_layers}L x dim {self.dim} x {self.num_heads}H "
            f"(head_dim {self.head_dim}), ffn {self.ffn_dim}; "
            f"{self.seq_len} tokens/step "
            f"({self.video_tokens} video + {self.register_tokens} register)"
        )


# Released DreamZero-DROID checkpoint.
WAN21_I2V_14B = WAMConfig(
    name="dreamzero-droid-wan2.1-i2v-14b",
    model_type="i2v",
    dim=5120,
    ffn_dim=13824,
    num_heads=40,
    num_layers=40,
    in_dim=36,
    out_dim=16,
    frame_seqlen=880,
    latent_height=22,
    latent_width=40,
    num_frame_per_block=2,
    num_action_per_block=24,
    num_state_per_block=1,
    action_dim=32,
    max_state_dim=64,
)

# Lower-VRAM backbone: 160x320 video -> 10x20 latent -> 5x10 = 50 tokens/frame.
WAN22_TI2V_5B = WAMConfig(
    name="dreamzero-wan2.2-ti2v-5b",
    model_type="ti2v",
    dim=3072,
    ffn_dim=14336,
    num_heads=24,
    num_layers=30,
    in_dim=48,
    out_dim=48,
    frame_seqlen=50,
    latent_height=5,
    latent_width=10,
    num_frame_per_block=1,
    num_action_per_block=32,
    num_state_per_block=1,
)

PRESETS = {"14b": WAN21_I2V_14B, "5b": WAN22_TI2V_5B}
