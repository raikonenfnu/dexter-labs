"""GR00T N1.7 architecture, derived from the released checkpoint.

Every number here was read out of `nvidia/GR00T-N1.7-3B`'s tensor shapes rather
than from NVIDIA's config classes. That is deliberate: their loader builds the
backbone by downloading `nvidia/Cosmos-Reason2-2B`, a **gated** repo, even
though all 494 backbone tensors are already in the GR00T checkpoint. Reading
the architecture from the weights removes that dependency entirely -- and is
what an inference engine should be doing anyway.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisionConfig:
    """Qwen3-VL vision tower."""

    dim: int = 1024
    depth: int = 24
    heads: int = 8                      # qkv is [3*dim, dim]; head_dim = dim // heads
    mlp_dim: int = 4096
    patch: tuple[int, int, int] = (2, 16, 16)   # (temporal, height, width)
    in_channels: int = 3
    spatial_merge: int = 2              # merger folds 2x2 patches -> 4*dim
    out_dim: int = 2048                 # merged into the LLM's width
    eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def merge_dim(self) -> int:
        return self.dim * self.spatial_merge**2


@dataclass(frozen=True)
class LanguageConfig:
    """Qwen3 decoder stack (grouped-query attention, per-head QK norm)."""

    dim: int = 2048
    depth: int = 16                     # layers stored in the checkpoint
    heads: int = 16                     # q_proj [2048, 2048] / head_dim
    kv_heads: int = 8                   # k_proj [1024, 2048] / head_dim -> GQA
    head_dim: int = 128
    mlp_dim: int = 6144                 # SwiGLU gate/up width
    eps: float = 1e-6
    rope_theta: float = 1_000_000.0

    @property
    def select_layer(self) -> int:
        """The layer whose hidden states feed the action head.

        N1.7 reads an *intermediate* layer, so the remaining blocks are dead
        weight at inference. Truncating there is the single largest saving
        available on this model and costs nothing.
        """
        return 12


@dataclass(frozen=True)
class ActionHeadConfig:
    """Diffusion-transformer action head with per-embodiment encoders."""

    dim: int = 1536                     # DiT width
    vlm_dim: int = 2048                 # backbone hidden size
    state_dim: int = 132                # padded across every embodiment
    action_dim: int = 132
    hidden: int = 1024                  # encoder/decoder hidden width
    num_embodiments: int = 32
    action_horizon: int = 40
    num_inference_steps: int = 4
    eps: float = 1e-5


@dataclass(frozen=True)
class GrootConfig:
    name: str = "gr00t_n1.7"
    vision: VisionConfig = VisionConfig()
    language: LanguageConfig = LanguageConfig()
    action_head: ActionHeadConfig = ActionHeadConfig()

    def describe(self) -> str:
        v, l, a = self.vision, self.language, self.action_head
        return (
            f"{self.name}: ViT {v.depth}x{v.dim} -> LLM {l.depth}x{l.dim} "
            f"(GQA {l.heads}/{l.kv_heads}, read at layer {l.select_layer}) "
            f"-> DiT {a.dim}, {a.action_horizon} actions x {a.action_dim} dims"
        )
