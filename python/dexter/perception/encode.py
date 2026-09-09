"""Turn camera frames and language into the DiT's conditioning.

The DiT is conditioned on three things, and getting any of their shapes or
scalings wrong produces plausible-looking garbage rather than an error, so the
exact contract is spelled out here:

``y`` -- ``[B, 20, T_lat, H/8, W/8]``, the i2v conditioning. It is a 4-channel
    mask concatenated with the 16-channel VAE latent of the observation, where
    the mask marks which latent frames are *known*. The first frame is the real
    observation; the rest are zeros the model is being asked to imagine. This
    is the tensor that becomes the DiT's 20 conditioning channels (80 after
    patchification), alongside the 16 noisy ones it denoises.

``clip_context`` -- ``[B, 257, 1280]`` from the CLIP ViT-H tower on the same
    first frame, cross-attended by every block's image branch.

``text`` -- ``[B, 512, 4096]`` from umt5-xxl.

Frames are ``[-1, 1]``, not ``[0, 1]``: the VAE was trained that way and a
half-scale input silently produces a washed-out latent.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from dexter.models.dreamzero.checkpoint import _ShardReader

# 352x640 video -> 44x80 latent -> 22x40 patches = 880 tokens per latent frame,
# which is the frame_seqlen the released DROID config was trained at.
VIDEO_HEIGHT = 352
VIDEO_WIDTH = 640
CLIP_TOKENS = 257


def _materialize(factory, reader: _ShardReader, prefix: str, device, dtype):
    """Build a module on ``meta`` and fill it tensor by tensor from the shards.

    Never assembles a full state dict, and never touches fp32. Both matter for
    umt5-xxl: at 11.4 GB in bf16 it is 22.7 GB in fp32, so reading the state
    dict at full precision on a 32 GB machine is an OOM before the module even
    exists. Building on meta and assigning device tensors directly keeps peak
    residency at the model's own bf16 size plus one tensor.
    """
    with torch.device("meta"):
        module = factory()

    keys = sorted((k for k in reader.weight_map if k.startswith(prefix)),
                  key=reader.shard_of)  # shard-grouped: map each file once
    if not keys:
        raise KeyError(f"no tensors under {prefix!r}")
    available = {k[len(prefix):] for k in keys}

    named = dict(module.named_parameters())
    named.update(dict(module.named_buffers()))
    missing = sorted(n for n in named if n not in available)
    if missing:
        raise KeyError(f"{prefix}: checkpoint is missing {len(missing)} tensors, "
                       f"e.g. {missing[:3]}")

    for key in keys:
        name = key[len(prefix):]
        if name not in named:
            continue  # checkpoint carries something this build does not use
        tensor = reader.get(key).to(device=device, dtype=dtype)
        parent = module.get_submodule(name.rpartition(".")[0]) if "." in name else module
        attr = name.rpartition(".")[2]
        if attr in parent._parameters:
            parent._parameters[attr] = torch.nn.Parameter(tensor, requires_grad=False)
        else:
            parent._buffers[attr] = tensor
    reader.release()
    return module.eval()


class ObservationEncoder:
    """VAE, CLIP and umt5, loaded on demand from the DreamZero checkpoint.

    Each is loaded only when first used and can be released independently.
    That matters here: umt5-xxl alone is 11.4 GB against a 9 GB int4 DiT on a
    32 GB machine, so a rollout encodes its instruction once, frees the text
    tower, and only then brings up the DiT.
    """

    def __init__(self, checkpoint: str | Path, device="cuda", dtype=torch.bfloat16) -> None:
        self.reader = _ShardReader(Path(checkpoint))
        self.device, self.dtype = device, dtype
        self._vae = None
        self._clip = None
        self._text = None

    # -- lazy components ---------------------------------------------------

    @property
    def vae(self):
        if self._vae is None:
            from dexter.perception.wan_vae import WanVideoVAE

            self._vae = _materialize(lambda: WanVideoVAE(z_dim=16), self.reader,
                                     "action_head.vae.", self.device, self.dtype)
        return self._vae

    @property
    def clip(self):
        if self._clip is None:
            from dexter.perception.wan_clip import WanImageEncoder

            self._clip = _materialize(WanImageEncoder, self.reader,
                                      "action_head.image_encoder.", self.device, self.dtype)
        return self._clip

    @property
    def text(self):
        if self._text is None:
            from dexter.perception.wan_t5 import WanTextEncoder

            self._text = _materialize(WanTextEncoder, self.reader,
                                      "action_head.text_encoder.", self.device, self.dtype)
        return self._text

    def release(self, *names: str) -> None:
        """Drop named components ("vae", "clip", "text") and their memory."""
        for name in names or ("vae", "clip", "text"):
            setattr(self, f"_{name}", None)
        torch.cuda.empty_cache()

    # -- encoding ----------------------------------------------------------

    @torch.no_grad()
    def encode_observation(self, frame: torch.Tensor, num_frames: int):
        """``frame`` is ``[3, H, W]`` in ``[-1, 1]``. Returns ``(clip, y)``.

        ``num_frames`` is the pixel-frame horizon the model imagines; the VAE
        compresses it 4x temporally, so ``T_lat = 1 + (num_frames - 1) // 4``.
        """
        frame = frame.to(device=self.device, dtype=self.dtype)
        if frame.shape[-2:] != (VIDEO_HEIGHT, VIDEO_WIDTH):
            frame = F.interpolate(frame[None], size=(VIDEO_HEIGHT, VIDEO_WIDTH),
                                  mode="bicubic", align_corners=False)[0]

        # CLIP wants [B, T, 3, H, W]; one frame.
        clip_context = self.clip.encode_image(frame[None, None]).to(self.dtype)

        # The VAE sees the observed frame followed by the frames to imagine.
        blanks = torch.zeros(1, 3, num_frames - 1, VIDEO_HEIGHT, VIDEO_WIDTH,
                             device=self.device, dtype=self.dtype)
        video = torch.cat((frame[None, :, None], blanks), dim=2)
        latent = self.vae.encode(video, tiled=False)

        # Mask channels: 1 where the latent frame is a real observation.
        mask = torch.zeros(1, 4, *latent.shape[2:], device=self.device, dtype=latent.dtype)
        mask[:, :, :1] = 1.0
        return clip_context, torch.cat((mask, latent), dim=1)

    @torch.no_grad()
    def encode_text(self, prompt: str, tokenizer, text_len: int = 512) -> torch.Tensor:
        tokens = tokenizer(prompt, padding="max_length", truncation=True,
                           max_length=text_len, return_tensors="pt")
        ids = tokens.input_ids.to(self.device)
        mask = tokens.attention_mask.to(self.device)
        out = self.text(ids, mask)
        # Padding positions are zeroed so they cannot contribute to attention.
        return (out * mask.unsqueeze(-1)).to(self.dtype)

    @torch.no_grad()
    def decode_latents(self, latent: torch.Tensor) -> torch.Tensor:
        """``[B, 16, T, H, W]`` latent -> ``[B, 3, T, H, W]`` video in ``[-1, 1]``."""
        return self.vae.decode(latent.to(self.dtype), tiled=False)


# -- patchification --------------------------------------------------------
#
# The input and the output do NOT use the same feature order, and assuming they
# do is a silent, plausible-looking failure.
#
#   input   patch_embedding is a Conv3d of stride (1, 2, 2), so a token's
#           features are the patch flattened as (c, kt, kh, kw) -- channel FIRST.
#   output  the head is a plain Linear and upstream reads its 64 outputs back
#           with `view(B, F, H, W, pt, ph, pw, c)` -- channel LAST.
#
# So `patchify` and `unpatchify` below are deliberately not inverses of each
# other: each matches the convention of the end it serves.


def patchify(latent: torch.Tensor, patch_size=(1, 2, 2)) -> torch.Tensor:
    """``[B, C, T, H, W]`` -> ``[B, T*H'*W', C*prod(patch)]``."""
    b, c, t, h, w = latent.shape
    pt, ph, pw = patch_size
    x = latent.view(b, c, t // pt, pt, h // ph, ph, w // pw, pw)
    # -> B, T', H', W', C, pt, ph, pw   (token axes first, then the conv order)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(b, (t // pt) * (h // ph) * (w // pw), c * pt * ph * pw)


def unpatchify(tokens: torch.Tensor, grid: tuple[int, int, int], channels: int,
               patch_size=(1, 2, 2)) -> torch.Tensor:
    """Video head output -> ``[B, C, T, H, W]``. ``grid`` is ``(T', H', W')``.

    Features are read as ``(pt, ph, pw, c)`` -- channel last -- matching
    upstream's ``view(B, *grid, *patch_size, c)`` followed by
    ``einsum('bfhwpqrc->bcfphqwr')``. This is *not* the order `patchify` writes;
    see the note above.
    """
    b = tokens.shape[0]
    tg, hg, wg = grid
    pt, ph, pw = patch_size
    x = tokens.view(b, tg, hg, wg, pt, ph, pw, channels)
    # bfhwpqrc -> bcfphqwr
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return x.reshape(b, channels, tg * pt, hg * ph, wg * pw)


def unpatchify_input(tokens: torch.Tensor, grid: tuple[int, int, int], channels: int,
                     patch_size=(1, 2, 2)) -> torch.Tensor:
    """True inverse of :func:`patchify` -- channel-first, the Conv3d order.

    Use this on anything that lives in the *input* layout, such as the noisy
    latent the sampler carries between denoising steps. Using the output-order
    :func:`unpatchify` on it instead leaves the coarse image intact but
    transposes the elements inside every patch, which shows up as a regular
    grid at the patch pitch.
    """
    b = tokens.shape[0]
    tg, hg, wg = grid
    pt, ph, pw = patch_size
    x = tokens.view(b, tg, hg, wg, channels, pt, ph, pw)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)  # -> b, c, f, pt, h, ph, w, pw
    return x.reshape(b, channels, tg * pt, hg * ph, wg * pw)


def patchify_output(video: torch.Tensor, patch_size=(1, 2, 2)) -> torch.Tensor:
    """Inverse of :func:`unpatchify` -- for comparing against a reference."""
    b, c, t, h, w = video.shape
    pt, ph, pw = patch_size
    x = video.view(b, c, t // pt, pt, h // ph, ph, w // pw, pw)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)  # bcfphqwr -> bfhwpqrc
    return x.reshape(b, (t // pt) * (h // ph) * (w // pw), c * pt * ph * pw)


def output_to_input_order(tokens: torch.Tensor, grid: tuple[int, int, int],
                          channels: int, patch_size=(1, 2, 2)) -> torch.Tensor:
    """Re-lay the head's output into the convention its input uses.

    The DiT consumes tokens channel-first and emits them channel-last, so a
    flow-matching update of the form ``x = x + dt * v`` is only meaningful once
    ``v`` has been moved into ``x``'s layout. Adding them directly type-checks,
    runs, and scrambles every patch.
    """
    return patchify(unpatchify(tokens, grid, channels, patch_size), patch_size)
