"""Model-side building blocks that route through :mod:`dexter.ops`."""

from __future__ import annotations

import torch
import torch.nn as nn

from dexter import ops
from dexter.quant import QuantizedWeight, quantize


class Linear(nn.Module):
    """``nn.Linear`` that can swap its weight for a packed int4/int8 one.

    Keeping both behind one module means quantisation is a deployment choice
    made after the graph is built -- ``model.quantize_(bits=4)`` walks the
    modules and repacks in place, and nothing in the forward pass changes shape.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype)) if bias else None
        self.packed: QuantizedWeight | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=self.in_features ** -0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @torch.no_grad()
    def quantize_(self, bits: int = 4, group_size: int = 128) -> None:
        """Repack the weight and drop the dense copy."""
        if self.in_features % group_size or (bits == 4 and (self.in_features // 2) % group_size):
            return  # K does not tile into groups cleanly; leave this one dense
        self.packed = quantize(
            self.weight.data, bits=bits, group_size=group_size, bias=self.bias
        ).to(self.weight.device)
        self.weight = None
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.packed is None:
            return ops.linear(x, self.weight, self.bias)
        p = self.packed
        return ops.linear(x, p.qweight, p.bias, scales=p.scales, zeros=p.zeros,
                          group_size=p.group_size, bits=p.bits)

    @property
    def weight_bytes(self) -> int:
        if self.packed is not None:
            return self.packed.nbytes
        n = self.weight.numel() * self.weight.element_size()
        return n + (self.bias.numel() * self.bias.element_size() if self.bias is not None else 0)


class RMSNorm(nn.Module):
    """Wan's per-head QK norm."""

    def __init__(self, dim: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ops.rms_norm(x, self.weight, self.eps)


class AffineLayerNorm(nn.Module):
    """LayerNorm with learned scale and shift (Wan's ``norm3``)."""

    def __init__(self, dim: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused as adaln: LN(x) * (1 + (w - 1)) + b.
        return ops.adaln_modulate(x, self.weight - 1, self.bias, self.eps)
