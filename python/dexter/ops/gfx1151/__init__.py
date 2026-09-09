"""Kernels for AMD RDNA3.5 (gfx1151 / gfx1150, Strix Halo and Strix Point).

What is different here, and why the kernels look the way they do:

* **wave32 + WMMA.** ``tl.dot`` lowers to ``v_wmma_f32_16x16x16_bf16`` over a
  32-lane wave. The MMA tile is 16x16x16, so ``BLOCK_M = 16`` is a full tile,
  not a quarter-empty one -- which suits a DiT step's ~125 query tokens.
* **No FP8 matrix path.** WMMA takes f16/bf16/int8/int4 only. Memory savings
  come from weight-only int4/int8 with a bf16 math path, not from FP8 GEMM.
* **~231 GB/s of measured LPDDR5X**, shared with the CPU, against ~59 bf16
  TFLOP/s of WMMA. The break-even arithmetic intensity is ~256 FLOP/byte; a
  batch-1 DiT step runs at about 2. Everything here is written to move fewer
  bytes, and nothing here is written to do fewer FLOPs.
"""

from dexter.ops.gfx1151 import attention, gemm, norm, rope  # noqa: F401
