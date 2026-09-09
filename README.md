# dexter

A minimal inference engine for **World Action Models** on **AMD RDNA3.5**.

Built to run [DreamZero](https://github.com/dreamzero0/dreamzero) (NVIDIA GEAR's
World Action Model) on a Strix Halo APU — `gfx1151`, 40 CU, 32 GB of unified
LPDDR5X. Structured after [TokenSpeed](https://github.com/lightseekorg/tokenspeed):
kernels are a pluggable subsystem behind a portable op API, a capability-gated
registry, and a priority-band selector, rather than being welded into the model.

Status: **working engine, honest numbers.** The full 5B backbone runs a closed
loop end to end. Some of the numbers are not the ones the design predicted, and
the interesting part of this README is where they diverge.

---

## The target, and why it dictates the design

Everything below is measured on this box, not quoted from a datasheet.

| | measured |
|---|---|
| Read bandwidth | **233 GB/s** (256 GB/s peak, 91%) |
| Dense bf16 matmul | **29.5 TFLOP/s** |
| **Machine balance** | **126 FLOP/byte** |
| Infinity Cache (MALL) | 32 MB |
| L2 | 2 MB |
| Wavefront | 32 |
| Matrix unit | `v_wmma_f32_16x16x16_bf16` (verified in emitted ISA) |

Three RDNA3.5 facts change kernel design, not just tuning constants:

1. **WMMA, not MFMA, over wave32.** The MMA tile is 16×16×16. `tl.dot` lowers
   to `v_wmma_f32_16x16x16_bf16` — confirmed by disassembling the generated
   code object.
2. **No FP8 matrix path.** WMMA covers f16/bf16/int8/int4 only. The MI300X FP8
   recipe does not port; the equivalent memory saving has to come from
   *weight-only* int4/int8 with bf16 math.
3. **Unified memory with a 32 MB MALL.** No PCIe hop, but no HBM either. The
   MALL is large enough to hold a per-layer weight tensor, which makes
   re-reading weights across M-blocks far cheaper than it would be on a part
   without one — the GEMM tile heuristic exploits this deliberately.

## Where a DreamZero step actually sits on the roofline

This was the finding that redirected the whole design. A closed-loop step is
**not** simply bandwidth bound.

```
$ dexter-bench roofline
measured: 233 GB/s read, 29.5 TFLOP/s bf16
machine balance: 126 FLOP/byte

model  bits  weights GB  FLOP/byte  bound by  ms/denoise  ms/control  Hz
-----  ----  ----------  ---------  --------  ----------  ----------  ---
14b    16    27.26       913        compute   843.5       3373.9      0.3
14b    8     14.06       1771       compute   843.5       3373.9      0.3
14b    4     7.24        3437       compute   843.5       3373.9      0.3
5b     16    10.19       83          memory    43.7        174.6       5.7
5b     8     5.26        161        compute   28.7        114.7       8.7
5b     4     2.71        312        compute   28.7        114.7       8.7
```

A weight-only quantised GEMM has an arithmetic intensity of `(16/bits) × M`
FLOP per weight byte. With `M = 83` tokens per step:

* **bf16 5B is memory bound** (83 < 126) — quantisation should help.
* **int8 and int4 are already compute bound** (161, 312 > 126) — they hit the
  same 28.7 ms floor. The extra bits saved buy nothing further in time.
* **The 14B is compute bound at any width.** At 913 tokens/step its floor is
  ~3.4 s per control step. On this part, int4 for the 14B buys the ability to
  *load it at all* (7.2 GB vs 27.3 GB in 32 GB shared with the OS), not speed.

So on RDNA3.5 the honest conclusion is: **weight-only quantisation is a
capacity technique here, and only marginally a latency one.** That is the
opposite of what the pure-bandwidth intuition suggests, and it is why this
README leads with the balance point.

## Results

### Fused elementwise ops — where Triton wins

```
$ dexter-bench kernels --model 5b
op              dexter ms  torch ms  speedup x  GB/s
--------------  ---------  --------  ---------  ----
qk_norm_rope    0.019      0.213     11.43      110
adaln           0.017      0.027     1.54       59
gated_residual  0.017      0.014     0.85       92
```

These are the ops with no fused torch equivalent. `qk_norm_rope` collapses four
passes (two RMSNorms, two rotaries) into one and is worth 11×.
`gated_residual` **loses** — `x + y * gate` is already a single fused
elementwise op in eager torch and the Triton launch is not cheaper. It stays
registered because it costs nothing to keep, but the engine is not pretending
it is a win.

A methodology note, because it changed the conclusion. These kernels run in
~20 microseconds; a Python call plus a HIP launch plus a synchronise is the
same order. Timed one call per sync, adaln reads as **0.59×**. Timed 200 calls
behind one sync, it reads as **1.54×**. The first number measures the harness.
`bench.harness.measure_kernel` does the second and takes the minimum across
trials, since the floor is the kernel and everything above it is a CPU
contending for the same package power budget.

### Dense GEMM — where hipBLASLt wins

Measured across M = 1…913 at Wan shapes, hand-written Triton lost to hipBLASLt
by a consistent ~1.4× at every size. The registry records that verdict:

```
$ dexter-bench platform
gemm/dense
  * hipblaslt_dense_gfx1151      SELECTED (pri=8)
  - triton_dense_gfx1151         viable (pri=4)
  - torch_mm                     viable (pri=0)
```

Vendor BLAS is the right answer for a plain dense GEMM on this part. Shipping
the Triton kernel as the default would have cost 40% on the hottest op in the
model.

### Quantised GEMM — correct, and currently too slow

The int4 kernel is numerically correct (0.7% relative error, consistent with a
4-bit grid) but does **not** yet beat dense bf16. Decomposing it at
`M=83, K=3072, N=14336`:

| stage | time | vs roofline |
|---|---|---|
| packed loads only | 0.094 ms | **at the 0.100 ms memory roofline** |
| + dequantise | 0.466 ms | 5× the load cost |
| + WMMA dot | 1.267 ms | vs 0.713 ms for dense bf16 |

The loads are perfect. **Dequantisation is the bottleneck**, and it is a
codegen problem, not an algorithmic one. Replacing the integer→float convert
with a bit-pattern construction (OR the nibble into the mantissa of `0x4300`,
which yields exactly `128 + n` in bf16, with the `+128` folded into the zero
point on the host) took the full kernel from 1.514 ms to 1.152 ms. That is a
real 1.3× and it is in the tree. Fixing the tile search to pick pipeline depth
before k-tile depth — the inner loop is dequantise-bound, so overlapping the
next tile's loads matters more than a deeper tile — was worth another 2.1×
(2.62 ms → 1.27 ms). A further ~2× is still needed and it is not reachable by
tile tuning: every configuration plateaus at the same ~23 GB/s.

The next step is packing into `int32` words rather than `uint8`: a streaming
microbenchmark shows int32 loads sustaining 426 GB/s against 248 GB/s for the
same bytes as uint8, and unpacking from a native register width should remove
the sub-dword conversions that dominate. That work is not done.

## Reproducing the vLLM-Omni post

The [post](https://andyluo7.github.io/rocm/amd/mi300x/vllm-omni/worldmodels/dreamzero/robotics/vla/2026/08/14/world-models-vllm-omni-rocm-dreamzero/)
reports DreamZero on **MI300X** going 564.7 ms → 398.3 ms per closed-loop step
(1.42×) from three changes. It cites PR numbers but publishes no commands,
configs or code, and vLLM-Omni is not public, so a literal re-run is not
possible. What is reproducible is the **methodology** — measure the closed-loop
step, not a kernel — and each optimisation class as it translates to RDNA3.5:

| MI300X change | RDNA3.5 translation | status |
|---|---|---|
| Host-side scheduling overlap (PR #5971) | HIP graph capture of the denoise step | implemented; **host is only 0.8 ms of a 377 ms step here, so there is nothing to hide** |
| FP8 DiT GEMMs (PR #6203) | no FP8 matrix path — weight-only int4/int8 instead | implemented, correct, slower than dense (above) |
| Fused VAE Conv3D | — | **not implemented** |

The host-overlap result is worth stating plainly: on MI300X the GPU is fast
enough that CPU-side work sits in front of it. On Strix Halo the device is slow
enough that 0.8 ms of host time disappears entirely behind 377 ms of device
time. The optimisation is correct and does not apply here.

### Measured closed-loop step, 5B backbone

Full Wan2.2-TI2V-5B shape (30 layers, dim 3072, 24 heads, ffn 14336), 83 tokens
per step, 4 denoising steps, batch 1:

```
$ dexter-bench e2e --model 5b --layers 30 --bits 16 8 4 --graph
precision  graph  weights GB  step ms  Hz   device ms  host ms
---------  -----  ----------  -------  ---  ---------  -------
bf16       False  10.35       361.1    2.8  362.1      0.7
bf16       True   10.35       362.2    2.8  365.4      1.6
int8       False  5.41        801.7    1.2  812.3      0.9
int8       True   5.41        818.0    1.2  802.3      0.5
int4       False  2.86        558.5    1.8  565.2      0.8
int4       True   2.86        569.9    1.8  576.9      0.6
```

Reading this honestly:

* **bf16 at 361 ms is 2.1x off its 174.6 ms roofline.** That gap is the
  remaining work, and it is mostly in attention and the unfused residual paths.
* **Graph capture changes nothing**, within noise, in all six rows. Host time
  is 0.5-1.6 ms against ~360 ms of device time. The optimisation is implemented
  and correct; on this hardware there is simply nothing behind the GPU to hide.
* **Quantisation currently costs time**, because the quantised GEMM is slower
  than hipBLASLt (above), not because of anything about the schedule. int4 at
  559 ms beats int8 at 802 ms for the same reason its kernel is faster.
* **int4 does deliver the capacity claim**: 2.86 GB against 10.35 GB, a 3.6x
  reduction, at 30 layers of a 5B backbone. That is the axis on which it is
  the enabling technique for the 14B, which does not otherwise fit.

## Layout

```
python/dexter/
  platform.py          capability detection; add an arch = one row in _AMD_ARCHS
  registry.py          KernelSpec, priority bands, capability-gated selection
  quant.py             weight-only int4/int8 packing (K-major, split-half)
  ops/
    __init__.py        the public op surface the model is allowed to call
    reference/         torch ground truth, Priority.REFERENCE
    gfx1151/           RDNA3.5 kernels: norm, rope, gemm, attention
  models/
    layers.py          Linear that swaps dense <-> packed in place
    dreamzero/         config (from upstream Hydra), DiT, closed-loop policy
  runtime/graph.py     HIP graph capture of the denoise step
  bench/               roofline, kernel microbench, end-to-end
test/                  28 tests: every kernel against its reference
```

Adding an architecture is one row in `platform._AMD_ARCHS` plus a directory of
kernels; nothing in the model changes. Kernels declare what they need
(`mma:wmma`, `matrix:int4`, `memory:unified`) and the selector filters on
capability before it ever consults priority. `DEXTER_KERNEL=gemm/w4a16:name`
pins a choice for bisection.

## Install and run

```bash
python3 -m venv ~/nod/venv/robo.venv
source ~/nod/venv/robo.venv/bin/activate

# gfx1151 needs AMD's TheRock build. The stock rocm7.0 wheels list gfx1151 in
# get_arch_list() but segfault inside ROCr at first dispatch on this box.
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch torchvision
pip install -e .

dexter-bench platform     # capabilities, bandwidth, kernel selection
dexter-bench roofline     # where each config sits on the roofline
dexter-bench kernels      # fused kernels vs torch references
dexter-bench e2e --model 5b --layers 30 --bits 16 8 4 --graph
pytest test -q
```

## What is not here

* **Real checkpoint loading.** The model is built to the upstream config shapes
  and benchmarked with initialised weights. Loading `GEAR-Dreams/DreamZero-DROID`
  needs a parameter-name mapping that is not written, and the 14B checkpoint is
  45 GB against 32 GB of shared memory — it needs int4 packing during load, not
  after.
* **VAE and text encoder.** The engine serves the DiT, which is where a step
  spends its time. Observations enter as latents and instructions as embeddings.
  The fused Conv3D from the post is not implemented.
* **A competitive int4 GEMM.** Diagnosed, not fixed. See above.
* **Multi-GPU.** Upstream shards across 2+ GPUs; this is a single-APU engine.
