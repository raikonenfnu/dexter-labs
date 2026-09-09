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

## Robotics demo: the latency that actually reaches the arm

`dexter-demo` drives a 7-DoF arm through the same closed-loop contract as
upstream's `eval_utils/run_sim_eval.py`: the policy returns a 32-action chunk,
only `open_loop_horizon` (8) of them execute, then it is re-queried. Upstream
that query is a **blocking** websocket call, so the robot stops moving for the
whole inference.

At DROID's 15 Hz, 8 actions is 533 ms of motion, and a 5B step is ~400 ms. The
arm therefore spends nearly half its life frozen:

```
$ dexter-demo --model 5b --ticks 150 --horizon 8 --prefetch 4 8
scheduler       rate      calls              stalled       worst      deadline misses
blocking          9.0 Hz   19 calls  stall  7.94s (47.6%)  worst  488.4 ms  misses  19/150
pipelined/pf4    12.7 Hz   19 calls  stall  3.10s (26.2%)  worst  405.8 ms  misses  19/150
pipelined/pf8    14.5 Hz   19 calls  stall  0.40s ( 3.9%)  worst  399.1 ms  misses   1/150
```

**Blocking loses 47.6% of wall time to inference stalls** — 9.0 Hz against a
15 Hz target, and a deadline miss on every chunk boundary. Starting the next
inference `prefetch` actions before the current chunk runs dry overlaps it with
motion the arm already has queued. A prefetch of 8 buys 533 ms of cover for a
~400 ms step and takes the arm to **14.5 Hz with one miss in 150**.

The residual 0.40 s is a cold start, and irreducible: the first chunk has no
predecessor to hide behind. Every later inference is fully overlapped.

The tempting alternative is to lengthen the open loop instead. It works, and it
is worse:

```
$ dexter-demo --model 5b --ticks 150 --horizon 16 --prefetch 8 16
blocking         11.1 Hz   10 calls  stall  4.12s (30.6%)  worst  456.3 ms  misses  10/150
pipelined/pf8    14.5 Hz   10 calls  stall  0.40s ( 3.9%)  worst  402.7 ms  misses   1/150
```

Doubling the horizon does cut blocking's stalls (47.6% → 30.6%) by querying
half as often, but it never reaches pipelining's 3.9%, and it pays for the
improvement in **observation staleness** — 16 actions is 1.07 s of acting on a
picture of the world that old. Pipelining reaches a better number while still
re-planning every 533 ms. On this hardware, overlap the inference; do not
lengthen the open loop to hide it.

Worth putting next to the post's headline: its three optimisations moved
MI300X from 564.7 ms to 398.3 ms, a 1.42x on inference latency. Scheduling the
same inference against the chunk it already has took **9.0 Hz to 14.5 Hz, a
1.6x on the rate the arm actually achieves**, without touching a kernel. Both
matter, but on a bandwidth-poor part the scheduling one is available first and
costs a thread.

Caveat, stated plainly: the weights are uninitialised, so the actions are not
meaningful robot commands and the arm does not accomplish the reach. What is
real here is the timing — when chunks arrive, how long the arm waits, and
whether the control period is met. That is the same quantity the post measured.

## Running the released 14B checkpoint

`GEAR-Dreams/DreamZero-DROID` loads and runs. The DiT is 27.3 GB in bf16 on a
32 GB machine that also holds the OS, so it is packed to int4 *as it streams*
rather than after:

```
$ dexter-demo --model 14b --checkpoint ~/nod/checkpoints/DreamZero-DROID --bits 4
loaded in 27.4s
resident weights : 9.06 GB      (bf16 would be 27.26 GB)
peak GPU alloc   : 10.24 GB     (of 34.4 GB)
```

The parameter mapping validates **1317 to 1317 tensors, nothing unfilled and
nothing unused** — the check that makes a silently half-loaded model impossible.
Building the model to match the release rather than the paper turned up six
real corrections, each of which would otherwise have been a quiet wrong answer:

| what the release actually does | what a sketch would assume |
|---|---|
| `patch_embedding` is a Conv3d of stride `(1,2,2)` — a linear over `in_dim * 4 = 144` | a linear over `in_dim` |
| text is projected to `dim` *before* any block, so cross-attn k/v are `dim -> dim` | `text_dim -> dim` |
| i2v cross-attends to CLIP image features too (`k_img`/`v_img`), summed with text | text only |
| action/state encoders are per-embodiment (category-specific) linears | plain linears |
| `WanRMSNorm` reduces over the full 5120 `dim`, across all heads | per-head over `head_dim` |
| `in_dim` 36 = 16 denoised + 20 conditioning channels; `out_dim` is only the 16 | the whole input is noisy |

The fifth of those meant the fused `qk_norm_rope` kernel was computing a
different model, and it was rewritten as a `[heads, head_dim/2]` tile with a
whole-plane reduction.

### Measured, real weights

```
run 0: step 20407 ms = device 20405 (denoise: 5598 4917 4930 4960) + host 2.5
run 1: step 19853 ms = device 19851 (denoise: 4947 4967 4970 4966) + host 2.1

action chunk (1, 24, 32) finite=True
  action[0] (7 joints + gripper): 0.023 -0.023 0.034 -0.036 -0.039 -0.031 0.068 -0.039
  chunk drift |a[23]-a[0]| = 0.049
```

20.1 s per control step, against a 6.6 s compute floor. The gap is the
quantised GEMM being ~1.8x off dense, and at 1785 tokens this model is firmly
compute bound — so int4 here buys the ability to *load* the model, and costs
time. bf16 would be faster per step and does not fit.

The joint deltas are small and smooth, and drift gently across the chunk, which
is what a trained policy should emit. They are still not *meaningful* commands,
for a reason worth stating precisely: the DiT is real, but its three encoders
are not. The VAE (observation to latent), CLIP (frame conditioning) and umt5
(language) live in shards 1-2 of the release and are neither downloaded nor
implemented here, so the conditioning fed to the DiT is a placeholder of the
right shape. Real weights, synthetic observations.

### A bug this surfaced

The first real-weights run returned NaN. Activations stayed bounded through 32
layers and then went non-finite at block 33 — sudden, not a gradual overflow.
It was the online-softmax rescale in the attention kernel: before the first
visible tile there is nothing accumulated and `running_max` is `-inf`, which the
kernel replaced with `0.0` and then used as `exp(0 - safe_max)`. With scores
near -5e4, that is `+inf`, and `0 * inf` in the accumulator is NaN. The
correction has to be *zero* in that case, not an exponential. Randomly
initialised weights never produce scores extreme enough to reach it; the real
checkpoint does by layer 33. There is now a regression test that drives
attention with deliberately extreme scores.

## World-model rollout on real robot data

`dexter-rollout` takes a real DROID camera frame and a real instruction, and
lets the model dream forward: it predicts the video its own actions would
produce, decodes it, and feeds the last frame back as the next observation.

```
$ dexter-rollout --checkpoint ~/nod/checkpoints/DreamZero-DROID \
    --video exterior_image_1_left.mp4 \
    --prompt "pick up the black bowl and place it on the plate" --steps 3

encoding instruction: "pick up the black bowl and place it on the plate"
  text (1, 512, 4096), text tower released
loading DiT as int4 ...
  9.06 GB resident
dreaming 3 blocks forward ...
wrote dream.mp4: 16 frames (23.6 s per block)
action trajectory (72, 32)
  joint deltas |mean| = 0.0433, max = 0.1328
```

The predicted frames are coherent: the stove, pan, grates and counter stay put
and stay recognisable, and the scene drifts gradually as the rollout proceeds.
Divergence from the observation grows smoothly — mean absolute difference 6.2
at the first predicted frame, 25.4 by the sixteenth — which is what a world
model dreaming forward should do, rather than either freezing or exploding.

The full stack runs on one 32 GB APU by never holding two large models at once:
umt5-xxl (11.4 GB) encodes the instruction and is released before the 9 GB int4
DiT is loaded; the VAE and CLIP tower are 1.5 GB together.

### Getting there: four bugs that all ran fine

Every one of these produced finite, plausible-looking output and passed the
test suite. None of them threw.

1. **Flat 1D rotary.** Wan gives video tokens a *3D* rotary -- `head_dim` split
   44/42/42 across time, height and width -- with the action register on its
   own 1D rotary. A flat rotary over sequence position leaves the model unable
   to tell which patch is where.
2. **Split-half rotary pairing.** Wan pairs *adjacent* elements
   (`reshape(..., d/2, 2)`), not `i` with `i + d/2`. The two are not
   interchangeable.
3. **No cache priming.** A causal video model denoises by attending to clean
   history. Upstream first runs the observed frame at timestep 0, with no
   action register, purely to fill the KV cache. Denoising the first block
   against an empty cache asks the model to imagine a scene it was never shown.
   Reading and writing the cache are also separate decisions: every denoising
   step reads, only a priming pass writes, and it writes clean latents.
4. **Two patch conventions, silently mixed.** This was the expensive one. The
   DiT reads its input channel-first -- `patch_embedding` is a Conv3d, so a
   token is `(c, kt, kh, kw)` -- and writes its output channel-*last*, because
   upstream reads the head back with `view(B, f, h, w, pt, ph, pw, c)`. Input
   and output are therefore **not** inverses. Mixing them in
   `x = x + dt * v` type-checks, runs, and scrambles every patch; using the
   wrong inverse when decoding leaves the image intact but adds a regular grid
   at the patch pitch. `patchify`/`unpatchify_input` and
   `patchify_output`/`unpatchify` are now separate pairs, each matching the end
   it serves, with a test asserting they are distinct.

What made these findable was running upstream's own `CausalWanModel` on
identical weights and inputs and diffing the outputs. That immediately split
the problem: `action_noise` matched at **corr +0.9999** while `video_noise` sat
at **+0.07**, which proved the entire trunk -- attention, modulation,
cross-attention, the action path -- was already correct and put the bug in the
video output alone. Both now match at **+0.9999**. Guessing had cost hours
before that; the differential took one run.

## A robot completing a task

The 14B world model predicts; it cannot drive anything here. One control block
is ~24 s against DROID's 66 ms period, and that gap is arithmetic rather than
engineering: 1785 tokens x 13.6B params x 4 denoising steps is ~195 TFLOP per
control step, so even at 100% of this machine's measured 29.5 TFLOP/s the floor
is 6.6 s. Two orders of magnitude from FLOPs alone.

So to watch a policy actually finish something, `dexter-pusht` runs a small
trained policy on a real simulated task — PushT, where a 2-DoF end effector
must shove a T-block onto a target outline, scored by coverage with a 0.95
success threshold.

```
$ dexter-pusht --episodes 10
lerobot/diffusion_pusht: 262.7M params on cuda

  episode 2: best coverage 0.952  SOLVED    127 steps in 33.2s (3.8 Hz)
  episode 5: best coverage 0.954  SOLVED    289 steps in 76.8s (3.8 Hz)
  episode 9: best coverage 0.963  SOLVED    138 steps in 37.3s (3.7 Hz)
  episode 8: best coverage 0.234  unsolved  300 steps in 78.8s (3.8 Hz)

7/10 episodes solved (coverage > 0.95); median best coverage 0.951
```

It shares the shape of DreamZero's problem — observe, denoise an action chunk,
execute part of it, re-plan — at three orders of magnitude less model. It does
not share the backbone, so it runs through LeRobot rather than dexter ops.

### Two things that made it look broken

**The normalisation statistics silently did not load.** `from_pretrained` on
the published checkpoint emits `Unexpected key(s): normalize_inputs.*` as a
warning and continues, because this lerobot version moved normalisation out of
the policy. The result is a policy running on unnormalised inputs, which scores
**0.00 coverage on every episode** — indistinguishable from a policy that
simply cannot do the task. The demo now reads the statistics out of the
checkpoint itself and applies them explicitly, which is also version-proof: the
arithmetic is fixed by the checkpoint rather than by whichever lerobot is
installed.

**Reward is not coverage.** PushT's reward is
`clip(coverage / 0.95, 0, 1)`, so it saturates at 1.00 for anything at or above
threshold. Reporting it as "coverage" showed episodes at 1.00 that the env had
not marked solved. The demo reports `info["coverage"]` and `info["is_success"]`
instead — which lowered the headline number and made it true.

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
    dreamzero/         config, DiT, closed-loop policy
      checkpoint.py    streaming loader: meta build, pack-as-you-go, one shard mapped
  perception/          vendored Wan VAE, CLIP and umt5 + conditioning encoder
  runtime/graph.py     HIP graph capture of the denoise step
  demo/robot.py        closed-loop arm, blocking vs pipelined chunk scheduling
  demo/rollout.py      world-model rollout from a real frame
  demo/pusht.py        a small policy solving a real task, end to end
  bench/               roofline, kernel microbench, end-to-end
test/                  41 tests: kernels vs references, model, scheduler, layouts
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
dexter-demo  --model 5b --ticks 150 --prefetch 4 8   # closed-loop robot demo
pytest test -q
```

## What is not here

* **DreamZero driving a robot.** The rollout is the model imagining, not an arm
  executing, and `dexter-pusht` is a different, much smaller policy. Closing
  that gap needs a ~5B backbone: DreamZero ships the recipe
  (`docs/WAN22_BACKBONE.md`, 83 tokens/step instead of 1785) but no released
  weights for it, so it would have to be trained. The engine work here —
  kernels, int4, chunk scheduling — applies unchanged if such a checkpoint
  existed; the 5B shape already measures 361 ms/step and pipelines to 14.5 Hz.
* **The fused VAE Conv3D** from the post. The VAE is vendored torch, which is
  the right call while it is 2% of the step, and the wrong one once the DiT
  gets faster.
* **A competitive int4 GEMM.** Diagnosed, not fixed. See above.
* **Multi-GPU.** Upstream shards across 2+ GPUs; this is a single-APU engine.
