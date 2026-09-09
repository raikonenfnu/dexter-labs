"""Graph capture for the denoising step -- the host-side half of the problem.

The vLLM-Omni write-up found that a meaningful slice of DreamZero's step time
on MI300X was *host* work sitting in front of the GPU rather than beside it,
and restructuring the scheduler to overlap them was worth more than any single
kernel change. The same effect is larger here, for a structural reason:

A DreamZero step is 40 layers x ~10 dispatches x 4 denoising steps, so roughly
1600 kernel launches for ~50 ms of device work. At a few microseconds of
Python and HIP launch overhead each, the host is not comfortably ahead of the
device -- it is the thing the device is waiting for. Strix Halo makes this
worse than a discrete part would: the CPU issuing those launches is sharing
both the memory controller and the power budget with the GPU executing them.

Capturing the step into a HIP graph replays the entire dispatch sequence from
one call, which removes per-launch host cost instead of trying to overlap it.
What it costs is rigidity: shapes, addresses and control flow are frozen at
capture time. That is an acceptable trade here precisely because a control
loop is shape-stable by construction -- the same block size, the same action
chunk, every step, for the whole episode.
"""

from __future__ import annotations

import torch


class GraphRunner:
    """A captured forward, replayed against fixed input buffers.

    Inputs are copied into the buffers the graph was captured with, so callers
    keep passing ordinary tensors and never see the static-address requirement.
    """

    def __init__(self, graph: torch.cuda.CUDAGraph, inputs: tuple[torch.Tensor, ...],
                 outputs: tuple[torch.Tensor, ...]) -> None:
        self.graph = graph
        self.inputs = inputs
        self.outputs = outputs

    def __call__(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if len(args) != len(self.inputs):
            raise ValueError(f"graph takes {len(self.inputs)} inputs, got {len(args)}")
        for buffer, arg in zip(self.inputs, args):
            buffer.copy_(arg)
        self.graph.replay()
        return self.outputs


def capture_denoise_step(model, ctx, sample_inputs: tuple[torch.Tensor, ...],
                         warmup: int = 3) -> GraphRunner:
    """Capture one denoising step (the KV-cache-free variant) into a graph.

    Only the steps that do *not* write the KV cache are captured. The final
    step mutates cache state whose length changes between control steps, and
    freezing that into a graph would silently pin the history length.
    """
    static = tuple(t.clone() for t in sample_inputs)

    # Warm up on a side stream: Triton autotuning, module loading and the
    # allocator all have to settle before capture, or they get recorded.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            model(*static, ctx, None)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = model(*static, ctx, None)

    return GraphRunner(graph, static, tuple(outputs))
