"""Every gfx1151 kernel against its torch reference."""
import pytest, torch
import dexter.ops as ops
from dexter.platform import current_platform
from dexter.registry import select
from dexter.quant import quantize

DEV = "cuda"
DT = torch.bfloat16


def close(a, b, tol=2e-2):
    a, b = a.float(), b.float()
    err = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)
    assert err < tol, f"max rel err {err:.4g}"


def ref(family, mode):
    """The REFERENCE implementation, bypassing selection."""
    from dexter.registry import _REGISTRY
    return next(s.impl for s in _REGISTRY[f"{family}/{mode}"] if s.priority == 0)


def test_platform():
    p = current_platform()
    print("\n" + p.describe())
    assert p.is_amd and p.arch == "gfx1151" and p.family == "RDNA3.5"
    assert p.supports("mma:wmma", "matrix:int4", "memory:unified")
    assert not p.supports("matrix:fp8")   # RDNA3.5 has no FP8 matrix path


@pytest.mark.parametrize("shape", [(1, 125, 3072), (2, 64, 5120)])
def test_adaln(shape):
    x = torch.randn(shape, device=DEV, dtype=DT)
    s = torch.randn(shape[0], 1, shape[2], device=DEV, dtype=DT)
    b = torch.randn(shape[0], 1, shape[2], device=DEV, dtype=DT)
    close(ops.adaln_modulate(x, s, b), ref("norm", "adaln")(x, s, b, 1e-6))


def test_gated_residual():
    x = torch.randn(1, 125, 3072, device=DEV, dtype=DT)
    y = torch.randn_like(x)
    g = torch.randn(1, 1, 3072, device=DEV, dtype=DT)
    close(ops.gated_residual(x, y, g), ref("norm", "gated_residual")(x, y, g))


def test_rms_norm():
    x = torch.randn(1, 125, 3072, device=DEV, dtype=DT)
    w = torch.randn(3072, device=DEV, dtype=DT)
    close(ops.rms_norm(x, w), ref("norm", "rms_norm")(x, w, 1e-6))


def test_qk_norm_rope():
    b, l, h, d = 1, 125, 24, 128
    q = torch.randn(b, l, h, d, device=DEV, dtype=DT)
    k = torch.randn_like(q)
    # Wan's QK norm spans the whole projection, so the weight is h*d long.
    qw = torch.randn(h * d, device=DEV, dtype=DT)
    kw = torch.randn(h * d, device=DEV, dtype=DT)
    pos = torch.arange(l, device=DEV, dtype=torch.float32)[:, None]
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, device=DEV, dtype=torch.float32) / d))[None, :]
    cos, sin = torch.cos(pos * inv), torch.sin(pos * inv)
    gq, gk = ops.qk_norm_rope(q, k, qw, kw, cos, sin)
    rq, rk = ref("rope", "qk_norm_rope")(q, k, qw, kw, cos, sin, 1e-6)
    close(gq, rq); close(gk, rk)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("symmetric", [False, True])
def test_quantized_gemm(bits, symmetric):
    m, k, n = 125, 3072, 8192
    x = torch.randn(m, k, device=DEV, dtype=DT) * 0.1
    w = torch.randn(n, k, device=DEV, dtype=DT) * 0.05
    qw = quantize(w, bits=bits, symmetric=symmetric).to(DEV)
    mode = "w4a16" if bits == 4 else "w8a16"
    got = select("gemm", mode)(x, qw.qweight, qw.scales, qw.zeros, None, qw.group_size)
    exp = ref("gemm", mode)(x, qw.qweight, qw.scales, qw.zeros, None, qw.group_size)
    close(got, exp, tol=3e-2)
    # And the quantiser itself must track the original weight.
    close(got, x @ w.t().float().to(DT), tol=0.25 if bits == 4 else 0.05)


def test_ffn_gelu():
    m, c, f = 125, 3072, 8192
    x = torch.randn(m, c, device=DEV, dtype=DT) * 0.1
    up = torch.randn(f, c, device=DEV, dtype=DT) * 0.03
    down = torch.randn(c, f, device=DEV, dtype=DT) * 0.03
    ub = torch.randn(f, device=DEV, dtype=DT) * 0.1
    db = torch.randn(c, device=DEV, dtype=DT) * 0.1
    close(ops.ffn_gelu(x, up, ub, down, db), ref("gemm", "ffn_gelu")(x, up, ub, down, db), tol=4e-2)


@pytest.mark.parametrize("cache_len", [0, 256])
@pytest.mark.parametrize("blocks", [None, [0, 50, 100]])
def test_attention(cache_len, blocks):
    b, l, h, d = 1, 125, 24, 128
    q = torch.randn(b, l, h, d, device=DEV, dtype=DT) * 0.3
    k = torch.randn_like(q); v = torch.randn_like(q)
    cache = torch.randn(2, b, 512, h, d, device=DEV, dtype=DT) * 0.3 if cache_len else None
    bb = torch.tensor(blocks, device=DEV) if blocks else None
    got = ops.blockwise_causal_attention(q, k, v, kv_cache=cache, cache_len=cache_len, block_boundaries=bb)
    exp = ref("attention", "blockwise_causal")(q, k, v, cache, cache_len, bb, None)
    close(got, exp, tol=3e-2)


@pytest.mark.parametrize("bits", [4, 8])
def test_linear_dispatches_on_bits_not_dtype(bits):
    """int4 and int8 both store uint8; only the declared width tells them apart."""
    from dexter.models.layers import Linear
    m = Linear(256, 128, bias=False, dtype=DT).to(DEV)
    dense = m(torch.randn(8, 256, device=DEV, dtype=DT))
    m.quantize_(bits=bits, group_size=128)
    assert m.packed.bits == bits
    out = m(torch.randn(8, 256, device=DEV, dtype=DT))
    assert out.shape == dense.shape and torch.isfinite(out).all()


@pytest.mark.parametrize("scale", [1.0, 60.0])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_attention_survives_extreme_scores(scale, sign):
    """Online softmax must not produce NaN when scores are large or very negative.

    A 14B DiT reaches |q.k| in the tens of thousands by its deeper layers, and
    the first tile's rescale is the place that overflows if the -inf running
    max is handled as a number rather than as "nothing accumulated yet".
    """
    b, l, h, d = 1, 96, 4, 128
    q = torch.full((b, l, h, d), sign * scale, device=DEV, dtype=DT)
    k = torch.full((b, l, h, d), scale, device=DEV, dtype=DT)
    v = torch.randn(b, l, h, d, device=DEV, dtype=DT)
    bb = torch.tensor([0, 48], device=DEV)
    got = ops.blockwise_causal_attention(q, k, v, block_boundaries=bb)
    assert torch.isfinite(got.float()).all(), "attention produced non-finite output"
    exp = ref("attention", "blockwise_causal")(q, k, v, None, 0, bb, None)
    close(got, exp, tol=3e-2)
