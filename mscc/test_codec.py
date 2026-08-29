#!/usr/bin/env python3
"""Tests for the product codec.

The load-bearing test is EQUIVALENCE: decode(encode(X)) must be numerically identical
to lib_inject.CPCACodec(X), the object every quality number in this project was
measured with. If it is not, the product is a lookalike and none of the measurements
transfer to it.

Run:  python3 -m mscc.test_codec       (from the repository root)
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "lib"))

from mscc import codec as C
from mscc.format import FrameHeader, read_frame, write_frame
from mscc.guard import check

PASS, FAIL = [], []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def synth_states(n=6000, hidden=256, seed=0):
    """Synthetic states with a realistic residual-stream shape: a heavy-tailed
    eigenspectrum plus a few massive-activation channels."""
    g = torch.Generator().manual_seed(seed)
    k = 64
    A = torch.randn(hidden, k, generator=g)
    z = torch.randn(n, k, generator=g) * torch.linspace(6.0, 0.2, k)
    X = z @ A.T + 0.35 * torch.randn(n, hidden, generator=g)
    X[:, 3] *= 40.0          # massive activation channel
    X[:, 17] *= 25.0
    return X + 1.7


def t_equivalence():
    print("\n-- EQUIVALENCE: product codec == the measured CPCACodec")
    import lib_inject as li
    X = synth_states()
    for dims, bits in [(64, 256), (128, 512), (200, 1024)]:
        torch.manual_seed(0)
        mu = X.mean(0, keepdim=True)
        ref_codec = li.CPCACodec(X - mu, dims, bits)
        torch.manual_seed(0)
        ref = ref_codec(X - mu) + mu

        # clip="quantile" IS the historical codec. Every pre-Gate-12 number was
        # measured with it, so it stays reproducible even though the product
        # default moved on.
        cb = C.fit(X, dims, bits, seed=0, clip="quantile")
        got = cb.decode(cb.encode(X), shape=X.shape)

        # the reference subsamples with the global RNG; equality of the fitted
        # objects is what matters, so compare the reconstruction directly
        md = (got - ref).abs().max().item()
        rel = ((got - ref) ** 2).sum().item() / (ref ** 2).sum().item()
        ok(f"dims={dims} bits={bits}: reconstruction matches CPCACodec",
           md < 1e-3, f"max|diff|={md:.2e} rel={rel:.2e}")
        ok(f"dims={dims} bits={bits}: funded dims agree",
           cb.n_dims == int(ref_codec.V.shape[1]),
           f"{cb.n_dims} vs {int(ref_codec.V.shape[1])}")
        ok(f"dims={dims} bits={bits}: bit total agrees",
           cb.bits_per_token == int(ref_codec.bits),
           f"{cb.bits_per_token} vs {int(ref_codec.bits)}")


def t_mse_clip():
    """The default clip strategy must never lose to the one it replaced.

    The quantile range is offered to mse_clip as a candidate, so "never worse" is
    structural. This test exists because the failure it guards against was silent:
    the old range produced a reconstruction with relative error > 1 -- worse than
    emitting the mean -- and nothing in the pipeline complained.
    """
    print("\n-- MSE clipping: strictly better at the bit widths that matter")
    X = synth_states()
    for dims, bits in [(200, 200), (200, 400), (200, 1024), (128, 512)]:
        q = C.fit(X, dims, bits, seed=0, clip="quantile")
        m = C.fit(X, dims, bits, seed=0, clip="mse")
        eq = ((q.decode(q.encode(X), shape=X.shape) - X) ** 2).mean().item()
        em = ((m.decode(m.encode(X), shape=X.shape) - X) ** 2).mean().item()
        var = X.var().item()
        ok(f"dims={dims} bits={bits}: mse clip <= quantile clip",
           em <= eq * 1.001, f"mse {em:.4g} vs quantile {eq:.4g}")
        ok(f"dims={dims} bits={bits}: beats emitting the mean",
           em < var, f"mse {em:.4g} vs signal variance {var:.4g}")
        ok(f"dims={dims} bits={bits}: same rate and shape as quantile clip",
           m.bits_per_token == q.bits_per_token and m.n_dims == q.n_dims,
           f"{m.bits_per_token}b/{m.n_dims}d vs {q.bits_per_token}b/{q.n_dims}d")


def t_bitpack():
    print("\n-- bit packing: the frame weighs what we say it weighs")
    rng = np.random.default_rng(0)
    widths = rng.integers(1, 9, size=300)
    codes = np.stack([rng.integers(0, 2 ** int(w), size=500) for w in widths], axis=1)
    packed = C.pack_codes(codes, widths)
    back = C.unpack_codes(packed, widths, 500)
    ok("pack/unpack is lossless", np.array_equal(codes.astype(np.uint32), back))
    expect = int(np.ceil(500 * widths.sum() / 8))
    ok("packed size == n_tokens * sum(widths) / 8",
       packed.size == expect, f"{packed.size} vs {expect}")
    naive = codes.astype(np.uint16).nbytes
    ok("packing beats a naive uint16 dump", packed.size < naive,
       f"{packed.size} vs {naive} bytes ({naive / packed.size:.1f}x)")

    # round-trip through a real codebook
    X = synth_states(n=2000, hidden=256)
    cb = C.fit(X, 128, 512)
    q = cb.encode(X)
    ok("codes fit inside their declared widths",
       bool((q < (2 ** cb.b.astype(np.int64))).all()),
       f"max code {int(q.max())}, max width {int(cb.b.max())}")
    p = C.pack_codes(q, cb.b)
    q2 = C.unpack_codes(p, cb.b, q.shape[0])
    ok("real codes survive packing", np.array_equal(q, q2))
    got_bpt = p.size * 8 / q.shape[0]
    ok("on-disk bits/token matches the declared figure",
       abs(got_bpt - cb.bits_per_token) < 1.0,
       f"{got_bpt:.1f} vs declared {cb.bits_per_token}")


def t_codebook_disk():
    print("\n-- codebook artifact carries its own provenance")
    X = synth_states(n=2000, hidden=256)
    cb = C.fit(X, 128, 512, meta={"model_sha": "d" * 64, "corpus_sha": "e" * 64,
                                  "layer": 23, "n_layers": 28})
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "cb.npz")
        cb.save(p)
        cb2 = C.Codebook.load(p)
        ok("codebook survives round-trip", cb.sha() == cb2.sha())
        ok("codebook records the model it was fitted against",
           cb2.meta["model_sha"] == "d" * 64)
        ok("codebook records the corpus digest", cb2.meta["corpus_sha"] == "e" * 64)
        ok("codebook records how many states it saw", cb2.meta["n_states"] == 2000)
        r1 = cb.decode(cb.encode(X), shape=X.shape)
        r2 = cb2.decode(cb2.encode(X), shape=X.shape)
        ok("reloaded codebook decodes identically",
           torch.equal(r1, r2))


def t_end_to_end_frame():
    print("\n-- end to end: fit -> encode -> frame -> guard -> decode")
    X = synth_states(n=4000, hidden=256)
    cb = C.fit(X, 200, 1024, meta={"model_sha": "f" * 64, "layer": 23, "n_layers": 28})
    doc = X[:512]
    q = cb.encode(doc)
    packed = C.pack_codes(q, cb.b)
    hdr = FrameHeader(model_sha="f" * 64, codebook_sha=cb.sha(), model_id="synthetic",
                      layer=23, n_layers=28, n_dims=cb.n_dims,
                      hidden_dim=cb.hidden_dim, bits_per_token=cb.bits_per_token,
                      codec=cb.meta["codec_name"], n_tokens=512,
                      created_utc="2026-08-27T00:00:00Z")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "doc.mscc")
        write_frame(p, hdr, {"codes": packed, "widths": cb.b})
        f = read_frame(p)
        r = check(f.header, model_sha="f" * 64, codebook_sha=cb.sha(),
                  min_dims=64)   # synthetic hidden=256, so relax the width floor
        ok("frame passes the guard", r.ok, r.reason())
        rej = check(f.header, model_sha="0" * 64, codebook_sha=cb.sha(), min_dims=64)
        ok("wrong model rejects the frame", not rej.ok, rej.reason())
        q2 = C.unpack_codes(f.payload["codes"], f.payload["widths"], 512)
        rec = cb.decode(q2, shape=doc.shape)
        ok("decoded frame == direct decode",
           torch.equal(rec, cb.decode(q, shape=doc.shape)))
        ok("frame beats a float dump on size",
           os.path.getsize(p) < doc.numel() * 4,
           f"{os.path.getsize(p)} vs {doc.numel() * 4} bytes float32")


def main():
    t_equivalence(); t_mse_clip(); t_bitpack(); t_codebook_disk(); t_end_to_end_frame()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print("  FAILED:", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
