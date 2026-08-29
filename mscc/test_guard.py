#!/usr/bin/env python3
"""Regression tests for the MSCC frame format and guard.

Run:  python3 -m mscc.test_guard        (from the repository root)
      python3 -m mscc.test_guard --real (also fingerprints two real models on disk;
                                         slower, hashes multi-GB weight files once)

The load-bearing test is ACCEPTANCE: a frame written against one model must be
REJECTED by another, not silently degraded. That is the whole point of the guard.
"""
import argparse
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mscc import (FrameError, FrameHeader, FrameRejected, check, codebook_fingerprint,
                  model_fingerprint, read_frame, read_header, require, write_frame)
from mscc.guard import MIN_DIMS, REF_BITS_PER_TOKEN

PASS, FAIL = [], []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def good_header(**kw):
    d = dict(model_sha="a" * 64, codebook_sha="b" * 64, model_id="qwen3-1.7b",
             layer=23, n_layers=28, n_dims=1024, hidden_dim=2048,
             bits_per_token=2048, codec="cpca1024b2048", n_tokens=512,
             created_utc="2026-08-27T00:00:00Z")
    d.update(kw)
    return FrameHeader(**d)


def t_roundtrip():
    print("\n-- container round-trip")
    hdr = good_header()
    payload = {"codes": np.arange(24, dtype=np.int16).reshape(2, 12),
               "scale": np.array([1.5, 2.5], dtype=np.float32)}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "f.mscc")
        n = write_frame(p, hdr, payload)
        ok("write returns byte count", n == os.path.getsize(p), f"{n} bytes")
        h2 = read_header(p)
        ok("header survives round-trip", h2 == hdr)
        f2 = read_frame(p)
        ok("payload survives round-trip",
           all(np.array_equal(payload[k], f2.payload[k]) for k in payload))
        # header-only read must not need the payload
        ok("read_header ignores payload", read_header(p).n_dims == 1024)


def t_malformed():
    print("\n-- malformed input is an error, not a guess")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "junk.mscc")
        open(p, "wb").write(b"this is not a frame at all, not even close")
        try:
            read_header(p); ok("non-frame rejected", False, "no exception raised")
        except FrameError:
            ok("non-frame rejected", True)

        p2 = os.path.join(d, "trunc.mscc")
        hdr = good_header()
        write_frame(p2, hdr, {"codes": np.zeros(4, np.int8)})
        raw = open(p2, "rb").read()
        open(p2, "wb").write(raw[:14])          # cut inside the header
        try:
            read_header(p2); ok("truncated header rejected", False, "no exception")
        except FrameError:
            ok("truncated header rejected", True)


def t_guard_pass():
    print("\n-- a well-formed frame passes cleanly")
    r = check(good_header(), model_sha="a" * 64, codebook_sha="b" * 64)
    ok("valid frame passes", r.ok, r.reason())
    ok("valid frame has no warnings", not r.warnings, str([str(w) for w in r.warnings]))
    r4b = check(good_header(layer=30, n_layers=36), "a" * 64, "b" * 64)
    ok("4B tap L30/36 also passes clean", r4b.ok and not r4b.warnings, r4b.reason())


def t_hard_failures():
    print("\n-- hard failures (frame is unusable -> fall back to a full read)")
    cases = [
        ("model mismatch", good_header(), dict(model_sha="c" * 64, codebook_sha="b" * 64),
         "MODEL_MISMATCH"),
        ("codebook mismatch", good_header(), dict(model_sha="a" * 64, codebook_sha="z" * 64),
         "CODEBOOK_MISMATCH"),
        ("512 dims", good_header(n_dims=512), dict(model_sha="a" * 64, codebook_sha="b" * 64),
         "DIMS_TOO_NARROW"),
        ("640 dims (inside the collapse region)", good_header(n_dims=640),
         dict(model_sha="a" * 64, codebook_sha="b" * 64), "DIMS_TOO_NARROW"),
        ("unknown format version", good_header(format_version=99),
         dict(model_sha="a" * 64, codebook_sha="b" * 64), "FORMAT_VERSION"),
        ("layer past end of stack", good_header(layer=99),
         dict(model_sha="a" * 64, codebook_sha="b" * 64), "LAYER_OUT_OF_RANGE"),
        ("no codec recorded", good_header(codec=""),
         dict(model_sha="a" * 64, codebook_sha="b" * 64), "CODEC_MISSING"),
    ]
    for name, hdr, kw, code in cases:
        r = check(hdr, **kw)
        ok(f"{name} -> hard reject", (not r.ok) and any(f.code == code for f in r.hard),
           r.reason())
        ok(f"{name} -> require() raises",
           _raises(lambda: require(hdr, **kw), FrameRejected))
        got = [f for f in r.hard if f.code == code]
        ok(f"{name} -> reject states why", bool(got and got[0].provenance)
           or code in ("LAYER_OUT_OF_RANGE", "CODEC_MISSING", "FORMAT_VERSION"))


def t_warnings():
    print("\n-- warnings (outside the measured envelope, but usable)")
    r = check(good_header(layer=14), "a" * 64, "b" * 64)
    ok("deep tap warns but passes", r.ok and any(f.code == "TAP_TOO_DEEP" for f in r.warnings),
       r.reason())
    r = check(good_header(bits_per_token=6144), "a" * 64, "b" * 64)
    ok("off-reference bits warns", r.ok and any(f.code == "BITS_OFF_REFERENCE" for f in r.warnings))
    # Phase C (gate9 + gate9b) measured ternary OUTSIDE its pre-registered +/-0.01
    # threshold at every context length, so it is a hard rejection, not a warning.
    r = check(good_header(codec="tcpca1024b2048"), "a" * 64, "b" * 64)
    ok("ternary codec rejected, not warned", (not r.ok)
       and any(f.code == "CODEC_RETRACTED" for f in r.hard), r.reason())
    r = check(good_header(codec="madeup9000"), "a" * 64, "b" * 64)
    ok("unmeasured codec warns", r.ok and any(f.code == "CODEC_UNKNOWN" for f in r.warnings))
    # Gate 9 measured 908 dims BEATING 1024 on book prose, so the old 1024 hard floor
    # would have rejected a configuration that is better than the default.
    r = check(good_header(n_dims=908), "a" * 64, "b" * 64)
    ok("908 dims passes with a warning, not a rejection",
       r.ok and any(f.code == "DIMS_BELOW_DEFAULT" for f in r.warnings), r.reason())
    r = check(good_header(n_dims=768), "a" * 64, "b" * 64)
    ok("768 dims (knee edge) passes with a warning", r.ok, r.reason())


def t_codebook_fingerprint():
    print("\n-- codebook fingerprint binds decoder to encoder")
    V = np.random.default_rng(0).normal(size=(64, 8)).astype(np.float32)
    meta = {"dims": 1024, "bits": 2048, "codec": "cpca"}
    s1 = codebook_fingerprint({"V": V}, meta)
    s2 = codebook_fingerprint({"V": V.copy()}, meta)
    ok("identical codebook -> identical sha", s1 == s2)
    V2 = V.copy(); V2[0, 0] += 1e-3
    ok("one perturbed weight -> different sha", codebook_fingerprint({"V": V2}, meta) != s1)
    ok("different meta -> different sha",
       codebook_fingerprint({"V": V}, {**meta, "dims": 512}) != s1)


def t_acceptance_real(models):
    print("\n-- ACCEPTANCE: real models, frame from one must be rejected by the other")
    shas = {}
    for m in models:
        if not os.path.isdir(m):
            ok(f"model present: {os.path.basename(m)}", False, "not on disk"); return
        shas[m] = model_fingerprint(m)
        print(f"     {os.path.basename(m)}  sha={shas[m][:16]}...")
    a, b = models[0], models[1]
    ok("two different models -> two different shas", shas[a] != shas[b])
    ok("fingerprint is stable across calls", model_fingerprint(a) == shas[a])
    ok("fingerprint is content-derived, not cache-derived",
       model_fingerprint(a, use_cache=False) == shas[a])

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sender.mscc")
        hdr = good_header(model_sha=shas[a], model_id=os.path.basename(a))
        write_frame(p, hdr, {"codes": np.zeros((512, 1024), np.int8)})
        h = read_header(p)
        ok("same model accepts its own frame", check(h, shas[a], h.codebook_sha).ok)
        r = check(h, shas[b], h.codebook_sha)
        ok("OTHER model REJECTS the frame", not r.ok, r.reason())
        ok("rejection names the mismatch", any(f.code == "MODEL_MISMATCH" for f in r.hard))
        ok("require() raises for the other model",
           _raises(lambda: require(h, shas[b], h.codebook_sha), FrameRejected))


def _raises(fn, exc):
    try:
        fn(); return False
    except exc:
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="also fingerprint real models on disk (hashes GBs, cached after)")
    ap.add_argument("--models", nargs=2,
                    default=[], help="model directories to fingerprint (acceptance only)")
    a = ap.parse_args()

    t_roundtrip(); t_malformed(); t_guard_pass(); t_hard_failures()
    t_warnings(); t_codebook_fingerprint()
    if a.real:
        t_acceptance_real(a.models)
    else:
        print("\n-- ACCEPTANCE skipped (pass --real to fingerprint models on disk)")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
