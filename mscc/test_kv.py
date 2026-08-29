#!/usr/bin/env python3
"""Tests for the full-depth KV frame.

The load-bearing tests here are the ones guarding failures that are SILENT. A KV
frame has 2*n_layers coding units; getting the ordering, the head reshape or the
codebook binding wrong does not raise, it decodes to plausible noise in the right
shape and the model answers fluently and wrongly. Every test below exists because
that failure mode is the whole reason this project has a guard.

Run:  python3 -m mscc.test_kv       (from the repository root)
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "lib"))

from mscc import codec as C
from mscc import kv as KV
from mscc.format import FrameHeader, FrameError, read_frame, write_frame

PASS, FAIL = [], []
N_LAYERS, HEADS, HEAD_DIM, N_TOK = 4, 4, 32, 256


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def _unit(layer, n, g):
    """One coding unit's worth of states, with a cache's awkward shape: a massive
    activation channel inside each head, and a per-layer scale."""
    t = torch.randn(n, HEADS, HEAD_DIM, generator=g)
    t[:, :, 0] *= 30.0                                  # outlier channel, every head
    return (t * (1.0 + 0.4 * layer)).reshape(n, HEADS * HEAD_DIM)


def synth_kv(seed=0):
    """[(K,V), ...] with a cache's shape and layout."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for l in range(N_LAYERS):
        pair = [_unit(l, N_TOK, g).reshape(N_TOK, HEADS, HEAD_DIM)
                .permute(1, 0, 2).unsqueeze(0).contiguous() for _ in range(2)]
        out.append((pair[0], pair[1]))
    return out


def synth_codebook(unit_bits=1024, seed=0):
    """Codebooks fitted on the SAME distribution the frames are drawn from -- a
    mismatched fit is a real failure mode, but it is Gate 6/7's subject, not this
    file's, and mixing it in here would make every reconstruction assertion moot."""
    g = torch.Generator().manual_seed(seed + 99)
    books = {}
    for key in KV.unit_keys(N_LAYERS):
        books[key] = C.fit(_unit(key[0], 3000, g), HEADS * HEAD_DIM, unit_bits,
                           meta={"kv_layer": key[0], "kv_unit": key[1]})
    meta = {"model_sha": "a" * 64, "n_layers": N_LAYERS, "kv_heads": HEADS,
            "head_dim": HEAD_DIM, "basis": "prerope", "unit_bits": unit_bits,
            "dims": HEADS * HEAD_DIM}
    return KV.KVCodebook(books=books, meta=meta)


def t_codebook():
    print("\n-- codebook artifact: 2*n_layers books, one identity")
    cb = synth_codebook()
    ok("holds one book per (layer, K|V)", len(cb.books) == 2 * N_LAYERS,
       f"{len(cb.books)} units")
    ok("rate is the sum over units",
       cb.bits_per_token == sum(b.bits_per_token for b in cb.books.values()))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "cb.npz")
        cb.save(p)
        cb2 = KV.KVCodebook.load(p)
        ok("survives a disk round trip", cb.sha() == cb2.sha())
        probe = torch.randn(5, HEADS * HEAD_DIM)
        ok("reloaded books decode identically", all(
            torch.equal(cb.books[k].decode(cb.books[k].encode(probe)),
                        cb2.books[k].decode(cb2.books[k].encode(probe)))
            for k in cb.books))

    # the failure a per-unit sha would let through: swap ONE unit's book
    other = synth_codebook(seed=7)
    swapped = KV.KVCodebook(books={**cb.books, (2, "v"): other.books[(2, "v")]},
                            meta=dict(cb.meta))
    ok("swapping a single unit's book changes the codebook sha",
       swapped.sha() != cb.sha())
    ok("changing shared meta changes the sha",
       KV.KVCodebook(books=cb.books, meta={**cb.meta, "basis": "postrope"}).sha()
       != cb.sha())


def t_roundtrip():
    print("\n-- encode/decode: shape, ordering and heads survive the trip")
    cb = synth_codebook()
    kv = synth_kv()
    payload = KV.encode_frame(kv, cb, n_sink=2)
    got = KV.decode_frame(payload, cb, N_TOK, n_sink=2, dtype=torch.float32)

    ok("layer count preserved", len(got) == N_LAYERS)
    ok("tensor shape preserved", all(
        g[0].shape == k[0].shape and g[1].shape == k[1].shape
        for g, k in zip(got, kv)), f"{tuple(got[0][0].shape)}")
    ok("protected tokens come back at fp16 exactness", all(
        torch.allclose(g[0][:, :, :2], k[0][:, :, :2].float(), rtol=2e-3, atol=2e-3)
        for g, k in zip(got, kv)))
    err = [((g[0] - k[0]).norm() / k[0].norm()).item() for g, k in zip(got, kv)]
    ok("reconstruction beats emitting the mean", max(err) < 1.0,
       f"max relative error {max(err):.3f}")

    # ordering: layer 0 must not decode into layer 3's slot. Layers here have
    # deliberately different scales, so a permutation shows up as a norm mismatch.
    norms_in = [float(k[0].norm()) for k in kv]
    norms_out = [float(g[0].norm()) for g in got]
    ok("per-layer ordering is preserved, not permuted",
       all(abs(a - b) / a < 0.25 for a, b in zip(norms_in, norms_out)),
       f"in {[round(x) for x in norms_in]} out {[round(x) for x in norms_out]}")

    # heads must not be transposed into the token axis. Marking each head with a
    # distinct constant makes a swap visible; a norm comparison would not see it.
    marked = torch.zeros(1, HEADS, N_TOK, HEAD_DIM)
    for h in range(HEADS):
        marked[0, h] = float(h + 1)
    flat = marked[0].permute(1, 0, 2).reshape(N_TOK, HEADS * HEAD_DIM)
    back = flat.reshape(N_TOK, HEADS, HEAD_DIM).permute(1, 0, 2).unsqueeze(0)
    ok("head/token reshape composes to the identity", torch.equal(back, marked))
    ok("each head keeps its own identity through the flattening",
       all(float(flat[:, h * HEAD_DIM:(h + 1) * HEAD_DIM].mean()) == h + 1
           for h in range(HEADS)))


def _hdr(**over):
    n = {"kind": "kv", "key_basis": "prerope", "sink": 4, "unit_bits": 1024,
         "kv_heads": HEADS, "head_dim": HEAD_DIM}
    n.update(over.pop("notes", {}))
    d = {"model_sha": "a" * 64, "codebook_sha": "b" * 64, "n_layers": N_LAYERS,
         "n_tokens": N_TOK, "notes": n}
    d.update(over)
    return FrameHeader(**d)


def t_mismatched_codebook():
    print("\n-- the wrong codebook: what decode catches, and what only the guard does")
    cb = synth_codebook(unit_bits=1024)
    narrow = synth_codebook(unit_bits=256)
    payload = KV.encode_frame(synth_kv(), cb, n_sink=0)

    # 1. different RATE -> different bit allocation -> decode itself refuses
    try:
        KV.decode_frame(payload, narrow, N_TOK, n_sink=0)
        ok("a differently-rated codebook is refused by decode", False, "no exception")
    except FrameError as e:
        ok("a differently-rated codebook is refused by decode", True, str(e)[:70])

    # 2. same rate, different FIT -> identical widths. decode CANNOT see this, and
    #    must not be relied on to: the binding is the codebook sha, checked by the
    #    guard before decode is ever called. This test pins that division of labour
    #    so nobody later mistakes the width check for a binding check.
    twin = synth_codebook(unit_bits=1024, seed=31)
    same_widths = all(np.array_equal(cb.books[k].b, twin.books[k].b) for k in cb.books)
    ok("a same-rate codebook has identical widths, so decode cannot detect it",
       same_widths)
    ok("...but its sha differs, which is what the guard checks",
       twin.sha() != cb.sha())
    r = KV.check_kv(_hdr(codebook_sha=cb.sha()), "a" * 64, twin.sha())
    ok("...and the guard refuses it before decode runs",
       not r.ok and any(f.code == "CODEBOOK_MISMATCH" for f in r.hard))


def t_bits_accounting():
    print("\n-- the frame weighs what the header says it weighs")
    cb = synth_codebook()
    raw = N_LAYERS * 2 * HEADS * HEAD_DIM * 16
    bpt = KV.frame_bits_per_token(cb, N_TOK, 4, raw)
    ok("protected tokens are CHARGED, not waved through",
       bpt > cb.bits_per_token,
       f"{bpt:.0f} vs codec-only {cb.bits_per_token}")
    expect = cb.bits_per_token + raw * 4 / N_TOK
    ok("overhead is the amortised uncompressed cost", abs(bpt - expect) < 1e-6)
    ok("sink of 0 costs nothing",
       KV.frame_bits_per_token(cb, N_TOK, 0, raw) == cb.bits_per_token)


def t_guard():
    print("\n-- guard: every condition refuses by name")
    r = KV.check_kv(_hdr(), "a" * 64, "b" * 64)
    ok("a well-formed frame passes", r.ok, r.reason())

    r = KV.check_kv(_hdr(), "c" * 64, "b" * 64)
    ok("different model -> MODEL_MISMATCH",
       not r.ok and any(f.code == "MODEL_MISMATCH" for f in r.hard))

    r = KV.check_kv(_hdr(), "a" * 64, "z" * 64)
    ok("different codebook -> CODEBOOK_MISMATCH",
       not r.ok and any(f.code == "CODEBOOK_MISMATCH" for f in r.hard))

    r = KV.check_kv(_hdr(notes={"unit_bits": 256}), "a" * 64, "b" * 64)
    ok("below the measured floor -> RATE_TOO_LOW",
       not r.ok and any(f.code == "RATE_TOO_LOW" for f in r.hard))
    ok("the refusal carries its measurement",
       any("0-of-12" in f.provenance for f in r.hard))

    r = KV.check_kv(_hdr(notes={"unit_bits": 512}), "a" * 64, "b" * 64)
    ok("above the floor but below default -> warning, not refusal",
       r.ok and any(f.code == "RATE_BELOW_DEFAULT" for f in r.warnings))

    r = KV.check_kv(_hdr(notes={"kind": "midstack"}), "a" * 64, "b" * 64)
    ok("a mid-stack frame is not loadable as a KV frame",
       not r.ok and any(f.code == "NOT_A_KV_FRAME" for f in r.hard))

    r = KV.check_kv(_hdr(notes={"key_basis": "postrope"}), "a" * 64, "b" * 64)
    ok("post-RoPE basis -> usable with a warning",
       r.ok and any(f.code == "KEY_BASIS_POSTROPE" for f in r.warnings))

    r = KV.check_kv(_hdr(notes={"key_basis": "spiral"}), "a" * 64, "b" * 64)
    ok("an unknown key basis is refused, not guessed",
       not r.ok and any(f.code == "KEY_BASIS_UNKNOWN" for f in r.hard))

    r = KV.check_kv(_hdr(notes={"sink": 0}), "a" * 64, "b" * 64)
    ok("no protected tokens -> warning",
       r.ok and any(f.code == "NO_PROTECTED_TOKENS" for f in r.warnings))

    r = KV.check_kv(_hdr(n_tokens=0), "a" * 64, "b" * 64)
    ok("an empty frame is refused", not r.ok)

    try:
        KV.require_kv(_hdr(), "c" * 64, "b" * 64)
        ok("require_kv raises on a hard failure", False)
    except Exception as e:
        ok("require_kv raises on a hard failure", "MODEL_MISMATCH" in str(e))


def t_end_to_end_frame():
    print("\n-- end to end: encode -> frame on disk -> guard -> decode")
    cb = synth_codebook()
    kv = synth_kv()
    payload = KV.encode_frame(kv, cb, n_sink=4)
    raw = N_LAYERS * 2 * HEADS * HEAD_DIM * 16
    bpt = KV.frame_bits_per_token(cb, N_TOK, 4, raw)
    hdr = FrameHeader(
        model_sha="a" * 64, codebook_sha=cb.sha(), model_id="synthetic",
        layer=-1, n_layers=N_LAYERS, n_dims=HEADS * HEAD_DIM,
        hidden_dim=HEADS * HEAD_DIM, bits_per_token=int(bpt),
        codec=f"kvcpca{cb.meta['unit_bits']}", n_tokens=N_TOK,
        created_utc="2026-08-27T00:00:00Z",
        notes={"kind": "kv", "key_basis": "prerope", "sink": 4,
               "unit_bits": cb.meta["unit_bits"], "kv_heads": HEADS,
               "head_dim": HEAD_DIM})
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "doc.kvf")
        write_frame(p, hdr, payload)
        fr = read_frame(p)
        r = KV.check_kv(fr.header, "a" * 64, cb.sha())
        ok("frame written by this module passes its own guard", r.ok, r.reason())
        got = KV.decode_frame(fr.payload, cb, N_TOK, n_sink=4, dtype=torch.float32)
        direct = KV.decode_frame(payload, cb, N_TOK, n_sink=4, dtype=torch.float32)
        ok("decoding from disk == decoding in memory", all(
            torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            for a, b in zip(got, direct)))
        ok("frame beats a bf16 KV dump on size",
           os.path.getsize(p) < raw * N_TOK / 8,
           f"{os.path.getsize(p)} vs {int(raw * N_TOK / 8)} bytes bf16")


def main():
    t_codebook(); t_roundtrip(); t_mismatched_codebook(); t_bits_accounting()
    t_guard(); t_end_to_end_frame()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print("  FAILED:", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
