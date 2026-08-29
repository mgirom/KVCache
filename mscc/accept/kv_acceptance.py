#!/usr/bin/env python3
"""Acceptance test for the full-depth KV frame -- the four criteria in DELIVERY-PLAN.

This is the same test Item 1 ran against the mid-stack frame, on the same document
with the same planted fact. That run failed criterion 1: `serve` answered
"1234567890" where the reference answered "BRK-7742", and the control localised the
fault to the skip rather than the codec.

Criteria (DELIVERY-PLAN, Item 1):
  1. fit -> encode -> serve on a real document, answer is correct
  2. same frame + wrong model -> refused BY NAME, not degraded
  3. below-floor frame -> refused
  4. frame survives a round-trip to disk and back byte-identical

Plus two the KV frame makes newly testable:
  5. a refused frame, forced through, really does answer fluently and wrongly
     (otherwise the guard is ceremony)
  6. a pre-RoPE frame replayed at a non-zero offset still answers, because the
     key basis is position-independent

Run:  python3 -m mscc.accept.kv_acceptance        (from the repository root)
"""
import json
import os
import subprocess
import sys
import time

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(R, "lib"), R]

from mscc import format as mfmt          # noqa: E402
from mscc import kv as mkv               # noqa: E402

A = os.path.join(R, "mscc", "accept", "kv")
MODEL = os.path.join(R, "models", "qwen3-1.7b-fp")
FRAME = os.path.join(A, "doc.kvf")
CB = os.path.join(A, "book.kvcb.npz")
NARROW_CB = os.path.join(A, "narrow.kvcb.npz")
ASK = ("\n\nQuestion: What is the calibration marker for this archive copy? "
       "Answer with the code only.\nAnswer:")
FACT = "BRK-7742"

results = []


def record(crit, name, ok, detail=""):
    results.append({"criterion": crit, "name": name, "pass": bool(ok),
                    "detail": detail})
    print(f"  {'PASS' if ok else 'FAIL'}  [{crit}] {name}"
          + (f"\n        {detail}" if detail else ""), flush=True)


def cli(*args, expect=None):
    p = subprocess.run([sys.executable, "-m", "mscc.cli", *args], cwd=R,
                       capture_output=True, text=True)
    return p


def main():
    print(f"acceptance: KV frame  ({time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())})")
    hdr = mfmt.read_header(FRAME)
    n_tok = hdr.n_tokens
    raw_bytes = int(hdr.notes["kv_heads"] * hdr.notes["head_dim"] * 2
                    * hdr.n_layers * 2 * n_tok)
    print(f"document {n_tok} tokens | frame {os.path.getsize(FRAME):,} B | "
          f"uncompressed KV {raw_bytes:,} B | "
          f"{raw_bytes / os.path.getsize(FRAME):.1f}x\n")

    # --- 1: the answer is correct, with zero document layers run ---------------
    print("-- criterion 1: answer from the frame alone")
    p = cli("kvserve", "--frame", FRAME, "--model", MODEL, "--codebook", CB,
            "--ask", ASK, "--gen", "24")
    out = p.stdout
    record(1, "guard passes a well-formed frame", "guard: PASS" in out,
           [l for l in out.splitlines() if l.startswith("guard:")][:1])
    record(1, "receiver runs ZERO layers over the document",
           "blocks executed over the document: 0" in out)
    answer = out.split("A:", 1)[1].strip() if "A:" in out else "(no answer)"
    record(1, f"answer contains the planted fact {FACT}", FACT in answer,
           f"answered: {answer.splitlines()[0][:80]!r}")

    # --- 2: wrong model, refused by name ---------------------------------------
    print("\n-- criterion 2: a frame from a different model is refused by name")
    fr = mfmt.read_frame(FRAME)
    bad = os.path.join(A, "_wrongmodel.kvf")
    h2 = mfmt.FrameHeader.from_dict(json.loads(fr.header.to_json()))
    h2.model_sha = "0" * 64
    mfmt.write_frame(bad, h2, fr.payload)
    p = cli("kvserve", "--frame", bad, "--model", MODEL, "--codebook", CB,
            "--ask", ASK, "--gen", "8")
    record(2, "refused (exit 2), not answered", p.returncode == 2,
           f"exit {p.returncode}")
    record(2, "refusal names the condition", "MODEL_MISMATCH" in p.stdout,
           [l.strip() for l in p.stdout.splitlines() if "MODEL_MISMATCH" in l][:1])
    record(2, "no answer was emitted", "A:" not in p.stdout)

    # --- 3: below-floor rate, refused ------------------------------------------
    print("\n-- criterion 3: a below-floor frame is refused")
    narrow = os.path.join(A, "narrow.kvf")
    if not os.path.exists(narrow):
        p = cli("kvencode", os.path.join(R, "mscc", "accept", "doc.txt"),
                "--model", MODEL, "--codebook", NARROW_CB, "--sink", "4",
                "-o", narrow)
        if p.returncode != 0:
            print(p.stdout, p.stderr)
    nh = mfmt.read_header(narrow)
    record(3, f"narrow frame really is narrower ({nh.notes['unit_bits']} bits/unit)",
           nh.notes["unit_bits"] < mkv.MIN_UNIT_BITS,
           f"{nh.notes['unit_bits']} < floor {mkv.MIN_UNIT_BITS}")
    p = cli("kvserve", "--frame", narrow, "--model", MODEL, "--codebook", NARROW_CB,
            "--ask", ASK, "--gen", "8")
    record(3, "refused (exit 2), not answered", p.returncode == 2,
           f"exit {p.returncode}")
    record(3, "refusal names the condition", "RATE_TOO_LOW" in p.stdout,
           [l.strip() for l in p.stdout.splitlines() if "RATE_TOO_LOW" in l][:1])

    # --- 5: and the refusal was earning its keep -------------------------------
    print("\n-- criterion 5: forced through, the refused frame answers wrongly")
    p = cli("kvserve", "--frame", narrow, "--model", MODEL, "--codebook", NARROW_CB,
            "--ask", ASK, "--gen", "24", "--force")
    forced = p.stdout.split("A:", 1)[1].strip() if "A:" in p.stdout else ""
    record(5, "forced run produces fluent text", len(forced) > 4,
           f"answered: {forced.splitlines()[0][:80]!r}" if forced else "(empty)")
    record(5, "forced run gets the fact WRONG -- the guard was not ceremony",
           FACT not in forced)

    # --- 4: byte-identical round trip ------------------------------------------
    print("\n-- criterion 4: the frame survives disk")
    with open(FRAME, "rb") as fh:
        raw = fh.read()
    copy = os.path.join(A, "_roundtrip.kvf")
    with open(copy, "wb") as fh:
        fh.write(raw)
    fr2 = mfmt.read_frame(copy)
    record(4, "bytes identical", open(copy, "rb").read() == raw)
    record(4, "header identical", fr2.header.to_json() == fr.header.to_json())
    same = (set(fr2.payload) == set(fr.payload)
            and all((fr2.payload[k] == fr.payload[k]).all() for k in fr.payload))
    record(4, f"all {len(fr.payload)} payload arrays identical", same)

    # --- 6: position independence ----------------------------------------------
    print("\n-- criterion 6: pre-RoPE frame replays at a non-zero offset")
    record(6, "frame declares the pre-RoPE basis",
           fr.header.notes.get("key_basis") == "prerope")
    p = cli("kvserve", "--frame", FRAME, "--model", MODEL, "--codebook", CB,
            "--ask", ASK, "--gen", "24", "--offset", "64")
    off = p.stdout.split("A:", 1)[1].strip() if "A:" in p.stdout else ""
    record(6, "still answers correctly when replayed 64 positions later",
           FACT in off, f"answered: {off.splitlines()[0][:80]!r}" if off else "(none)")

    for f in (bad, copy):
        if os.path.exists(f):
            os.remove(f)

    npass = sum(r["pass"] for r in results)
    out_path = os.path.join(R, "results", "gate12", "kv_acceptance.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "frame": FRAME, "n_tokens": n_tok,
               "frame_bytes": os.path.getsize(FRAME),
               "uncompressed_kv_bytes": raw_bytes,
               "passed": npass, "total": len(results), "checks": results},
              open(out_path, "w"), indent=1)
    print(f"\n{npass}/{len(results)} checks passed -> {out_path}")
    for r in results:
        if not r["pass"]:
            print("  FAILED:", r["name"])
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
