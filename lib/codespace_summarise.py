#!/usr/bin/env python3
"""One table across every model the code-space harness has run on.

Columns are what a reader needs to judge the live path, in this order: did attending
over packed codes give the SAME ANSWER as decode-then-attend (the fold's correctness
on a real model), did both give the dense model's answer (the codec's cost), how much
smaller the cache actually was, and how much slower this PyTorch implementation is.
Whole-sequence identity is also shown because it is the stricter test; where it is
lower than answer agreement the divergence came after the answer, at a near-tie.
"""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codespace import _repo_root
ROOT = _repo_root(__file__)
sys.path.insert(0, os.path.join(ROOT, "auditor", "runner"))
import assemble as A

tasks = {it["id"]: it for it in json.load(open(os.path.join(ROOT, "auditor/workload/tasks.json")))["items"]}
files = sorted(glob.glob(os.path.join(ROOT, "results", "codespace_*_r*.json")))
if not files:
    sys.exit("no results/codespace_*_r*.json yet")
hdr = f"{'model':<16} {'fmt':<4} {'n':>3} {'answer B==C':>11} {'seq B==C':>9} {'correct A/B/C':>14} {'KV smaller':>10} {'decode ms/tok A/B/C':>20} {'q-prefill ms A/C':>16}"
print(hdr); print("-" * len(hdr))
for f in files:
    d = json.load(open(f)); s = d["summary"]; rows = d["rows"]
    ans_agree = sum(A.check(tasks[r["id"]], r["text"]["B"])["first"] == A.check(tasks[r["id"]], r["text"]["C"])["first"] for r in rows)
    fmt = s.get("prompt_format", "raw")[:4]
    if "decode_ms_per_tok_mean" in s:
        m, pf = s["decode_ms_per_tok_mean"], s["prefill_ms_mean"]
        f3 = lambda v: "-" if v is None else f"{v:.1f}"                     # noqa: E731
        tim = f"{f3(m['A']):>7}/{f3(m['B'])}/{f3(m['C'])}"; pre = f"{pf['A']:>8.0f}/{pf['C']:.0f}"
    else:                                                                    # older files
        m = s["ms_per_tok_mean"]; tim = f"{m['A']:>7.1f}/{m['B']:.1f}/{m['C']:.1f}*"; pre = f"{'n/a':>14}"
    print(f"{s['model']:<16} {fmt:<4} {s['n']:>3} {ans_agree:>8}/{s['n']:<2} {s['identical_BC']:>6}/{s['n']:<2} "
          f"{s['correct']['A']:>4}/{s['correct']['B']}/{s['correct']['C']:<5} {s['ratio_mean']:>9.2f}x {tim:>20} {pre:>16}")
print("* = older harness: whole-generation ms/token including question prefill")
