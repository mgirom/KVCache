#!/usr/bin/env python3
"""Regenerate the <!-- codespace-table --> block in every file given, from results/.

The table in the README and the research note must be the result files and nothing
else; a hand-edited number is a number nobody can trace. Answer agreement is computed
from the saved texts with the auditor's own checker. Timing columns show only files
produced by the current harness (decode timed apart from question prefill); older
files show a dash rather than a number that means something different.
"""
import glob, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codespace import _repo_root
ROOT = _repo_root(__file__)
sys.path.insert(0, os.path.join(ROOT, "auditor", "runner"))
import assemble as A

NAMES = {"qwen3-1.7b-fp": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B", "smollm2-1.7b": "SmolLM2-1.7B",
         "olmo2-1b": "OLMo2-1B", "bitnet-2b-bf16": "BitNet-2B"}
tasks = {it["id"]: it for it in json.load(open(os.path.join(ROOT, "auditor/workload/tasks.json")))["items"]}

rows, ratios, slow = [], [], []
for m, name in NAMES.items():
    f = os.path.join(ROOT, "results", f"codespace_{m}_r4.json")
    if not os.path.exists(f):
        continue
    d = json.load(open(f)); s = d["summary"]; rr = d["rows"]
    agree = sum(A.check(tasks[r["id"]], r["text"]["B"])["first"] == A.check(tasks[r["id"]], r["text"]["C"])["first"] for r in rr)
    c = s["correct"]; ratios.append(s["ratio_mean"])
    if "decode_ms_per_tok_mean" in s and s["decode_ms_per_tok_mean"]["A"] and s["decode_ms_per_tok_mean"]["C"]:
        a_, c_ = s["decode_ms_per_tok_mean"]["A"], s["decode_ms_per_tok_mean"]["C"]
        tim = f"{a_:.0f} → {c_:.0f}"; slow.append(c_ / a_)
    else:
        tim = "—"
    bold = "**" if not rows else ""
    rows.append(f"| {name} | {agree}/{s['n']} | {c['A']} / {c['B']} / {c['C']} | {bold}{s['ratio_mean']:.2f}× smaller{bold} | {tim} |")

table = ("<!-- codespace-table -->\n"
         "| model | same answer as decode-then-attend | correct: dense / decoded / code-space | KV memory | decode ms/token: dense → code-space |\n"
         "|---|---:|---:|---:|---:|\n" + "\n".join(rows) + "\n<!-- /codespace-table -->")
pat = re.compile(r"<!-- codespace-table -->.*?<!-- /codespace-table -->", re.S)
for path in sys.argv[1:]:
    p = os.path.abspath(path); s = open(p).read()
    if not pat.search(s):
        print("no table block in", path); continue
    open(p, "w").write(pat.sub(lambda _: table, s)); print("refreshed", path)
print(table)
if slow:
    print(f"\ncode-space decode is {min(slow):.1f}x to {max(slow):.1f}x slower per token "
          f"(mean {sum(slow)/len(slow):.1f}x over {len(slow)} models with split timing)")
