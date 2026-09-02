#!/usr/bin/env python3
"""Does attention over PACKED CODES give the same answers as decode-then-attend, and
how much memory does the cache actually occupy while it does?

Three paths on the same document and question, same model, same run:
  A  dense       : f16 cache, ordinary attention             (the model as shipped)
  B  decode      : codes -> f16 cache -> ordinary attention   (every earlier number)
  C  code-space  : codes stay packed; attention via the fold  (the live path)

B and C compute the same function up to f16 rounding of B's reconstructed cache, so
their generated tokens should agree almost everywhere; where they do not, the first
step's logit gap says how far apart they are. C's memory is read from the tensors.
Speed is reported and is expected to be worse -- this is PyTorch, not a kernel.
"""
import argparse, json, os, sys
# Some modeling code (BitNet) calls torch.compile in its forward; the inductor's gcc
# step fails on this machine and the harness gains nothing from compilation anyway.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codespace import _repo_root                                     # noqa: E402
ROOT = _repo_root(__file__)
sys.path[:0] = [ROOT, os.path.join(ROOT, "auditor", "runner")]

import gpulock
from lib_inject import load_model, DEV
import lib_kv as K
import codespace as CS
from mscc.kv import KVCodebook
import assemble as A

QM = "\n\nQuestion: "

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(ROOT, "models/qwen3-1.7b-fp"))
ap.add_argument("--codebook", default=os.path.join(ROOT, "mscc/accept/kv/perhead2048_post.kvcb.npz"))
ap.add_argument("--context", type=int, default=1024)
ap.add_argument("--n", type=int, default=12, help="items, strided across the workload")
ap.add_argument("--maxnew", type=int, default=24)
ap.add_argument("--out", default="")
ap.add_argument("--raw", action="store_true",
                help="use the bare 'Question:/Answer:' prompt even if the tokenizer has a chat "
                     "template. Instruct models given a bare prompt may emit their end token at once.")
a = ap.parse_args()

gpulock.acquire("codespace-test")
tok, model = load_model(a.model)
cb = KVCodebook.load(a.codebook)
assert cb.meta.get("basis") == "postrope" and cb.meta.get("per_head"), \
    "the fold needs a per-head POST-RoPE codebook"
heads, hd = int(cb.meta["kv_heads"]), int(cb.meta["head_dim"])
eos = tok.eos_token_id

tasks = json.load(open(os.path.join(ROOT, "auditor/workload/tasks.json")))
hay = open(os.path.join(ROOT, "auditor/workload/haystack.txt")).read()
items = [it for it in tasks["items"] if it["context"] == a.context]
step = max(1, len(items) // a.n)
items = items[::step][: a.n]
n_tokens = lambda s: len(tok(s).input_ids)                                  # noqa: E731

use_chat = bool(getattr(tok, "chat_template", None)) and not a.raw

def split_prompt(it, doc):
    """(document text, question text, add_special_tokens for the document).
    With a chat template the whole task is one user turn and the model answers as the
    assistant; the split still falls at the question marker, so the template's header
    and the document are the coded segment and the question plus the template's tail
    are the recent one. Thinking is disabled where the template knows the switch."""
    raw = A.prompt_for(it, doc)
    if not use_chat:
        d, q = raw.split(QM, 1); return d, QM + q, True
    full = tok.apply_chat_template([{"role": "user", "content": raw}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    d, q = full.split(QM, 1); return d, QM + q, False

# warm-up: the first CUDA forward pays for allocator and kernel selection; keep it
# out of every path's timing
with torch.inference_mode():
    _w = tok("warm up", return_tensors="pt").input_ids.to(DEV)
    model(input_ids=_w, use_cache=False); torch.cuda.synchronize()
print(f"prompt format: {'chat template' if use_chat else 'raw Question/Answer'}", flush=True)

rows = []
for i, it in enumerate(items):
    doc = A.assemble(it, hay, a.context, n_tokens, offset_chars=i * 9973)
    doc_text, q_text, special = split_prompt(it, doc)
    d_ids = tok(doc_text, return_tensors="pt", add_special_tokens=special).input_ids.to(DEV)
    q_ids = tok(q_text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)

    with torch.inference_mode():
        kv = K.capture_kv(model, d_ids)
        dense_bytes = sum(t.numel() * t.element_size() for kk, vv in kv for t in (kk, vv))

        tm = {"A": {}, "B": {}, "C": {}}
        # A: dense
        outA = K.gen_from_cache(model, kv, q_ids, a.maxnew, eos=eos, timings=tm["A"])

        # B: decode-then-attend
        kvB = K.merge_exact(kv, K.roundtrip_kv_perhead_postrope(kv, cb.books, heads, hd), n_sink=4)
        outB = K.gen_from_cache(model, kvB, q_ids, a.maxnew, eos=eos, timings=tm["B"])
        del kvB

        # C: code-space -- the cache is packed codes from here on
        torch.cuda.synchronize(); m0 = torch.cuda.memory_allocated()
        cache = CS.CodeSpaceCache(kv, cb.books, heads, hd, sink=4)
        del kv; torch.cuda.empty_cache(); torch.cuda.synchronize()
        held = cache.bytes()
        prev = CS.install(model, cache)
        try:
            outC = CS.generate_codespace(model, cache, q_ids, a.maxnew, eos=eos, timings=tm["C"])
        finally:
            CS.uninstall(model, prev)
        del cache; torch.cuda.empty_cache()

    txt = {k: tok.decode(v, skip_special_tokens=True) for k, v in (("A", outA), ("B", outB), ("C", outC))}
    ok = {k: A.check(it, v)["hit"] for k, v in txt.items()}
    n_cmp = min(len(outB), len(outC))
    agree = sum(int(x == y) for x, y in zip(outB[:n_cmp], outC[:n_cmp]))
    row = dict(id=it["id"], tier=it["tier"], dense_bytes=dense_bytes, code_bytes=held,
               ratio=round(dense_bytes / held, 2), tok_agree_BC=f"{agree}/{n_cmp}",
               identical_BC=(outB == outC), correct=ok,
               n_gen={k: len(v) for k, v in (("A", outA), ("B", outB), ("C", outC))},
               prefill_ms={k: round(tm[k]["prefill_s"] * 1e3, 1) for k in "ABC"},
               decode_ms_per_tok={k: (round(tm[k]["decode_s"] / tm[k]["n_decode"] * 1e3, 1) if tm[k]["n_decode"] else None) for k in "ABC"},
               text=txt)
    rows.append(row)
    print(f"[{i+1}/{len(items)}] {it['id']:<8} {row['ratio']:>5}x  B==C:{str(row['identical_BC']):<5} "
          f"agree {row['tok_agree_BC']:>6}  correct A/B/C {int(ok['A'])}/{int(ok['B'])}/{int(ok['C'])}  "
          f"decode ms/tok A {row['decode_ms_per_tok']['A']} B {row['decode_ms_per_tok']['B']} C {row['decode_ms_per_tok']['C']}  "
          f"q-prefill ms A {row['prefill_ms']['A']} C {row['prefill_ms']['C']}", flush=True)

n = len(rows)
summ = dict(model=os.path.basename(a.model), codebook=os.path.basename(a.codebook), context=a.context, n=n,
            identical_BC=sum(r["identical_BC"] for r in rows),
            correct=dict(A=sum(r["correct"]["A"] for r in rows), B=sum(r["correct"]["B"] for r in rows),
                         C=sum(r["correct"]["C"] for r in rows)),
            ratio_mean=round(sum(r["ratio"] for r in rows) / n, 2),
            prompt_format="chat" if use_chat else "raw",
            prefill_ms_mean={k: round(sum(r["prefill_ms"][k] for r in rows) / n, 1) for k in "ABC"},
            decode_ms_per_tok_mean={k: (lambda v: round(sum(v) / len(v), 1) if v else None)(
                [r["decode_ms_per_tok"][k] for r in rows if r["decode_ms_per_tok"][k] is not None]) for k in "ABC"})
print(json.dumps(summ, indent=1))
if a.out:
    json.dump(dict(summary=summ, rows=rows), open(a.out, "w"), indent=1)
