#!/usr/bin/env python3
"""Milestone 3 of LLAMACPP-CPCA-DESIGN.md: does llama.cpp with the fitted rotation
answer like the PyTorch emulation of the same codebook?

Four arms on the same items, same prompts:
  f16        llama.cpp, uncompressed cache                       (the model as shipped)
  q4_0       llama.cpp, q4_0 cache with its own Hadamard rotation (what -ctk q4_0 does today)
  q4_0+cpca  llama.cpp, q4_0 cache with the fitted rotation      (this work)
  emu        PyTorch, HF weights, q4_0 emulated exactly on the same codebook, decode-then-attend

The two right-hand arms compute the same quantisation of the same rotated states, so
their answers should agree far more often than either agrees with q4_0 Hadamard; the
model weights differ (GGUF Q4_K_M vs bf16 safetensors), so bit-identity is not the
bar -- answer agreement is, and the server log must say the codebook was loaded.
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codespace import _repo_root
ROOT = _repo_root(__file__)
sys.path[:0] = [ROOT, os.path.join(ROOT, "auditor", "runner")]
import gpulock
import assemble as A
import backends as BE

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", default=os.path.join(ROOT, "models/qwen3-1.7b-fp/q3-1.7b-Q4_K_M.gguf"))
ap.add_argument("--hf", default=os.path.join(ROOT, "models/qwen3-1.7b-fp"))
ap.add_argument("--codebook-gguf", default=os.path.join(ROOT, "mscc/accept/kv/qwen3-1.7b_pca128q4_0.cpca.gguf"))
ap.add_argument("--codebook-npz", default=os.path.join(ROOT, "mscc/accept/kv/pca128q4_0_post.kvcb.npz"))
ap.add_argument("--binary", default=os.path.join(ROOT, "llama.cpp/build-cuda/bin/llama-server"))
ap.add_argument("--context", type=int, default=1024)
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--out", default="")
a = ap.parse_args()

gpulock.acquire("cpca-check")
tasks = json.load(open(os.path.join(ROOT, "auditor/workload/tasks.json")))
hay = open(os.path.join(ROOT, "auditor/workload/haystack.txt")).read()
items = [it for it in tasks["items"] if it["context"] == a.context]
items = items[:: max(1, len(items) // a.n)][: a.n]
log_dir = os.path.join(ROOT, "logs", "cpca_check"); os.makedirs(log_dir, exist_ok=True)

def run_arm(name, make):
    be = make().start()
    try:
        n_tok = lambda s: be.n_tokens(s)                                     # noqa: E731
        out = {}
        for i, it in enumerate(items):
            doc = A.assemble(it, hay, a.context, n_tok, offset_chars=i * 9973)
            r = be.complete(A.prompt_for(it, doc), n_predict=24)
            out[it["id"]] = r["content"]
        meta = {"kv_bytes_per_token": (be.kv_bytes_per_token(a.context) if hasattr(be, "kv_bytes_per_token") else None)}
        return out, meta, getattr(getattr(be, "srv", None), "log_path", None)
    finally:
        be.stop()

arms = {
    "f16":       lambda: BE.LlamaCppBackend(a.binary, a.gguf, a.context, ctk="f16", ctv="f16", log_dir=log_dir),
    "q4_0":      lambda: BE.LlamaCppBackend(a.binary, a.gguf, a.context, ctk="q4_0", ctv="q4_0", log_dir=log_dir),
    "q4_0+cpca": lambda: BE.LlamaCppBackend(a.binary, a.gguf, a.context, ctk="q4_0", ctv="q4_0", log_dir=log_dir, codebook=a.codebook_gguf),
    "emu":       lambda: BE.MsccBackend(a.hf, a.codebook_npz, a.context, unit_bits=int(json.loads(__import__("numpy").load(a.codebook_npz)["_meta"].tobytes())["meta"]["unit_bits"])),
}
texts, logs = {}, {}
for name, make in arms.items():
    t0 = time.perf_counter(); texts[name], meta, logp = run_arm(name, make); logs[name] = logp
    print(f"{name:<10} done in {time.perf_counter()-t0:5.0f}s", flush=True)

# the server must have said it loaded the codebook, else the +cpca arm is a lie
loaded = False
if logs.get("q4_0+cpca") and os.path.exists(logs["q4_0+cpca"]):
    loaded = "cpca codebook" in open(logs["q4_0+cpca"], errors="replace").read()
print(f"\ncodebook loaded per server log: {loaded}")

ans = {n: {i: A.check(it, texts[n][it["id"]])["first"] for it in items for i in [it["id"]]} for n in arms}
hit = {n: sum(A.check(it, texts[n][it["id"]])["hit"] for it in items) for n in arms}
def agree(x, y): return sum(ans[x][i] == ans[y][i] for i in ans[x])
print(f"\ncorrect of {len(items)}: " + "  ".join(f"{n} {hit[n]}" for n in arms))
print(f"answer agreement: cpca==emu {agree('q4_0+cpca','emu')}/{len(items)}   cpca==f16 {agree('q4_0+cpca','f16')}   "
      f"q4_0==f16 {agree('q4_0','f16')}   emu==f16 {agree('emu','f16')}   q4_0==cpca {agree('q4_0','q4_0+cpca')}")
for it in items:
    i = it["id"]
    if not (ans["q4_0+cpca"][i] == ans["emu"][i]):
        print(f"  differ {i:<8} want {it['answer']!r}: cpca {ans['q4_0+cpca'][i]!r}  emu {ans['emu'][i]!r}  f16 {ans['f16'][i]!r}  q4_0 {ans['q4_0'][i]!r}")
if a.out:
    json.dump({"items": [it["id"] for it in items], "texts": texts, "hits": hit, "codebook_loaded": loaded}, open(a.out, "w"), indent=1)
