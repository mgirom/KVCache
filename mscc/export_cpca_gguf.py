#!/usr/bin/env python3
"""Export a fitted per-head codebook (.kvcb.npz) as the GGUF file llama.cpp's cpca
cache reads: per layer, the rotation matrices and biases in exactly the orientation
ggml_mul_mat consumes, so the C++ side does no algebra.

ggml_mul_mat(a, b) with a as numpy (n_out, n_in) computes y = a @ x for each column x
of b. So every matrix is stored as (heads, n_out, n_in). Per-head tensors are
[n_in, n_out, H_kv] in ggml order and broadcast by division over query heads, which
is the GQA grouping llama.cpp uses for K.Q. Elementwise biases that land on n_head
tensors (the attention output) are pre-expanded here because ggml_add broadcasts by
modulo, not division.
"""
import argparse, hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "llama.cpp", "gguf-py")]
import gguf                                                              # noqa: E402
from mscc.kv import KVCodebook                                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("codebook")
ap.add_argument("-o", "--out", required=True)
ap.add_argument("--n-head", type=int, required=True, help="query heads (for output-side biases)")
ap.add_argument("--model-gguf", default="", help="the model file this codebook is for; its identity is written so llama.cpp refuses a mismatch")
ap.add_argument("--fp16", action="store_true", help="store the matrices as f16 (halves the file; llama.cpp converts on load)")
a = ap.parse_args()

cb = KVCodebook.load(a.codebook)
m = cb.meta
assert m.get("per_head") and m.get("basis") == "postrope", "cpca needs a per-head post-RoPE codebook"
assert m.get("quant") in ("q4_0", "q8_0"), f"codebook quant {m.get('quant')!r}: fit with --quant q4_0"
H, d = int(m["kv_heads"]), int(m["head_dim"])
n_layers = cb.n_layers if hasattr(cb, "n_layers") else 1 + max(l for l, _ in cb.books)
groups = a.n_head // H
assert a.n_head % H == 0

w = gguf.GGUFWriter(a.out, arch="cpca")
w.add_string("cpca.source_codebook", os.path.basename(a.codebook))
w.add_string("cpca.codebook_sha", cb.sha() if hasattr(cb, "sha") else "")
w.add_string("cpca.quant", m["quant"]); w.add_bool("cpca.whiten", bool(m.get("whiten", False)))
# where the states came from: the served file's own cache (capture_gguf_kv) or HF weights (kvfit).
# A Q4_K_M model's keys are not its bf16 twin's; a codebook fitted on the served file fits better.
w.add_string("cpca.fit_source", str(m.get("captured_from", "huggingface weights (kvfit)")))
w.add_uint32("cpca.n_head", a.n_head)
if not a.model_gguf: w.add_uint32("cpca.n_layer", n_layers)
for key in ("model", "corpus_sha256", "n_states"):
    if key in m: w.add_string(f"cpca.{key}", str(m[key]))
w.add_uint32("cpca.n_head_kv", H); w.add_uint32("cpca.head_dim", d)
if a.model_gguf:
    from mscc.ggufmeta import gguf_metadata, model_geometry
    kv = gguf_metadata(a.model_gguf); g = model_geometry(a.model_gguf)
    assert g["n_head_kv"] == H and g["head_dim"] == d and g["n_head"] == a.n_head, ("model/codebook geometry mismatch", g, H, d, a.n_head)
    w.add_string("cpca.model_arch", g["arch"]); w.add_string("cpca.model_name", kv.get("general.name", ""))
    w.add_uint32("cpca.n_layer", g["n_layer"])
    w.add_string("cpca.model_file", os.path.basename(a.model_gguf))
    print(f"bound to {g['arch']} '{kv.get('general.name', '')}' {g['n_layer']}L x {H}kv x {d}d")
dtype = np.float16 if a.fp16 else np.float32

def stack(unit, fn):
    return np.stack([fn(cb.books[(l, f"{unit}{h}")]) for h in range(H)]).astype(np.float32)

for l in range(n_layers):
    if (l, "k0") not in cb.books:
        continue
    def zs(b):              # per-component code scale (whitening); ones when absent
        return np.ones(b.V.shape[1]) if b.zs is None else b.zs.reshape(-1).astype(np.float64)
    def rot(b, inv):        # write: diag(1/zs) V^T diag(1/s)   query: diag(zs) V^T diag(s)
        V, s = b.V.astype(np.float64), b.s.reshape(-1).astype(np.float64)
        M = V.T * (1.0 / s if inv else s)[None, :]
        return M * ((1.0 / zs(b)) if inv else zs(b))[:, None]
    def unrot(b):           # diag(s) V diag(zs)
        return (b.V.astype(np.float64) * b.s.reshape(-1).astype(np.float64)[:, None]) * zs(b)[None, :]
    q_rot  = np.stack([rot(cb.books[(l, f"k{h}")], inv=False) for h in range(H)])
    k_rot  = np.stack([rot(cb.books[(l, f"k{h}")], inv=True)  for h in range(H)])
    k_mean = np.stack([cb.books[(l, f"k{h}")].mu.reshape(-1) for h in range(H)]).astype(np.float64)
    k_bias = -np.einsum("hij,hj->hi", k_rot, k_mean)
    k_unrot = np.stack([unrot(cb.books[(l, f"k{h}")]) for h in range(H)])
    v_rot  = np.stack([rot(cb.books[(l, f"v{h}")], inv=True) for h in range(H)])
    v_mean = np.stack([cb.books[(l, f"v{h}")].mu.reshape(-1) for h in range(H)]).astype(np.float64)
    v_bias = -np.einsum("hij,hj->hi", v_rot, v_mean)
    v_unrot = np.stack([unrot(cb.books[(l, f"v{h}")]) for h in range(H)])
    v_mean_full = np.repeat(v_mean, groups, axis=0)                     # [n_head, d], head h -> kv h//groups
    for name, arr in (("q_rot", q_rot), ("k_rot", k_rot), ("k_bias", k_bias), ("k_unrot", k_unrot),
                      ("k_mean", k_mean), ("v_rot", v_rot), ("v_bias", v_bias), ("v_unrot", v_unrot),
                      ("v_mean", v_mean_full)):
        w.add_tensor(f"blk.{l}.cpca_{name}", np.ascontiguousarray(arr, dtype=dtype))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"wrote {a.out}: {os.path.getsize(a.out)/1e6:.1f} MB, {n_layers} layers, {H} kv heads x {d}, {a.n_head} query heads")

# self-check: the exported matrices reproduce the codec's own encode/decode on random data
import torch
l0 = 14 if (14, "k0") in cb.books else min(l for l, _ in cb.books)
b = cb.books[(l0, "k0")]
x = torch.randn(7, d); z = ((x - torch.as_tensor(b.mu)) / torch.as_tensor(b.s).reshape(-1)) @ torch.as_tensor(b.V)
if b.zs is not None: z = z / torch.as_tensor(b.zs).reshape(-1)
kr = np.stack([rot(cb.books[(l0, f"k{h}")], inv=True) for h in range(H)])[0]
kb = -(kr @ cb.books[(l0, "k0")].mu.reshape(-1).astype(np.float64))
z2 = x.numpy() @ kr.T + kb
print("exported k_rot reproduces codec projection:", np.allclose(z.numpy(), z2, atol=1e-4))
q = torch.randn(3, d); qr = np.stack([rot(cb.books[(l0, f"k{h}")], inv=False) for h in range(H)])[0]
lhs = (q.numpy() @ qr.T) @ z2.T; rhs = q.numpy() @ (x.numpy() - cb.books[(l0, "k0")].mu.reshape(-1)).T
print("q_rot.k_rot preserves q.(k-mu):", np.allclose(lhs, rhs, atol=1e-3))
ur = np.stack([unrot(cb.books[(l0, f"k{h}")]) for h in range(H)])[0]
print("k_unrot inverts k_rot:", np.allclose(z2 @ ur.T + cb.books[(l0, "k0")].mu.reshape(-1), x.numpy(), atol=1e-4))
