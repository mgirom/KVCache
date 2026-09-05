#!/usr/bin/env python3
"""Fit a cpca codebook for a model that exists only as a GGUF.

The fitter needs post-RoPE keys and values for a corpus. HuggingFace capture cannot
serve a model whose bf16 twin does not fit the card, and the ternary models here
have no such twin at all. llama.cpp can write its own cache to a sequence state file
(`--slot-save-path` + POST /slots/0?action=save), layer by layer, in the cache's own
type. With an f16 cache the rotation is off, so what lands in the file is the cache as
the model leaves it: exactly the states the live path will see.

File layout (llama.h LLAMA_STATE_SEQ_VERSION 2, llama-kv-cache.cpp state_write):
  u32 magic 'GGSQ' | u32 version | u32 n_tokens | i32 tokens[n]
  u32 n_stream | per stream: u32 cell_count | per cell: i32 pos, u32 n_seq_id, i32 seq_ids[...]
  u32 v_trans | u32 n_layer | per layer: i32 k_type, u64 k_row_bytes, rows[cell_count]
  (v_trans == 0) per layer: i32 v_type, u64 v_row_bytes, rows[cell_count]
Rows are [head_dim x n_head_kv] f16, which is head_slices() order.
"""
import argparse, json, os, struct, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "alphabet", "scripts"), os.path.join(ROOT, "auditor", "runner"),
                os.path.join(ROOT, "llama.cpp", "gguf-py")]
import gguf, requests                                                   # noqa: E402
import gpulock, lib_kv as K                                             # noqa: E402
from server import LlamaServer                                          # noqa: E402
from mscc.kv import KVCodebook                                          # noqa: E402
from mscc.codec import GGML_BLOCK_BYTES, corpus_digest                  # noqa: E402
from mscc.ggufmeta import gguf_metadata, model_geometry                   # noqa: E402

GGML_F16 = 1

def parse_state(path, geo):
    b = open(path, "rb").read(); o = 0
    def u32():
        nonlocal o; v = struct.unpack_from("<I", b, o)[0]; o += 4; return v
    def i32():
        nonlocal o; v = struct.unpack_from("<i", b, o)[0]; o += 4; return v
    def u64():
        nonlocal o; v = struct.unpack_from("<Q", b, o)[0]; o += 8; return v
    magic, ver = u32(), u32()
    assert magic == 0x67677371, f"not a GGSQ state file: {magic:#x}"          # LLAMA_FILE_MAGIC_GGSQ
    assert ver == 2, f"state version {ver}, this reader knows 2"
    n_tok = u32(); o += 4 * n_tok
    n_stream = u32(); assert n_stream == 1, n_stream
    n_cells = u32()
    kv_layers = geo["kv_layers"]
    H, d = geo["n_head_kv"], geo["head_dim"]
    # Per cell: pos i32, n_seq_id u32, [ext: extra positions, present when the model has
    # more than one position per embedding, e.g. multi-axis RoPE], seq_ids i32[n_seq_id].
    # The ext size is not in the file; try candidates and keep the one under which the
    # first K header validates (f16, row = n_head_kv * head_dim * 2).
    meta_start = o
    chosen = None
    for ext in (0, 4, 8, 12, 16, 32):
        o = meta_start
        try:
            for _ in range(n_cells):
                i32(); n_seq = u32(); o += ext + 4 * n_seq
            v_trans, n_layer = u32(), u32()
            t, row = struct.unpack_from("<iQ", b, o)
            if n_layer == len(kv_layers) and t == GGML_F16 and row == H * d * 2:
                chosen = ext; break
        except struct.error:
            continue
    assert chosen is not None, "could not align the cell records; unknown per-cell layout"
    if chosen:
        print(f"note: {chosen}-byte extra position record per cell (multi-axis positions)", flush=True)
    states = {}
    # K: one row of n_embd f16 per cell
    for j in range(n_layer):
        l = kv_layers[j]                                                # model layer index, as blk.{l}
        t, row = i32(), u64()
        assert t == GGML_F16 and row == H * d * 2, ("k", l, t, row, H, d)
        arr = np.frombuffer(b, dtype=np.float16, count=n_cells * H * d, offset=o).reshape(n_cells, H * d); o += n_cells * row
        states[(l, "k")] = torch.from_numpy(arr.astype(np.float32))
    if not v_trans:
        for j in range(n_layer):
            l = kv_layers[j]
            t, row = i32(), u64()
            assert t == GGML_F16 and row == H * d * 2, ("v", l, t, row, H, d)
            arr = np.frombuffer(b, dtype=np.float16, count=n_cells * H * d, offset=o).reshape(n_cells, H * d); o += n_cells * row
            states[(l, "v")] = torch.from_numpy(arr.astype(np.float32))
    else:
        # flash attention was off: V is stored channel-major, one run of n_cells per
        # channel. Same numbers, transposed.
        print("note: V cache is transposed (flash attention was off for this model)", flush=True)
        for j in range(n_layer):
            l = kv_layers[j]
            t, el, n_embd = i32(), u32(), u32()
            assert t == GGML_F16 and el == 2 and n_embd == H * d, ("v", l, t, el, n_embd, H, d)
            arr = np.frombuffer(b, dtype=np.float16, count=n_embd * n_cells, offset=o).reshape(n_embd, n_cells); o += n_embd * n_cells * el
            states[(l, "v")] = torch.from_numpy(np.ascontiguousarray(arr.T).astype(np.float32))
    if o != len(b):
        # hybrid models append their recurrent state after the attention cache
        print(f"note: {len(b) - o} trailing bytes after the KV cache (recurrent state?)", flush=True)
    return states, n_cells

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True); ap.add_argument("--corpus", default=os.path.join(ROOT, "mscc/accept/corpus.txt"))
ap.add_argument("--tokens", type=int, default=16384); ap.add_argument("--codes", type=int, default=128)
ap.add_argument("--quant", default="q4_0", choices=list(GGML_BLOCK_BYTES))
ap.add_argument("--binary", default=os.path.join(ROOT, "llama.cpp/build-cuda/bin/llama-server"))
ap.add_argument("-o", "--out", required=True, help="output .kvcb.npz (a .cpca.gguf is written beside it)")
ap.add_argument("--state-file", default="", help="reuse a saved state file instead of running the server")
ap.add_argument("--whiten", action="store_true", help="scale code components to unit spread (see kvfit --whiten)")
ap.add_argument("--whiten-power", type=float, default=1.0, help="partial whitening: divide by spread^power (1 = full)")
ap.add_argument("--window", type=int, default=0, help="prefill the corpus in windows of this many tokens, each as its own sequence "
                                                       "(positions restart at 0), so the fit sees the positions a served context does; 0 = one sequence")
a = ap.parse_args()

geo = model_geometry(a.gguf); print("model:", geo, flush=True)
assert a.codes <= geo["head_dim"] and a.codes % 32 == 0
save_dir = os.path.join(ROOT, "logs", "kvstate"); os.makedirs(save_dir, exist_ok=True)
text = open(a.corpus, encoding="utf-8", errors="replace").read()
fname = "capture.bin"
if a.state_file:
    save_dir, fname = os.path.dirname(os.path.abspath(a.state_file)), os.path.basename(a.state_file)
    print("reusing state file", a.state_file, flush=True)
else:
    gpulock.acquire("capture-gguf")
    srv = LlamaServer(a.binary, a.gguf, a.tokens + 64, cache_type_k="f16", cache_type_v="f16",
                      extra=["--slot-save-path", save_dir + "/", "-fa", "on"], log_dir=save_dir)
    srv.start()
    try:
        ids = requests.post(f"{srv.base}/tokenize", json={"content": text}).json()["tokens"][: a.tokens]
        win = a.window or len(ids)
        windows = [ids[i:i + win] for i in range(0, len(ids), win) if len(ids[i:i + win]) >= min(win, 256)]
        fnames = []
        t0 = time.perf_counter()
        for wi, chunk in enumerate(windows):
            # cache_prompt False: each window is a fresh sequence, positions from 0
            r = requests.post(f"{srv.base}/completion", json={"prompt": chunk, "n_predict": 1, "cache_prompt": False,
                                                              "temperature": 0}, timeout=3600).json()
            fn = f"capture_w{wi}.bin" if a.window else fname
            r = requests.post(f"{srv.base}/slots/0?action=save", json={"filename": fn}, timeout=600).json()
            fnames.append(fn)
        print(f"prefilled {len(windows)} window(s) of up to {win} tokens in {time.perf_counter()-t0:.0f}s", flush=True)
    finally:
        srv.stop()

if a.state_file or not a.window:
    states, n_cells = parse_state(os.path.join(save_dir, fname), geo)
else:
    parts = [parse_state(os.path.join(save_dir, fn), geo) for fn in fnames]
    states = {k: torch.cat([p[0][k] for p in parts]) for k in parts[0][0]}
    n_cells = sum(p[1] for p in parts)
print(f"parsed {n_cells} cells x {len(geo['kv_layers'])} KV layers of {geo['n_layer']}; K row norm mean {float(states[(geo['kv_layers'][0],'k')].norm(dim=1).mean()):.2f}", flush=True)
H, d = geo["n_head_kv"], geo["head_dim"]
bph = a.codes * GGML_BLOCK_BYTES[a.quant] * 8 // 32
books = K.fit_kv_codebooks_perhead(states, bph, H, d, quant=a.quant, codes=a.codes, whiten=a.whiten, whiten_power=a.whiten_power,
                                   progress=lambda i, n, key: (i % 16 == 0 and print(f"  fitted {i+1}/{n} units", flush=True)))
meta = {"model": os.path.basename(a.gguf), "arch": geo["arch"], "basis": "postrope", "per_head": True,
        "quant": a.quant, "codes_per_head": a.codes, "bits_per_head": bph, "unit_bits": H * bph, "whiten": bool(a.whiten), "whiten_power": a.whiten_power if a.whiten else 0.0,
        "kv_heads": H, "head_dim": d, "n_head": geo["n_head"], "n_layers": geo["n_layer"],
        "n_states": n_cells, "corpus_sha256": corpus_digest(text), "captured_from": "llama.cpp state file",
        "capture_window": int(a.window)}
cb = KVCodebook(books=books, meta=meta); cb.save(a.out)
print(f"wrote {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB)")
out_gguf = a.out.replace(".kvcb.npz", ".cpca.gguf")
os.system(f"{sys.executable} {os.path.join(HERE, 'export_cpca_gguf.py')} {a.out} --n-head {geo['n_head']} --model-gguf {a.gguf} --fp16 -o {out_gguf}")
