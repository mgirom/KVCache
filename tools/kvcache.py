#!/usr/bin/env python3
"""kvcache: pull a fitted codebook, serve a model with it, audit the result.

  python3 tools/kvcache.py list
  python3 tools/kvcache.py pull bonsai-8b                      # codebook -> ~/.kvcache/codebooks/
  python3 tools/kvcache.py serve bonsai-8b --model /path/to/Ternary-Bonsai-8B-Q2_0_g64.gguf [-- any llama-server options]
  python3 tools/kvcache.py audit bonsai-8b --model /path/to/... [--profile quick]

The model file comes from its publisher; the registry records the sha256 of the exact
file each row was measured on, and `serve` checks it before starting. The server is the
patched llama.cpp (mscc/llamacpp/); point KVCACHE_LLAMA_SERVER at its llama-server
binary or put it on PATH.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.join(os.path.expanduser("~"), ".kvcache", "codebooks")

def registry():
    return json.load(open(os.path.join(ROOT, "registry", "models.json")))

def entry(reg, mid):
    for m in reg["models"]:
        if m["id"] == mid:
            return m
    sys.exit(f"unknown model id {mid!r}; try: list")

def sha256(path, bar=True):
    h = hashlib.sha256(); n = 0; size = os.path.getsize(path)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk); n += len(chunk)
            if bar and size > (1 << 28):
                print(f"\r  hashing {n/size:5.1%}", end="", file=sys.stderr, flush=True)
    if bar and size > (1 << 28): print("", file=sys.stderr)
    return h.hexdigest()

def server_binary():
    b = os.environ.get("KVCACHE_LLAMA_SERVER") or shutil.which("llama-server")
    if not b:
        sys.exit("no llama-server found: build the patched llama.cpp (mscc/llamacpp/README.md) and set KVCACHE_LLAMA_SERVER")
    out = subprocess.run([b, "--help"], capture_output=True, text=True).stdout + subprocess.run([b, "--help"], capture_output=True, text=True).stderr
    if "--kv-codebook" not in out:
        sys.exit(f"{b} is not the patched llama.cpp (no --kv-codebook flag)")
    return b

def cmd_list(a):
    reg = registry()
    print(f"llama.cpp {reg['llama_cpp_commit']} + {reg['patches']}")
    for m in reg["models"]:
        print(f"  {m['id']:<12} {m['weights']:<38} codebook {m['codebook_bytes']/1e6:5.1f} MB  " + "; ".join(f"{k}: {v}" for k, v in m["measured"].items()))

def pull(m, reg):
    os.makedirs(HOME, exist_ok=True)
    dst = os.path.join(HOME, m["codebook_file"])
    if os.path.exists(dst) and sha256(dst, bar=False) == m["codebook_sha256"]:
        return dst
    url = reg["codebook_base_url"] + m["codebook_file"]
    print(f"  fetching {url}", file=sys.stderr)
    urllib.request.urlretrieve(url, dst + ".part")
    got = sha256(dst + ".part", bar=False)
    if got != m["codebook_sha256"]:
        os.remove(dst + ".part"); sys.exit(f"codebook sha256 mismatch: got {got[:12]}, registry says {m['codebook_sha256'][:12]}")
    os.replace(dst + ".part", dst)
    return dst

def cmd_pull(a):
    reg = registry(); m = entry(reg, a.id); print(pull(m, reg))

def check_model(m, path):
    if not os.path.exists(path): sys.exit(f"model file not found: {path}")
    got = sha256(path)
    if got != m["model_sha256"]:
        print(f"warning: this is not the exact file the rows were measured on (sha256 {got[:12]} vs {m['model_sha256'][:12]}); "
              "the codebook is bound by architecture and geometry, so it may still load", file=sys.stderr)

def server_args(m, cb, a):
    r = m["recommended"]
    return ["-m", a.model, "-ctk", a.ctk or r["ctk"], "-ctv", a.ctv or r["ctv"], "-fa", r["flash_attn"], "--kv-codebook", cb]

def cmd_serve(a):
    reg = registry(); m = entry(reg, a.id); check_model(m, a.model); cb = pull(m, reg); b = server_binary()
    extra = [x for x in a.extra if x != "--"]                         # unknown args go to llama-server
    cmd = [b] + server_args(m, cb, a) + ["-c", str(a.ctx), "--host", a.host, "--port", str(a.port)] + extra
    print("  " + " ".join(cmd), file=sys.stderr)
    os.execv(b, cmd)

def cmd_audit(a):
    reg = registry(); m = entry(reg, a.id); check_model(m, a.model); cb = pull(m, reg); b = server_binary()
    out = a.out or f"kvcache-audit-{a.id}.json"
    cmd = [sys.executable, os.path.join(ROOT, "auditor", "runner", "run.py"), "--backend", "llamacpp", "--binary", b, "--model", a.model,
           "--codebook", cb, "--arms", a.arms, "--profile", a.profile, "--contexts", a.contexts, "-o", out]
    print("  " + " ".join(cmd), file=sys.stderr); sys.exit(subprocess.call(cmd))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest="cmd", required=True)
sub.add_parser("list").set_defaults(fn=cmd_list)
p = sub.add_parser("pull"); p.add_argument("id"); p.set_defaults(fn=cmd_pull)
for name, fn in (("serve", cmd_serve), ("audit", cmd_audit)):
    p = sub.add_parser(name); p.add_argument("id"); p.add_argument("--model", required=True); p.add_argument("--ctk"); p.add_argument("--ctv")
    if name == "serve":
        p.add_argument("--ctx", type=int, default=8192); p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8080)
    else:
        p.add_argument("--arms", default="q4_0,q4_0+cpca"); p.add_argument("--profile", default="quick"); p.add_argument("--contexts", default="1024,4096"); p.add_argument("--out", default="")
    p.set_defaults(fn=fn)
a, extra = ap.parse_known_args()
if extra and a.cmd != "serve":
    ap.error("unrecognized arguments: " + " ".join(extra))
a.extra = extra                                                      # serve: passed through to llama-server (e.g. -- -ngl 0)
a.fn(a)
