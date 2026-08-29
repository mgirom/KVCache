#!/usr/bin/env python3
"""Item 3 -- the agent-to-agent handoff demo, openable from a phone on the LAN.

    Sender    reads a document, writes a full-depth KV frame.
    Receiver  loads the frame THROUGH THE GUARD, never sees the text,
                      and answers questions about it.

Gate 10 declared this demo "not buildable on a single-depth frame" -- the receiver
could not attend to the document below the tap, so it invented answers. Gate 12
replaced the frame with a compressed full-depth KV cache and the receiver now runs
ZERO layers over the document and gets the answer right.

WHY THE RECEIVER IS A SUBPROCESS
--------------------------------
"the receiver never sees the document" is a claim if sender and receiver are two functions in
one process, and a fact if they are two processes and one of them is only ever
handed a file path. This server holds no model and no plaintext during the answer:
it shells out to the shipped CLI, `mscc kvserve --frame ... --ask ...`, which
receives the frame path, the codebook path and the question, and nothing else. That
also means the demo exercises the real product surface and the real guard, and each
GPU step takes the tree's lock in turn, so the "one job on the card" rule holds.

The cost is a model load per step (~10 s). That is visible in the timings, and it is
honest: it is the price of the isolation the demo is claiming.

Run:   python3 demo/handoff_server.py [--bind 0.0.0.0] [--port 8093]
Then:  http://<this-box>:8093/
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "demo" / "run"
MODEL = ROOT / "models" / "qwen3-1.7b-fp"
CB_GOOD = ROOT / "mscc" / "accept" / "kv" / "book.kvcb.npz"
CB_NARROW = ROOT / "mscc" / "accept" / "kv" / "narrow.kvcb.npz"
PY = sys.executable

SAMPLE = """Field report -- Dunraven relay station, week 34.

The relief crew arrived on Tuesday with supplies for the month. Weather was clear
all week and the swell stayed under two metres, so the tender was able to come
alongside on the first attempt for the first time since spring.

The maintenance access code for the Dunraven lighthouse is 47-19-83. It was reset
this week after the old code appeared in a contractor's printed handover pack.

Fuel state at the end of the week: 1,340 litres in the main tank and 210 in the
day tank. The generator ran 41 hours, mostly overnight. The optic drive motor is
still drawing more current than the log says it should and is scheduled for
replacement in the spring window.
"""


def sh(args, timeout=900):
    """Run a subprocess with an argument LIST. Never shell=True: the document text
    and the question are attacker-controlled in any LAN-exposed demo."""
    t0 = time.perf_counter()
    p = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True,
                       timeout=timeout)
    return p, time.perf_counter() - t0


def last_json(text):
    """The CLI prints a human preamble then a JSON block. Take the JSON."""
    i = text.rfind("{")
    while i != -1:
        try:
            return json.loads(text[i:])
        except json.JSONDecodeError:
            i = text.rfind("{", 0, i)
    return None


# --------------------------------------------------------------------- sender

def encode(text: str) -> dict:
    RUN.mkdir(parents=True, exist_ok=True)
    doc = RUN / "doc.txt"
    doc.write_text(text, encoding="utf-8")
    frame = RUN / "doc.kvf"
    p, secs = sh([PY, "-m", "mscc.cli", "kvencode", str(doc), "--model", str(MODEL),
                  "--codebook", str(CB_GOOD), "--sink", "4", "-o", str(frame)])
    if p.returncode != 0:
        return {"ok": False, "error": (p.stderr or p.stdout)[-1200:]}
    j = last_json(p.stdout) or {}
    j.update({"ok": True, "wall_seconds": round(secs, 1),
              "frame_path": str(frame.relative_to(ROOT)),
              "narrow_ready": False})
    # the poisoned twin is built lazily, on demand
    for stale in ("doc.narrow.kvf", "doc.wrongmodel.kvf"):
        (RUN / stale).unlink(missing_ok=True)
    return j


def build_poison(kind: str) -> tuple[Path, Path] | None:
    """Return (frame, codebook) for a deliberately broken handoff."""
    src = RUN / "doc.kvf"
    if not src.exists():
        return None
    if kind == "narrow":
        out = RUN / "doc.narrow.kvf"
        if not out.exists():
            p, _ = sh([PY, "-m", "mscc.cli", "kvencode", str(RUN / "doc.txt"),
                       "--model", str(MODEL), "--codebook", str(CB_NARROW),
                       "--sink", "4", "-o", str(out)])
            if p.returncode != 0:
                return None
        return out, CB_NARROW
    if kind == "wrongmodel":
        out = RUN / "doc.wrongmodel.kvf"
        if not out.exists():
            sys.path[:0] = [str(ROOT)]
            from mscc import format as mfmt
            fr = mfmt.read_frame(str(src))
            h = mfmt.FrameHeader.from_dict(json.loads(fr.header.to_json()))
            h.model_sha = "0" * 64
            h.model_id = "some-other-1.7b (a fine-tune of the same family)"
            mfmt.write_frame(str(out), h, fr.payload)
        return out, CB_GOOD
    return None


# ------------------------------------------------------------------- receiver

BLOCKS = re.compile(r"blocks executed over the document: (\d+) of (\d+)")
DECODE = re.compile(r"frame decode: ([\d.]+) ms\s+generation: ([\d.]+) ms")


#: What each staged failure does and does NOT demonstrate. The wrong-model case is
#: the one that needs a caveat: forging the sha in the header does not change which
#: model actually produced the frame, so the forced answer stays right. The case the
#: check really exists for -- a fine-tune of the same family, which produces fluent
#: WRONG text -- needs a second set of weights this box does not have.
POISON_NOTE = {
    "narrow": "The codec really is below the measured floor here, so the forced "
              "answer really is wrong. This is the failure the floor prevents.",
    "wrongmodel": "Only the header's model fingerprint was forged -- the frame was "
                  "still produced by this model, so the forced answer stays correct. "
                  "What this shows is the identity check firing. The dangerous case "
                  "it exists for is a fine-tune of the same family, whose frames "
                  "decode to fluent WRONG text; staging that needs a second set of "
                  "weights, which this box does not have.",
}


def ask(question: str, poison: str = "") -> dict:
    frame, cb = (RUN / "doc.kvf", CB_GOOD)
    if poison:
        got = build_poison(poison)
        if not got:
            return {"ok": False, "error": f"could not build the {poison} frame"}
        frame, cb = got
    if not frame.exists():
        return {"ok": False, "error": "no frame yet -- send a document first"}

    args = [PY, "-m", "mscc.cli", "kvserve", "--frame", str(frame), "--model",
            str(MODEL), "--codebook", str(cb), "--ask",
            f"\n\nQuestion: {question}\nAnswer:", "--gen", "48"]
    if poison:
        # show what refusal prevents, rather than only that it happened
        args.append("--force")
    p, secs = sh(args)
    out = p.stdout
    guard = out.split("\n\nblocks executed")[0].strip()
    refused = "guard: REJECT" in guard
    answer = out.split("A:", 1)[1].strip() if "A:" in out else ""
    b = BLOCKS.search(out)
    d = DECODE.search(out)
    return {"ok": True, "refused": refused, "forced": bool(poison) and refused,
            "guard": guard, "answer": answer, "note": POISON_NOTE.get(poison, ""),
            "blocks_run": int(b.group(1)) if b else None,
            "blocks_total": int(b.group(2)) if b else None,
            "decode_ms": float(d.group(1)) if d else None,
            "gen_ms": float(d.group(2)) if d else None,
            "wall_seconds": round(secs, 1),
            "stderr": p.stderr[-400:] if p.stderr else ""}


# ------------------------------------------------------------------------ page

# NOTE: PAGE is a plain str, so any backslash escape meant for the BROWSER
# must be doubled here. A bare \n inside a JS string literal below becomes a
# real newline and takes the whole <script> down with a parse error.
PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MSCC handoff -- sender to receiver</title>
<style>
:root{--bg:#fbfaf8;--fg:#1a1a18;--dim:#6b6862;--line:#e0ddd6;--card:#fff;
      --ok:#1a7f4b;--bad:#b3261e;--accent:#2f5f8f}
@media (prefers-color-scheme:dark){:root{--bg:#16161a;--fg:#e9e7e2;--dim:#9a968e;
      --line:#2e2e34;--card:#1d1d22;--ok:#5cc98d;--bad:#ef6b62;--accent:#7fb0e0}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 -apple-system,
     BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;padding:16px}
.wrap{max-width:820px;margin:0 auto}
h1{font-size:1.35rem;margin:0 0 2px}
.sub{color:var(--dim);font-size:.9rem;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
      padding:14px 16px;margin-bottom:14px}
.card h2{font-size:.78rem;letter-spacing:.09em;text-transform:uppercase;
      color:var(--dim);margin:0 0 10px;font-weight:600}
textarea,input{width:100%;font:inherit;color:var(--fg);background:var(--bg);
      border:1px solid var(--line);border-radius:8px;padding:9px 11px}
textarea{min-height:150px;font-size:.9rem;resize:vertical}
button{font:inherit;font-weight:600;border:0;border-radius:8px;padding:10px 16px;
      background:var(--accent);color:#fff;cursor:pointer;margin-top:10px}
button.ghost{background:transparent;color:var(--bad);border:1px solid var(--bad)}
button:disabled{opacity:.5;cursor:progress}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px}
.stat{border:1px solid var(--line);border-radius:8px;padding:8px 10px}
.stat b{display:block;font-size:1.15rem;font-variant-numeric:tabular-nums}
.stat span{color:var(--dim);font-size:.72rem;text-transform:uppercase;
      letter-spacing:.06em}
pre{white-space:pre-wrap;word-break:break-word;background:var(--bg);
      border:1px solid var(--line);border-radius:8px;padding:10px;
      font-size:.82rem;margin:0;overflow-x:auto}
.verdict{font-weight:700}.pass{color:var(--ok)}.reject{color:var(--bad)}
.answer{font-size:1.05rem;border-left:3px solid var(--accent);padding-left:12px;
      margin:10px 0}
.answer.bad{border-left-color:var(--bad)}
.note{color:var(--dim);font-size:.82rem;margin-top:8px}
.hide{display:none}
</style></head><body><div class="wrap">
<h1>Document handoff &mdash; sender &rarr; receiver</h1>
<div class="sub">The sender reads the document once and writes a full-depth KV frame.
The receiver is a separate process: it is handed the frame and the question, never the text,
and runs <b>zero</b> layers over the document.</div>

<div class="card"><h2>1 &middot; the sender reads the document</h2>
<textarea id="doc"></textarea>
<button id="enc">Read &amp; frame it</button>
<div id="encout" class="hide"></div></div>

<div class="card"><h2>2 &middot; the receiver answers, having never seen the text</h2>
<input id="q" value="What is the maintenance access code for the Dunraven lighthouse?">
<div class="row">
  <button id="askb">Ask the receiver</button>
  <button id="poison1" class="ghost">Poisoned: narrow codec</button>
  <button id="poison2" class="ghost">Poisoned: forged model identity</button>
</div>
<div class="note">The poisoned buttons hand the receiver a frame that violates a measured
condition, and then <b>override the guard</b> so you can see what refusal prevents:
fluent, confident, wrong.</div>
<div id="askout" class="hide"></div></div>
</div><script>
const $=i=>document.getElementById(i);
$('doc').value=SAMPLE_TEXT;
const fmt=n=>n==null?'--':n.toLocaleString();
const esc=s=>(s||'').replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));
function busy(on,el){document.querySelectorAll('button').forEach(b=>b.disabled=on);
  if(on&&el){el.classList.remove('hide');el.innerHTML='<p class="note">running on the GPU (loads the model, takes the lock) &hellip;</p>';}}
async function post(u,b){const r=await fetch(u,{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});return r.json();}

$('enc').onclick=async()=>{const o=$('encout');busy(true,o);
  const j=await post('/api/encode',{text:$('doc').value});busy(false);
  if(!j.ok){o.innerHTML='<pre>'+j.error+'</pre>';return;}
  o.innerHTML=`<div class="stats">
   <div class="stat"><b>${fmt(j.tokens)}</b><span>document tokens</span></div>
   <div class="stat"><b>${(j.bytes/1048576).toFixed(1)} MB</b><span>frame on disk</span></div>
   <div class="stat"><b>${(j.uncompressed_kv_bytes/1048576).toFixed(0)} MB</b><span>uncompressed KV</span></div>
   <div class="stat"><b>${j.compression_vs_bf16}&times;</b><span>vs bf16 KV</span></div>
   <div class="stat"><b>${j.read_seconds}s</b><span>sender read time</span></div>
   <div class="stat"><b>${fmt(j.text_bytes)} B</b><span>the text itself</span></div></div>
   <p class="note">The frame is far bigger than the text &mdash; that is the trade.
   Inside one operator the wire is loopback; what is bought is the receiver's prefill.</p>`;};

async function doAsk(poison,label){const o=$('askout');busy(true,o);
  const j=await post('/api/ask',{question:$('q').value,poison:poison});busy(false);
  if(!j.ok){o.innerHTML='<pre>'+j.error+'</pre>';return;}
  const rej=j.refused;
  o.innerHTML=`<p class="verdict ${rej?'reject':'pass'}">${label} &mdash;
     guard says ${rej?'REJECT':'PASS'}</p>
   <pre>${esc(j.guard)}</pre>
   ${rej?'<p class="note"><b>In production this is where it stops.</b> The answer below only exists because the guard was overridden, to show what it prevents.</p>':''}
   <div class="answer ${rej?'bad':''}">${esc(j.answer.split('\\n\\n')[0])||'(refused)'}</div>
   ${j.note?'<p class="note">'+esc(j.note)+'</p>':''}
   <div class="stats">
    <div class="stat"><b>${j.blocks_run}/${j.blocks_total}</b><span>blocks run over doc</span></div>
    <div class="stat"><b>${j.decode_ms?j.decode_ms.toFixed(0):'--'} ms</b><span>frame decode</span></div>
    <div class="stat"><b>${j.gen_ms?j.gen_ms.toFixed(0):'--'} ms</b><span>generation</span></div>
    <div class="stat"><b>${j.wall_seconds}s</b><span>wall, incl. model load</span></div></div>`;}

$('askb').onclick=()=>doAsk('','Honest handoff');
$('poison1').onclick=()=>doAsk('narrow','Poisoned: 256 bits/unit, below the measured floor');
$('poison2').onclick=()=>doAsk('wrongmodel','Poisoned: forged model fingerprint');
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"[{self.address_string()}] {fmt % args}", flush=True)

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            return self._send(404, b"not found", "text/plain")
        page = PAGE.replace("SAMPLE_TEXT", json.dumps(SAMPLE))
        self._send(200, page.encode(), "text/html; charset=utf-8")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 2_000_000:
            return self._send(413, b'{"ok":false,"error":"document too large"}')
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, b'{"ok":false,"error":"bad json"}')
        try:
            if self.path == "/api/encode":
                out = encode(str(req.get("text", ""))[:200_000])
            elif self.path == "/api/ask":
                poison = str(req.get("poison", ""))
                if poison not in ("", "narrow", "wrongmodel"):
                    poison = ""
                out = ask(str(req.get("question", ""))[:2000], poison)
            else:
                return self._send(404, b'{"ok":false,"error":"no such endpoint"}')
        except subprocess.TimeoutExpired:
            out = {"ok": False, "error": "the GPU step timed out"}
        except Exception as e:                                  # noqa: BLE001
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self._send(200, json.dumps(out).encode())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind", default="0.0.0.0",
                    help="0.0.0.0 makes it reachable from a phone on the LAN, which "
                         "is what Item 3 asks for. Use 127.0.0.1 to keep it local.")
    ap.add_argument("--port", type=int, default=8093)
    a = ap.parse_args()

    for p, what in ((MODEL, "model"), (CB_GOOD, "codebook"), (CB_NARROW, "narrow codebook")):
        if not p.exists():
            print(f"missing {what}: {p}\n"
                  f"build it with:  python3 -m mscc.cli kvfit --corpus <text> "
                  f"--model {MODEL} -o {p}", file=sys.stderr)
            return 2
    RUN.mkdir(parents=True, exist_ok=True)
    if a.bind == "0.0.0.0":
        print("!! bound to 0.0.0.0 with no authentication. Anything on the LAN can\n"
              "!! submit documents and start GPU jobs on this box. Item 3 asks for\n"
              "!! phone access; use --bind 127.0.0.1 when you do not need it.",
              file=sys.stderr)
    print(f"handoff demo on http://{a.bind}:{a.port}/  (Ctrl-C to stop)", flush=True)
    ThreadingHTTPServer((a.bind, a.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
