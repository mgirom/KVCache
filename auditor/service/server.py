#!/usr/bin/env python3
"""The submission service. Accept, validate, screen, store, answer queries.

Deliberately stdlib-only and deliberately small. The hard part of this service was
done before it was written: `validate.py` already decides what is admissible, so the
server is mostly plumbing plus the two things a server must own -- refusing bad input
and never trusting the client about anything it can check itself.

    python3 auditor/service/server.py --root submissions --port 8095

Endpoints
    POST   /api/v1/submission            a run, as produced by the runner
    DELETE /api/v1/submission/<run_id>   with X-Deletion-Token
    GET    /api/v1/query?...             the join: hardware x model x method
    GET    /api/v1/stats                 what is in the store
    GET    /                             the generated site
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE)),
                os.path.join(os.path.dirname(HERE), "runner")]

import validate as V                                    # noqa: E402
from store import Store, StoreError, screen             # noqa: E402

MAX_BODY = 8 << 20          # a submission is tens of KB; this is generous
STORE: Store | None = None


def pooled(arm):
    h = n = 0
    for r in arm.get("rungs", []):
        if r.get("ran"):
            ts = r.get("quality", {}).get("task_success", {}).get("overall", {})
            h += ts.get("hits", 0)
            n += ts.get("n", 0)
    return h, n


TRUST_ORDER = {"unverified": 0, "plausible": 1, "reproduced": 2}


def query(store: Store, gpu="", model="", backend="", method="", context=0,
          min_trust="plausible"):
    """The join the whole thing exists for: given this hardware and this model, what
    does each method cost? Rows are grouped, never averaged across models or
    workloads -- those are different measurements wearing the same units."""
    groups: dict[tuple, dict] = {}
    excluded_by_trust = 0
    floor = TRUST_ORDER.get(min_trust, 1)
    for doc in store.all():
        # Rows that failed plausibility screening are NOT pooled into an aggregate by
        # default. They are not hidden either -- the count is reported, and
        # min_trust=unverified returns them. Silently averaging a flagged row into a
        # headline is how one bad submission moves a recommendation.
        if TRUST_ORDER.get(doc.get("integrity", {}).get("trust", "unverified"), 0) < floor:
            excluded_by_trust += 1
            continue
        s = doc.get("system", {})
        m = doc.get("workload", {}).get("model", {})
        if gpu and gpu.lower() not in (s.get("gpu_model", "") or "").lower():
            continue
        if backend and backend != s.get("backend"):
            continue
        if model and model.lower() not in (m.get("name", "") or "").lower():
            continue
        ref_h, ref_n = pooled(doc.get("reference", {}))
        for arm in doc.get("arms", []):
            if method and arm.get("name") != method:
                continue
            for r in arm.get("rungs", []):
                if not r.get("ran"):
                    continue
                if context and r["context"] != context:
                    continue
                key = (s.get("gpu_model") or s.get("cpu_model", "?"),
                       s.get("backend", "?"), m.get("sha256", "?")[:12],
                       arm["name"], r["context"])
                g = groups.setdefault(key, {
                    "hardware": key[0], "backend": key[1], "model": m.get("name", "?"),
                    "model_sha": key[2], "method": key[3], "context": key[4],
                    "n_submissions": 0, "hits": 0, "n": 0,
                    "ref_hits": 0, "ref_n": 0,
                    "kv_bytes_per_token": [], "decode_tok_per_s": [],
                    "prefill_ms": [], "trust": set()})
                ts = r["quality"]["task_success"]["overall"]
                g["n_submissions"] += 1
                g["hits"] += ts["hits"]
                g["n"] += ts["n"]
                g["ref_hits"] += ref_h
                g["ref_n"] += ref_n
                g["kv_bytes_per_token"].append(r["cost"]["kv_bytes_per_token_measured"])
                g["decode_tok_per_s"].append(r["cost"]["decode_tok_per_s"])
                g["prefill_ms"].append(r["cost"]["prefill_ms"])
                g["trust"].add(doc.get("integrity", {}).get("trust", "unverified"))

    out = []
    for g in groups.values():
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None      # noqa: E731
        lo, hi = V.wilson_upper, None
        rate = g["hits"] / g["n"] if g["n"] else None
        ref_rate = g["ref_hits"] / g["ref_n"] if g["ref_n"] else None
        out.append({
            "hardware": g["hardware"], "backend": g["backend"],
            "model": g["model"], "model_sha": g["model_sha"],
            "method": g["method"], "context": g["context"],
            "n_submissions": g["n_submissions"], "n_items": g["n"],
            "task_success": round(rate, 4) if rate is not None else None,
            "reference_task_success": round(ref_rate, 4) if ref_rate is not None else None,
            # the number that matters: the delta against that machine's own reference
            "delta_vs_reference": (round(rate - ref_rate, 4)
                                   if rate is not None and ref_rate is not None else None),
            "kv_bytes_per_token": med(g["kv_bytes_per_token"]),
            "decode_tok_per_s": med(g["decode_tok_per_s"]),
            "prefill_ms": med(g["prefill_ms"]),
            "trust": sorted(g["trust"]),
        })
    out.sort(key=lambda r: (r["model"], r["context"], r["method"]))
    return {"rows": out, "excluded_by_trust": excluded_by_trust,
            "min_trust": min_trust}


def stats(store: Store):
    docs = list(store.all())
    models, hw, backends = set(), set(), set()
    for d in docs:
        models.add(d.get("workload", {}).get("model", {}).get("name", "?"))
        s = d.get("system", {})
        hw.add(s.get("gpu_model") or s.get("cpu_model", "?"))
        backends.add(s.get("backend", "?"))
    return {"submissions": len(docs), "models": sorted(models),
            "hardware": sorted(hw), "backends": sorted(backends)}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "kv-audit-service/0.1"

    def log_message(self, fmt, *args):
        print(f"[{self.address_string()}] {fmt % args}", flush=True)

    def _send(self, code, obj, ctype="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj, indent=1).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------- routes
    def do_POST(self):
        if self.path.rstrip("/") != "/api/v1/submission":
            return self._send(404, {"ok": False, "error": "no such endpoint"})
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            return self._send(413, {"ok": False, "error": "submission too large"})
        try:
            doc = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"ok": False, "error": "body is not valid JSON"})

        # 1. admissibility: the same rules the runner checks itself against
        errs = V.validate(doc)
        if errs:
            return self._send(422, {"ok": False, "error": "submission rejected",
                                    "reasons": errs})
        # 2. integrity: recompute rather than believe the client's hash
        want = doc.get("integrity", {}).get("result_hash", "")
        got = V.canonical_hash(doc)
        if want and want != got:
            return self._send(422, {"ok": False,
                                    "error": "result_hash does not match the content"})
        # 3. plausibility: flag, never silently trust or silently drop
        peers = list(STORE.all())
        notes = screen(doc, peers)
        trust = "unverified" if notes else "plausible"
        try:
            res = STORE.put(doc, trust=trust)
        except StoreError as e:
            return self._send(409, {"ok": False, "error": str(e)})
        res["screen_notes"] = notes
        return self._send(201, res)

    def do_DELETE(self):
        parts = self.path.strip("/").split("/")
        if len(parts) != 4 or parts[:3] != ["api", "v1", "submission"]:
            return self._send(404, {"ok": False, "error": "no such endpoint"})
        token = self.headers.get("X-Deletion-Token", "")
        try:
            gone = STORE.delete(parts[3], token)
        except StoreError as e:
            return self._send(403, {"ok": False, "error": str(e)})
        return self._send(200 if gone else 404, {"ok": gone})

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        if u.path.rstrip("/") == "/api/v1/query":
            return self._send(200, {"ok": True, **query(
                STORE, gpu=q.get("gpu", ""), model=q.get("model", ""),
                backend=q.get("backend", ""), method=q.get("method", ""),
                context=int(q.get("context", 0) or 0),
                min_trust=q.get("min_trust", "plausible"))})
        if u.path.rstrip("/") == "/api/v1/stats":
            return self._send(200, {"ok": True, **stats(STORE)})
        if u.path in ("/", "/index.html"):
            import site_gen
            return self._send(200, site_gen.render(list(STORE.all())).encode(),
                              "text/html; charset=utf-8")
        return self._send(404, {"ok": False, "error": "no such endpoint"})


def main():
    global STORE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="submissions")
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8095)
    a = ap.parse_args()
    STORE = Store(a.root)
    print(f"store {os.path.abspath(a.root)}  ({STORE.count()} submissions)")
    if a.bind == "0.0.0.0":
        print("!! bound to 0.0.0.0 with no authentication. Anything that can reach "
              "this port can submit rows.", file=sys.stderr)
    print(f"listening on http://{a.bind}:{a.port}/", flush=True)
    ThreadingHTTPServer((a.bind, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
