#!/usr/bin/env python3
"""Submission storage. Files in a directory, not rows in a database, on purpose.

WHY FILES
---------
For a benchmark specifically, the storage medium is part of the credibility argument:

- **The audit trail is the storage.** Kept in a git repo, the history is tamper
  evidence nobody had to build. You can see when a row appeared and that it has not
  been edited since.
- **Anyone can recompute the aggregates.** The leaderboard stops being "trust my SQL"
  and becomes "here are the rows and here is the script". That is the difference
  between a scoreboard and a source.
- **Nothing to operate.** No database to back up, migrate, or leak.

This is not a claim that files beat a database in general. It is a claim that at the
scale this needs to work -- tens of thousands of submissions -- the audit and
reproducibility properties are worth more than query speed, and a database can be
built from these files later without losing anything. The reverse is not true.

DELETION
--------
PRIVACY.md promises a submitter can remove their own row without proving who they are.
That is implemented as a capability token: the server stores only a SHA-256 of it, so
possession of the token is the only way to delete, and the stored digest identifies
nobody. Losing the token means the row cannot be attributed back to the submitter by us
either -- which is the same property that makes it non-identifying.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import time

RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class StoreError(RuntimeError):
    pass


class Store:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    # ------------------------------------------------------------------ layout
    def _dir_for(self, utc: str) -> str:
        # YYYY/MM from the submission's own timestamp, so the tree stays browsable
        m = re.match(r"^(\d{4})-(\d{2})", utc or "")
        y, mo = (m.group(1), m.group(2)) if m else ("unknown", "unknown")
        d = os.path.join(self.root, y, mo)
        os.makedirs(d, exist_ok=True)
        return d

    def path_for(self, run_id: str) -> str | None:
        for dirpath, _, files in os.walk(self.root):
            if f"{run_id}.json" in files:
                return os.path.join(dirpath, f"{run_id}.json")
        return None

    # ------------------------------------------------------------------- write
    def put(self, doc: dict, trust: str = "unverified") -> dict:
        run_id = doc.get("run_id", "")
        if not RUN_ID_RE.match(run_id):
            raise StoreError("run_id must be 32 lowercase hex characters")
        if self.path_for(run_id):
            raise StoreError(f"run_id {run_id} already exists; a run is submitted once")

        token = secrets.token_urlsafe(32)
        doc = json.loads(json.dumps(doc))          # never mutate the caller's object
        doc.setdefault("integrity", {})["trust"] = trust
        doc["_server"] = {
            "received_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            # only the digest is kept: possession of the token is the capability, and
            # the digest on its own identifies nobody
            "deletion_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        }
        path = os.path.join(self._dir_for(doc.get("utc", "")), f"{run_id}.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)                      # atomic: no half-written rows
        return {"ok": True, "run_id": run_id, "deletion_token": token,
                "trust": trust, "path": os.path.relpath(path, self.root)}

    def delete(self, run_id: str, token: str) -> bool:
        path = self.path_for(run_id)
        if not path:
            return False
        doc = json.load(open(path))
        want = doc.get("_server", {}).get("deletion_token_sha256", "")
        got = hashlib.sha256((token or "").encode()).hexdigest()
        if not want or not secrets.compare_digest(want, got):
            raise StoreError("deletion token does not match")
        os.remove(path)
        return True

    # -------------------------------------------------------------------- read
    def all(self):
        for dirpath, _, files in os.walk(self.root):
            for fn in sorted(files):
                if fn.endswith(".json") and not fn.endswith(".tmp"):
                    try:
                        yield json.load(open(os.path.join(dirpath, fn)))
                    except (json.JSONDecodeError, OSError):
                        continue

    def count(self) -> int:
        return sum(1 for _ in self.all())


# ------------------------------------------------------------ plausibility screen
#
# Full anti-cheat is impossible for a tool that runs on the submitter's machine, and
# pretending otherwise would be worse than saying so. What IS achievable is noticing
# that a row disagrees with physics or with everything else in the store. Flagged rows
# are marked, never silently dropped and never silently trusted.

def screen(doc: dict, peers: list | None = None) -> list[str]:
    """Reasons to doubt this row. Empty means nothing looked wrong."""
    notes = []
    sysd = doc.get("system", {})
    backend = sysd.get("backend", "")
    model = doc.get("workload", {}).get("model", {})
    n_layers = model.get("n_layers", 0)
    kv_heads = model.get("n_kv_heads", 0)
    head_dim = model.get("head_dim", 0)

    for arm in [doc.get("reference", {})] + doc.get("arms", []):
        for r in arm.get("rungs", []):
            if not r.get("ran"):
                continue
            cost = r.get("cost", {})
            tps = cost.get("decode_tok_per_s", 0)
            if tps <= 0:
                notes.append(f"{arm.get('name')}@{r['context']}: decode rate is zero")
            elif backend == "cpu" and tps > 400:
                notes.append(f"{arm.get('name')}@{r['context']}: {tps:.0f} tok/s on a "
                             "CPU backend is implausibly fast")
            elif tps > 20000:
                notes.append(f"{arm.get('name')}@{r['context']}: {tps:.0f} tok/s "
                             "exceeds anything this workload can produce")

            # the cache cannot be smaller than its own geometry allows
            bpt = cost.get("kv_bytes_per_token_measured", 0)
            if bpt and n_layers and kv_heads and head_dim:
                floor_bits = n_layers * 2 * kv_heads * head_dim * 1.0   # 1 bit/element
                if bpt * 8 < floor_bits * 0.9:
                    notes.append(
                        f"{arm.get('name')}@{r['context']}: {bpt:,.0f} B/token is below "
                        "one bit per cache element for this geometry")

    if doc.get("workload", {}).get("model", {}).get("n_layers", 0) <= 0:
        notes.append("model geometry was not read; cache figures cannot be checked")

    # agreement with peers on the same (workload, model, arm) -- quality is nearly
    # invariant across machines even though per-item outcomes are not, so a row far
    # from its peers is worth a look
    if peers:
        for arm in doc.get("arms", []):
            mine = _pooled(arm)
            same = [_pooled(a) for p in peers for a in p.get("arms", [])
                    if a.get("name") == arm.get("name")
                    and p.get("workload", {}).get("sha256") == doc["workload"]["sha256"]
                    and p.get("workload", {}).get("model", {}).get("sha256")
                    == model.get("sha256")]
            same = [x for x in same if x is not None]
            if mine is not None and len(same) >= 3:
                med = sorted(same)[len(same) // 2]
                if abs(mine - med) > 0.15:
                    notes.append(f"arm {arm.get('name')}: {mine:.3f} against a peer "
                                 f"median of {med:.3f} over {len(same)} rows")
    return notes


def _pooled(arm: dict):
    h = n = 0
    for r in arm.get("rungs", []):
        if r.get("ran"):
            ts = r.get("quality", {}).get("task_success", {}).get("overall", {})
            h += ts.get("hits", 0)
            n += ts.get("n", 0)
    return h / n if n else None
