#!/usr/bin/env python3
"""Rebuild a submission from its per-item record under the current scoring rules.

Scoring rules change -- the tier-exclusion rule just did. Re-running three models on
three rungs to apply a change in aggregation would cost hours of GPU for answers
already on disk: every item's tier, depth, arm, context and hit are in the .items.json
beside the result. This recomputes the submission from those.

What it may NOT do is invent anything the run did not measure. It only re-aggregates,
so a rescored submission is exactly as trustworthy as the run that produced it, and it
carries `rescored_from` to say so.

    python3 auditor/runner/rescore.py results/auditor/foo.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE))]

from run import rate, wilson, TIERS          # noqa: E402
import validate as V                          # noqa: E402


def rescore(result: dict, items: list, floor: float) -> dict:
    doc = json.loads(json.dumps(result))          # deep copy; never mutate the input

    # 1. which (tier, context) pairs the reference can be trusted on
    excluded: dict[str, dict] = {}
    live: dict[int, set] = {}
    ref_rows = [d for d in items if d["arm"] == doc["reference"]["name"]]
    for rung in doc["reference"]["rungs"]:
        if not rung.get("ran"):
            continue
        ctx = rung["context"]
        live[ctx] = set()
        for t in TIERS:
            rows = [d for d in ref_rows if d["ctx"] == ctx and d["tier"] == t]
            if not rows:
                continue
            h, n = sum(r["hit"] for r in rows), len(rows)
            if wilson(h, n)[1] < floor:
                e = excluded.setdefault(t, {"contexts": [], "reference": {},
                                            "reason": "reference arm's 95% upper "
                                                      "bound is below reference_floor"})
                e["contexts"].append(ctx)
                e["reference"][str(ctx)] = {"hits": h, "n": n,
                                            "rate": round(h / n, 4),
                                            "ci95_upper": round(wilson(h, n)[1], 4)}
            else:
                live[ctx].add(t)

    # A rescore may only NARROW the tier set, never widen it. The reference arm ran
    # every tier in order to discover which ones to drop; the arms under test ran only
    # the survivors. Admitting a tier the arms never measured would give the reference
    # data its comparators do not have, and the delta would be against a different
    # workload. Widening requires re-running, not re-aggregating.
    arm_names = [x["name"] for x in doc["arms"]]
    for ctx in list(live):
        measured_by_all = {t for t in live[ctx]
                           if all(any(d["ctx"] == ctx and d["tier"] == t
                                      and d["arm"] == an for d in items)
                                  for an in arm_names)}
        dropped = live[ctx] - measured_by_all
        for t in sorted(dropped):
            e = excluded.setdefault(t, {"contexts": [], "reference": {},
                                        "reason": "not measured by every arm in this "
                                                  "run; a rescore may narrow the tier "
                                                  "set but never widen it"})
            if ctx not in e["contexts"]:
                e["contexts"].append(ctx)
        live[ctx] = measured_by_all

    # 2. re-aggregate every arm over the surviving pairs
    for arm in [doc["reference"]] + doc["arms"]:
        rows_all = [d for d in items if d["arm"] == arm["name"]]
        for rung in arm["rungs"]:
            if not rung.get("ran"):
                continue
            ctx = rung["context"]
            rows = [d for d in rows_all
                    if d["ctx"] == ctx and d["tier"] in live.get(ctx, set())]
            if not rows:
                rung["ran"] = False
                rung["skip_reason"] = "unsupported"
                rung.pop("quality", None)
                rung.pop("cost", None)
                continue
            bt, bd = {}, {}
            for d in rows:
                bt.setdefault(d["tier"], [0, 0])
                bt[d["tier"]][0] += d["hit"]; bt[d["tier"]][1] += 1
                k = str(d["depth"])
                bd.setdefault(k, [0, 0])
                bd[k][0] += d["hit"]; bd[k][1] += 1
            h = sum(v[0] for v in bt.values()); n = sum(v[1] for v in bt.values())
            rung["quality"]["task_success"] = {
                "overall": rate(h, n),
                "by_tier": {t: rate(*v) for t, v in sorted(bt.items())},
                "by_depth": {d: rate(*v) for d, v in sorted(bd.items())}}

    doc["workload"]["excluded_tiers"] = excluded
    doc["workload"]["reference_floor"] = floor
    doc["notes_rescored"] = {
        "rescored_from": result.get("integrity", {}).get("result_hash", "")[:16],
        "rule": "exclude a (tier, context) only when the reference arm's 95% Wilson "
                "upper bound is below the floor",
        "why": "a hard threshold on the point estimate is a knife-edge: 43/48 vs "
               "44/48 on the same model and items decided a tier's existence between "
               "two backends"}
    doc["integrity"]["result_hash"] = V.canonical_hash(doc)
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--floor", type=float, default=0.9)
    ap.add_argument("-o", "--out", default="")
    a = ap.parse_args()
    items_path = a.result.replace(".json", ".items.json")
    if not os.path.exists(items_path):
        print(f"no per-item record beside {a.result}; cannot rescore", file=sys.stderr)
        return 2
    doc = rescore(json.load(open(a.result)), json.load(open(items_path)), a.floor)
    out = a.out or a.result
    json.dump(doc, open(out, "w"), indent=1)
    errs = V.validate(doc)
    ex = doc["workload"]["excluded_tiers"]
    print(f"{os.path.basename(out)}")
    print(f"  excluded: " + ("; ".join(f"{t} at {e['contexts']}" for t, e in ex.items())
                             or "(nothing)"))
    for arm in [doc["reference"]] + doc["arms"]:
        h = sum(r["quality"]["task_success"]["overall"]["hits"]
                for r in arm["rungs"] if r.get("ran"))
        n = sum(r["quality"]["task_success"]["overall"]["n"]
                for r in arm["rungs"] if r.get("ran"))
        lo, hi = wilson(h, n)
        print(f"  {arm['name']:<6} {h:>3}/{n:<3} = {h/n:.3f}  [{lo:.3f},{hi:.3f}]")
    print("  " + ("VALID" if not errs else "REJECTED: " + "; ".join(errs)))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
