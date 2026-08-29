#!/usr/bin/env python3
"""Validate a KV-Audit submission: schema shape plus the rules the schema cannot express.

JSON Schema can say "task_success is required". It cannot say "the reference arm must
have scored above the floor" or "declared bytes must match measured bytes". Those are
section 5 of SPEC-v0.1.md, and they are the rules that actually decide whether a row
means anything, so they are enforced here and every one of them REJECTS rather than
down-weights.

Run:  python3 auditor/validate.py result.json
      python3 auditor/validate.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

TIERS = ("t1_retrieve", "t2_link", "t3_aggregate", "t4_distractor")

#: Local-only diagnostics: written into the result file, never transmitted (they can
#: describe the machine's momentary state more than the benchmark needs). They are
#: excluded from the hash for a specific reason -- the hash must cover EXACTLY what is
#: sent and stored, or it cannot attest to the stored row. Hashing something that never
#: leaves the machine made every upload fail its own integrity check, which is how this
#: was found. upload.py imports this list rather than keeping its own.
NON_TRANSMITTED = ("_capability",)


def canonical_hash(doc: dict) -> str:
    """sha256 over the transmissible content, with `integrity` itself removed."""
    d = {k: v for k, v in doc.items()
         if k != "integrity" and k not in NON_TRANSMITTED}
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _rate(r: dict) -> float:
    return r["hits"] / r["n"] if r.get("n") else 0.0


def wilson_upper(hits: int, n: int, z: float = 1.96) -> float:
    """Upper bound of the 95% Wilson interval. The exclusion rule reads this rather
    than the point estimate: a hard threshold on a noisy rate is a knife-edge, and
    43/48 vs 44/48 on the same model and items decided a tier's existence between two
    backends."""
    if n <= 0:
        return 0.0
    ph = hits / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / d
    return min(1.0, c + h)


def _check_rate(r: dict, where: str) -> list[str]:
    """Arithmetic a fabricated row cannot satisfy.

    Re-hashing a tampered submission makes it internally consistent and defeats the
    integrity check -- that is what the hash is for, and it is not a fraud detector.
    These invariants are: a rate cannot exceed 1, hits cannot exceed n, and the
    per-tier hits must sum to the overall. Cheap, and they catch the naive edit that
    the hash alone does not.
    """
    out = []
    h, n = r.get("hits"), r.get("n")
    if not isinstance(h, int) or not isinstance(n, int):
        return [f"{where}: hits and n must be integers"]
    if n <= 0:
        out.append(f"{where}: n must be positive")
    elif h > n:
        out.append(f"{where}: {h} hits out of {n} items is impossible")
    if h < 0:
        out.append(f"{where}: negative hits")
    declared = r.get("rate")
    if declared is not None and n > 0 and abs(declared - h / n) > 1e-6:
        out.append(f"{where}: declared rate {declared} does not match {h}/{n}")
    return out


def validate(doc: dict) -> list[str]:
    """Return a list of rejection reasons. Empty means the submission is valid."""
    bad: list[str] = []

    if doc.get("schema_version") != "0.1.0":
        return [f"unsupported schema_version {doc.get('schema_version')!r}"]

    for k in ("run_id", "utc", "workload", "system", "reference", "arms", "integrity"):
        if k not in doc:
            bad.append(f"missing required top-level field {k!r}")
    if bad:
        return bad

    wl = doc["workload"]
    floor = wl.get("reference_floor", 0.9)
    ctxs = set(wl.get("contexts", []))

    # --- rule 4: the reference arm establishes which tiers this MODEL can be audited
    # on. A tier it fails measures the model rather than the optimisation, so it must
    # be declared in workload.excluded_tiers and must not appear in any arm. Silent
    # omission is the failure this rule exists to catch; declared exclusion is fine and
    # is how the benchmark stays usable across a capability range.
    # {tier: {contexts: [...]}} -- exclusion is per (tier, context), because model
    # capability is context-dependent and a tier can be sound at 1k and unusable at 16k.
    exmap = wl.get("excluded_tiers", {}) or {}
    def is_excluded(tier, ctx):
        e = exmap.get(tier)
        return bool(e) and ctx in (e.get("contexts") or [])
    ref = doc["reference"]
    if ref.get("method", {}).get("family") not in (None, "none"):
        bad.append("reference arm must have method.family == 'none'")
    ref_ok_ctx = set()
    ref_tiers_seen = set()
    for rung in ref.get("rungs", []):
        if not rung.get("ran"):
            continue
        ref_ok_ctx.add(rung["context"])
        rts = rung.get("quality", {}).get("task_success", {})
        if "overall" in rts:
            bad += _check_rate(rts["overall"], f"reference ctx {rung['context']} overall")
        by_tier = rts.get("by_tier", {})
        for tier, r in by_tier.items():
            bad += _check_rate(r, f"reference ctx {rung['context']} {tier}")
            if is_excluded(tier, rung["context"]):
                bad.append(f"tier {tier} is declared excluded at ctx "
                           f"{rung['context']} but the reference arm still reports it")
            elif wilson_upper(r.get("hits", 0), r.get("n", 0)) < floor:
                bad.append(
                    f"reference arm scored {_rate(r):.3f} on {tier} at ctx "
                    f"{rung['context']} with a 95% upper bound of "
                    f"{wilson_upper(r['hits'], r['n']):.3f}, confidently below "
                    f"reference_floor {floor}, and it is not declared in "
                    "workload.excluded_tiers. A tier the reference cannot do measures "
                    "the model, not the method.")
        ref_tiers_seen.update(by_tier)
    if not ref_ok_ctx:
        bad.append("reference arm ran no context rungs")
    if ref_ok_ctx and not ref_tiers_seen:
        bad.append("every tier was excluded; this model cannot be audited by this "
                   "workload and the run reports nothing")

    # --- rules 1, 3, 5, 6 over every arm under test
    for arm in doc.get("arms", []):
        name = arm.get("name", "?")
        seen = set()
        for rung in arm.get("rungs", []):
            ctx = rung.get("context")
            seen.add(ctx)
            if not rung.get("ran"):
                if not rung.get("skip_reason"):
                    bad.append(f"arm {name!r} ctx {ctx}: not run and no skip_reason")
                continue

            # rule 1 (in part): a rung with no reference twin has no baseline
            if ctx not in ref_ok_ctx:
                bad.append(f"arm {name!r} ctx {ctx}: no reference arm result at this "
                           "context, so the delta is undefined")

            q = rung.get("quality", {})
            ts = q.get("task_success", {})
            if "overall" in ts:
                bad += _check_rate(ts["overall"], f"arm {name!r} ctx {ctx} overall")
            tier_h = tier_n = 0
            for tname, tr in ts.get("by_tier", {}).items():
                bad += _check_rate(tr, f"arm {name!r} ctx {ctx} {tname}")
                tier_h += tr.get("hits", 0) or 0
                tier_n += tr.get("n", 0) or 0
            ov = ts.get("overall", {})
            if ts.get("by_tier") and ov.get("n"):
                if tier_h != ov.get("hits") or tier_n != ov.get("n"):
                    bad.append(
                        f"arm {name!r} ctx {ctx}: per-tier totals {tier_h}/{tier_n} do "
                        f"not sum to the overall {ov.get('hits')}/{ov.get('n')}")
            for tier in q.get("task_success", {}).get("by_tier", {}):
                if is_excluded(tier, ctx):
                    bad.append(f"arm {name!r} ctx {ctx}: reports tier {tier}, which is "
                               "declared excluded at this context")
            # rule 3: agreement alone certifies nothing. This is the 0.833/0-of-12 rule.
            if "task_success" not in q:
                bad.append(f"arm {name!r} ctx {ctx}: quality reported without "
                           "task_success")
            elif "agreement" in q and not q["task_success"].get("by_tier"):
                bad.append(f"arm {name!r} ctx {ctx}: agreement reported but "
                           "task_success carries no per-tier breakdown")

            tr = q.get("_truncated_misses")
            if tr and tr.get("n") and _rate(tr) > 0.05:
                bad.append(
                    f"arm {name!r} ctx {ctx}: {tr['hits']}/{tr['n']} misses were cut "
                    "off part-way through the correct answer. Above 5% the run is "
                    "measuring n_predict, not the cache -- raise it and re-run.")

            c = rung.get("cost", {})
            # rule 5: declared bytes must match measured bytes within 1%
            dec, mea = c.get("kv_bytes_per_token_declared"), \
                c.get("kv_bytes_per_token_measured")
            if dec and mea and abs(dec - mea) / mea > 0.01:
                bad.append(f"arm {name!r} ctx {ctx}: declared {dec:.1f} B/token vs "
                           f"measured {mea:.1f} ({abs(dec-mea)/mea:.1%} apart, max 1%)")
            # restore_ms is mandatory for anything that stores or moves a cache
            fam = arm.get("method", {}).get("family", "")
            if c.get("store_bytes") and c.get("restore_ms") is None:
                bad.append(f"arm {name!r} ctx {ctx}: method stores a cache "
                           f"({fam}) but reports no restore_ms")

        # rule 6: every declared context is accounted for, ran or skipped
        for missing in sorted(ctxs - seen):
            bad.append(f"arm {name!r}: context {missing} absent with no skipped record")

    # --- integrity
    got = canonical_hash(doc)
    want = doc.get("integrity", {}).get("result_hash")
    if want and want != got:
        bad.append(f"result_hash mismatch: file says {want[:12]}..., content hashes to "
                   f"{got[:12]}...")
    return bad


# ------------------------------------------------------------------- self-test

def _rate_obj(hits, n):
    return {"hits": hits, "n": n, "rate": hits / n}


def _rung(ctx, tier_hits, n=12, declared=None, measured=3584.0, store=None,
          restore=None, agreement=None):
    # overall must be the sum of the tiers actually present, not a hardcoded n*4 --
    # the arithmetic invariants added to validate() correctly flagged the old fixture
    q = {"task_success": {
        "overall": _rate_obj(sum(tier_hits.values()), n * max(1, len(tier_hits))),
        "by_tier": {t: _rate_obj(h, n) for t, h in tier_hits.items()}}}
    if agreement is not None:
        q["agreement"] = {"top1": agreement, "n_positions": 1536}
    c = {"kv_bytes_per_token_measured": measured, "prefill_ms": 337.0,
         "decode_tok_per_s": 42.0}
    if declared is not None:
        c["kv_bytes_per_token_declared"] = declared
    if store is not None:
        c["store_bytes"] = store
    if restore is not None:
        c["restore_ms"] = restore
    return {"context": ctx, "ran": True, "quality": q, "cost": c}


def _doc(arms, ref_hits=None):
    # `is None`, not `or`: an empty dict is a meaningful fixture (every tier excluded)
    # and `or` silently replaced it with the full set.
    ref_hits = {t: 12 for t in TIERS} if ref_hits is None else ref_hits
    d = {
        "schema_version": "0.1.0", "run_id": "0" * 32,
        "utc": "2026-08-27T20:00:00Z",
        "workload": {"id": "kvaudit-2026.1", "sha256": "a" * 64,
                     "model": {"sha256": "b" * 64, "n_layers": 28, "n_kv_heads": 8,
                               "head_dim": 128, "kv_dtype": "f16"},
                     "contexts": [1024], "reference_floor": 0.9},
        "system": {"os": "linux", "arch": "x86_64", "backend": "cuda"},
        "reference": {"name": "fp16", "method": {"family": "none"},
                      "rungs": [_rung(1024, ref_hits, measured=114688.0)]},
        "arms": arms, "integrity": {},
    }
    d["integrity"]["result_hash"] = canonical_hash(d)
    return d


def selftest() -> int:
    cases = []

    good = _doc([{"name": "int4", "method": {"family": "int4"},
                  "rungs": [_rung(1024, {t: 12 for t in TIERS}, agreement=0.947)]}])
    cases.append(("a clean submission validates", good, None))

    # the headline rule, in the exact shape that motivated it
    blur = _doc([{"name": "cpca256", "method": {"family": "cpca"},
                  "rungs": [{"context": 1024, "ran": True,
                             "quality": {"agreement": {"top1": 0.833, "n_positions": 1536}},
                             "cost": {"kv_bytes_per_token_measured": 2240.0,
                                      "prefill_ms": 337.0, "decode_tok_per_s": 42.0}}]}])
    cases.append(("agreement without task_success is rejected", blur, "task_success"))

    lie = _doc([{"name": "int4", "method": {"family": "int4"},
                 "rungs": [_rung(1024, {t: 12 for t in TIERS},
                                 declared=28672.0, measured=30464.0)]}])
    cases.append(("declared bytes that omit the scales are rejected", lie, "apart"))

    quiet = _doc([{"name": "frames", "method": {"family": "cpca"},
                   "rungs": [_rung(1024, {t: 12 for t in TIERS},
                                   store=12968248)]}])
    cases.append(("a stored cache with no restore_ms is rejected", quiet, "restore_ms"))

    brokenref = _doc([{"name": "int4", "method": {"family": "int4"},
                       "rungs": [_rung(1024, {t: 12 for t in TIERS})]}],
                     ref_hits={**{t: 12 for t in TIERS}, "t3_aggregate": 4})
    cases.append(("a tier the reference fails, undeclared, is rejected",
                  brokenref, "excluded_tiers"))

    # the same failure, declared: the run stays valid on the tiers that survived.
    # This is what lets a 1.7B model be audited on 3 tiers and a 7B on 4.
    live = [t for t in TIERS if t != "t3_aggregate"]
    declared = _doc([{"name": "int4", "method": {"family": "int4"},
                      "rungs": [_rung(1024, {t: 12 for t in live}, n=12)]}],
                    ref_hits={t: 12 for t in live})
    declared["workload"]["excluded_tiers"] = {
        "t3_aggregate": {"reference_rate": 0.33, "hits": 4, "n": 12,
                         "reason": "reference arm below reference_floor"}}
    declared["integrity"]["result_hash"] = canonical_hash(declared)
    cases.append(("the same tier, DECLARED excluded, stays valid", declared, None))

    # ...but it may not then be reported as if it had run
    sneaky = _doc([{"name": "int4", "method": {"family": "int4"},
                    "rungs": [_rung(1024, {t: 12 for t in TIERS})]}],
                  ref_hits={t: 12 for t in live})
    sneaky["workload"]["excluded_tiers"] = {"t3_aggregate": {"contexts": [1024]}}
    sneaky["integrity"]["result_hash"] = canonical_hash(sneaky)
    cases.append(("an excluded tier reported anyway is rejected", sneaky,
                  "declared excluded at this context"))

    allgone = _doc([{"name": "int4", "method": {"family": "int4"},
                     "rungs": [_rung(1024, {}, n=1)]}], ref_hits={})
    allgone["workload"]["excluded_tiers"] = {t: {"reference_rate": 0.0} for t in TIERS}
    allgone["integrity"]["result_hash"] = canonical_hash(allgone)
    cases.append(("every tier excluded means the run reports nothing",
                  allgone, "every tier was excluded"))

    absent = _doc([{"name": "int4", "method": {"family": "int4"}, "rungs": []}])
    cases.append(("an omitted context rung is rejected", absent, "absent"))

    tampered = _doc([{"name": "int4", "method": {"family": "int4"},
                      "rungs": [_rung(1024, {t: 12 for t in TIERS})]}])
    tampered["arms"][0]["rungs"][0]["cost"]["decode_tok_per_s"] = 9999.0
    cases.append(("an edited row fails its own hash", tampered, "result_hash"))

    npass = 0
    for name, doc, expect in cases:
        errs = validate(doc)
        if expect is None:
            ok = not errs
        else:
            ok = any(expect in e for e in errs)
        npass += ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"        got: {errs}")
    print(f"\n{npass}/{len(cases)} self-tests passed")
    return 0 if npass == len(cases) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.result:
        ap.error("give a result file or --selftest")
    errs = validate(json.load(open(a.result)))
    if errs:
        print("REJECTED:")
        for e in errs:
            print("  -", e)
        return 2
    print("valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
