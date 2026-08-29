#!/usr/bin/env python3
"""Generate the pinned task set for a KV-Audit workload. Deterministic, then frozen.

DESIGN CONSTRAINT, and it is the one that makes the benchmark mean anything:
**no task may be answerable from the model's own knowledge.** Every answer is planted
in the document by this generator, using invented names and invented numbers. A model
that has never encountered the subject matter scores exactly the same as one that has,
because there is nothing to have encountered. What remains is the only thing we want
to measure: whether the KV optimisation still lets the model see its own context.

The second constraint is the opposite of what "hard benchmark" usually means. Every
item must be **trivial at full precision** -- the reference arm should sit at or near
1.0 on every tier. A task the reference arm fails is a broken task and gets cut, not
kept for being difficult. Difficulty here comes from the compression, never from the
question.

Four tiers, escalating in what they demand of attention:

  T1 retrieve    one span, one place in the context
  T2 link        two spans, far apart, joined WITHOUT arithmetic ("the same as", never
                 "twice as much") -- arithmetic failures are model failures and would
                 contaminate the signal
  T3 aggregate   the whole span rather than one point: count scattered markers
  T4 distractor  the real fact and a plausible near-twin, both planted. This tier
                 exists because a degraded cache does not go blank, it goes
                 confidently adjacent -- the failure this project met as an invented
                 "1234567890" delivered in the format of a real answer.

Answers are checked by exact normalised string match. No judge model, no rubric,
nothing that drifts between versions of the benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random

# Invented throughout. Any collision with a real place is coincidence, and harmless:
# the facts attached to them are invented too, so there is nothing to recall.
SITES = ["Dunraven", "Ferngate", "Kilvern", "Marlow", "Ashcombe", "Brindlow",
         "Calderwick", "Eastmere", "Fallowgate", "Grimsby Reach", "Harrowfield",
         "Ilsington", "Jarrowmoor", "Kestrelby", "Lowmarsh", "Netherby"]
ASSETS = ["lighthouse", "relay station", "pumping house", "signal tower",
          "weather mast", "cable landing", "tide gauge", "beacon"]
GATES = ["north", "south", "east", "west", "upper", "lower", "inner", "outer"]
# (quantity, unit). Slotted into one frame that stays grammatical for all four, rather
# than a verb per unit: drawing verb and unit independently gave "the consignment
# weighed 7,648 hours", and patching the verb forms gave "did the consignment ran".
# One frame, no conjugation, nothing to get wrong later.
#   "The X survey recorded a {quantity} of {v} {unit}."
#   "The Y survey recorded the same {quantity} as the X survey."
#   "What {quantity} did the Y survey record?"
MEASURES = [("mass", "kilograms"), ("length", "metres"),
            ("volume", "litres"), ("duration", "hours")]


def code(rng):
    return f"{rng.randint(10, 99)}-{rng.randint(10, 99)}-{rng.randint(10, 99)}"


def qty(rng):
    return f"{rng.randint(1, 9)},{rng.randint(100, 999)}"


def t1_retrieve(rng, i):
    site, asset = rng.choice(SITES), rng.choice(ASSETS)
    c = code(rng)
    return {
        "tier": "t1_retrieve", "id": f"t1-{i}",
        "exclusive": False,
        "plants": [{"at": "primary",
                    "text": f"The maintenance access code for the {site} {asset} is {c}."}],
        "question": f"What is the maintenance access code for the {site} {asset}? "
                    f"Answer with the code only.",
        "answer": c,
    }


def t2_link(rng, i):
    """Two plants, far apart. The link is identity, never arithmetic."""
    a, b = rng.sample(SITES, 2)
    v = qty(rng)
    quantity, unit = rng.choice(MEASURES)
    return {
        "tier": "t2_link", "id": f"t2-{i}",
        "exclusive": False,
        "plants": [
            {"at": "primary",
             "text": f"The {a} survey recorded a {quantity} of {v} {unit}."},
            {"at": "secondary",
             "text": f"The {b} survey recorded the same {quantity} as the {a} survey."},
        ],
        "question": f"What {quantity} did the {b} survey record? "
                    f"Answer with the number only.",
        "answer": v,
    }


def t3_aggregate(rng, i):
    """Counting scattered markers: demands the whole span, not one location."""
    n = rng.randint(4, 9)
    sites = [rng.choice(SITES) for _ in range(n)]
    return {
        "tier": "t3_aggregate", "id": f"t3-{i}",
        # A counting task owns its document. Two of them assembled into one haystack
        # would each count the other's markers and both answers would be wrong -- for
        # a reason that has nothing to do with the optimisation under test.
        "exclusive": True,
        "plants": [{"at": "scatter",
                    "text": f"Vessel logged: inspection run to {s}."} for s in sites],
        "question": "How many lines beginning 'Vessel logged:' appear in the document "
                    "above? Answer with the number only.",
        "answer": str(n),
    }


def t4_distractor(rng, i):
    """The real answer and a plausible near-twin, both present. Blur picks the twin."""
    site = rng.choice(SITES)
    g1, g2 = rng.sample(GATES, 2)
    real, decoy = code(rng), code(rng)
    return {
        "tier": "t4_distractor", "id": f"t4-{i}",
        "plants": [
            {"at": "decoy", "text": f"The {g1} gate at {site} uses access code {decoy}."},
            {"at": "primary", "text": f"The {g2} gate at {site} uses access code {real}."},
        ],
        "question": f"What is the access code for the {g2} gate at {site}? "
                    f"Answer with the code only.",
        "answer": real,
        "near_miss": decoy,
        "exclusive": False,
    }


TIERS = [t1_retrieve, t2_link, t3_aggregate, t4_distractor]


def build(seed: int, contexts, depths, per_cell: int):
    rng = random.Random(seed)
    items = []
    for ctx in contexts:
        for depth in depths:
            for tier in TIERS:
                for k in range(per_cell):
                    it = tier(rng, len(items))
                    it["context"] = ctx
                    it["depth"] = depth
                    items.append(it)
    return items


def normalise(s: str) -> str:
    """The single answer-checking rule. Frozen: changing it changes every score ever
    submitted, so it lives here and is versioned with the workload."""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--contexts", default="1024,4096,16384")
    ap.add_argument("--depths", default="10,50,90")
    ap.add_argument("--per-cell", type=int, default=4)
    ap.add_argument("-o", "--out", default="auditor/workload/tasks.json")
    a = ap.parse_args()

    contexts = [int(x) for x in a.contexts.split(",")]
    depths = [int(x) for x in a.depths.split(",")]
    items = build(a.seed, contexts, depths, a.per_cell)

    doc = {"workload_task_version": "0.1.0", "seed": a.seed, "contexts": contexts,
           "depths": depths, "per_cell": a.per_cell,
           "answer_normalisation": "lowercase, strip all non-alphanumerics",
           "assembly_rules": [
               "One item per document unless every item in it has exclusive=false.",
               "An item with exclusive=true is assembled alone.",
               "'primary' is planted at the item's depth; 'secondary' at least 25% of "
               "the context away from it; 'decoy' on the opposite side of 'primary'; "
               "'scatter' spread evenly across the whole span.",
               "The haystack is public-domain prose, pinned by hash, and is never "
               "allowed to contain the answer string by chance -- the assembler must "
               "check and reroll the plant values if it does.",
           ],
           "n_items": len(items), "items": items}
    blob = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    doc["sha256"] = hashlib.sha256(blob).hexdigest()
    with open(a.out, "w") as fh:
        json.dump(doc, fh, indent=1)

    per_tier = {}
    for it in items:
        per_tier[it["tier"]] = per_tier.get(it["tier"], 0) + 1
    print(f"wrote {a.out}")
    print(f"  items      {len(items)}  ({per_tier})")
    print(f"  contexts   {contexts}   depths {depths}   per cell {a.per_cell}")
    print(f"  sha256     {doc['sha256']}")
    print("\nsample of each tier:")
    seen = set()
    for it in items:
        if it["tier"] in seen:
            continue
        seen.add(it["tier"])
        print(f"  [{it['tier']}] plants={len(it['plants'])}")
        for p in it["plants"][:2]:
            print(f"      ({p['at']}) {p['text']}")
        print(f"      Q: {it['question']}")
        print(f"      A: {it['answer']!r}"
              + (f"   near-miss: {it['near_miss']!r}" if "near_miss" in it else ""))


if __name__ == "__main__":
    main()
