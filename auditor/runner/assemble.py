#!/usr/bin/env python3
"""Assemble a task item into a document of an exact token length, and score the reply.

TOKENISATION IS THE MODEL'S JOB, NOT OURS. Documents are built against the server's
own /tokenize endpoint, so the workload ships as text plus tasks (both hashed) and the
token counts are derived per model. Shipping pre-tokenised documents would silently
bind the workload to one tokeniser.

SCORING. Every answer in this workload is numeric or a code, so the check extracts
number-like spans from the reply and compares the FIRST one to the expected answer,
after stripping non-alphanumerics. Substring matching was tried first and is wrong:
"I count 17" contains "7", so a wrong answer scores as right on every t3 item. The
first-span rule matches what the question asks for ("answer with the code only") and
`any_match` is recorded alongside as a diagnostic, so if the two ever diverge that is
visible in the data rather than baked into the score.
"""
from __future__ import annotations

import re

NUMLIKE = re.compile(r"\d[\d,\-]*")


def normalise(s: str) -> str:
    """Frozen with the workload: changing this changes every score ever submitted."""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def check(item: dict, reply: str) -> dict:
    want = normalise(item["answer"])
    cands = [normalise(c) for c in NUMLIKE.findall(reply)]
    cands = [c for c in cands if c]
    first = cands[0] if cands else ""
    # A miss is only a HARNESS artifact if the reply was cut off part-way through the
    # right answer. "83-11-69" for "83-15-69" is complete and wrong -- that is the
    # method blurring a digit, and it is precisely what this benchmark exists to see.
    # A reply that loops without ever producing a number is also a real failure: more
    # budget buys more looping, not an answer.
    partial = bool(first) and first != want and want.startswith(first)
    out = {"hit": first == want, "any_match": want in cands, "first": first,
           "n_candidates": len(cands), "partial_answer": partial}
    if "near_miss" in item:
        # t4 only: did it grab the decoy? This is the number that tells you HOW a
        # cache is failing, not just that it failed.
        out["took_decoy"] = first == normalise(item["near_miss"])
    return out


def _paragraphs(text: str) -> list[str]:
    return [p for p in text.split("\n\n") if p.strip()]


def _fit_chars(haystack: str, start: int, target_tokens: int, n_tokens,
               hint_cpt: float, tol: float) -> tuple[str, float]:
    """Binary-search a character count whose token count hits the target.

    Sizing by whole paragraphs was tried first and cannot converge: Moby-Dick's
    paragraphs run to a few hundred tokens, so the smallest legal document overshot a
    1,024-token rung by 57% and every request 400'd. Characters are a continuous knob
    and always can.
    """
    lo, hi = 0, min(len(haystack) - start, int(target_tokens * hint_cpt * 3) + 4000)
    best, best_err, cpt = "", None, hint_cpt
    for _ in range(14):
        mid = (lo + hi) // 2
        cand = haystack[start:start + mid]
        n = n_tokens(cand)
        if n:
            cpt = mid / n
        err = abs(n - target_tokens)
        if best_err is None or err < best_err:
            best, best_err = cand, err
        if err <= max(1, target_tokens * tol):
            return cand, cpt
        if n > target_tokens:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 8:
            break
    return best, cpt


def assemble(item: dict, haystack: str, target_tokens: int, n_tokens, *,
             offset_chars: int = 0, tol: float = 0.02, hint_cpt: float = 4.2):
    """Build one document: haystack prose sized to `target_tokens`, plants at depth.

    `offset_chars` walks each item to a different stretch of the haystack, so a run is
    not 144 questions about the same paragraph. The plants' own tokens are subtracted
    from the body budget first, so the finished document lands on the rung size rather
    than overshooting by however much was planted into it.
    """
    plants = item["plants"]
    plant_tok = n_tokens("\n\n".join(p["text"] for p in plants))
    body_target = max(64, target_tokens - plant_tok - 8)

    span = int(target_tokens * hint_cpt * 3) + 8000
    start = offset_chars % max(1, len(haystack) - span - 1)
    body, cpt = _fit_chars(haystack, start, body_target, n_tokens, hint_cpt, tol)

    paras = _paragraphs(body)
    if len(paras) < 6:
        step = max(200, len(body) // 12)
        paras = [body[i:i + step] for i in range(0, len(body), step)]

    depth = item.get("depth", 50) / 100.0
    inserts: list[tuple[int, str]] = []
    scatter: list[str] = []
    for p in plants:
        at = p["at"]
        if at == "primary":
            pos = depth
        elif at == "secondary":
            # at least ~30% of the document away, so two hops cannot be served by one
            # local window of attention
            pos = depth + 0.3 if depth <= 0.5 else depth - 0.3
        elif at == "decoy":
            pos = 1.0 - depth if abs(1.0 - 2 * depth) > 0.2 else depth + 0.35
        else:
            scatter.append(p["text"])
            continue
        inserts.append((min(max(pos, 0.02), 0.97), p["text"]))
    for k, t in enumerate(scatter):
        inserts.append(((k + 1) / (len(scatter) + 1), t))

    out = list(paras)
    for pos, t in sorted(inserts, key=lambda x: -x[0]):
        out.insert(min(int(pos * len(out)), len(out)), t)
    return "\n\n".join(out), cpt


def prompt_for(item: dict, doc: str) -> str:
    return f"{doc}\n\nQuestion: {item['question']}\nAnswer:"


def haystack_is_clean(item: dict, doc: str) -> bool:
    """The planted answer must not also occur in the prose by accident.

    For codes this never fires. For a t3 count the answer is a small integer and the
    check would fire constantly, which is exactly why scoring reads the model's FIRST
    number rather than searching the document -- so this is a plant-collision check
    over the OTHER plants' text, not a containment check over the haystack.
    """
    planted = " ".join(p["text"] for p in item["plants"])
    body = doc.replace(planted, "")
    if item["tier"] == "t3_aggregate":
        return True
    return normalise(item["answer"]) not in normalise(body)
