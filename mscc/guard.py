"""MSCC guard -- refuse rather than degrade.

Every threshold in this file is a measured number with a citation, not a preference.
The reason the guard exists at all: a violated condition does not produce an error,
it produces fluent confident wrong text. That has now happened twice in this project
under settings that looked fine (Gate 4's 512-dim collapse; Gate 5's floor-identical
cells), and both times only a reference comparison caught it. Production has no
reference to compare against, so the conditions have to be enforced up front.

Hard failure  -> the frame is unusable; fall back to a full read.
Warning       -> outside the measured envelope; usable, but nothing is guaranteed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .format import FORMAT_VERSION, FrameHeader

# ---------------------------------------------------------------- policy + provenance

#: Hard floor on retained subspace width -- below this the codec has been measured
#: landing bit-identical to the do-nothing floor at the SAME wire cost.
MIN_DIMS = 768
#: Recommended default. Not a floor: 908 dims measured BETTER than 1024 on book prose
#: (Gate 9), so 1024 is a safe default, not an optimum.
DEFAULT_DIMS = 1024
PROV_MIN_DIMS = (
    "Collapse region: Gate 5 (Qwen3-4B, L30/36) 512 dims -> 0.372 agreement, "
    "bit-identical to the mean-vector floor, while 1024 -> 0.948 at the same 2048 "
    "bits. Gate 6 width knee (Qwen3-1.7B, L23, mismatched fit): 512 -> 0.4088, "
    "768 -> 0.8544, 1024 -> 0.9000, 1536 -> 0.9526. The knee sits between 512 and "
    "768, so 768 is the lowest width measured outside the collapse."
)
PROV_DEFAULT_DIMS = (
    "1024 is the validated default, but it is not the peak. Gate 6 matched fit: "
    "1024 -> 0.9877, 1536 -> 0.9772 (wider was worse). Gate 9 (paired 5-fold, book "
    "prose, matched fit): 908 dims at 2.26 bits each BEAT 1024 dims at 2.00 bits by "
    "+0.039 / +0.028 / +0.006 at ctx 1k / 4k / 16k. The width-vs-bits-per-dim "
    "tradeoff has an interior optimum whose location depends on the corpus, and it "
    "has not been mapped. Widths in [768, 1024) are measured-safe but off-default."
)

#: Reference wire budget. Not a hard failure -- more bits has been measured to buy
#: nothing, and under a mismatched codebook to actively hurt.
REF_BITS_PER_TOKEN = 2048
PROV_BITS = (
    "Gate 4 (Qwen3-4B): 2048/4096/6144 bits -> 0.5632/0.5632/0.5614 under a "
    "mismatched codebook. Flat, then slightly negative. Gate 3 (1.7B): 2048 -> 0.4140, "
    "6144 -> 0.3912."
)

#: The tap must sit near the top of the stack.
TAP_TOP_FRACTION = 0.15
PROV_TAP = (
    "Validated taps: L23/28 (17.9% from top) and L30/36 (16.7%). Depth sweep into "
    "Qwen3: own L22 0.8680, own L20 0.7146, own L14 0.4208 -- quality falls away "
    "sharply as the tap moves down. Mid-stack injection collapses."
)

#: Codecs whose behaviour has been measured end to end.
VALIDATED_CODECS = frozenset({"cpca"})
#: Measured and REJECTED. Not "unproven" -- proven worse, against a threshold set
#: before the run. Kept nameable so old result files stay readable.
RETRACTED_CODECS = frozenset({"tcpca"})
#: Measured, but under an open question -- usable with a warning only.
PROVISIONAL_CODECS: frozenset[str] = frozenset()
RETRACTED_CODEC = (
    "tcpca (base-3) is RETRACTED as of Phase C (gate9_ternary.py, paired 5-fold on "
    "book prose, 2026-08-27). Threshold set before the run: return to the default "
    "only if the paired mean delta vs binary stays within +/-0.01 at every context "
    "length. Measured delta: -0.0299 @1k, -0.0498 @4k, -0.0875 @16k -- outside at "
    "all three, and widening with length. Matched-width control (tcpca_floor, 1024 "
    "funded components, identical to binary) lost by the same margin (-0.0334 / "
    "-0.0570 / -0.0867), so this is not the allocation-floor artifact -- base 3 "
    "genuinely costs on long natural prose. The earlier 'ternary is free' result "
    "holds only for short benchmark items and is superseded here."
)
PROV_CODEC = (
    "cpca (binary transform coder) is the measured default -- condition 6 of seven. "
    "Any other family has not been measured end to end at this tap and width."
)

SUPPORTED_FORMAT_VERSIONS = frozenset({1})


class FrameRejected(Exception):
    """Raised by require() when a frame fails a hard check."""


@dataclass
class Finding:
    code: str
    message: str
    provenance: str = ""

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


@dataclass
class GuardResult:
    ok: bool
    hard: list[Finding] = field(default_factory=list)
    warnings: list[Finding] = field(default_factory=list)

    def reason(self) -> str:
        if self.hard:
            return "; ".join(str(f) for f in self.hard)
        return "ok" if not self.warnings else "ok (with warnings): " + \
            "; ".join(str(f) for f in self.warnings)

    def report(self) -> str:
        lines = [f"guard: {'PASS' if self.ok else 'REJECT'}"]
        for f in self.hard:
            lines.append(f"  HARD  {f.code}: {f.message}")
            if f.provenance:
                lines.append(f"        why: {f.provenance}")
        for f in self.warnings:
            lines.append(f"  WARN  {f.code}: {f.message}")
        return "\n".join(lines)


def check(header: FrameHeader, model_sha: str, codebook_sha: str | None = None,
          min_dims: int = MIN_DIMS) -> GuardResult:
    """Validate a frame header against the model and codebook about to be used.

    `model_sha` is the fingerprint of the RECEIVER's model, from
    format.model_fingerprint(). Passing the sender's own sha would defeat the check.
    """
    hard: list[Finding] = []
    warn: list[Finding] = []

    # --- condition 0: we understand this file at all
    if header.format_version not in SUPPORTED_FORMAT_VERSIONS:
        hard.append(Finding(
            "FORMAT_VERSION",
            f"frame format v{header.format_version}, this reader supports "
            f"{sorted(SUPPORTED_FORMAT_VERSIONS)}",
            "An unknown version may place fields differently; guessing is how you get "
            "a confident wrong answer."))
        return GuardResult(False, hard, warn)  # nothing below can be trusted

    # --- conditions 1 + 7: same model both ends / regenerate on model change
    if not header.model_sha or not model_sha:
        hard.append(Finding("MODEL_SHA_MISSING",
                            "frame or receiver has no model fingerprint"))
    elif header.model_sha != model_sha:
        hard.append(Finding(
            "MODEL_MISMATCH",
            f"frame was encoded against model {header.model_sha[:12]}..., receiver is "
            f"{model_sha[:12]}...",
            "A mid-stack state is one specific model's internal coordinate system. A "
            "different family is meaningless; a fine-tune or re-quantisation of the "
            "SAME family is the dangerous case, because it will still produce fluent "
            "text. Cost of rejecting: one full read."))

    # --- codebook binding
    if codebook_sha is not None:
        if not header.codebook_sha:
            hard.append(Finding("CODEBOOK_SHA_MISSING", "frame carries no codebook sha"))
        elif header.codebook_sha != codebook_sha:
            hard.append(Finding(
                "CODEBOOK_MISMATCH",
                f"frame encoded with codebook {header.codebook_sha[:12]}..., decoder "
                f"holds {codebook_sha[:12]}...",
                "The codebook IS the decoder. A mismatched pair decodes to plausible "
                "noise in the right shape."))

    # --- condition 3: subspace width
    if header.n_dims < 0:
        hard.append(Finding("DIMS_MISSING", "frame does not record n_dims"))
    elif header.n_dims < min_dims:
        hard.append(Finding(
            "DIMS_TOO_NARROW",
            f"n_dims={header.n_dims}, hard floor is {min_dims}",
            PROV_MIN_DIMS))
    elif header.n_dims < DEFAULT_DIMS:
        warn.append(Finding(
            "DIMS_BELOW_DEFAULT",
            f"n_dims={header.n_dims} is outside the collapse region but below the "
            f"{DEFAULT_DIMS} default",
            PROV_DEFAULT_DIMS))

    # --- structural sanity on the tap
    if header.layer < 0 or header.n_layers <= 0:
        hard.append(Finding("LAYER_MISSING", "frame does not record layer/n_layers"))
    elif header.layer >= header.n_layers:
        hard.append(Finding(
            "LAYER_OUT_OF_RANGE",
            f"layer {header.layer} >= n_layers {header.n_layers}"))
    else:
        # --- condition 2: tap near the top (warning: measured worse, not broken)
        from_top = (header.n_layers - 1 - header.layer) / header.n_layers
        if from_top > TAP_TOP_FRACTION:
            warn.append(Finding(
                "TAP_TOO_DEEP",
                f"layer {header.layer}/{header.n_layers} is {from_top:.1%} from the "
                f"top; validated taps sit within {TAP_TOP_FRACTION:.0%}",
                PROV_TAP))

    # --- condition 4: wire budget
    if header.bits_per_token > 0 and header.bits_per_token != REF_BITS_PER_TOKEN:
        warn.append(Finding(
            "BITS_OFF_REFERENCE",
            f"bits_per_token={header.bits_per_token}, reference is "
            f"{REF_BITS_PER_TOKEN}",
            PROV_BITS))

    # --- condition 6: codec
    # Codec names are <family><params>, e.g. cpca1024b2048 / tcpca1024b2048 / pq512k256.
    # The family is the leading alphabetic run -- stripping trailing digits is wrong
    # ("cpca1024b2048" -> "cpca1024b"), which the regression test caught.
    m = re.match(r"^[a-z]+", header.codec.lower())
    fam = m.group(0) if m else ""
    if not header.codec:
        hard.append(Finding("CODEC_MISSING", "frame does not record its codec"))
    elif fam in RETRACTED_CODECS:
        hard.append(Finding("CODEC_RETRACTED",
                            f"codec '{header.codec}' was measured and rejected",
                            RETRACTED_CODEC))
    elif fam in PROVISIONAL_CODECS:
        warn.append(Finding("CODEC_PROVISIONAL",
                            f"codec '{header.codec}' is provisional", PROV_CODEC))
    elif fam not in VALIDATED_CODECS:
        warn.append(Finding("CODEC_UNKNOWN",
                            f"codec '{header.codec}' has not been measured end to end",
                            PROV_CODEC))

    return GuardResult(not hard, hard, warn)


def require(header: FrameHeader, model_sha: str, codebook_sha: str | None = None,
            min_dims: int = MIN_DIMS) -> GuardResult:
    """check(), but raise FrameRejected on hard failure. Use at load time."""
    r = check(header, model_sha, codebook_sha, min_dims)
    if not r.ok:
        raise FrameRejected(r.reason())
    return r
