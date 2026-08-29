"""MSCC-KV -- the full-depth KV frame: codebook artifact, frame payload, guard rules.

WHY THE FRAME CHANGED SHAPE
---------------------------
The mid-stack frame stored one layer's hidden state per token. It was measured, in
three separate runs, to produce fluent confident wrong answers whenever the receiver
had a question the sender had not already asked (Item 1 acceptance, Gate 10 recall
0/12 at five of six tap depths, Gate 11 handoff 0/9). The cause is structural: a
question token cannot attend to the document below the tap, because those keys and
values were never computed at the receiver.

A full-depth KV frame does not have that failure mode available to it. The receiver
is handed K and V at every layer, so a question attends to the document at every
depth exactly as if the document were in its context, and the receiver runs ZERO
layers over the document instead of 14%.

WHAT THIS MODULE ADDS OVER format.py
------------------------------------
A mid-stack frame is one codebook and one code matrix. A KV frame is 2*n_layers of
each -- one coding unit per (layer, K|V) -- plus a handful of tokens carried at full
precision, plus the fact that keys are coded in their PRE-RoPE basis. Each of those
is a condition that has to be recorded and checked, not remembered.

    unit ordering       fixed by lib_kv.unit_keys; a frame that decodes units in a
                        different order than they were encoded is silent garbage.
    key basis           "prerope" frames are position-independent and must be
                        re-rotated at the offset they are replayed at. "postrope"
                        frames are welded to their capture offset.
    protected tokens    the leading `sink` tokens travel uncompressed. Dropping them
                        on decode costs ~0.31 top-1 agreement (Gate 12 ablation),
                        which is more than the entire codec is worth.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .codec import Codebook, pack_codes, unpack_codes
from .format import FrameError

UNITS = ("k", "v")
#: Key bases a decoder in this file knows how to replay.
VALID_BASES = frozenset({"prerope", "postrope"})


def unit_keys(n_layers: int) -> list[tuple[int, str]]:
    """Canonical coding-unit order. One definition, imported by everything that
    encodes, decodes or fingerprints, so the three can never drift apart."""
    return [(l, u) for l in range(n_layers) for u in UNITS]


def _tag(layer: int, unit: str) -> str:
    return f"{layer}.{unit}"


# --------------------------------------------------------------- codebook artifact

@dataclass
class KVCodebook:
    """2*n_layers CPCA codebooks plus the provenance that binds them to one model."""
    books: dict[tuple[int, str], Codebook]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_layers(self) -> int:
        return 1 + max(l for l, _ in self.books)

    @property
    def bits_per_token(self) -> int:
        """Codec bits only. Protected tokens are charged separately, by the frame."""
        return sum(cb.bits_per_token for cb in self.books.values())

    @property
    def basis(self) -> str:
        return self.meta.get("basis", "postrope")

    def sha(self) -> str:
        """One digest over every unit, in canonical order, plus the shared meta.

        A frame binds to this. Swapping any single unit's codebook -- the failure a
        per-unit sha would let through if the units were checked one at a time --
        changes it.
        """
        h = hashlib.sha256()
        h.update(json.dumps(self.meta, sort_keys=True,
                            separators=(",", ":")).encode())
        # iterate the books actually present, in canonical order, rather than assuming
        # the (layer, k|v) layout -- a per-head codebook has units "k0".."k7"
        for key in sorted(self.books):
            h.update(_tag(*key).encode() + b"\0" + self.books[key].sha().encode())
        return h.hexdigest()

    def save(self, path: str) -> int:
        arrays: dict[str, np.ndarray] = {}
        per_unit_meta = {}
        for key, cb in self.books.items():
            for name, arr in cb.arrays().items():
                arrays[f"{_tag(*key)}:{name}"] = arr
            per_unit_meta[_tag(*key)] = cb.meta
        blob = json.dumps({"meta": self.meta, "units": per_unit_meta},
                          sort_keys=True).encode()
        arrays["_meta"] = np.frombuffer(blob, dtype=np.uint8)
        np.savez_compressed(path, **arrays)
        return os.path.getsize(path)

    @staticmethod
    def load(path: str) -> "KVCodebook":
        with np.load(path) as z:
            blob = json.loads(bytes(z["_meta"]).decode())
            books: dict[tuple[int, str], Codebook] = {}
            for tag, m in blob["units"].items():
                l, u = tag.split(".")
                books[(int(l), u)] = Codebook(
                    mu=z[f"{tag}:mu"], s=z[f"{tag}:s"], V=z[f"{tag}:V"],
                    b=z[f"{tag}:b"], lo=z[f"{tag}:lo"], hi=z[f"{tag}:hi"], meta=m)
        return KVCodebook(books=books, meta=blob["meta"])


# ------------------------------------------------------------------ frame payload

def encode_frame(pairs, cb: KVCodebook, n_sink: int = 0):
    """Coding-space [(K,V), ...] -> the arrays that go in a frame.

    `pairs` must already be in the codebook's basis: pre-RoPE keys for a "prerope"
    codebook. lib_kv.coding_pairs() is what produces them; passing post-RoPE keys to
    a pre-RoPE codebook decodes to noise in the right shape, which is exactly the
    class of failure this project exists to make impossible, so the basis is written
    into the header and checked on load.
    """
    import torch
    n_layers = cb.n_layers
    n = pairs[0][0].shape[2]
    out: dict[str, np.ndarray] = {}
    for l in range(n_layers):
        for j, u in enumerate(UNITS):
            t = pairs[l][j]
            b, h, _, d = t.shape
            x = t[0].permute(1, 0, 2).reshape(n, h * d).float().contiguous()
            unit = cb.books[(l, u)]
            codes = unit.encode(x)
            out[f"codes:{_tag(l, u)}"] = pack_codes(codes, unit.b)
            out[f"widths:{_tag(l, u)}"] = np.asarray(unit.b, dtype=np.int32)
            if n_sink:
                s = min(n_sink, n)
                out[f"sink:{_tag(l, u)}"] = (
                    t[:, :, :s].to(torch.float16).cpu().numpy())
    return out


def decode_frame(payload, cb: KVCodebook, n_tokens: int, n_sink: int = 0,
                 dtype=None, device="cpu"):
    """Frame arrays -> coding-space [(K,V), ...]. Inverse of encode_frame."""
    import torch
    dtype = dtype or torch.float32
    n_layers = cb.n_layers
    kv = []
    for l in range(n_layers):
        pair = []
        for u in UNITS:
            tag = _tag(l, u)
            unit = cb.books[(l, u)]
            widths = payload[f"widths:{tag}"]
            fw = np.asarray(widths).ravel()
            bw = np.asarray(unit.b).ravel()
            if not np.array_equal(fw, bw):
                if fw.size != bw.size:
                    why = (f"frame declares {fw.size} code widths, the codebook "
                           f"holds {bw.size}")
                else:
                    j = int(np.flatnonzero(fw != bw)[0])
                    why = (f"same width count ({fw.size}) but they differ from "
                           f"component {j}: frame {fw[j]} bits, codebook {bw[j]}")
                raise FrameError(
                    f"unit {tag}: {why}. The frame and the codebook were not "
                    "produced together. Note this check only sees the bit "
                    "ALLOCATION -- two codebooks fitted on different corpora at the "
                    "same rate have identical widths and are indistinguishable "
                    "here. The binding is the codebook sha, checked by the guard "
                    "before decode runs.")
            codes = unpack_codes(payload[f"codes:{tag}"], widths, n_tokens)
            x = unit.decode(codes, device=device)
            h = unit.hidden_dim // _head_dim(cb)
            t = x.reshape(n_tokens, h, _head_dim(cb)).permute(1, 0, 2)
            t = t.unsqueeze(0).to(dtype=dtype, device=device)
            if n_sink:
                s = min(n_sink, n_tokens)
                sk = torch.as_tensor(payload[f"sink:{tag}"], device=device).to(dtype)
                t[:, :, :s] = sk[:, :, :s]
            pair.append(t)
        kv.append((pair[0], pair[1]))
    return kv


def _head_dim(cb: KVCodebook) -> int:
    hd = cb.meta.get("head_dim")
    if not hd:
        raise FrameError("codebook meta has no head_dim; cannot reshape a unit "
                         "back into attention heads")
    return int(hd)


def frame_bits_per_token(cb: KVCodebook, n_tokens: int, n_sink: int,
                         raw_bits_per_token: float) -> float:
    """Everything the frame actually costs, protected tokens included.

    Quoting the codec rate alone while shipping full-precision sink tokens would
    understate the wire by ~12% at 1k context, and by more on short documents. The
    figure this returns is the one that goes in the header.
    """
    prot = min(n_sink, n_tokens)
    return cb.bits_per_token + raw_bits_per_token * prot / max(n_tokens, 1)


# ------------------------------------------------------------------------- guard
#
# These are the KV-frame analogues of guard.py's seven conditions. They live here
# rather than in guard.py because they check fields guard.py's header does not have,
# but the discipline is identical: refuse rather than degrade.

#: Below this the frame has been measured to answer questions wrongly while still
#: sounding fluent. See PROV_KV_RATE.
MIN_UNIT_BITS = 512
DEFAULT_UNIT_BITS = 1024
PROV_KV_RATE = (
    "Gate 12 (Qwen3-1.7B, 28 layers, ctx 1024, pre-RoPE basis + 4 protected "
    "tokens), agreement / needle recall against the document-in-context reference: "
    "2048 bits/unit 0.956 / 12-of-12 at 7.8x; 1024 -> 0.925 / 12-of-12 at 15.1x; "
    "512 -> 0.873 / 10-of-12 at 28.4x; 256 -> 0.833 / 0-of-12 at 51.2x. The "
    "incumbent uniform quantisers on the same windows: int8 (2.0x) 0.980 / 12-of-12, "
    "int4 (3.8x) 0.947 / 12-of-12, int2 (7.1x) 0.669 / 9-of-12. 256 bits/unit is "
    "the floor case this project keeps rediscovering: agreement still reads a "
    "respectable 0.833 while recall is ZERO, so a frame that looks fine on the "
    "proxy metric answers every question wrongly and fluently. That gap is why the "
    "floor is 512 and why agreement alone never clears a rate."
)
PROV_KV_SINK = (
    "Gate 12 ablation at 512 bits/unit: full product 0.873 agreement / 10-of-12 "
    "recall, without the protected tokens 0.560 / 7-of-12. Attention parks a large share of every head's probability mass "
    "on the first token, so an error there is not one token's error, it moves every "
    "downstream attention denominator."
)
#: The health check (DELIVERY-PLAN Item 2, measured in Gate 13) did NOT clear its
#: pre-registered bar. It is recorded and surfaced, and it is never a gate.
HEALTH_STAMP_THRESHOLD = 0.8854
PROV_KV_HEALTH = (
    "Gate 13. The sender frames all but the last P document tokens, teacher-forces "
    "those P through the decoded cache and stamps the agreement into the header. "
    "Pre-registered bar, set before the run: flag >=90% of known-bad frames AND pass "
    ">=90% of known-good, on windows the threshold was not fitted on. Measured on 192 "
    "frames (12 configurations x 16 windows, half in-domain and half out-of-domain): "
    "held-out pass_good 0.929, flag_bad 0.897. It MISSES the bar on flag_bad, by 0.003. "
    "The reference-free receiver-side signal (code saturation) did worse -- pass_good "
    "0.857, flag_bad 0.912 -- and is close to a restatement of the rate the header "
    "already declares. So: this stamp is advisory. It is not a certification, it does "
    "not gate anything, and roughly one bad frame in ten passes it. Verify your own "
    "frames."
)
PROV_KV_BASIS = (
    "Gate 12 ablation at 512 bits/unit: pre-RoPE key basis 0.873 agreement / "
    "10-of-12 recall, post-RoPE 0.812 / 2-of-12 -- a 0.06 agreement gap hiding a "
    "5x recall gap. "
    "RoPE smears each key channel across every rotation angle in the document, "
    "which is variance the quantiser then has to spend bits covering. The pre-RoPE "
    "frame is also position-independent: re-rotation reproduces the live cache "
    "bit-exactly (max|diff| = 0.0), so the same frame can be replayed at any offset."
)


def check_kv(header, model_sha: str, codebook_sha: str | None = None,
             min_unit_bits: int = MIN_UNIT_BITS):
    """Validate a KV frame header. Same contract as guard.check: hard vs warning."""
    from .guard import Finding, GuardResult, SUPPORTED_FORMAT_VERSIONS

    hard, warn = [], []
    n = header.notes or {}

    if header.format_version not in SUPPORTED_FORMAT_VERSIONS:
        hard.append(Finding("FORMAT_VERSION",
                            f"frame format v{header.format_version}, this reader "
                            f"supports {sorted(SUPPORTED_FORMAT_VERSIONS)}"))
        return GuardResult(False, hard, warn)

    if n.get("kind") != "kv":
        hard.append(Finding(
            "NOT_A_KV_FRAME",
            f"frame kind is {n.get('kind', 'midstack')!r}, this loader wants 'kv'",
            "A mid-stack frame loaded as a KV frame would be reshaped into the "
            "wrong tensor and decode to noise."))
        return GuardResult(False, hard, warn)

    if not header.model_sha or not model_sha:
        hard.append(Finding("MODEL_SHA_MISSING",
                            "frame or receiver has no model fingerprint"))
    elif header.model_sha != model_sha:
        hard.append(Finding(
            "MODEL_MISMATCH",
            f"frame was encoded against model {header.model_sha[:12]}..., receiver "
            f"is {model_sha[:12]}...",
            "K and V are one specific model's internal coordinates at every layer. "
            "A fine-tune of the same family is the dangerous case: it still "
            "produces fluent text. Cost of rejecting: one full read."))

    if codebook_sha is not None:
        if not header.codebook_sha:
            hard.append(Finding("CODEBOOK_SHA_MISSING", "frame carries no codebook sha"))
        elif header.codebook_sha != codebook_sha:
            hard.append(Finding(
                "CODEBOOK_MISMATCH",
                f"frame encoded with codebook {header.codebook_sha[:12]}..., decoder "
                f"holds {codebook_sha[:12]}...",
                "The codebook is the decoder, and there are 2*n_layers of them. A "
                "mismatched pair decodes to plausible noise in the right shape."))

    basis = n.get("key_basis")
    if basis not in VALID_BASES:
        hard.append(Finding("KEY_BASIS_UNKNOWN",
                            f"frame declares key basis {basis!r}, known: "
                            f"{sorted(VALID_BASES)}", PROV_KV_BASIS))
    elif basis == "postrope":
        warn.append(Finding("KEY_BASIS_POSTROPE",
                            "post-RoPE frame: measured worse AND welded to the "
                            "offset it was captured at", PROV_KV_BASIS))

    if header.n_layers <= 0 or n.get("kv_heads", 0) <= 0 or n.get("head_dim", 0) <= 0:
        hard.append(Finding("SHAPE_MISSING",
                            "frame does not record n_layers / kv_heads / head_dim"))

    ub = n.get("unit_bits", 0)
    if ub <= 0:
        hard.append(Finding("RATE_MISSING", "frame does not record its unit rate"))
    elif ub < min_unit_bits:
        hard.append(Finding("RATE_TOO_LOW",
                            f"unit_bits={ub}, hard floor is {min_unit_bits}",
                            PROV_KV_RATE))
    elif ub < DEFAULT_UNIT_BITS:
        warn.append(Finding("RATE_BELOW_DEFAULT",
                            f"unit_bits={ub} is above the floor but below the "
                            f"{DEFAULT_UNIT_BITS} default", PROV_KV_RATE))

    if not n.get("sink", 0):
        warn.append(Finding("NO_PROTECTED_TOKENS",
                            "frame carries no full-precision sink tokens",
                            PROV_KV_SINK))

    # The health stamp is INFORMATION, not a verdict. It failed its pre-registered
    # bar, so it may not be promoted to a hard check no matter how convenient that
    # would be -- that is precisely the move this project's guard exists to prevent.
    stamp = n.get("health_probe")
    if stamp is None:
        warn.append(Finding("NO_HEALTH_STAMP",
                            "frame carries no self-declared health probe",
                            PROV_KV_HEALTH))
    elif stamp < HEALTH_STAMP_THRESHOLD:
        warn.append(Finding(
            "HEALTH_STAMP_LOW",
            f"sender's own probe agreement is {stamp:.3f}, below the {HEALTH_STAMP_THRESHOLD:.3f} "
            "calibration point -- ADVISORY ONLY, this is not a verdict",
            PROV_KV_HEALTH))

    if header.n_tokens <= 0:
        hard.append(Finding("EMPTY_FRAME", "frame declares no tokens"))

    return GuardResult(not hard, hard, warn)


def require_kv(header, model_sha: str, codebook_sha: str | None = None,
               min_unit_bits: int = MIN_UNIT_BITS):
    from .guard import FrameRejected
    r = check_kv(header, model_sha, codebook_sha, min_unit_bits)
    if not r.ok:
        raise FrameRejected(r.reason())
    return r
