"""MSCC frame container -- a self-describing mid-stack context frame.

A frame stops being a bare tensor and becomes a file that states the conditions it
was produced under. Every field here exists because violating it was measured to
produce fluent, confident, WRONG output rather than an error. The reader's job
(guard.py) is to refuse rather than degrade silently.

Container layout:
    magic      8 bytes   b"MSCC\\x00\\x00\\x00\\x01"
    hlen       4 bytes   big-endian uint32, header length
    header     hlen      UTF-8 JSON
    payload    rest      .npz (numpy) archive of the codec's arrays

No torch dependency: a frame can be inspected, fingerprinted and rejected on a
machine with no GPU and no model loaded.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import struct
from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np

FORMAT_VERSION = 1
MAGIC = b"MSCC\x00\x00\x00\x01"
_HDR = struct.Struct(">I")

# Files that define a model's identity. Tokeniser files are deliberately excluded:
# a tokeniser change that leaves the weights alone does not invalidate a frame.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt")
_CONFIG_NAMES = ("config.json",)


class FrameError(Exception):
    """Malformed container."""


def _sha256_file(path: str, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def model_fingerprint(model_dir: str, use_cache: bool = True) -> str:
    """sha256 over config.json + every weight file, by CONTENT.

    A fine-tune, a re-quantisation or a different checkpoint all change this, which
    is what makes condition 7 (regenerate frames on model change) fall out of
    condition 1 (same model both ends) for free -- they are one mechanism.

    Hashing multi-GB weights takes seconds, so the digest is cached in a sidecar
    keyed on (relpath, size, mtime_ns). The cache is only a cache: delete it and
    the same digest is recomputed from content.
    """
    model_dir = os.path.abspath(model_dir)
    if not os.path.isdir(model_dir):
        raise FrameError(f"not a model directory: {model_dir}")

    parts = []
    for name in sorted(os.listdir(model_dir)):
        p = os.path.join(model_dir, name)
        if not os.path.isfile(p):
            continue
        if name in _CONFIG_NAMES or name.endswith(_WEIGHT_SUFFIXES):
            st = os.stat(p)
            parts.append((name, st.st_size, st.st_mtime_ns, p))
    if not parts:
        raise FrameError(f"no config/weight files found in {model_dir}")

    cache_path = os.path.join(model_dir, ".mscc-fingerprint.json")
    stamp = [[n, sz, mt] for n, sz, mt, _ in parts]
    if use_cache and os.path.exists(cache_path):
        try:
            c = json.load(open(cache_path))
            if c.get("stamp") == stamp and c.get("format_version") == FORMAT_VERSION:
                return c["model_sha"]
        except (json.JSONDecodeError, KeyError, OSError):
            pass  # a corrupt cache must never be load-bearing

    h = hashlib.sha256()
    for name, size, _, p in parts:
        h.update(name.encode() + b"\0" + str(size).encode() + b"\0")
        h.update(bytes.fromhex(_sha256_file(p)))
    digest = h.hexdigest()

    if use_cache:
        try:
            json.dump({"format_version": FORMAT_VERSION, "model_sha": digest,
                       "stamp": stamp}, open(cache_path, "w"))
        except OSError:
            pass  # read-only model dir is fine, just slower
    return digest


def codebook_fingerprint(arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> str:
    """sha256 over the fitted codebook arrays plus the metadata that shaped them.

    Binds frame -> codebook -> model. A frame encoded with one codebook cannot be
    decoded with another by accident: the shas will not match.
    """
    h = hashlib.sha256()
    h.update(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode())
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(k.encode() + b"\0" + str(a.dtype).encode() + b"\0"
                 + str(a.shape).encode() + b"\0")
        h.update(hashlib.sha256(a.tobytes()).digest())
    return h.hexdigest()


@dataclass
class FrameHeader:
    # --- identity: who produced this, against what
    model_sha: str
    codebook_sha: str
    model_id: str = ""            # human label only, never trusted for identity
    producer: str = "mscc"

    # --- the seven operating conditions, recorded so they can be checked
    layer: int = -1               # cond 2: tap depth
    n_layers: int = -1
    n_dims: int = -1              # cond 3: retained subspace width
    hidden_dim: int = -1
    bits_per_token: int = -1      # cond 4
    codec: str = ""               # cond 6

    # --- payload description
    n_tokens: int = 0
    created_utc: str = ""
    format_version: int = FORMAT_VERSION
    notes: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "FrameHeader":
        known = {f for f in FrameHeader.__dataclass_fields__}
        extra = {k: v for k, v in d.items() if k not in known}
        base = {k: v for k, v in d.items() if k in known}
        hdr = FrameHeader(**base)
        if extra:
            hdr.notes = {**hdr.notes, "_unknown_fields": extra}
        return hdr


@dataclass
class Frame:
    header: FrameHeader
    payload: dict[str, np.ndarray]


def write_frame(path: str, header: FrameHeader, payload: dict[str, np.ndarray]) -> int:
    """Write a frame. Returns bytes written."""
    hb = header.to_json().encode()
    buf = io.BytesIO()
    np.savez_compressed(buf, **payload)
    body = buf.getvalue()
    with open(path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(_HDR.pack(len(hb)))
        fh.write(hb)
        fh.write(body)
    return len(MAGIC) + _HDR.size + len(hb) + len(body)


def read_header(path: str) -> FrameHeader:
    """Read only the header. Cheap: no payload decompression, no model needed.

    This is what lets the guard reject a frame before anything expensive happens.
    """
    with open(path, "rb") as fh:
        magic = fh.read(len(MAGIC))
        if magic[:4] != MAGIC[:4]:
            raise FrameError(f"not an MSCC frame: {path}")
        (hlen,) = _HDR.unpack(fh.read(_HDR.size))
        raw = fh.read(hlen)
    if len(raw) != hlen:
        raise FrameError(f"truncated header: {path}")
    try:
        d = json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise FrameError(f"unreadable header in {path}: {e}") from e
    return FrameHeader.from_dict(d)


def read_frame(path: str) -> Frame:
    hdr = read_header(path)
    with open(path, "rb") as fh:
        fh.read(len(MAGIC))
        (hlen,) = _HDR.unpack(fh.read(_HDR.size))
        fh.read(hlen)
        body = fh.read()
    with np.load(io.BytesIO(body)) as z:
        payload = {k: z[k] for k in z.files}
    return Frame(hdr, payload)
