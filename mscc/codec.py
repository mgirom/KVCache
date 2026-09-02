"""The product codec: CPCA with an explicit encode/decode split and real bit packing.

Every experiment in alphabet/ used `CPCACodec.__call__`, which encodes and decodes in
one step and returns a reconstructed float tensor. That is the right shape for
measuring quality and the wrong shape for a product: it never produces the CODES, so
a "2048 bits per token" frame written that way would actually be a float dump 32x
larger than the number we quote.

This module splits it:

    fit(states, dims, bits)   -> Codebook          (CPU, no gradients, minutes)
    codebook.encode(states)   -> integer codes     (what goes on the wire)
    codebook.decode(codes)    -> reconstructed states

and packs the variable-width codes into a bitstream so the frame on disk is the size
we claim it is. `test_codec.py` asserts decode(encode(X)) is numerically identical to
the measured CPCACodec(X), so the product codec IS the measured codec, not a
lookalike.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .format import codebook_fingerprint


# ------------------------------------------------------------------ bit packing

def pack_codes(codes: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Pack [n_tokens, n_dims] integer codes with per-dim bit widths into a bitstream.

    Row-major, MSB-first within each field. Returns uint8. The whole point is that
    len(result)*8 == n_tokens * widths.sum() (rounded up), so the frame's size on
    disk is the bits-per-token figure we quote, not 16 bits per component.
    """
    codes = np.ascontiguousarray(codes, dtype=np.uint32)
    widths = np.asarray(widths, dtype=np.int64)
    n, d = codes.shape
    if d != widths.size:
        raise ValueError(f"codes has {d} dims, widths has {widths.size}")
    if np.any(widths < 1) or np.any(widths > 16):
        raise ValueError("widths must be in 1..16")
    total = int(widths.sum())
    bitmat = np.zeros((n, total), dtype=np.uint8)
    off = 0
    for j, w in enumerate(widths):
        w = int(w)
        shifts = np.arange(w - 1, -1, -1, dtype=np.uint32)
        bitmat[:, off:off + w] = ((codes[:, j, None] >> shifts) & 1).astype(np.uint8)
        off += w
    return np.packbits(bitmat.reshape(-1))


def unpack_codes(packed: np.ndarray, widths: np.ndarray, n_tokens: int) -> np.ndarray:
    widths = np.asarray(widths, dtype=np.int64)
    total = int(widths.sum())
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8))[: n_tokens * total]
    bitmat = bits.reshape(n_tokens, total)
    out = np.zeros((n_tokens, widths.size), dtype=np.uint32)
    off = 0
    for j, w in enumerate(widths):
        w = int(w)
        chunk = bitmat[:, off:off + w].astype(np.uint32)
        shifts = np.arange(w - 1, -1, -1, dtype=np.uint32)
        out[:, j] = (chunk << shifts).sum(axis=1)
        off += w
    return out


# ------------------------------------------------------------------ the codebook

@dataclass
class Codebook:
    """A fitted CPCA codebook. Everything needed to encode and decode, plus the
    provenance that binds it to one model and one corpus."""
    mu: np.ndarray        # [1, hidden]  mean state
    s: np.ndarray         # [hidden]     per-channel std
    V: np.ndarray         # [hidden, k]  retained basis
    b: np.ndarray         # [k]          bits per component
    lo: np.ndarray        # [k]
    hi: np.ndarray        # [k]
    meta: dict[str, Any]

    # -- properties the guard reads
    @property
    def n_dims(self) -> int: return int(self.V.shape[1])
    @property
    def bits_per_token(self) -> int:
        if self.quant in GGML_BLOCK_BYTES:
            # ggml block types: 32 codes per block, fp16 scale + nibbles or int8
            return int(self.V.shape[1] * GGML_BLOCK_BYTES[self.quant] * 8 // 32)
        return int(self.b.sum())
    @property
    def quant(self) -> str:
        """"cpca" (variable-width uniform codes, the storage codec) or "q8_0" (ggml's
        block quantiser emulated exactly, the llama.cpp live-path codec)."""
        return str(self.meta.get("quant", "cpca"))
    @property
    def hidden_dim(self) -> int: return int(self.V.shape[0])

    def arrays(self) -> dict[str, np.ndarray]:
        return {"mu": self.mu, "s": self.s, "V": self.V,
                "b": self.b, "lo": self.lo, "hi": self.hi}

    def sha(self) -> str:
        return codebook_fingerprint(self.arrays(), self.meta)

    # -- encode / decode. Torch in, torch out; the arrays stay numpy on disk.
    def _t(self, x, dev, dt=torch.float32):
        return torch.as_tensor(x, dtype=dt, device=dev)

    def encode(self, states: torch.Tensor):
        """[..., hidden] float states -> [n, k] integer codes. For quant="q8_0" the
        return is (int8 codes [n, k], fp16 scales [n, k//32]) -- ggml's layout."""
        dev = states.device
        X = states.reshape(-1, states.shape[-1]).float()
        Z = ((X - self._t(self.mu, dev)) / self._t(self.s, dev)) @ self._t(self.V, dev)
        if self.quant == "q8_0":
            return q8_0_quantize(Z)
        if self.quant == "q4_0":
            return q4_0_quantize(Z)
        lo, hi = self._t(self.lo, dev), self._t(self.hi, dev)
        lev = (2.0 ** self._t(self.b, dev) - 1).clamp(min=1)
        q = ((Z.clamp(lo, hi) - lo) / (hi - lo) * lev).round()
        return q.to(torch.int32).cpu().numpy().astype(np.uint32)

    def decode(self, codes, device="cpu", shape=None) -> torch.Tensor:
        dev = torch.device(device)
        if self.quant == "q8_0":
            Zq = q8_0_dequantize(*codes).to(dev)
        elif self.quant == "q4_0":
            Zq = q4_0_dequantize(*codes).to(dev)
        else:
            q = torch.as_tensor(np.asarray(codes, dtype=np.int64), device=dev).float()
            lo, hi = self._t(self.lo, dev), self._t(self.hi, dev)
            lev = (2.0 ** self._t(self.b, dev) - 1).clamp(min=1)
            Zq = lo + q / lev * (hi - lo)
        X = (Zq @ self._t(self.V, dev).T) * self._t(self.s, dev) + self._t(self.mu, dev)
        return X.reshape(shape) if shape is not None else X

    # -- disk
    def save(self, path: str) -> int:
        np.savez_compressed(path, _meta=np.frombuffer(
            json.dumps(self.meta, sort_keys=True).encode(), dtype=np.uint8),
            **self.arrays())
        import os
        return os.path.getsize(path)

    @staticmethod
    def load(path: str) -> "Codebook":
        with np.load(path) as z:
            meta = json.loads(bytes(z["_meta"]).decode())
            return Codebook(mu=z["mu"], s=z["s"], V=z["V"], b=z["b"],
                            lo=z["lo"], hi=z["hi"], meta=meta)


def mse_clip(Z: torch.Tensor, b: torch.Tensor, *, fallback: tuple | None = None,
             sample: int = 16384, n_grid: int = 33) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-component clipping range that minimises quantisation MSE at width b.

    WHY THIS IS NOT COSMETIC. The original range was the 0.1%/99.9% quantiles --
    roughly +/-3.3 sigma -- regardless of how many levels the component was funded.
    A component allotted ONE bit then has exactly two reconstruction values, both
    at +/-3.3 sigma, so a sample near zero is reconstructed 3.3 sigma away. The
    measured consequence on a full-depth KV frame: relative reconstruction error
    1.27 at 512 bits/unit, i.e. WORSE than emitting the mean and calling it a day,
    and downstream top-1 agreement of exactly 0.000.

    A uniform quantiser's overload point has to shrink as levels get scarce (for a
    Gaussian, ~0.80 sigma at 1 bit against ~2.0 sigma at 3 bits). Rather than assume
    a distribution, this searches a grid of ranges and keeps the empirical minimum,
    with the historical quantile range included as a candidate so the result can
    never be worse than what it replaces.
    """
    dev = Z.device
    if Z.shape[0] > sample:
        g = torch.Generator(device=dev).manual_seed(0)
        Z = Z[torch.randperm(Z.shape[0], generator=g, device=dev)[:sample]]
    m = Z.mean(0)
    sd = Z.std(0).clamp(min=1e-8)
    lev = (2.0 ** b.float() - 1).clamp(min=1)

    cands = [(m - c * sd, m + c * sd)
             for c in torch.linspace(0.30, 6.0, n_grid).tolist()]
    if fallback is not None:
        cands.append(fallback)

    best_lo = best_hi = None
    best = torch.full((Z.shape[1],), float("inf"), device=dev)
    for lo, hi in cands:
        rng = (hi - lo).clamp(min=1e-8)
        q = ((Z.clamp(lo, hi) - lo) / rng * lev).round()
        err = ((lo + q / lev * rng) - Z).pow(2).mean(0)
        take = err < best
        if best_lo is None:
            best_lo, best_hi = lo.clone(), hi.clone()
        best_lo = torch.where(take, lo, best_lo)
        best_hi = torch.where(take, hi, best_hi)
        best = torch.where(take, err, best)
    return best_lo, best_hi


def fit_many(states: torch.Tensor, dims: int, bits_list, *, max_bits: int = 12,
             sample: int = 200_000, meta: dict[str, Any] | None = None,
             seed: int = 0, clip: str = "mse") -> list["Codebook"]:
    """Fit one CPCA basis and derive a codebook for each bit budget in `bits_list`.

    The rotation and the per-channel scales do not depend on the bit budget -- only
    the reverse-water-filling allocation does. A rate sweep that re-fits the PCA per
    rate pays for a 1024x1024 eigendecomposition it already has, and worse, lets the
    basis drift between rate points so the curve mixes two effects. Sharing the basis
    makes rate the only variable.

    `fit()` is this function with one budget, so the sweep measures the shipping
    codec by construction rather than by a lookalike that has to be asserted equal.
    """
    from importlib import import_module
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "alphabet", "scripts"))
    li = import_module("lib_inject")

    X = states.reshape(-1, states.shape[-1]).float()
    mu = X.mean(0, keepdim=True)
    Xc = X - mu
    s = Xc.std(0).clamp(min=1e-4)
    Xs = Xc / s
    V0, evals = li.fit_pca(Xs, dims)
    g = torch.Generator(device=Xs.device).manual_seed(seed)
    idx = torch.randperm(Xs.shape[0], generator=g, device=Xs.device)[:sample]
    Z0 = Xs[idx] @ V0

    out = []
    for bits in bits_list:
        b = li.alloc_bits(evals, bits, max_bits=max_bits)
        keep = b > 0
        V, bk = V0[:, keep].contiguous(), b[keep]
        Zk = Z0[:, keep]
        lo, hi = Zk.quantile(0.001, dim=0), Zk.quantile(0.999, dim=0)
        if clip == "mse":
            lo, hi = mse_clip(Zk, bk, fallback=(lo, hi))
        elif clip != "quantile":
            raise ValueError(f"unknown clip strategy: {clip!r}")

        m = dict(meta or {})
        m.update({"codec": "cpca", "clip": clip, "requested_dims": int(dims),
                  "requested_bits": int(bits), "max_bits": int(max_bits),
                  "funded_dims": int(V.shape[1]), "actual_bits": int(bk.sum().item()),
                  "n_states": int(X.shape[0]), "hidden_dim": int(X.shape[1]),
                  "seed": int(seed),
                  "codec_name": f"cpca{int(V.shape[1])}b{int(bk.sum().item())}"})
        out.append(Codebook(mu=mu.cpu().numpy(), s=s.cpu().numpy(),
                            V=V.cpu().numpy(), b=bk.cpu().numpy().astype(np.int32),
                            lo=lo.cpu().numpy(), hi=hi.cpu().numpy(), meta=m))
    return out


def fit(states: torch.Tensor, dims: int, bits: int, *, max_bits: int = 12,
        sample: int = 200_000, meta: dict[str, Any] | None = None,
        seed: int = 0, clip: str = "mse") -> Codebook:
    """Fit a CPCA codebook. Runs on whatever device `states` is on -- pass CPU
    tensors. Fitting on the GPU next to the model weights is what OOM'd the 4B run
    twice, so the CLI always hands this CPU tensors.

    `clip="quantile"` reproduces lib_inject.CPCACodec bit for bit -- that is the
    object every pre-Gate-12 number was measured with, and test_codec.py asserts
    the equivalence still holds. `clip="mse"` is the default because the quantile
    range is a measured defect at low bit widths (see mse_clip).
    """
    return fit_many(states, dims, [bits], max_bits=max_bits, sample=sample,
                    meta=meta, seed=seed, clip=clip)[0]


QK8_0 = 32
#: bytes per 32-code block for the ggml types the live path can use
GGML_BLOCK_BYTES = {"q8_0": 34, "q4_0": 18}


def q4_0_quantize(Z: torch.Tensor):
    """ggml quantize_row_q4_0_ref, exactly: per block of 32, d = max/-8 where max is
    the signed value of largest magnitude, stored as fp16; q = min(15, int8(z/d + 8.5))
    with C's truncation toward zero. Returns (uint8 [n, k] nibble values, fp16 [n, k//32])."""
    n, k = Z.shape
    assert k % QK8_0 == 0, f"q4_0 needs k % 32 == 0, got {k}"
    blocks = Z.float().reshape(n, k // QK8_0, QK8_0)
    idx = blocks.abs().argmax(-1, keepdim=True)
    mx = torch.gather(blocks, -1, idx)                        # signed max-magnitude value
    d32 = mx / -8.0
    d = d32.to(torch.float16)
    idv = torch.where(d32 != 0, 1.0 / d32, torch.zeros_like(d32))
    x = blocks * idv + 8.5
    q = torch.clamp(x.trunc(), max=15).clamp(min=0).to(torch.uint8)  # (int8_t) truncates toward zero
    return q.reshape(n, k), d.reshape(n, k // QK8_0)


def q4_0_dequantize(q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    n, k = q.shape
    return ((q.reshape(n, k // QK8_0, QK8_0).float() - 8.0) * d.float().unsqueeze(-1)).reshape(n, k)


def q8_0_quantize(Z: torch.Tensor):
    """ggml quantize_row_q8_0_ref, exactly: per block of 32, d = amax/127 rounded to
    fp16, q = roundf(z/d) as int8. Rows are code vectors; k must be a multiple of 32.
    Returns (int8 [n, k], fp16 [n, k//32]) as torch tensors on Z's device."""
    n, k = Z.shape
    assert k % QK8_0 == 0, f"q8_0 needs k % 32 == 0, got {k}"
    blocks = Z.float().reshape(n, k // QK8_0, QK8_0)
    amax = blocks.abs().amax(-1, keepdim=True)
    d = (amax / 127.0).to(torch.float16)                      # ggml stores fp16(d)
    # ggml computes id = 1/d from the float32 d, before the fp16 rounding of what it stores
    idv = torch.where(amax > 0, 1.0 / (amax / 127.0), torch.zeros_like(amax))
    q = torch.round(blocks * idv).clamp(-128, 127).to(torch.int8)
    return q.reshape(n, k), d.reshape(n, k // QK8_0)


def q8_0_dequantize(q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    n, k = q.shape
    return (q.reshape(n, k // QK8_0, QK8_0).float() * d.float().unsqueeze(-1)).reshape(n, k)


def fit_fixed(states: torch.Tensor, dims: int, k: int, *, quant: str = "q8_0",
              meta: dict[str, Any] | None = None, seed: int = 0) -> Codebook:
    """The llama.cpp live-path codebook: the same standardise+PCA as fit(), truncated
    to a FIXED k components, each stored as ggml q8_0. No bit allocation, no clipping:
    the block scale adapts per token. k must be a multiple of 32 (a q8_0 block)."""
    assert k % QK8_0 == 0 and 0 < k <= dims, f"k={k} must be a multiple of 32 and <= {dims}"
    from importlib import import_module
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "alphabet", "scripts"))
    li = import_module("lib_inject")
    X = states.reshape(-1, states.shape[-1]).float()
    mu = X.mean(0, keepdim=True)
    Xc = X - mu
    s = Xc.std(0).clamp(min=1e-4)
    V0, evals = li.fit_pca(Xc / s, dims)
    V = V0[:, :k].contiguous()
    m = dict(meta or {})
    assert quant in GGML_BLOCK_BYTES, quant
    m.update({"codec": "cpca", "quant": quant, "requested_dims": int(dims), "k": int(k),
              "funded_dims": int(k), "actual_bits": int(k * GGML_BLOCK_BYTES[quant] * 8 // 32),
              "n_states": int(X.shape[0]), "hidden_dim": int(X.shape[1]), "seed": int(seed),
              "explained_var": float(evals[:k].sum() / evals.sum()),
              "codec_name": f"pca{k}{quant}"})
    b = np.full(k, 8 if quant == "q8_0" else 4, dtype=np.int32)
    zeros = np.zeros(k, dtype=np.float32)
    return Codebook(mu=mu.cpu().numpy(), s=s.cpu().numpy(), V=V.cpu().numpy(), b=b,
                    lo=zeros, hi=zeros, meta=m)


def fit_q8(states, dims, k, *, meta=None, seed=0):
    return fit_fixed(states, dims, k, quant="q8_0", meta=meta, seed=seed)


def corpus_digest(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def bits_math_check(cb: Codebook, n_tokens: int) -> dict[str, float]:
    """What the frame SHOULD weigh, so a claim can be checked against the file."""
    payload_bits = n_tokens * cb.bits_per_token
    return {"bits_per_token": cb.bits_per_token,
            "payload_bytes": math.ceil(payload_bits / 8),
            "n_tokens": n_tokens}
