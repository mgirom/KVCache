#!/usr/bin/env python3
"""Code-space attention: attend over a KV cache that is held as PACKED CODES.

WHAT THIS PROVES, AND WHAT IT DOES NOT
--------------------------------------
Every earlier number for the codec came from decode-then-attend: reconstruct the full
f16 cache, then run ordinary attention. That saves storage and saves nothing live. This
module never reconstructs the cache. The document's keys and values live on the GPU as
bit-packed codes, and attention is computed from them directly by folding the codebook
into the query:

    q . k_i  =  z_i . (V^T (s * q))  +  mu . q        (one projection per step)
    out      =  ((sum_i p_i z_i) V^T) * s  +  mu       (accumulate in code space)

So the memory the cache occupies IS the packed-code memory, and that is measured here
from the tensors, not computed from a formula.

What it does not prove: speed. This is PyTorch, unpacking bits with tensor ops and
looping over kv-head groups in Python. It will be slower than f16 attention, and that
is expected and stated. The claim it can establish is narrower and is the one that
matters first: that generation from packed codes is CORRECT -- identical to
decode-then-attend, since the fold is exact -- while the cache is actually smaller. A
fused kernel is what would turn the memory saving into a speed saving, and nothing here
stands in for one.

TWO REQUIREMENTS, BOTH MEASURED ELSEWHERE IN THIS TREE
- The codes must be POST-RoPE. On pre-RoPE codes the projected query depends on each
  position's rotation and the fold is wrong (measured error 67.7 vs 0.000).
- The basis must be PER-HEAD. A joint basis across heads makes every query head dot
  against the whole joint code, 7x the compute of standard attention.
Both cost compression. Whether the combination still holds task success at a useful
rate is what the auditor decides; this module is what makes the question worth asking.

HOW IT PLUGS IN
The document goes into a CodeSpaceCache. The model's own DynamicCache holds only the
tokens that come after it -- the question and whatever is generated -- as ordinary f16.
A custom attention function, registered with transformers' AttentionInterface, scores
both segments and takes one softmax over the union. The leading `sink` tokens of the
document stay dense, exactly as merge_exact() keeps them in the decode-then-attend path,
so the two paths compute the same thing and can be compared to the bit.
"""
from __future__ import annotations

import os
import sys

import torch

def _repo_root(start):
    """Walk up until the directory that holds `mscc/`: two levels above this file in
    the research tree, one level in the published package."""
    p = os.path.abspath(start)
    for _ in range(5):
        p = os.path.dirname(p)
        if os.path.isdir(os.path.join(p, "mscc")):
            return p
    raise RuntimeError("cannot find the repository root (no mscc/ above this file)")


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root(__file__))

from lib_inject import DEV                                            # noqa: E402
import lib_kv as K                                                    # noqa: E402


# ------------------------------------------------------------------ packed codes

class PackedUnit:
    """One (layer, K|V, head)'s codes, bit-packed on the GPU, plus what decodes them."""

    def __init__(self, cb, X: torch.Tensor, device):
        # X: [n, head_dim] float, post-RoPE for K
        import numpy as np
        codes = cb.encode(X)                                      # numpy [n, k] uint32
        n = X.shape[0]
        widths = torch.as_tensor(np.asarray(cb.b), dtype=torch.int64)
        # a 0-bit component still encodes to {0,1} in Codebook.encode (lev clamps to 1);
        # dropping its bit would silently change what decode() would have produced
        assert bool((widths > 0).all()), "codebook has unfunded components; cannot pack exactly"
        total = int(widths.sum())
        nbytes = (total + 7) // 8
        # Pack each POSITION to a whole number of bytes (at most 7 bits of padding per
        # row) so a row is one gatherable unit. Big-endian within each component and
        # within each byte, matching np.packbits' default order.
        bits = np.zeros((n, nbytes * 8), dtype=np.uint8)
        off = 0
        for j, bj in enumerate(widths.tolist()):
            for t in range(bj):
                bits[:, off + t] = (codes[:, j] >> (bj - 1 - t)) & 1
            off += bj
        self.packed = torch.as_tensor(np.packbits(bits, axis=1), device=device)  # [n, nbytes]
        self.n, self.k = n, int(widths.numel())
        self.widths = widths.to(device)
        # per-component bit offsets and an index table for vectorised unpacking
        off = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.int64), widths[:-1]]), 0)
        maxb = int(widths.max())
        t = torch.arange(maxb)
        self.idx = (off[:, None] + t[None, :]).to(device)                # [k, maxb]
        valid = t[None, :] < widths[:, None]
        self.weight = torch.where(valid, 2 ** (widths[:, None] - 1 - t[None, :]),
                                  torch.zeros_like(valid, dtype=torch.int64)
                                  ).to(device).float()                  # [k, maxb]
        self.idx = torch.where(valid.to(device), self.idx, torch.zeros_like(self.idx))
        # dequantisation + basis, resident on device in float32
        f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
        self.lo, self.hi = f(cb.lo), f(cb.hi)
        self.lev = (2.0 ** f(cb.b) - 1).clamp(min=1)
        self.V, self.s, self.mu = f(cb.V), f(cb.s).view(-1), f(cb.mu).view(-1)

    def bytes(self) -> int:
        return int(self.packed.numel() * self.packed.element_size())

    def unpack(self) -> torch.Tensor:
        """Packed bytes -> dequantised codes Z [n, k] float32. Transient; the resident
        tensor is self.packed."""
        bits = ((self.packed.unsqueeze(-1) >> torch.arange(7, -1, -1, device=self.packed.device,
                                                           dtype=torch.uint8)) & 1)
        bits = bits.reshape(self.n, -1).float()                        # [n, nbytes*8]
        codes = (bits[:, self.idx] * self.weight).sum(-1)              # [n, k]
        return self.lo + codes / self.lev * (self.hi - self.lo)

    # -- the fold
    def key_scores(self, q: torch.Tensor) -> torch.Tensor:
        """q: [..., head_dim] post-RoPE queries -> scores [..., n]. One projection per
        query, then a dot with the codes; the reconstructed key never exists."""
        Z = self.unpack()                                              # [n, k]
        w = (q * self.s) @ self.V                                      # [..., k]
        return w @ Z.T + (q @ self.mu).unsqueeze(-1)                   # [..., n]

    def value_out(self, p: torch.Tensor) -> torch.Tensor:
        """p: [..., n] attention probabilities over the coded positions -> [..., head_dim].
        Accumulates in code space and projects once."""
        Z = self.unpack()
        acc = p @ Z                                                    # [..., k]
        return (acc @ self.V.T) * self.s + self.mu * p.sum(-1, keepdim=True)


class CodeSpaceCache:
    """The document's KV, per layer per head, as packed codes -- plus `sink` leading
    tokens kept dense so this path computes exactly what merge_exact() computes."""

    def __init__(self, kv_doc, books, heads, head_dim, sink=4, device=DEV):
        self.heads, self.head_dim, self.sink = heads, head_dim, sink
        self.n_doc = kv_doc[0][0].shape[2]
        self.units: dict = {}
        self.dense_sink: dict = {}
        for l, (kk, vv) in enumerate(kv_doc):
            for unit, t in (("k", kk), ("v", vv)):
                x = K.to_matrix(t).to(device)                          # [n, heads*dim]
                self.dense_sink[(l, unit)] = t[:, :, :sink].clone()    # [1, heads, sink, d]
                for h, xh in enumerate(K.head_slices(x, heads, head_dim)):
                    self.units[(l, unit, h)] = PackedUnit(
                        books[(l, f"{unit}{h}")], xh[sink:], device)

    def bytes(self) -> int:
        packed = sum(u.bytes() for u in self.units.values())
        dense = sum(t.numel() * t.element_size() for t in self.dense_sink.values())
        return packed + dense

    def dense_bytes_equivalent(self) -> int:
        """What the same cache costs as f16, for the ratio."""
        n_layers = 1 + max(l for l, _, _ in self.units)
        return n_layers * 2 * self.heads * self.n_doc * self.head_dim * 2


# ------------------------------------------------------------ the attention hook

def make_codespace_attention(cache: CodeSpaceCache):
    """An AttentionInterface-compatible function that attends over the CodeSpaceCache
    for the document and over the model's own (recent-only) cache for everything after
    it, with one softmax across both."""

    def attn(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        # query: [b, Hq, qlen, d] already rotated; key/value: [b, Hkv, n_rec, d] recent
        layer = module._cs_layer
        b, Hq, qlen, d = query.shape
        Hkv = key.shape[1]
        groups = Hq // Hkv
        qf = query[0].float()                                          # [Hq, qlen, d]
        kf, vf = key[0].float(), value[0].float()                      # [Hkv, n_rec, d]
        out = torch.empty(Hq, qlen, d, device=query.device, dtype=torch.float32)
        ks, vs = cache.dense_sink[(layer, "k")][0].float(), cache.dense_sink[(layer, "v")][0].float()

        for g in range(Hkv):
            qg = qf[g * groups:(g + 1) * groups]                       # [groups, qlen, d]
            uk, uv = cache.units[(layer, "k", g)], cache.units[(layer, "v", g)]
            s_sink = qg @ ks[g].T                                      # [groups, qlen, sink]
            s_code = uk.key_scores(qg)                                 # [groups, qlen, n_code]
            s_rec = qg @ kf[g].T                                       # [groups, qlen, n_rec]
            if attention_mask is not None:
                # the mask is built for the recent segment only -- the document precedes
                # every query token and is always fully visible
                m = attention_mask[0, 0, :, -s_rec.shape[-1]:]
                if m.dtype == torch.bool:
                    m = torch.zeros_like(m, dtype=torch.float32).masked_fill(~m, float("-inf"))
                s_rec = s_rec + m.float()
            scores = torch.cat([s_sink, s_code, s_rec], -1) * scaling
            p = torch.softmax(scores, -1)
            n_sink, n_code = s_sink.shape[-1], s_code.shape[-1]
            p_sink, p_code, p_rec = (p[..., :n_sink], p[..., n_sink:n_sink + n_code],
                                     p[..., n_sink + n_code:])
            out[g * groups:(g + 1) * groups] = (p_sink @ vs[g]
                                                + uv.value_out(p_code)
                                                + p_rec @ vf[g])
        return out.to(query.dtype).unsqueeze(0).transpose(1, 2).contiguous(), None

    return attn


def install(model, cache: CodeSpaceCache):
    """Register the hook and tag every attention layer with its index."""
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import eager_mask
    AttentionInterface.register("codespace", make_codespace_attention(cache))
    # the model builds its causal mask through a per-implementation table too; an
    # unregistered name is a KeyError there. The additive float mask is what we consume.
    AttentionMaskInterface.register("codespace", eager_mask)
    layers = model.model.layers
    for i, l in enumerate(layers):
        l.self_attn._cs_layer = i
    prev = model.config._attn_implementation
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("codespace")
    else:
        model.config._attn_implementation = "codespace"
    return prev


def uninstall(model, prev):
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(prev)
    else:
        model.config._attn_implementation = prev


@torch.inference_mode()
def generate_codespace(model, cache: CodeSpaceCache, q_ids, maxnew, eos=None, timings=None):
    """Generate from the packed document + a question, never reconstructing the cache.

    The model's DynamicCache starts EMPTY and only ever holds the question and the
    generated tokens. RoPE positions continue from the end of the document, so the
    coded keys (rotated at 0..n_doc-1) and the recent keys line up exactly as they
    would in one contiguous context.
    """
    from transformers.cache_utils import DynamicCache
    dev = q_ids.device
    n_doc = cache.n_doc
    past = DynamicCache()
    nq = q_ids.shape[1]
    import time as _t
    slot = torch.arange(0, nq, device=dev)                             # recent-cache slots
    t0 = _t.perf_counter()
    o = model(input_ids=q_ids, past_key_values=past, cache_position=slot,
              position_ids=(slot + n_doc).unsqueeze(0), use_cache=True)
    past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
    got = [int(nxt)]
    if timings is not None:
        torch.cuda.synchronize() if nxt.is_cuda else None
        timings["prefill_s"] = _t.perf_counter() - t0
        t0 = _t.perf_counter()
    for i in range(maxnew - 1):
        if eos is not None and got[-1] == eos:
            break
        p = nq + i
        o = model(input_ids=nxt, past_key_values=past,
                  cache_position=torch.tensor([p], device=dev),
                  position_ids=torch.tensor([[p + n_doc]], device=dev), use_cache=True)
        past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
        got.append(int(nxt))
    if timings is not None:
        torch.cuda.synchronize() if nxt.is_cuda else None
        timings["decode_s"], timings["n_decode"] = _t.perf_counter() - t0, len(got) - 1
    return got
