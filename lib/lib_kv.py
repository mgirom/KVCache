#!/usr/bin/env python3
"""Full-depth KV frame: capture, compress, hand over, resume.

WHY THIS EXISTS
---------------
Gates 10 and 11 killed the single-depth mid-stack frame. Both landed on the same
diagnosis and the same one-line fix, which nobody had built:

    "Hand over the full-depth KV cache, compressed, instead of a single-depth
     hidden state. Then receiver-side tokens attend to the document at every
     depth by construction, and the receiver skips the document read entirely."
     -- GATE-11-HANDOFF-RESULTS.md

This module is that fix. The difference from the mid-stack frame is structural,
not a tuning knob:

    mid-stack frame   receiver runs layers L+1..N over the document state.
                      The question's tokens have no document keys/values below
                      layer L, because those were never computed. Attention to
                      the document simply does not exist down there. -> answers
                      fluent, confident and wrong (Gate 10, 0/12 recall).

    KV frame          receiver runs ZERO layers over the document. It is handed
                      K and V at every layer, so a question token attends to
                      the document at every depth exactly as if the document
                      were in its context. The query path is intact by
                      construction, not by luck.

The trade moves from "can it answer at all" (it can) to "how hard does the cache
compress", which is a measurement, not an architecture risk.

REFERENCE ARITHMETIC (Qwen3-1.7B, from models/qwen3-1.7b-fp/config.json)
    28 layers x 2 (K,V) x 8 kv-heads x 128 head_dim x 16 bits = 917,504 bits/token
                                                              = 114,688 bytes/token
Measured, not quoted: kv_bits_per_token() recomputes it from the live cache.

CODING UNIT
-----------
One codebook per (layer, K|V) pair -- 2*n_layers of them. Each unit is a
[n_tokens, n_kv_heads*head_dim] matrix, i.e. 1024 dims on this model, which is
the same shape the CPCA transform coder in lib_inject/mscc.codec already handles.
Heads are flattened together on purpose: the basis is then free to exploit
cross-head correlation, which per-head coding throws away.

Keys are coded PRE-RoPE (capture_kv_prerope). Coding them as the cache holds them
is supported and measured, and is worse on both counts: RoPE smears each key
channel across every rotation angle in the document, and it welds the frame to the
offset it was captured at. Which basis a frame used is written into its header and
checked on load, never inferred.
"""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from lib_inject import DEV  # noqa: E402


# ------------------------------------------------------------------ cache shape

def cache_layers(cache):
    """The per-layer (keys, values) views of a transformers Cache, version-tolerantly.

    transformers 5.x exposes cache.layers[i].keys/.values; 4.x exposed
    cache.key_cache[i]/.value_cache[i]. Both are handled so a frame written on
    one does not become unreadable on the other.
    """
    if hasattr(cache, "layers"):
        return [(l.keys, l.values) for l in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


def build_cache(kv, config=None):
    """[(K,V), ...] per layer -> a Cache the model will accept as past_key_values."""
    from transformers.cache_utils import DynamicCache
    return DynamicCache([(k, v) for k, v in kv])


@torch.inference_mode()
def capture_kv(model, ids, chunk=0):
    """Run the document once, return [(K,V), ...] per layer, detached and cloned.

    K/V are [1, n_kv_heads, n_tokens, head_dim]. With chunk>0 the document is
    prefilled in slices so a long document does not need one giant attention
    matrix; the resulting cache is the same object either way.

    logits_to_keep=1 matters more than it looks: without it the model materialises a
    [1, n_tokens, vocab] logits tensor that nothing here reads. At 8k tokens on this
    vocab that is 4.6 GB, and it OOM'd a 12 GB card during a capture that otherwise
    fits comfortably.
    """
    if chunk and ids.shape[1] > chunk:
        from transformers.cache_utils import DynamicCache
        cache, n = DynamicCache(), ids.shape[1]
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            cp = torch.arange(s, e, device=ids.device)
            model(input_ids=ids[:, s:e], past_key_values=cache, cache_position=cp,
                  position_ids=cp.unsqueeze(0), use_cache=True, logits_to_keep=1)
        out = cache
    else:
        out = model(input_ids=ids, use_cache=True,
                    logits_to_keep=1).past_key_values
    return [(k.detach().clone(), v.detach().clone()) for k, v in cache_layers(out)]


@torch.inference_mode()
def capture_kv_prerope(model, ids, chunk=0):
    """Capture the cache AND the keys as they are BEFORE RoPE is applied.

    RoPE rotates each (2i, 2i+d/2) channel pair by an angle that depends on the
    token's position. That does two bad things to a codec. It smears each key
    channel's distribution across every rotation angle in the document, inflating
    the variance the quantiser has to cover; and it welds the frame to the offset
    it was captured at.

    Coding pre-RoPE keys and re-rotating at decode time fixes both. The rotation
    is reproduced bit-exactly (verified: max|diff| = 0.0 against the live cache),
    so this is a change of coding basis, not an approximation.

    Returns (kv, kpre) where kpre[l] is [1, heads, n, dim] in the pre-rotation basis.
    """
    layers = model.model.layers if hasattr(model, "model") else model.layers
    grab, handles = {}, []
    for i, l in enumerate(layers):
        if not hasattr(l.self_attn, "k_norm"):
            for h in handles:
                h.remove()
            raise RuntimeError("this architecture has no k_norm; pre-RoPE capture "
                               "would need an architecture-specific hook")
        handles.append(l.self_attn.k_norm.register_forward_hook(
            lambda m, a, o, i=i: grab.__setitem__(i, o.detach().clone())))
    try:
        kv = capture_kv(model, ids, chunk=chunk)
    finally:
        for h in handles:
            h.remove()
    # k_norm emits [b, n, heads, dim]; the cache convention is [b, heads, n, dim]
    kpre = [grab[i].transpose(1, 2).contiguous() for i in range(len(layers))]
    if kpre[0].shape[2] != kv[0][0].shape[2]:
        raise RuntimeError(f"pre-RoPE capture length {kpre[0].shape[2]} != cache "
                           f"length {kv[0][0].shape[2]}; chunked prefill only keeps "
                           "the last slice's hook output")
    return kv, kpre


@torch.inference_mode()
def rope_keys(model, kpre, start=0):
    """Re-apply RoPE to pre-rotation keys at absolute positions start..start+n.

    `start` is why a pre-RoPE frame is worth having: the same frame can be replayed
    at any offset, so a framed document can be spliced behind a system prompt
    without being re-encoded.
    """
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    n = kpre[0].shape[2]
    dev = kpre[0].device
    pos = torch.arange(start, start + n, device=dev).unsqueeze(0)
    dummy = torch.zeros(1, n, model.config.hidden_size, device=dev,
                        dtype=next(model.parameters()).dtype)
    cos, sin = model.model.rotary_emb(dummy, pos)
    out = []
    for k in kpre:
        _, kr = apply_rotary_pos_emb(k, k, cos, sin)
        out.append(kr)
    return out


def merge_exact(kv_ref, kv_q, n_sink=0, n_recent=0):
    """Splice full-precision tokens back into a quantised cache.

    Attention has a documented sink at the first token(s): a large share of every
    head's probability mass parks there, so an error in that one key or value is
    not one token's worth of error, it is a shift in every downstream token's
    attention denominator. Measured here at 16x compression: protecting a SINGLE
    token moved top-1 agreement 0.609 -> 0.812.

    The recent window is the other end -- the tokens a question attends to most
    when it is appended right after the document.

    The cost is real and is charged in exact_overhead_bits_per_token(): protected
    tokens travel uncompressed.
    """
    if not n_sink and not n_recent:
        return kv_q
    n = kv_ref[0][0].shape[2]
    out = []
    for (kr, vr), (kq, vq) in zip(kv_ref, kv_q):
        k, v = kq.clone(), vq.clone()
        if n_sink:
            s = min(n_sink, n)
            k[:, :, :s], v[:, :, :s] = kr[:, :, :s], vr[:, :, :s]
        if n_recent:
            s = min(n_recent, n)
            k[:, :, n - s:], v[:, :, n - s:] = kr[:, :, n - s:], vr[:, :, n - s:]
        out.append((k, v))
    return out


def exact_overhead_bits_per_token(raw_bpt, n_tokens, n_sink=0, n_recent=0):
    """Amortised wire cost of the protected tokens. Charged, not waved through."""
    prot = min(n_sink, n_tokens) + min(n_recent, n_tokens)
    prot = min(prot, n_tokens)
    return raw_bpt * prot / n_tokens


def kv_bits_per_token(kv):
    """Measured wire cost of an uncompressed cache, from the tensors themselves."""
    n = kv[0][0].shape[2]
    bits = sum((k.numel() + v.numel()) * k.element_size() * 8 for k, v in kv)
    return bits / n


def to_matrix(t):
    """[1, heads, n, dim] -> [n, heads*dim] float32, contiguous."""
    b, h, n, d = t.shape
    assert b == 1, "batch>1 frames are not defined"
    return t[0].permute(1, 0, 2).reshape(n, h * d).float().contiguous()


def from_matrix(x, heads, dim, dtype, device):
    """[n, heads*dim] -> [1, heads, n, dim] in the model's dtype."""
    n = x.shape[0]
    return x.reshape(n, heads, dim).permute(1, 0, 2).unsqueeze(0).to(dtype=dtype,
                                                                    device=device)


UNITS = ("k", "v")


def unit_keys(n_layers):
    """Canonical iteration order over coding units. One place, so the fitter, the
    encoder and the frame writer can never disagree about ordering."""
    return [(l, u) for l in range(n_layers) for u in UNITS]


# ------------------------------------------------------------------ codebook fit

def fit_kv_codebooks(states, dims, bits, *, progress=None, **kw):
    """{(layer, unit): [n, d] CPU float tensor} -> {(layer, unit): Codebook}.

    Fitting is per unit and sequential: only one [n, d] slice is on the GPU at a
    time. Fitting all 56 units at once is what OOM'd the 4B run twice, and the
    delivery plan froze codebook fitting to a CPU/one-at-a-time discipline.
    """
    from mscc import codec as mcodec
    out = {}
    for i, key in enumerate(sorted(states)):
        layer, unit = key
        X = states[key]
        cb = mcodec.fit(X, dims, bits,
                        meta={"kv_layer": int(layer), "kv_unit": unit,
                              "kv_dims": int(X.shape[1]), **kw})
        out[key] = cb
        if progress:
            progress(i, len(states), key, cb)
    return out


def frame_bits_per_token(books):
    """Sum of every unit's per-token budget. This is the number the frame weighs."""
    return sum(cb.bits_per_token for cb in books.values())


# ------------------------------------------------------------- encode / decode

def encode_kv(kv, books):
    """[(K,V), ...] -> {(layer, unit): uint32 codes}. What goes on the wire."""
    codes = {}
    for l, (K, V) in enumerate(kv):
        for unit, t in (("k", K), ("v", V)):
            cb = books[(l, unit)]
            codes[(l, unit)] = cb.encode(to_matrix(t).to(DEV))
    return codes


def decode_kv(codes, books, shapes, dtype, device=DEV):
    """Codes -> [(K,V), ...] ready to hand to build_cache()."""
    kv = []
    n_layers = len(shapes)
    for l in range(n_layers):
        pair = []
        for unit in UNITS:
            b, h, n, d = shapes[l]
            x = books[(l, unit)].decode(codes[(l, unit)], device=device)
            pair.append(from_matrix(x, h, d, dtype, device))
        kv.append((pair[0], pair[1]))
    return kv


def roundtrip_kv(kv, books):
    """encode then decode, in one step. Same numbers as the wire path, no packing."""
    dtype, device = kv[0][0].dtype, kv[0][0].device
    shapes = [tuple(k.shape) for k, _ in kv]
    return decode_kv(encode_kv(kv, books), books, shapes, dtype, device)


def coding_pairs(kv, kpre=None):
    """The tensors the codec actually sees: pre-RoPE keys when we have them."""
    if kpre is None:
        return [(k, v) for k, v in kv]
    return [(kp, v) for kp, (_, v) in zip(kpre, kv)]


def roundtrip_kv_prerope(model, kv, kpre, books, start=0):
    """Code the keys in the pre-rotation basis, then re-rotate into a usable cache."""
    dtype, device = kv[0][0].dtype, kv[0][0].device
    pairs = coding_pairs(kv, kpre)
    shapes = [tuple(k.shape) for k, _ in pairs]
    rec = decode_kv(encode_kv(pairs, books), books, shapes, dtype, device)
    krot = rope_keys(model, [k for k, _ in rec], start=start)
    return [(kr, v) for kr, (_, v) in zip(krot, rec)]


# ----------------------------------------------------- the incumbent baselines

def quant_uniform(x, bits, dim):
    """Uniform min/max quantise-dequantise along `dim`. This is what int8/int4 KV
    quantisation actually is in the serving stacks it ships in."""
    lo = x.amin(dim=dim, keepdim=True)
    hi = x.amax(dim=dim, keepdim=True)
    lev = float((1 << bits) - 1)
    scale = (hi - lo).clamp(min=1e-8) / lev
    q = ((x - lo) / scale).round().clamp(0, lev)
    return q * scale + lo


def quant_kv_baseline(kv, bits):
    """The incumbent: per-CHANNEL quantisation for K, per-TOKEN for V.

    That asymmetry is not decoration. Key channels have wildly different scales
    (the massive-activation channels), so a per-token key range is set by the
    outlier channel and crushes the rest; values are the other way round. Getting
    this backwards would hand the codec a strawman to beat.
    """
    out = []
    for K, V in kv:
        Kf, Vf = K.float(), V.float()
        Kq = quant_uniform(Kf, bits, dim=2)   # over tokens => per (head, channel)
        Vq = quant_uniform(Vf, bits, dim=3)   # over channels => per (head, token)
        out.append((Kq.to(K.dtype), Vq.to(V.dtype)))
    return out


def baseline_bits_per_token(kv, bits, group=0):
    """Wire cost of the int-N baseline, scales counted honestly.

    Per-channel K scales amortise over the whole document (2 fp16 per channel);
    per-token V scales cost 2 fp16 per (head, token) and do NOT amortise. With
    group>0 the K scales are per group of `group` tokens, as real int4 KV
    implementations do, and that cost is counted too.
    """
    n = kv[0][0].shape[2]
    total = 0.0
    for K, V in kv:
        _, h, _, d = K.shape
        total += K.numel() * bits + V.numel() * bits
        nk_groups = math.ceil(n / group) if group else 1
        total += h * d * 2 * 16 * nk_groups      # K scale+zero, fp16
        total += h * n * 2 * 16                  # V scale+zero, fp16, per token
    return total / n


# ------------------------------------------------------------------ generation

@torch.inference_mode()
def gen_from_cache(model, kv, q_ids, maxnew, eos=None, pos_offset=0):
    """THE PRODUCT PATH. The receiver never sees the document, runs zero layers
    over it, and generates from the handed-over cache plus its own question.

    cache_position and position_ids are deliberately NOT the same sequence.
    cache_position is a slot index into the handed-over cache and always starts at
    n_doc; position_ids is the RoPE position and starts at pos_offset + n_doc,
    because a pre-RoPE frame re-rotated to sit at `pos_offset` puts the document at
    pos_offset..pos_offset+n_doc. Conflating the two -- which is the easy mistake,
    and the one this project's acceptance test caught -- leaves the question
    overlapping the document in RoPE space while the mask still says it follows.
    """
    n_doc = kv[0][0].shape[2]
    cache = build_cache(kv)
    nq = q_ids.shape[1]
    dev = q_ids.device
    slot = torch.arange(n_doc, n_doc + nq, device=dev)
    o = model(input_ids=q_ids, past_key_values=cache, cache_position=slot,
              position_ids=(slot + pos_offset).unsqueeze(0), use_cache=True)
    past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
    got = [int(nxt)]
    for i in range(maxnew - 1):
        if eos is not None and got[-1] == eos:
            break
        p = n_doc + nq + i
        o = model(input_ids=nxt, past_key_values=past,
                  cache_position=torch.tensor([p], device=dev),
                  position_ids=torch.tensor([[p + pos_offset]], device=dev),
                  use_cache=True)
        past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
        got.append(int(nxt))
    return got


@torch.inference_mode()
def gen_ref(model, ids, maxnew, eos=None):
    """Ground truth: the whole document in context, no frame, no codec."""
    o = model(input_ids=ids, use_cache=True)
    past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
    got = [int(nxt)]
    for _ in range(maxnew - 1):
        if eos is not None and got[-1] == eos:
            break
        o = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
        got.append(int(nxt))
    return got


@torch.inference_mode()
def logits_from_cache(model, kv, ids, pos_offset=0):
    """Teacher-forced logits for `ids` continuing a framed document.

    This is the fine-grained rate-curve metric: needle recall over 12 items is a
    0/1 measurement with error bars the width of the table, whereas per-position
    top-1 agreement over hundreds of tokens shows the shape of the curve.
    """
    n_doc = kv[0][0].shape[2]
    cache = build_cache(kv)
    n = ids.shape[1]
    slot = torch.arange(n_doc, n_doc + n, device=ids.device)
    return model(input_ids=ids, past_key_values=cache, cache_position=slot,
                 position_ids=(slot + pos_offset).unsqueeze(0),
                 use_cache=True).logits.float()


@torch.inference_mode()
def logits_ref(model, doc_ids, tail_ids):
    """Reference logits for the same tail with the document really in context."""
    ids = torch.cat([doc_ids, tail_ids], 1)
    o = model(input_ids=ids, use_cache=False)
    return o.logits[:, doc_ids.shape[1]:].float()


def top1_agreement(a, b):
    """Fraction of positions where two logit tensors pick the same token."""
    return (a.argmax(-1) == b.argmax(-1)).float().mean().item()
