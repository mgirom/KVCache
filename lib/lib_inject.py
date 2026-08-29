#!/usr/bin/env python3
"""Shared harness: per-token hidden-state capture, codebook fit, mid-stack injection.

Convention (HuggingFace): hidden_states[0] = embeddings,
hidden_states[i] = OUTPUT of decoder layer i-1. So "inject at layer L" means:
replace the output of layers[L] (0-based) and let layers[L+1:] run normally.
Layers 0..L are then skippable at the receiver.
"""
import json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda:0"


# ---------------------------------------------------------------- model utils
def load_model(path, dtype=torch.bfloat16):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, device_map=DEV)
    m.eval()
    return tok, m


def decoder_layers(model):
    """Return the ModuleList of decoder blocks, architecture-agnostic."""
    for attr in ("model", "transformer", "gpt_neox"):
        base = getattr(model, attr, None)
        if base is None:
            continue
        for lattr in ("layers", "h", "blocks"):
            layers = getattr(base, lattr, None)
            if layers is not None:
                return layers
    raise RuntimeError("could not locate decoder layers")


def chunk_tokens(tok, text, seqlen, nseq, stride_skip=0):
    ids = tok(text, return_tensors="pt").input_ids[0]
    out = []
    i = stride_skip * seqlen
    while len(out) < nseq and i + seqlen + 1 <= len(ids):
        out.append(ids[i:i + seqlen])
        i += seqlen
    return torch.stack(out)


# ------------------------------------------------------------- forward passes
@torch.inference_mode()
def full_pass(model, batch, layer):
    """Returns (hidden at layer output, logits) for one batch on GPU."""
    o = model(input_ids=batch.to(DEV), output_hidden_states=True, use_cache=False)
    return o.hidden_states[layer + 1].float(), o.logits.float()


@torch.inference_mode()
def inject_pass(model, batch, layer, new_hidden):
    """Run the model but overwrite the output of decoder block `layer`."""
    layers = decoder_layers(model)
    holder = {"h": new_hidden}

    def hook(module, args, output):
        rep = holder["h"].to(output[0].dtype if isinstance(output, tuple) else output.dtype)
        if isinstance(output, tuple):
            return (rep,) + tuple(output[1:])
        return rep

    h = layers[layer].register_forward_hook(hook)
    try:
        o = model(input_ids=batch.to(DEV), use_cache=False)
    finally:
        h.remove()
    return o.logits.float()


# ------------------------------------------------------------------- metrics
def compare(logits_ref, logits_test, targets, chunk=2048):
    """top-1 agreement with reference, KL(ref||test), NLL vs true targets.

    Chunked over positions: a (bs,seq,151936) fp32 logit tensor is ~600 MB, and
    a naive softmax over both copies OOMs a 12 GB card.
    """
    V = logits_ref.shape[-1]
    lr = logits_ref[:, :-1].reshape(-1, V)
    lt = logits_test[:, :-1].reshape(-1, V)
    tg = targets[:, 1:].reshape(-1).to(lr.device)
    n = lr.shape[0]
    top1 = ov = 0.0
    kl = nll_ref = nll_test = 0.0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        a, b = lr[s:e], lt[s:e]
        top1 += (a.argmax(-1) == b.argmax(-1)).float().sum().item()
        t5a = a.topk(5, -1).indices
        t5b = b.topk(5, -1).indices
        ov += (t5a.unsqueeze(2) == t5b.unsqueeze(1)).any(2).float().mean(1).sum().item()
        pa = F.log_softmax(a, -1)
        pb = F.log_softmax(b, -1)
        kl += (pa.exp() * (pa - pb)).sum(-1).sum().item()
        idx = tg[s:e].unsqueeze(1)
        nll_ref += -pa.gather(1, idx).sum().item()
        nll_test += -pb.gather(1, idx).sum().item()
        del a, b, pa, pb, t5a, t5b
    return dict(top1_agree=round(top1 / n, 4), top5_overlap=round(ov / n, 4),
                kl=round(kl / n, 4), ppl_ref=round(math.exp(nll_ref / n), 3),
                ppl_test=round(math.exp(nll_test / n), 3))


# ----------------------------------------------------------------------- RVQ
def kmeans(X, k, iters=20, seed=0):
    """Vectorised k-means (scatter_add update, no python loop over clusters)."""
    g = torch.Generator(device=X.device).manual_seed(seed)
    n, d = X.shape
    C = X[torch.randperm(n, generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, C).argmin(1)
        S = torch.zeros_like(C)
        S.index_add_(0, a, X)
        cnt = torch.bincount(a, minlength=k).unsqueeze(1).float()
        empty = (cnt.squeeze(1) == 0)
        C = torch.where(cnt > 0, S / cnt.clamp(min=1), C)
        if empty.any():
            ridx = torch.randint(n, (int(empty.sum()),), generator=g, device=X.device)
            C[empty] = X[ridx]
    return C


def rvq_fit(X, k, D, iters=20, seed=0):
    R = X.clone()
    books = []
    for d in range(D):
        C = kmeans(R, k, iters=iters, seed=seed + d)
        a = torch.cdist(R, C).argmin(1)
        books.append(C)
        R = R - C[a]
    return books


def rvq_encode(books, X, chunk=8192):
    R = X.clone()
    codes = torch.empty((X.shape[0], len(books)), dtype=torch.int16, device=X.device)
    for d, C in enumerate(books):
        a = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
        for s in range(0, X.shape[0], chunk):
            e = min(s + chunk, X.shape[0])
            a[s:e] = torch.cdist(R[s:e], C).argmin(1)
        codes[:, d] = a.to(torch.int16)
        R = R - C[a]
    return codes


def rvq_decode(books, codes):
    out = torch.zeros((codes.shape[0], books[0].shape[1]), device=codes.device)
    for d, C in enumerate(books):
        out += C[codes[:, d].long()]
    return out


# ------------------------------------------------- product quantiser + codecs
def pq_fit(X, M, k, iters=20, seed=0):
    """Product quantisation: split d into M contiguous subspaces, k-means each."""
    d = X.shape[1]
    assert d % M == 0, f"{d} not divisible by {M}"
    sub = d // M
    return [kmeans(X[:, m*sub:(m+1)*sub].contiguous(), k, iters=iters, seed=seed+m)
            for m in range(M)]


def pq_encode(books, X, chunk=16384):
    M = len(books); sub = X.shape[1] // M
    codes = torch.empty((X.shape[0], M), dtype=torch.int32, device=X.device)
    for m, C in enumerate(books):
        Xs = X[:, m*sub:(m+1)*sub].contiguous()
        for s in range(0, X.shape[0], chunk):
            e = min(s + chunk, X.shape[0])
            codes[s:e, m] = torch.cdist(Xs[s:e], C).argmin(1).to(torch.int32)
    return codes


def pq_decode(books, codes):
    M = len(books); sub = books[0].shape[1]
    out = torch.empty((codes.shape[0], M * sub), device=codes.device)
    for m, C in enumerate(books):
        out[:, m*sub:(m+1)*sub] = C[codes[:, m].long()]
    return out


def split_norm(X, eps=1e-6):
    """Direction + log-norm. The residual stream has huge per-token norm spread;
    quantising direction and scale separately is far cheaper than either alone."""
    n = X.norm(dim=1, keepdim=True).clamp(min=eps)
    return X / n, n


def quant_scalar(v, bits, lo, hi):
    """Uniform scalar quantise-dequantise into `bits`, clamped to [lo,hi]."""
    levels = (1 << bits) - 1
    q = ((v.clamp(lo, hi) - lo) / (hi - lo) * levels).round()
    return q, lo + q / levels * (hi - lo)


# ------------------------------------------- transform coding for residual streams
# The residual stream is dominated by a few "massive activation" channels:
# on Qwen3-1.7B layer 23 the top channel has std 772 against a median of 22.
# Uniform quantisation sets its range from those outliers and crushes everything
# else, so both per-channel standardisation and a PCA rotation matter enormously.

def fit_pca(Xc, m):
    """Xc mean-centred. Returns (V[:, :m], eigenvalues[:m]) descending."""
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov.double())
    idx = torch.argsort(evals, descending=True)[:m]
    return evecs[:, idx].float().contiguous(), evals[idx].float().clamp(min=1e-12)


def alloc_bits(evals, total_bits, max_bits=12):
    """Reverse water-filling: b_i ~ 0.5*log2(lambda_i) + c, sum(b_i)=total_bits."""
    lg = 0.5 * torch.log2(evals)
    lo, hi = -60.0, 60.0
    for _ in range(60):
        c = (lo + hi) / 2
        b = (lg + c).round().clamp(0, max_bits)
        if b.sum() > total_bits: hi = c
        else: lo = c
    b = (lg + lo).round().clamp(0, max_bits)
    # spend any slack on the highest-variance components still under the cap
    slack = int(total_bits - b.sum().item())
    order = torch.argsort(evals, descending=True)
    i = 0
    while slack > 0 and i < len(order) * max_bits:
        j = order[i % len(order)].item()
        if b[j] < max_bits:
            b[j] += 1; slack -= 1
        i += 1
    return b.to(torch.int32)


class PCACodec:
    """Transform coding: rotate to the PCA basis, then spend a fixed bit budget
    across components by reverse water-filling. Side info (basis + scales) is a
    one-off shared artefact, not per-token wire cost."""

    def __init__(self, Xc, m, total_bits, max_bits=12):
        self.V, evals = fit_pca(Xc, m)
        Z = Xc @ self.V
        self.b = alloc_bits(evals, total_bits, max_bits)
        self.keep = self.b > 0
        self.V = self.V[:, self.keep].contiguous()
        self.b = self.b[self.keep]
        Zk = Z[:, self.keep]
        self.lo = Zk.quantile(0.001, dim=0)
        self.hi = Zk.quantile(0.999, dim=0)
        self.bits = int(self.b.sum().item())

    def __call__(self, Xc):
        Z = Xc @ self.V
        lev = (2 ** self.b.float() - 1).clamp(min=1)
        q = ((Z.clamp(self.lo, self.hi) - self.lo) / (self.hi - self.lo) * lev).round()
        Zq = self.lo + q / lev * (self.hi - self.lo)
        return Zq @ self.V.T


class StdPQCodec:
    """Per-channel standardise, then product-quantise. Cheapest fix for the
    outlier-channel problem: no rotation, just divide each channel by its std."""

    def __init__(self, Xc, M, K, iters=15):
        self.s = Xc.std(0).clamp(min=1e-4)
        Xs = Xc / self.s
        self.books = pq_fit(Xs, M, K, iters=iters)
        self.bits = round(M * math.log2(K))

    def __call__(self, Xc):
        Xs = Xc / self.s
        return pq_decode(self.books, pq_encode(self.books, Xs)) * self.s


class SQCCodec:
    """Per-CHANNEL uniform scalar quantisation. The obvious baseline, and the
    one a global-range quantiser fails to beat because outlier channels set the
    range for everybody. Side info: 2*hidden floats, amortised to ~0 per token."""

    def __init__(self, Xc, bits, sample=200_000):
        idx = torch.randperm(Xc.shape[0], device=Xc.device)[:sample]
        S = Xc[idx]
        self.lo = S.quantile(0.001, dim=0)
        self.hi = S.quantile(0.999, dim=0)
        self.nb = bits
        self.bits = Xc.shape[1] * bits

    def __call__(self, Xc):
        lev = float((1 << self.nb) - 1)
        q = ((Xc.clamp(self.lo, self.hi) - self.lo) / (self.hi - self.lo) * lev).round()
        return self.lo + q / lev * (self.hi - self.lo)


class CPCACodec:
    """Correlation-PCA transform coder: per-channel standardise FIRST (so the
    massive-activation channels stop monopolising the basis), then rotate to the
    principal basis and spend the bit budget by reverse water-filling."""

    def __init__(self, Xc, m, total_bits, max_bits=12, sample=200_000):
        self.s = Xc.std(0).clamp(min=1e-4)
        Xs = Xc / self.s
        self.V, evals = fit_pca(Xs, m)
        idx = torch.randperm(Xs.shape[0], device=Xs.device)[:sample]
        Z = Xs[idx] @ self.V
        self.b = alloc_bits(evals, total_bits, max_bits)
        keep = self.b > 0
        self.V = self.V[:, keep].contiguous()
        self.b = self.b[keep]
        Zk = Z[:, keep]
        self.lo = Zk.quantile(0.001, dim=0)
        self.hi = Zk.quantile(0.999, dim=0)
        self.bits = int(self.b.sum().item())

    def __call__(self, Xc):
        Z = (Xc / self.s) @ self.V
        lev = (2 ** self.b.float() - 1).clamp(min=1)
        q = ((Z.clamp(self.lo, self.hi) - self.lo) / (self.hi - self.lo) * lev).round()
        Zq = self.lo + q / lev * (self.hi - self.lo)
        return (Zq @ self.V.T) * self.s


class TernaryCPCACodec(CPCACodec):
    """Same transform coder, but every component is quantised to a power of 3
    (3, 9, 27, 81, 243, 729 levels = 1..6 trits) instead of a power of 2.
    Tests directly whether base 3 costs anything at matched wire bits."""

    def __init__(self, Xc, m, total_bits, max_trits=6, sample=200_000):
        self.s = Xc.std(0).clamp(min=1e-4)
        Xs = Xc / self.s
        self.V, evals = fit_pca(Xs, m)
        idx = torch.randperm(Xs.shape[0], device=Xs.device)[:sample]
        Z = Xs[idx] @ self.V
        # allocate in trits: a trit is log2(3)=1.585 bits
        t = alloc_bits(evals, total_bits / math.log2(3), max_bits=max_trits)
        keep = t > 0
        self.V = self.V[:, keep].contiguous()
        self.t = t[keep]
        Zk = Z[:, keep]
        self.lo = Zk.quantile(0.001, dim=0)
        self.hi = Zk.quantile(0.999, dim=0)
        self.trits = int(self.t.sum().item())
        self.bits = round(self.trits * math.log2(3))

    def __call__(self, Xc):
        Z = (Xc / self.s) @ self.V
        lev = (3.0 ** self.t.float() - 1).clamp(min=1)
        q = ((Z.clamp(self.lo, self.hi) - self.lo) / (self.hi - self.lo) * lev).round()
        Zq = self.lo + q / lev * (self.hi - self.lo)
        return (Zq @ self.V.T) * self.s


# --------------------------------------------------- binary-symmetric channel
def bsc(q, nbits, p, gen):
    """Flip each bit of integer tensor q (value < 2**nbits) with probability p.
    nbits may be a scalar or a per-column tensor."""
    if p <= 0:
        return q
    qi = q.to(torch.int64)
    nb = nbits if torch.is_tensor(nbits) else torch.full((q.shape[1],), int(nbits),
                                                         device=q.device)
    maxb = int(nb.max().item())
    out = qi.clone()
    for bit in range(maxb):
        active = (nb > bit).unsqueeze(0)
        flip = (torch.rand(q.shape, generator=gen, device=q.device) < p) & active
        out = out ^ (flip.to(torch.int64) << bit)
    return out


def _sqc_call(self, Xc, p=0.0, gen=None):
    lev = float((1 << self.nb) - 1)
    q = ((Xc.clamp(self.lo, self.hi) - self.lo) / (self.hi - self.lo) * lev).round()
    if p > 0:
        q = bsc(q, self.nb, p, gen).clamp(0, lev).float()
    return self.lo + q / lev * (self.hi - self.lo)


def _cpca_call(self, Xc, p=0.0, gen=None):
    Z = (Xc / self.s) @ self.V
    lev = (2 ** self.b.float() - 1).clamp(min=1)
    q = ((Z.clamp(self.lo, self.hi) - self.lo) / (self.hi - self.lo) * lev).round()
    if p > 0:
        q = bsc(q, self.b, p, gen).float().minimum(lev)
    Zq = self.lo + q / lev * (self.hi - self.lo)
    return (Zq @ self.V.T) * self.s


def _tcpca_call(self, Xc, p=0.0, gen=None):
    Z = (Xc / self.s) @ self.V
    lev = (3.0 ** self.t.float() - 1).clamp(min=1)
    q = ((Z.clamp(self.lo, self.hi) - self.lo) / (self.hi - self.lo) * lev).round()
    if p > 0:
        # ternary symbols still travel over a binary link: ceil(t*log2 3) bits
        nb = torch.ceil(self.t.float() * math.log2(3)).to(torch.int64)
        q = bsc(q, nb, p, gen).float().minimum(lev)
    Zq = self.lo + q / lev * (self.hi - self.lo)
    return (Zq @ self.V.T) * self.s


SQCCodec.__call__ = _sqc_call
CPCACodec.__call__ = _cpca_call
TernaryCPCACodec.__call__ = _tcpca_call
