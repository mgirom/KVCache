#!/usr/bin/env python3
"""mscc -- read a document once, hand over the state of having read it.

TWO FRAME FAMILIES, AND ONLY ONE OF THEM ANSWERS QUESTIONS
----------------------------------------------------------
  kvfit / kvencode / kvserve   FULL-DEPTH KV frame.  <-- use these
  fit / encode / serve         mid-stack frame, RETAINED. Kept because the
                               measurement history depends on it, but it cannot
                               answer a question the sender did not already ask:
                               the question's tokens have no document keys below
                               the tap, so it invents answers fluently (Gate 10
                               recall 0/12, Gate 11 handoff 0/9). `serve` now
                               says so on every invocation.

Commands:
  kvfit     fit 2*n_layers codebooks on a corpus       (GPU capture, CPU fit)
  kvencode  document -> full-depth KV frame            (GPU, one pass)
  kvserve   answer questions from a frame alone        (GPU, ZERO document layers)
  inspect   header + guard verdict, no model needed    (CPU, both families)

Every GPU command takes the tree's single lock first. Two jobs never share the
card: the second one waits. Codebook fitting always happens on CPU tensors --
fitting next to the model weights is what OOM'd the 4B run twice.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "alphabet", "scripts"))
sys.path.insert(0, _ROOT)

from mscc import codec as mcodec
from mscc import format as mfmt
from mscc import guard as mguard
from mscc import kv as mkv


# --------------------------------------------------------------------- helpers
def _lock(name):
    import gpulock
    return gpulock.acquire(name)


def _li():
    import lib_inject
    return lib_inject


def _lkv():
    import lib_kv
    return lib_kv


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _default_layer(n_layers: int) -> int:
    """Top ~15% of the stack: 23/28 and 30/36 both land here."""
    return int(round(n_layers * 0.857)) - 1


def _capture(li, tok, model, text, layer, seqlen, max_tokens, log=print):
    """Run the full stack over `text` in windows, keep the layer-L output."""
    import torch
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = min(len(ids), max_tokens) if max_tokens > 0 else len(ids)
    ids = ids[:n]
    out, t0 = [], time.perf_counter()
    for i in range(0, n, seqlen):
        w = ids[i:i + seqlen]
        if len(w) < 8:
            break
        h, _ = li.full_pass(model, w.unsqueeze(0), layer)
        out.append(h[0].cpu())
        del h
        torch.cuda.empty_cache()
        log(f"  captured {min(i + seqlen, n)}/{n} tokens", flush=True)
    states = torch.cat(out, 0)
    return states, ids[:states.shape[0]], time.perf_counter() - t0


# ------------------------------------------------------------------------- fit
def cmd_fit(a):
    import torch
    lk = _lock("mscc-fit"); li = _li()
    tok, model = li.load_model(a.model)
    layers = li.decoder_layers(model)
    L = len(layers)
    layer = a.layer if a.layer >= 0 else _default_layer(L)
    print(f"model: {a.model}  layers={L}  tap=layer {layer} "
          f"({(layer + 1) / L:.1%} of stack skipped at the receiver)", flush=True)

    text = _read_text(a.corpus)
    states, _, secs = _capture(li, tok, model, text, layer, a.seqlen, a.tokens)
    print(f"captured {tuple(states.shape)} in {secs:.1f}s -- fitting on CPU", flush=True)

    del model; torch.cuda.empty_cache()

    meta = dict(model_sha=mfmt.model_fingerprint(a.model), model_id=os.path.basename(a.model),
                layer=layer, n_layers=L, corpus=os.path.basename(a.corpus),
                corpus_sha=mcodec.corpus_digest(text)[:16], seqlen=a.seqlen)
    cb = mcodec.fit(states.float(), a.dims, a.bits, meta=meta)
    size = cb.save(a.out)
    print(json.dumps({"codebook": a.out, "bytes": size, "sha": cb.sha()[:16],
                      "funded_dims": cb.n_dims, "bits_per_token": cb.bits_per_token,
                      "hidden_dim": cb.hidden_dim, "layer": layer,
                      "n_states": int(states.shape[0])}, indent=1))
    del lk


# ---------------------------------------------------------------------- encode
def cmd_encode(a):
    import torch
    cb = mcodec.Codebook.load(a.codebook)
    layer = int(cb.meta["layer"])
    lk = _lock("mscc-encode"); li = _li()
    tok, model = li.load_model(a.model)
    L = len(li.decoder_layers(model))
    model_sha = mfmt.model_fingerprint(a.model)
    if model_sha != cb.meta.get("model_sha"):
        print(f"[REFUSED] codebook was fitted against {cb.meta.get('model_sha', '?')[:8]}..., "
              f"this model is {model_sha[:8]}...", file=sys.stderr)
        sys.exit(2)

    text = _read_text(a.doc)
    states, ids, secs = _capture(li, tok, model, text, layer, a.seqlen, a.tokens)
    n_tok = int(states.shape[0])
    windowed = n_tok > a.seqlen

    codes = cb.encode(states.to("cuda:0"))
    packed = mcodec.pack_codes(codes, cb.b)

    hdr = mfmt.FrameHeader(
        model_sha=model_sha, codebook_sha=cb.sha(), model_id=os.path.basename(a.model),
        layer=layer, n_layers=L, n_dims=cb.n_dims, hidden_dim=cb.hidden_dim,
        bits_per_token=cb.bits_per_token, codec=cb.meta.get("codec_name", "cpca"),
        n_tokens=n_tok, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        notes={"doc": os.path.basename(a.doc), "doc_sha": mcodec.corpus_digest(text)[:16],
               "seqlen": a.seqlen, "windowed": windowed,
               "read_seconds": round(secs, 2)})
    nbytes = mfmt.write_frame(a.out, hdr, {"codes": packed, "ids": ids.numpy().astype(np.int32)})
    if windowed:
        print(f"[WARN] document exceeded one {a.seqlen}-token window; positions repeat "
              f"per window. Untested configuration -- recorded in the header.", file=sys.stderr)
    print(json.dumps({"frame": a.out, "bytes": nbytes, "tokens": n_tok,
                      "bits_per_token": cb.bits_per_token, "layer": layer,
                      "read_seconds": round(secs, 2),
                      "text_bytes": len(text.encode())}, indent=1))
    del model, lk; torch.cuda.empty_cache()


# --------------------------------------------------------------------- inspect
def cmd_inspect(a):
    hdr = mfmt.read_header(a.frame)
    print(json.dumps(json.loads(hdr.to_json()), indent=1))
    is_kv = (hdr.notes or {}).get("kind") == "kv"
    if not a.model:
        print(f"\n({'KV' if is_kv else 'mid-stack'} frame; no --model given: "
              "identity not checked)")
        return
    if is_kv:
        cb_sha = mkv.KVCodebook.load(a.codebook).sha() if a.codebook else None
        r = mkv.check_kv(hdr, mfmt.model_fingerprint(a.model), cb_sha)
    else:
        cb_sha = mcodec.Codebook.load(a.codebook).sha() if a.codebook else None
        r = mguard.check(hdr, mfmt.model_fingerprint(a.model), cb_sha)
    print()
    print(r.report())
    sys.exit(0 if r.ok else 2)


# ================================================================== KV commands
#
# The difference from the mid-stack commands above is one line of behaviour: the
# receiver runs ZERO layers over the document instead of 14% of them, and in
# exchange the question can attend to the document at every depth.

def _kv_capture_corpus(li, lkv, tok, model, text, seqlen, max_tokens, log=print,
                       postrope=False):
    """Prefill the corpus in windows; keep pre-RoPE K and V for every layer.

    Windowed on purpose: the codebook wants the distribution of cache states, and
    a window boundary changes which states occur, not what they mean. One window
    at a time is also what keeps a long corpus off a 12 GB card.
    """
    import torch
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = min(len(ids), max_tokens) if max_tokens > 0 else len(ids)
    n_layers = len(li.decoder_layers(model))
    buf = {key: [] for key in lkv.unit_keys(n_layers)}
    got, t0 = 0, time.perf_counter()
    for i in range(0, n, seqlen):
        w = ids[i:i + seqlen]
        if len(w) < 8:
            break
        kpre = None
        if postrope:
            kv = lkv.capture_kv(model, w.unsqueeze(0).to(li.DEV))
            ksrc = [kk for kk, _ in kv]          # keys as the cache holds them
        else:
            kv, kpre = lkv.capture_kv_prerope(model, w.unsqueeze(0).to(li.DEV))
            ksrc = kpre
        for l, (_, vv) in enumerate(kv):
            buf[(l, "v")].append(lkv.to_matrix(vv).half().cpu())
            buf[(l, "k")].append(lkv.to_matrix(ksrc[l]).half().cpu())
        got += len(w)
        del kv, kpre
        torch.cuda.empty_cache()
        log(f"  captured {got}/{n} tokens", flush=True)
    return {k: torch.cat(v) for k, v in buf.items()}, got, time.perf_counter() - t0


def cmd_kvfit(a):
    import torch
    lk = _lock("mscc-kvfit"); li = _li(); lkv = _lkv()
    tok, model = li.load_model(a.model)
    n_layers = len(li.decoder_layers(model))
    cfg = model.config
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    print(f"model: {a.model}  layers={n_layers}  kv_heads={kv_heads} "
          f"head_dim={head_dim}  -> {2 * n_layers} coding units", flush=True)

    text = _read_text(a.corpus)
    states, n_tok, secs = _kv_capture_corpus(li, lkv, tok, model, text,
                                             a.seqlen, a.tokens, postrope=a.postrope)
    print(f"captured {n_tok} tokens x {len(states)} units in {secs:.1f}s "
          f"-- fitting on CPU", flush=True)
    del model
    torch.cuda.empty_cache()

    meta = dict(model_sha=mfmt.model_fingerprint(a.model),
                model_id=os.path.basename(a.model), n_layers=n_layers,
                kv_heads=int(kv_heads), head_dim=int(head_dim),
                basis="postrope" if a.postrope else "prerope",
                unit_bits=a.unit_bits, dims=a.dims,
                corpus=os.path.basename(a.corpus),
                corpus_sha=mcodec.corpus_digest(text)[:16],
                n_states=int(n_tok), seqlen=a.seqlen)
    if a.per_head:
        # same TOTAL budget, split across heads: the basis stops mixing heads, which
        # is what makes a code-space attention kernel affordable
        meta["per_head"] = True
        meta["bits_per_head"] = a.unit_bits // kv_heads
        books = lkv.fit_kv_codebooks_perhead(
            {k: v.float() for k, v in states.items()},
            a.unit_bits // kv_heads, int(kv_heads), int(head_dim),
            progress=lambda i, n, key: (i % 14 == 0 and
                                        print(f"  fitted {i+1}/{n} units", flush=True)))
        states.clear()
    else:
        books = {}
        for i, key in enumerate(sorted(states)):
            books[key] = mcodec.fit(states.pop(key).float(), a.dims, a.unit_bits,
                                    meta={"kv_layer": key[0], "kv_unit": key[1],
                                          **meta})
            if i % 14 == 0:
                print(f"  fitted {i + 1}/{2 * n_layers} units", flush=True)
    cb = mkv.KVCodebook(books=books, meta=meta)
    size = cb.save(a.out)
    raw = n_layers * 2 * kv_heads * head_dim * 16
    print(json.dumps({"codebook": a.out, "bytes": size, "sha": cb.sha()[:16],
                      "units": len(books), "unit_bits": a.unit_bits,
                      "bits_per_token": cb.bits_per_token,
                      "raw_bits_per_token": raw,
                      "compression_vs_bf16": round(raw / cb.bits_per_token, 1),
                      "basis": meta["basis"], "n_states": n_tok}, indent=1))
    del lk


def cmd_kvencode(a):
    import torch
    cb = mkv.KVCodebook.load(a.codebook)
    lk = _lock("mscc-kvencode"); li = _li(); lkv = _lkv()
    tok, model = li.load_model(a.model)
    model_sha = mfmt.model_fingerprint(a.model)
    if model_sha != cb.meta.get("model_sha"):
        print(f"[REFUSED] codebook was fitted against "
              f"{cb.meta.get('model_sha', '?')[:8]}..., this model is "
              f"{model_sha[:8]}...", file=sys.stderr)
        sys.exit(2)

    text = _read_text(a.doc)
    ids = tok(text, return_tensors="pt").input_ids
    if a.tokens > 0:
        ids = ids[:, :a.tokens]
    n_tok = int(ids.shape[1])
    t0 = time.perf_counter()
    kv, kpre = lkv.capture_kv_prerope(model, ids.to(li.DEV))
    read_s = time.perf_counter() - t0
    raw_bpt = lkv.kv_bits_per_token(kv)

    # --- the health stamp. The sender has the true cache, so it can measure what the
    # receiver will not be able to: frame all but the last P tokens, teacher-force
    # those P through the decoded cache, and record the agreement. It failed its
    # pre-registered bar (Gate 13) and is therefore ADVISORY, never a gate.
    stamp = None
    if a.probe > 0 and n_tok > a.probe + 16:
        import torch as _t
        pdoc = ids[:, :n_tok - a.probe].to(li.DEV)
        ptail = ids[:, n_tok - a.probe:].to(li.DEV)
        lref = lkv.logits_ref(model, pdoc, ptail)
        pkv, pkpre = lkv.capture_kv_prerope(model, pdoc)
        pq = lkv.merge_exact(pkv, lkv.roundtrip_kv_prerope(
            model, pkv, pkpre, cb.books), n_sink=a.sink)
        stamp = round(lkv.top1_agreement(
            lkv.logits_from_cache(model, pq, ptail), lref), 4)
        del pkv, pkpre, pq, lref
        _t.cuda.empty_cache()

    pairs = lkv.coding_pairs(kv, kpre)
    payload = mkv.encode_frame(pairs, cb, n_sink=a.sink)
    payload["ids"] = ids[0].cpu().numpy().astype(np.int32)
    bpt = mkv.frame_bits_per_token(cb, n_tok, a.sink, raw_bpt)

    n_layers = cb.meta["n_layers"]
    hdr = mfmt.FrameHeader(
        model_sha=model_sha, codebook_sha=cb.sha(),
        model_id=os.path.basename(a.model), layer=-1, n_layers=n_layers,
        n_dims=cb.meta["dims"], hidden_dim=cb.meta["kv_heads"] * cb.meta["head_dim"],
        bits_per_token=int(round(bpt)), codec=f"kvcpca{cb.meta['unit_bits']}",
        n_tokens=n_tok,
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        notes={"kind": "kv", "key_basis": cb.basis, "sink": a.sink,
               "unit_bits": cb.meta["unit_bits"], "kv_heads": cb.meta["kv_heads"],
               "head_dim": cb.meta["head_dim"], "position_offset": 0,
               "doc": os.path.basename(a.doc),
               "doc_sha": mcodec.corpus_digest(text)[:16],
               "health_probe": stamp, "health_probe_tokens": a.probe if stamp else 0,
               "read_seconds": round(read_s, 2)})
    nbytes = mfmt.write_frame(a.out, hdr, payload)
    print(json.dumps({"frame": a.out, "bytes": nbytes, "tokens": n_tok,
                      "bits_per_token": round(bpt, 1),
                      "raw_kv_bits_per_token": raw_bpt,
                      "compression_vs_bf16": round(raw_bpt / bpt, 1),
                      "uncompressed_kv_bytes": int(raw_bpt * n_tok / 8),
                      "text_bytes": len(text.encode()),
                      "health_probe": stamp,
                      "read_seconds": round(read_s, 2)}, indent=1))
    del model, kv, kpre, lk
    torch.cuda.empty_cache()


def cmd_kvserve(a):
    import torch
    fr = mfmt.read_frame(a.frame)
    hdr = fr.header
    cb = mkv.KVCodebook.load(a.codebook)
    lk = _lock("mscc-kvserve"); li = _li(); lkv = _lkv()
    tok, model = li.load_model(a.model)

    # the guard runs before anything expensive, and before any output
    r = mkv.check_kv(hdr, mfmt.model_fingerprint(a.model), cb.sha())
    print(r.report(), flush=True)
    if not r.ok:
        if not a.force:
            print("\nrefused: the receiver will read the document normally instead.",
                  file=sys.stderr)
            sys.exit(2)
        # --force exists for one reason: so a demo can SHOW that a rejected frame
        # does not fail loudly, it answers fluently and wrongly. Never a production path.
        print("\n[FORCED] guard overridden. Output below is the failure mode the "
              "guard exists to prevent -- treat it as a demonstration, not an answer.",
              file=sys.stderr)

    dt = next(model.parameters()).dtype
    n = hdr.n_tokens
    sink = (hdr.notes or {}).get("sink", 0)
    t0 = time.perf_counter()
    kv = mkv.decode_frame(fr.payload, cb, n, n_sink=sink, dtype=dt, device=li.DEV)
    if (hdr.notes or {}).get("key_basis") == "prerope":
        krot = lkv.rope_keys(model, [k for k, _ in kv], start=a.offset)
        kv = [(kr, v) for kr, (_, v) in zip(krot, kv)]
    decode_s = time.perf_counter() - t0

    q_ids = tok(a.ask, return_tensors="pt").input_ids.to(li.DEV)
    t0 = time.perf_counter()
    got = lkv.gen_from_cache(model, kv, q_ids, a.gen, eos=tok.eos_token_id,
                             pos_offset=a.offset)
    gen_s = time.perf_counter() - t0

    print(f"\nblocks executed over the document: 0 of {hdr.n_layers}  (100% skipped)")
    print(f"document tokens carried by the frame: {n}")
    print(f"frame decode: {decode_s * 1000:.0f} ms   generation: {gen_s * 1000:.0f} ms")
    print(f"\nQ: {a.ask}\nA: {tok.decode(got, skip_special_tokens=True).strip()}")
    del model, kv, lk
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------- serve
def cmd_serve(a):
    import torch, torch.nn as nn
    fr = mfmt.read_frame(a.frame)
    hdr = fr.header
    cb = mcodec.Codebook.load(a.codebook)
    lk = _lock("mscc-serve"); li = _li()
    tok, model = li.load_model(a.model)
    model_sha = mfmt.model_fingerprint(a.model)

    # --- the guard runs before anything expensive, and before any output
    r = mguard.check(hdr, model_sha, cb.sha())
    print(r.report(), flush=True)
    if not r.ok:
        print("\nrefused: the receiver will read the document normally instead.",
              file=sys.stderr)
        sys.exit(2)

    layer = hdr.layer
    base = model.model
    orig = nn.ModuleList(list(li.decoder_layers(model)))
    dt = next(model.parameters()).dtype

    codes = mcodec.unpack_codes(fr.payload["codes"], cb.b, hdr.n_tokens)
    doc_h = cb.decode(codes, device="cuda:0",
                      shape=(1, hdr.n_tokens, cb.hidden_dim)).to(dt)

    q_ids = tok(a.ask, return_tensors="pt").input_ids.to("cuda:0")
    with torch.inference_mode():
        # the question never saw the document in the lower layers -- by design.
        o = model(input_ids=q_ids, use_cache=True, output_hidden_states=True)
        q_h = o.hidden_states[layer + 1].to(dt)
        bottom_cache = o.past_key_values

        base.layers = nn.ModuleList(list(orig)[layer + 1:])
        try:
            t0 = time.perf_counter()
            o2 = model(inputs_embeds=torch.cat([doc_h, q_h], 1), use_cache=True)
            top_cache = o2.past_key_values
            nxt = o2.logits[:, -1:].argmax(-1)
            ttft = time.perf_counter() - t0
            got = [int(nxt)]
            for _ in range(a.gen - 1):
                base.layers = orig
                ob = model(input_ids=nxt, past_key_values=bottom_cache,
                           use_cache=True, output_hidden_states=True)
                hb = ob.hidden_states[layer + 1].to(dt)
                base.layers = nn.ModuleList(list(orig)[layer + 1:])
                ot = model(inputs_embeds=hb, past_key_values=top_cache, use_cache=True)
                top_cache = ot.past_key_values
                nxt = ot.logits[:, -1:].argmax(-1)
                if int(nxt) in (tok.eos_token_id,):
                    break
                got.append(int(nxt))
        finally:
            base.layers = orig

    print(f"\nblocks executed over the document: {len(orig) - layer - 1} of {len(orig)}"
          f"  ({(layer + 1) / len(orig):.1%} skipped)")
    print(f"time to first token: {ttft * 1000:.0f} ms")
    print(f"\nQ: {a.ask}\nA: {tok.decode(got, skip_special_tokens=True).strip()}")
    del model, lk; torch.cuda.empty_cache()


# ------------------------------------------------------------------------ main
def main(argv=None):
    p = argparse.ArgumentParser(prog="mscc", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit a codebook on a corpus")
    f.add_argument("--corpus", required=True); f.add_argument("--model", required=True)
    f.add_argument("-o", "--out", required=True)
    f.add_argument("--layer", type=int, default=-1)
    f.add_argument("--dims", type=int, default=1024)
    f.add_argument("--bits", type=int, default=2048)
    f.add_argument("--seqlen", type=int, default=1024)
    f.add_argument("--tokens", type=int, default=120_000)
    f.set_defaults(fn=cmd_fit)

    e = sub.add_parser("encode", help="turn a document into a frame")
    e.add_argument("doc"); e.add_argument("--model", required=True)
    e.add_argument("--codebook", required=True); e.add_argument("-o", "--out", required=True)
    e.add_argument("--seqlen", type=int, default=2048)
    e.add_argument("--tokens", type=int, default=0)
    e.set_defaults(fn=cmd_encode)

    i = sub.add_parser("inspect", help="header + guard verdict")
    i.add_argument("frame"); i.add_argument("--model", default="")
    i.add_argument("--codebook", default="")
    i.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("serve", help="answer a question from a frame alone")
    s.add_argument("--frame", required=True); s.add_argument("--model", required=True)
    s.add_argument("--codebook", required=True); s.add_argument("--ask", required=True)
    s.add_argument("--gen", type=int, default=48)
    s.set_defaults(fn=cmd_serve)

    kf = sub.add_parser("kvfit", help="fit the 2*n_layers KV codebooks on a corpus")
    kf.add_argument("--corpus", required=True); kf.add_argument("--model", required=True)
    kf.add_argument("-o", "--out", required=True)
    kf.add_argument("--unit-bits", type=int, default=1024,
                    help="bits per token per (layer,K|V) unit. 1024 is the default "
                         "(15.1x, full recall); 512 is the measured floor (28.4x, "
                         "10-of-12); 256 answers everything wrongly")
    kf.add_argument("--dims", type=int, default=1024)
    kf.add_argument("--postrope", action="store_true",
                    help="code keys as the cache holds them (rotated). Compresses "
                         "worse, but it is the only basis the query-side attention "
                         "fold works in -- and it needs no architecture-specific hook, "
                         "so it runs on any HF model.")
    kf.add_argument("--per-head", action="store_true",
                    help="fit one basis per attention head instead of one across all "
                         "of them, at the same total rate. Compresses worse; required "
                         "for the code-space attention fold.")
    kf.add_argument("--seqlen", type=int, default=512)
    kf.add_argument("--tokens", type=int, default=12_288)
    kf.set_defaults(fn=cmd_kvfit)

    ke = sub.add_parser("kvencode", help="document -> full-depth KV frame")
    ke.add_argument("doc"); ke.add_argument("--model", required=True)
    ke.add_argument("--codebook", required=True)
    ke.add_argument("-o", "--out", required=True)
    ke.add_argument("--sink", type=int, default=4,
                    help="leading tokens carried at full precision; 0 costs ~0.31 "
                         "top-1 agreement (Gate 12 ablation)")
    ke.add_argument("--tokens", type=int, default=0)
    ke.add_argument("--probe", type=int, default=64,
                    help="tokens used for the sender's self-declared health stamp; "
                         "0 to skip. Advisory only -- it missed its pre-registered "
                         "bar (Gate 13), so it never gates anything.")
    ke.set_defaults(fn=cmd_kvencode)

    ks = sub.add_parser("kvserve", help="answer from a KV frame; zero document layers")
    ks.add_argument("--frame", required=True); ks.add_argument("--model", required=True)
    ks.add_argument("--codebook", required=True); ks.add_argument("--ask", required=True)
    ks.add_argument("--gen", type=int, default=48)
    ks.add_argument("--force", action="store_true",
                    help="answer even if the guard rejects the frame. Only for "
                         "demonstrating what refusal prevents.")
    ks.add_argument("--offset", type=int, default=0,
                    help="absolute position to replay the frame at (pre-RoPE only)")
    ks.set_defaults(fn=cmd_kvserve)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    main()
