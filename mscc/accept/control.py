"""Why did true-skip serve get the planted fact wrong?

Three paths, same document, same question:
  A  reference   -- full model, doc+question in context, no codec at all
  B  hook inject -- FULL prefill, layer-23 output swapped for the codec's
                    reconstruction. This is what gate8 (12/12 needle) measured.
  C  true skip   -- layers 0..23 never run over the document; the question is
                    processed alone and joined at layer 23. This is the product.
If A and B are right and C is wrong, the codec is fine and the SKIP is what costs.
"""
import os, sys, torch, torch.nn as nn
import os
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(R, "lib"), R]
import lib_inject as li
from mscc import codec as mcodec, format as mfmt

Q = "\n\nQuestion: What is the calibration marker for this archive copy? Answer with the code only.\nAnswer:"
doc = open(f"{R}/mscc/accept/doc.txt").read()
cb = mcodec.Codebook.load(f"{R}/mscc/accept/book.cb.npz")
fr = mfmt.read_frame(f"{R}/mscc/accept/doc.mscc")
L = int(fr.header.layer)

tok, model = li.load_model(f"{R}/models/qwen3-1.7b-fp")
dt = next(model.parameters()).dtype
base, orig = model.model, nn.ModuleList(list(li.decoder_layers(model)))


def gen(fn, n=16):
    out, cache = [], None
    for _ in range(n):
        nxt, cache = fn(out, cache)
        if int(nxt) == tok.eos_token_id:
            break
        out.append(int(nxt))
    return tok.decode(out, skip_special_tokens=True).strip()


with torch.inference_mode():
    ids = tok(doc + Q, return_tensors="pt").input_ids.to("cuda:0")
    n_doc = tok(doc, return_tensors="pt").input_ids.shape[1]

    # ---- A: reference, no codec
    def a(out, cache):
        inp = ids if cache is None else torch.tensor([[out[-1]]], device="cuda:0")
        o = model(input_ids=inp, past_key_values=cache, use_cache=True)
        return o.logits[:, -1:].argmax(-1), o.past_key_values
    print("A reference      :", repr(gen(a)))

    # ---- B: hook injection, full prefill (the gate8 configuration)
    rec = cb.decode(mcodec.unpack_codes(fr.payload["codes"], cb.b, fr.header.n_tokens),
                    device="cuda:0", shape=(1, fr.header.n_tokens, cb.hidden_dim)).to(dt)
    def b(out, cache):
        if cache is None:
            h_full, _ = li.full_pass(model, ids[0].unsqueeze(0).cpu(), L)
            h = h_full.to(dt).clone()
            h[:, :rec.shape[1]] = rec           # doc positions only
            holder = {"h": h}
            def hook(m, args, o):
                r = holder["h"].to(o[0].dtype if isinstance(o, tuple) else o.dtype)
                return (r,) + tuple(o[1:]) if isinstance(o, tuple) else r
            hd = orig[L].register_forward_hook(hook)
            try:
                o = model(input_ids=ids, use_cache=True)
            finally:
                hd.remove()
        else:
            o = model(input_ids=torch.tensor([[out[-1]]], device="cuda:0"),
                      past_key_values=cache, use_cache=True)
        return o.logits[:, -1:].argmax(-1), o.past_key_values
    print("B hook, full pre :", repr(gen(b)))

    # ---- C: true skip, question joined at layer 23 (what mscc serve does)
    oq = model(input_ids=tok(Q, return_tensors="pt").input_ids.to("cuda:0"),
               use_cache=True, output_hidden_states=True)
    qh, bcache = oq.hidden_states[L + 1].to(dt), oq.past_key_values
    base.layers = nn.ModuleList(list(orig)[L + 1:])
    try:
        oc = model(inputs_embeds=torch.cat([rec, qh], 1), use_cache=True)
        tcache, nxt = oc.past_key_values, oc.logits[:, -1:].argmax(-1)
        got = [int(nxt)]
        for _ in range(15):
            base.layers = orig
            ob = model(input_ids=nxt, past_key_values=bcache, use_cache=True,
                       output_hidden_states=True)
            base.layers = nn.ModuleList(list(orig)[L + 1:])
            ot = model(inputs_embeds=ob.hidden_states[L + 1].to(dt),
                       past_key_values=tcache, use_cache=True)
            tcache, nxt = ot.past_key_values, ot.logits[:, -1:].argmax(-1)
            if int(nxt) == tok.eos_token_id: break
            got.append(int(nxt))
    finally:
        base.layers = orig
    print("C true skip      :", repr(tok.decode(got, skip_special_tokens=True).strip()))
print("\nexpected: BRK-7742")
