#!/usr/bin/env python3
"""Rotate a model's residual stream into a fitted (or Hadamard) basis before weight
quantisation. The weight-side twin of the fitted KV basis.

A pre-norm transformer's residual stream x is only ever read through RMSNorm followed by
a linear map, and only ever written by a linear map added back in. RMSNorm's scale is
rotation-invariant, so for any orthogonal R the model with x' = R^T x is identical if:
  embedding rows       E'      = E R                 (x' = R^T x for every token)
  every norm gain w    folds into the linear that follows: W_in' = W_in diag(w) R, gain = 1
  residual writers     W_out'  = R^T W_out           (o_proj, down_proj)
  final norm + head    W_lm'   = W_lm diag(w_f) R    (so tied embeddings must be untied)
Quantising W' instead of W is what changes: a basis where activation energy is spread
(Hadamard) or aligned (PCA of the residual stream) gives the 4-bit blocks a better
shape. This script writes the rotated model as an HF checkpoint; llama.cpp converts and
quantises it like any other, and the auditor judges the three 4-bit files against the
same f16 model.
"""
import argparse, json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
try:
    import gpulock                                       # one GPU job at a time on the research machine
except ImportError:                                      # the published package has no lock; run anyway
    gpulock = None
from lib_inject import load_model, DEV

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(HERE, "..", "..", "models", "qwen3-1.7b-fp"))
ap.add_argument("--corpus", default=os.path.join(HERE, "..", "..", "mscc", "accept", "corpus.txt"))
ap.add_argument("--tokens", type=int, default=8192); ap.add_argument("--window", type=int, default=1024)
ap.add_argument("--basis", choices=("pca", "hadamard"), required=True)
ap.add_argument("-o", "--out", required=True)
a = ap.parse_args()

if gpulock: gpulock.acquire("rotate-weights")
tok, model = load_model(a.model)
cfg = model.config; d = cfg.hidden_size; L = cfg.num_hidden_layers
layers = model.model.layers

# ---- 1. the basis
if a.basis == "hadamard":
    # Sylvester Hadamard with random signs (QuaRot's R1): d must be a power of two
    assert d & (d - 1) == 0, f"hidden size {d} is not a power of two"
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    g = torch.Generator().manual_seed(0)
    signs = torch.randint(0, 2, (d,), generator=g).float() * 2 - 1
    R = (H / d ** 0.5) * signs[None, :]                                  # orthogonal
    stats = {"basis": "hadamard"}
else:
    # covariance of the residual stream at every block input, accumulated in float64
    cov = torch.zeros(d, d, dtype=torch.float64, device=DEV); n = 0
    def hook(mod, args):
        global cov, n
        x = args[0].reshape(-1, d).double(); cov += x.T @ x; n += x.shape[0]
    hs = [l.register_forward_pre_hook(hook) for l in layers]
    ids = tok(open(a.corpus, encoding="utf-8", errors="replace").read(), return_tensors="pt").input_ids[0][: a.tokens]
    with torch.inference_mode():
        for i in range(0, len(ids), a.window):
            model(input_ids=ids[i:i + a.window].unsqueeze(0).to(DEV), use_cache=False, logits_to_keep=1)
    for h in hs: h.remove()
    cov /= n
    evals, evecs = torch.linalg.eigh(cov.cpu())                          # ascending
    R = evecs.flip(1).float()                                            # descending variance
    stats = {"basis": "pca", "states": int(n), "explained_top256": float(evals.flip(0)[:256].sum() / evals.sum())}
print("basis:", stats, flush=True)

# ---- 2. rotate the weights (float32, on CPU)
model = model.to("cpu").float(); R = R.cpu()
sd = model.state_dict()
def fold_in(lin, gain):      # W_in' = W_in diag(w) R   (W is [out, in])
    lin.weight.data = (lin.weight.data * gain[None, :]) @ R
def fold_out(lin):           # W_out' = R^T W_out
    lin.weight.data = R.T @ lin.weight.data
E = model.model.embed_tokens.weight.data
head = model.lm_head.weight.data.clone()                                 # before untying
model.model.embed_tokens.weight.data = E @ R
for l in layers:
    w1 = l.input_layernorm.weight.data.clone(); l.input_layernorm.weight.data.fill_(1.0)
    for lin in (l.self_attn.q_proj, l.self_attn.k_proj, l.self_attn.v_proj): fold_in(lin, w1)
    fold_out(l.self_attn.o_proj)
    w2 = l.post_attention_layernorm.weight.data.clone(); l.post_attention_layernorm.weight.data.fill_(1.0)
    for lin in (l.mlp.gate_proj, l.mlp.up_proj): fold_in(lin, w2)
    fold_out(l.mlp.down_proj)
wf = model.model.norm.weight.data.clone(); model.model.norm.weight.data.fill_(1.0)
model.lm_head.weight = torch.nn.Parameter((head * wf[None, :]) @ R)     # untied
model.config.tie_word_embeddings = False

# ---- 3. equivalence check against the original, float32 on CPU
tok0, orig = load_model(a.model); orig = orig.to("cpu").float()
probe = tok("The maintenance access code for the signal tower is", return_tensors="pt").input_ids
with torch.inference_mode():
    lo = orig(input_ids=probe, use_cache=False).logits[0, -1]; lr = model(input_ids=probe, use_cache=False).logits[0, -1]
diff = float((lo - lr).abs().max()); print(f"equivalence: max |logit diff| {diff:.2e}  top1 same: {bool(lo.argmax() == lr.argmax())}", flush=True)
assert diff < 5e-2, "rotated model is not equivalent; refusing to save"

# ---- 4. save as bf16 HF checkpoint
os.makedirs(a.out, exist_ok=True)
model.to(torch.bfloat16).save_pretrained(a.out, safe_serialization=True); tok.save_pretrained(a.out)
json.dump({"rotation": stats, "source": os.path.abspath(a.model), "equivalence_max_logit_diff": diff}, open(os.path.join(a.out, "rotation.json"), "w"), indent=1)
torch.save(R, os.path.join(a.out, "R.pt"))
print("saved", a.out, flush=True)
