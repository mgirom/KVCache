#!/usr/bin/env python3
"""Self-test for code-space attention. Runs on CPU in seconds, no model download.

Two checks, both against the codec's own decode path, both exact by construction:

  1. PACKING. Bit-packed rows unpack to precisely the values Codebook.decode would
     produce, and the key fold and value fold equal dotting with the decoded keys
     and values.
  2. THE HOOK. A tiny randomly-initialised Qwen3 (2 layers, GQA 4:2) answers a
     question from a packed cache and from a reconstructed dense cache; the logits
     must agree to float32 roundoff. This exercises the mask, the grouped-head
     mapping, RoPE position continuation and per-layer tagging.

If either fails, do not trust any number the GPU harness prints.
"""
import os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from codespace import _repo_root                                     # noqa: E402
sys.path.insert(0, _repo_root(__file__))
import lib_kv as K                                                    # noqa: E402
import codespace as CS                                                # noqa: E402
from mscc.codec import Codebook                                       # noqa: E402

torch.manual_seed(0); np.random.seed(0)
ok = True
def report(name, cond, detail=""):
    global ok; ok &= bool(cond)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}  {detail}")

# ---------------------------------------------------------------- 1. packing
d, k, n = 128, 40, 300
V, _ = np.linalg.qr(np.random.randn(d, d)); V = V[:, :k].astype(np.float32)
b = np.array([12] * 4 + [8] * 8 + [4] * 12 + [2] * 10 + [1] * 6, dtype=np.int64)
cb = Codebook(mu=np.random.randn(1, d).astype(np.float32), s=(0.5 + np.random.rand(d)).astype(np.float32),
              V=V, b=b, lo=-np.full(k, 3.0, np.float32), hi=np.full(k, 3.0, np.float32), meta={})
X = torch.randn(n, d) * 1.5
u = CS.PackedUnit(cb, X, "cpu"); codes = cb.encode(X); Xhat = cb.decode(codes)
Zq = torch.as_tensor(cb.lo) + torch.as_tensor(codes.astype(np.float32)) / (2.0 ** torch.as_tensor(cb.b).float() - 1) * torch.as_tensor(cb.hi - cb.lo)
report("unpack == codec dequant", torch.equal(u.unpack(), Zq))
q = torch.randn(3, 5, d); p = torch.softmax(torch.randn(3, 5, n), -1)
report("key fold == q.k_hat", torch.allclose(u.key_scores(q), q @ Xhat.T, atol=1e-3), f"max {float((u.key_scores(q) - q @ Xhat.T).abs().max()):.1e}")
report("value fold == p.v_hat", torch.allclose(u.value_out(p), p @ Xhat, atol=1e-4), f"max {float((u.value_out(p) - p @ Xhat).abs().max()):.1e}")
report("packed smaller than f16", u.bytes() < n * d * 2, f"{n*d*2/u.bytes():.1f}x")

# ---------------------------------------------------------------- 2. the hook
from transformers import Qwen3Config, Qwen3ForCausalLM                # noqa: E402
from transformers.cache_utils import DynamicCache                     # noqa: E402
cfg = Qwen3Config(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
model = Qwen3ForCausalLM(cfg).eval()
heads, hd, sink, n_doc, nq = 2, 16, 4, 64, 6
doc = torch.randint(0, 300, (1, n_doc)); qi = torch.randint(0, 300, (1, nq))
with torch.inference_mode():
    kv = K.capture_kv(model, doc)
    states = {(l, unit): K.to_matrix(t) for l, (kk, vv) in enumerate(kv) for unit, t in (("k", kk), ("v", vv))}
    books = K.fit_kv_codebooks_perhead(states, bits_per_head=4 * hd, heads=heads, head_dim=hd)
    kvB = K.merge_exact(kv, K.roundtrip_kv_perhead_postrope(kv, books, heads, hd), n_sink=sink)
    past = DynamicCache()
    for l, (kk, vv) in enumerate(kvB): past.update(kk, vv, l)
    pos = torch.arange(n_doc, n_doc + nq)
    logB = model(input_ids=qi, past_key_values=past, cache_position=pos, position_ids=pos.unsqueeze(0), use_cache=True).logits[0].float()
    cache = CS.CodeSpaceCache(kv, books, heads, hd, sink=sink, device="cpu")
    prev = CS.install(model, cache)
    slot = torch.arange(nq)
    logC = model(input_ids=qi, past_key_values=DynamicCache(), cache_position=slot, position_ids=(slot + n_doc).unsqueeze(0), use_cache=True).logits[0].float()
    outC = CS.generate_codespace(model, cache, qi, 4)
    CS.uninstall(model, prev)
diff = float((logB - logC).abs().max())
report("hook logits == dense-on-reconstructed", diff < 1e-4, f"max |diff| {diff:.1e} over {nq} positions")
report("generate_codespace runs", len(outC) == 4, f"{len(outC)} tokens")
report("attention implementation restored", model.config._attn_implementation == prev, repr(prev))
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
