#!/usr/bin/env python3
"""Backends the auditor can measure. The seam that makes this a benchmark.

Until now the runner spoke only to llama.cpp, which made it a llama.cpp test rather
than a benchmark. The surface an arm actually needs turned out to be five methods --
start, stop, n_tokens, complete, vram_bytes -- so anything that can answer a prompt
under a named KV strategy can enter, and be measured by the same reference-arm rule,
the same tier exclusion and the same intervals as everything else.

The first non-llama.cpp entrant is this project's own codec, and that is the point.
MSCC's published compression figure rests on n=12 with no error bars, while the
validator here rejects submissions for exactly that. Holding your own method to the bar
you built for everyone else is not optional, and the honest way to find out whether
15.1x survives contact with n=432 is to run it.

WHAT DOES NOT TRANSFER BETWEEN BACKENDS
---------------------------------------
Absolute rates. A llama.cpp arm runs a GGUF; the MSCC arm runs HuggingFace safetensors
in a different runtime at a different weight precision. Those are different models by
the protocol's own rule, so their scores are never pooled. What compares is each arm's
**delta against its own reference**, measured in its own run -- which is weaker than a
within-runtime comparison and is stated as such wherever the number appears.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "alphabet", "scripts")]


class Backend:
    """What an arm needs. Implement these five and the auditor can measure you."""

    name = "backend"

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def n_tokens(self, text: str) -> int:
        raise NotImplementedError

    def complete(self, prompt: str, n_predict: int = 64) -> dict:
        """Return {content, stop_type, timings:{prompt_ms, predicted_per_second,
        prompt_n}} -- the same shape llama-server returns, because that is what the
        runner already reads."""
        raise NotImplementedError

    def vram_bytes(self):
        return None

    def model_meta(self) -> dict:
        return {}


# --------------------------------------------------------------------- llama.cpp

class LlamaCppBackend(Backend):
    """The original path, unchanged in behaviour -- now just one implementation."""

    def __init__(self, binary, model, ctx, ctk="f16", ctv="f16", port=8099,
                 ngl=None, log_dir=None, codebook=None):
        from server import LlamaServer
        self.name = f"{ctk}" if ctk == ctv else f"{ctk}/{ctv}"
        env = {}
        if codebook:
            # the cpca prototype: llama.cpp's quantised cache with the fitted rotation
            # in place of its Hadamard, selected by environment (see LLAMACPP-CPCA-DESIGN.md)
            self.name += "+cpca"
            env["LLAMA_KV_CODEBOOK"] = os.path.abspath(codebook)
        self.codebook = codebook
        self.srv = LlamaServer(binary, model, ctx, port=port, cache_type_k=ctk,
                               cache_type_v=ctv, log_dir=log_dir, ngl=ngl, env=env)

    def start(self):
        self.srv.start()
        return self

    def stop(self):
        self.srv.stop()

    def n_tokens(self, text):
        return self.srv.n_tokens(text)

    def complete(self, prompt, n_predict=64):
        return self.srv.complete(prompt, n_predict=n_predict)

    def vram_bytes(self):
        return self.srv.vram_bytes()

    def model_meta(self):
        return self.srv.model_meta()


# -------------------------------------------------------------------------- MSCC

#: The runner builds prompts as "{document}\n\nQuestion: {q}\nAnswer:". MSCC's whole
#: claim is that the document is framed separately from the question, so the backend
#: has to split there. This is the one place a backend is allowed to know the prompt's
#: shape, and it is asserted rather than assumed.
QUESTION_MARKER = "\n\nQuestion: "


class MsccBackend(Backend):
    """This project's own KV codec, measured by this project's own benchmark.

    An arm here is: frame the document, decode the frame, answer the question from the
    decoded cache. That is `mscc kvserve` without the file round-trip, so it measures
    the shipped path rather than a lookalike.

    `unit_bits=0` is the reference arm: capture the cache and use it uncompressed. It
    is not a no-op -- it still hands over a full-depth cache and skips the document
    prefill -- so it isolates the codec from the handover, which is the comparison that
    matters for this method.
    """

    def __init__(self, model_dir, codebook, ctx, unit_bits=1024, sink=4,
                 device="cuda:0"):
        self.name = f"cpca{unit_bits}" if unit_bits else "kv_exact"
        self.model_dir, self.codebook_path = model_dir, codebook
        self.ctx, self.unit_bits, self.sink = ctx, unit_bits, sink
        self.device = device
        self.tok = self.model = self.cb = None

    def start(self):
        import torch
        from lib_inject import load_model
        self.tok, self.model = load_model(self.model_dir)
        self._torch = torch
        if self.unit_bits:
            from mscc.kv import KVCodebook
            self.cb = KVCodebook.load(self.codebook_path)
            got = self.cb.meta.get("unit_bits")
            if got != self.unit_bits:
                raise RuntimeError(
                    f"codebook is {got} bits/unit, this arm wants {self.unit_bits}. "
                    "Fit one per rate rather than reusing across rates.")
        return self

    def stop(self):
        self.model = self.tok = self.cb = None
        try:
            self._torch.cuda.empty_cache()
        except Exception:                                              # noqa: BLE001
            pass

    def n_tokens(self, text):
        return len(self.tok(text, return_tensors="pt").input_ids[0])

    def vram_bytes(self):
        try:
            return int(self._torch.cuda.memory_allocated())
        except Exception:                                              # noqa: BLE001
            return None

    def kv_bytes_per_token(self, n_tokens: int) -> float:
        """Exact, not probed: a CPCA frame's size is the sum of its code widths, which
        is what the bit packer emits, plus the leading tokens carried at full
        precision. The llama.cpp arms need a memory-slope probe because their cache
        size is an allocation detail; here it is a property of the codebook and is
        computed rather than estimated. Reporting 0 because the probe does not apply
        would read as 'free', which is the opposite of true."""
        c = self.model.config
        n_kv = getattr(c, "num_key_value_heads", c.num_attention_heads)
        hd = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
        raw_bits = c.num_hidden_layers * 2 * n_kv * hd * 16
        if not self.unit_bits:
            return raw_bits / 8
        codec_bits = self.cb.bits_per_token
        sink_bits = raw_bits * min(self.sink, n_tokens) / max(n_tokens, 1)
        return (codec_bits + sink_bits) / 8

    def model_meta(self):
        c = self.model.config
        return {"arch": c.model_type, "n_layers": c.num_hidden_layers,
                "n_kv_heads": getattr(c, "num_key_value_heads",
                                      c.num_attention_heads),
                "head_dim": getattr(c, "head_dim",
                                    c.hidden_size // c.num_attention_heads),
                "n_heads": c.num_attention_heads}

    def complete(self, prompt, n_predict=64):
        import time
        import lib_kv as K
        torch = self._torch

        if QUESTION_MARKER not in prompt:
            raise RuntimeError("prompt has no question marker; this backend must know "
                               "where the document ends and the question begins")
        doc_text, q_text = prompt.split(QUESTION_MARKER, 1)
        q_text = QUESTION_MARKER + q_text

        dev = self.device
        doc = self.tok(doc_text, return_tensors="pt").input_ids.to(dev)
        q = self.tok(q_text, return_tensors="pt",
                     add_special_tokens=False).input_ids.to(dev)

        t0 = time.perf_counter()
        postrope = bool(self.cb and self.cb.meta.get("basis") == "postrope")
        if postrope:
            kv, kpre = K.capture_kv(self.model, doc), None
        else:
            kv, kpre = K.capture_kv_prerope(self.model, doc)
        if self.unit_bits:
            per_head = self.cb.meta.get("per_head")
            hh, hd = int(self.cb.meta["kv_heads"]), int(self.cb.meta["head_dim"])
            if postrope and per_head:
                q_ = K.roundtrip_kv_perhead_postrope(kv, self.cb.books, hh, hd)
            elif postrope:
                q_ = K.roundtrip_kv(kv, self.cb.books)
            elif per_head:
                q_ = K.roundtrip_kv_prerope_perhead(self.model, kv, kpre,
                                                    self.cb.books, hh, hd)
            else:
                q_ = K.roundtrip_kv_prerope(self.model, kv, kpre, self.cb.books)
            kv = K.merge_exact(kv, q_, n_sink=self.sink)
        prompt_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        out = K.gen_from_cache(self.model, kv, q, n_predict,
                               eos=self.tok.eos_token_id)
        gen_s = time.perf_counter() - t1
        del kv, kpre
        torch.cuda.empty_cache()

        text = self.tok.decode(out, skip_special_tokens=True)
        return {"content": text,
                # the model stopped on its own only if it emitted EOS before the budget
                "stop_type": "limit" if len(out) >= n_predict else "eos",
                "timings": {"prompt_ms": round(prompt_ms, 2),
                            "prompt_n": int(doc.shape[1] + q.shape[1]),
                            "predicted_n": len(out),
                            "predicted_per_second": (len(out) / gen_s) if gen_s else 0.0}}


def make(kind, **kw) -> Backend:
    if kind == "llamacpp":
        return LlamaCppBackend(**kw)
    if kind == "mscc":
        return MsccBackend(**kw)
    raise ValueError(f"unknown backend {kind!r}")
