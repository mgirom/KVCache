# mscc — hand a document's *read state* to another agent

Read a document once. Hand the second agent the state of having read it, rather than
the text. The second agent answers questions about the document while running **zero**
transformer layers over it.

## Which frame to use

There are two frame families in this tree. Only one of them works.

| | mid-stack frame (`fit`/`encode`/`serve`) | **KV frame (`kvfit`/`kvencode`/`kvserve`)** |
|---|---|---|
| what is carried | one layer's hidden state per token | K and V at **every** layer |
| receiver skips | 86% of the document read | **100%** |
| can answer a question the sender did not anticipate | **no** — 0/12 needle recall | **yes** — 12/12 |
| status | retained for the measurement record only | the product |

The mid-stack frame is kept because every pre-Gate-12 number in this repository was
measured with it. It is not a smaller or cheaper option: a question's tokens have no
document keys below the tap, so it does not fail loudly, it invents answers fluently.
That is documented in `findings/2026-08-26-alphabet/GATE-{10,11}-*.md` and reproduced
by `mscc/accept/control.py`.

## Use it

```bash
# 1. one codebook per model, fitted once (GPU capture, CPU fit, ~2 min, ~128 MB)
python3 -m mscc.cli kvfit \
    --corpus mscc/accept/corpus.txt \
    --model models/qwen3-1.7b-fp \
    --unit-bits 1024 \
    -o mscc/accept/kv/book.kvcb.npz
```

```bash
# 2. the sender reads the document once and writes a frame
python3 -m mscc.cli kvencode mscc/accept/doc.txt \
    --model models/qwen3-1.7b-fp \
    --codebook mscc/accept/kv/book.kvcb.npz \
    --sink 4 -o mscc/accept/kv/doc.kvf
```

```bash
# 3. the receiver answers from the frame alone, running 0 of 28 blocks over the doc
python3 -m mscc.cli kvserve \
    --frame mscc/accept/kv/doc.kvf \
    --model models/qwen3-1.7b-fp \
    --codebook mscc/accept/kv/book.kvcb.npz \
    --ask "What is the calibration marker for this archive copy?"
```

```bash
# header + guard verdict, no model load, no GPU
python3 -m mscc.cli inspect mscc/accept/kv/doc.kvf --model models/qwen3-1.7b-fp
```

The LAN demo of the same flow, with a poisoned run shown refusing:

```bash
python3 demo/handoff_server.py --bind 127.0.0.1 --port 8093
```

## What it costs, honestly

Measured on the 1,803-token acceptance document (Qwen3-1.7B):

| | bytes |
|---|---:|
| the text itself | 8,042 |
| uncompressed bf16 KV | 206,782,464 |
| **the frame** | **12,968,248** |

So the frame is **16× smaller than the KV cache** it replaces and **1,612× larger
than the text**. Sending text is always cheaper on the wire. What the frame buys is
the receiver's prefill: it never reads the document. That trade only pays inside one
operator's boundary, where the wire is loopback or PCIe — which is the scope decision
this project already made (`findings/.../SCOPE-DECISION.md`).

The honest competitor is not bf16 KV, it is whatever KV quantisation an operator
already runs. Against int4 at equal needle recall, the frame is **4× smaller**.

## Rates

`--unit-bits` is bits per token per (layer, K|V) coding unit. Measured at ctx 1024
(Gate 12):

| unit-bits | vs bf16 KV | agreement | needle recall | |
|---:|---:|---:|---:|---|
| 2048 | 7.8× | 0.956 | 12/12 | |
| **1024** | **15.1×** | **0.925** | **12/12** | default |
| 512 | 28.4× | 0.873 | 10/12 | hard floor |
| 256 | 51.2× | 0.833 | **0/12** | refused |

Read the last row before choosing a rate. At 256 bits/unit the frame still scores a
respectable-looking 0.833 agreement and answers **every** question wrongly. Agreement
is a proxy; it does not certify a frame, and the guard refuses below 512 for that
reason.

## The guard

A violated condition does not produce an error, it produces fluent confident wrong
text. So the conditions are recorded in the frame header and checked before anything
expensive happens. `mscc/kv.py` refuses on: unknown format version, wrong frame kind,
model mismatch, codebook mismatch, unknown key basis, rate below the floor, missing
shape, empty frame. It warns on: post-RoPE key basis, rate below default, no protected
tokens.

Every refusal carries the measurement that justifies it, so `inspect` prints *why*,
not just *no*.

## The health stamp — advisory, not a verdict

`kvencode` records a self-declared quality stamp (`notes.health_probe`): the sender
frames all but the last 64 tokens, teacher-forces those through the decoded cache, and
records the agreement. `inspect` surfaces it.

**It missed its pre-registered bar and does not gate anything.** Gate 13 measured
held-out pass_good 0.929 / flag_bad 0.897 against a 0.90/0.90 bar set before the run,
so roughly one bad frame in ten passes it. The reference-free receiver-side signal did
worse. If you do not trust the sender, there is currently no working health signal:
verify your own frames.

## Tests

The live path has its own self-test, CPU only, seconds:

```bash
python3 lib/codespace_selftest.py       # packing bit-exact, folds exact, hook == dense on a tiny model
```

```bash
python3 -m mscc.test_codec && python3 -m mscc.test_guard && python3 -m mscc.test_kv
```

```bash
# end-to-end acceptance against a real model (needs the codebooks from kvfit)
python3 -m mscc.accept.kv_acceptance
```
