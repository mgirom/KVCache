# KVCache

[![build llama.cpp with the cpca patches](https://github.com/mgirom/KVCache/actions/workflows/build-llamacpp.yml/badge.svg)](https://github.com/mgirom/KVCache/actions/workflows/build-llamacpp.yml)

Two things, and the second exists because the first needed proving.

| | |
|---|---|
| **[`mscc/`](mscc/README.md)** | **A KV cache codec.** Compresses a full-depth KV cache **15× for storage and transport**, with no measurable loss of task success. Read a document once, hand the frame to another process, and it answers questions with the receiver running **zero layers** over that document. And a **live path**: in llama.cpp, a fitted basis for the quantised cache — **`iq4_nl` at 3.6× smaller with no measurable loss (231/240 vs 231/240)** on a model where every 4-bit cache had cost 11 to 22 points (patch series included); in PyTorch, attention over the codes themselves at 3.9×. |
| **[`auditor/`](auditor/README.md)** | **The benchmark that measures it** — and any other KV method. Reports tokens per second *and what the speed cost you*, because they come apart. Results: **https://mgirom.github.io/KVCache/** |

The order matters. The codec came first; the benchmark exists because its original
evidence was twelve items with no error bars, and that is not enough to ask anyone to
believe something. Now both are measured the same way.

---

## Use it in three commands

The live path, on a model the registry has measured (Qwen3-1.7B, the ternary
Bonsai-8B and Bonsai-27B), with the patched llama.cpp built once:

```bash
./reproduce-llamacpp.sh build                                   # llama.cpp at the pinned commit + the cpca patches
python3 tools/kvcache.py pull bonsai-8b                         # the fitted codebook, hash-checked
python3 tools/kvcache.py serve bonsai-8b --model Ternary-Bonsai-8B-Q2_0_g64.gguf   # q4_0 cache, fitted basis, OpenAI-compatible server
```

`python3 tools/kvcache.py audit <id> --model FILE` runs the benchmark on your own
machine and tells you what the cache setting cost there. The model file comes from its
publisher; the registry ([`registry/models.json`](registry/models.json)) carries the
sha256 of the exact file every row was measured on, and the codebook refuses a model
it was not fitted for. For a model that is not in the registry:
`python3 mscc/capture_gguf_kv.py --gguf MODEL.gguf -o MODEL.kvcb.npz` fits one from
llama.cpp's own saved cache. All documents are indexed in [`docs/INDEX.md`](docs/INDEX.md).

---

## The codec: 15× smaller cache, no measurable cost

```bash
python3 -m mscc.cli kvfit    --corpus CORPUS --model MODEL -o cb.npz   # once per model
python3 -m mscc.cli kvencode DOC --model MODEL --codebook cb.npz -o doc.kvf
python3 -m mscc.cli kvserve  --frame doc.kvf --model MODEL --codebook cb.npz --ask "..."
```

Audited by `auditor/`, same workload and rules as everything else:

| arm | B/token | vs f16 | task success |
|---|---:|---:|---:|
| reference — uncompressed cache | 114,688 | 1.00× | 236/240 |
| **`cpca1024`** | **7,616** | **15.06×** | **236/240** |

Identical hit counts at 15× compression, n=240. For comparison, on the same benchmark
llama.cpp's own `q8_0` manages 1.8× and `q4_0` 3.2×.

**What it is for, and what it is not.** The shipped `kvfit`/`kvencode`/`kvserve` path
compresses a cache **at rest and in transit** — a prompt cache on disk, a document
handed between agents — and decodes back to full precision to use. That path **does not
reduce live VRAM during inference**. It competes against storing the cache
uncompressed, where there is no standard alternative.

### In llama.cpp: the fitted basis for the quantised cache

llama.cpp already stores a quantised KV cache in a rotated basis (a Hadamard, since
PR #21038). The codec's basis is the same kind of object, fitted to the model's own
keys and values, with a per-channel scale and a mean. Supplied in place of the
Hadamard, through the hooks llama.cpp already has, it changes what `-ctk q4_0` costs.
Standard profile, one run, the same 240 items, Qwen3-1.7B GGUF:

| cache | ctx 1k | ctx 4k | KV smaller |
|---|---:|---:|---:|
| f16 reference | 136/144 | 95/96 | 1.00× |
| `f32` (twice the bytes, half the speed) | 140/144 | 95/96 | 0.50× |
| `q8_0` | 141/144 | 95/96 | 1.80× |
| `q4_0`, Hadamard basis (llama.cpp today) | 120/144 | 82/96 | 3.24× |
| `q4_0`, fitted basis (codebook fitted on HF bf16 states) | 134/144 | 90/96 | 3.19× |
| **`q4_0`, fitted basis (fitted on this GGUF's own states, whitened)** | **134/144** | **96/96** | **3.23×** |
| `q5_0`, Hadamard basis | 138/144 | 96/96 | 2.94× |
| `q5_0`, fitted basis | 140/144 | 95/96 | 2.94× |
| `iq4_nl`, Hadamard basis | 104/144 | 84/96 | 3.60× |
| **`iq4_nl`, fitted basis** | **138/144** | **93/96** | **3.60×** |

The rotation does not depend on the block type, so one codebook drives every
quantised cache type. With `q4_0` the fitted basis takes the loss from 11 points to
1.4 at 1k, and at 4k from 13.5 points to 5 with a codebook fitted on the model's bf16
HuggingFace states, to none with one fitted on the served GGUF's own cache and
whitened: a Q4_K_M model's keys are not its bf16 twin's, so fit on the file you serve. With `iq4_nl`, the non-linear 4-bit type that
loses 22 points on this model in its Hadamard basis, the fitted basis lands on
**231/240 against the reference's 231/240 at 3.6× smaller**: no measurable loss, on
the model where every 4-bit cache had cost between 11 and 22 points. (Three runs, three
references: the `q4_0` rows come from one run, the `q5_0`/`iq4_nl` rows from another
and the `f32` row from a third, each against its own reference, which scored
identically each time. The `f32` row answers a question readers ask: more precision
than 16 bits buys nothing measurable, at twice the memory and half the speed.)

**A speed caveat on the block types.** In this llama.cpp CUDA build only `q4_0` and
`q8_0` caches have a fast flash-attention path. `q5_0` and `iq4_nl` fall back to a slow
one: on the 1.7B at 4k context they prefill in 72 to 83 seconds against 1.1 for f16 and
decode at 13 to 17 tokens per second against 74, with or without the fitted basis. The
`iq4_nl` accuracy result stands; using it today means paying that kernel gap, which is
llama.cpp's to close, not the codebook's.

The same measurement on the **ternary 8B** (Bonsai-8B, 2.3 GB GGUF, no HuggingFace
twin; its codebook was fitted from llama.cpp's own saved cache), standard profile,
288 items:

| cache | ctx 1k | ctx 4k | KV smaller |
|---|---:|---:|---:|
| f16 reference | 144/144 | 143/144 | 1.00× |
| `q4_0`, Hadamard basis | 143/144 | 142/144 | 3.48× |
| **`q4_0`, fitted basis** | **143/144** | **144/144** | **3.38×** |
| `iq4_nl`, Hadamard basis | 143/144 | 144/144 | 3.57× |
| `iq4_nl`, fitted basis | 143/144 | 144/144 | 3.57× |

At 16k context, where the cache is the memory that matters, the same 8B (96 items; two
tiers the reference itself fails at this length are excluded):

| cache | ctx 16k | KV smaller |
|---|---:|---:|
| f16 reference | 93/96 | 1.00× |
| `q4_0`, Hadamard basis | 90/96 | 3.48× |
| **`q4_0`, fitted basis** | **94/96** | **3.38×** |

Here `q4_0` was already free, as the earlier audit found for 8B-class models, and the
fitted basis keeps it free; `iq4_nl` is free too, at 3.57×, with the kernel speed cost
noted above.

And the largest model this 12 GB card can hold: the **ternary 27B** (Bonsai-27B, a
Qwen3.5 hybrid with attention on 16 of its 64 layers, 7.6 GB GGUF, codebook from its
own saved cache), standard profile, 288 items:

| cache | ctx 1k | ctx 4k | KV smaller |
|---|---:|---:|---:|
| f16 reference | 144/144 | 142/144 | 1.00× |
| `q4_0`, Hadamard basis | 142/144 | 143/144 | 3.14× |
| **`q4_0`, fitted basis** | **143/144** | **143/144** | **3.27×** |

And at 16k context on the same 27B, 144 items: reference 142/144, Hadamard `q4_0`
143/144, fitted `q4_0` **143/144** at 3.27× smaller, decoding at 15.5 tokens per
second. A 27B model at 16k context, its cache under a third of f16, no measurable
loss, on a consumer card. That is the combination this project set out to find. Its price is time, now small: with the per-head multiply as one fused operator, the
fitted basis decodes 4 to 5 percent slower than the Hadamard `q4_0` and prefills 6 to 9
percent slower on the 1.7B, down from 10 to 13 and 13 to 20 with the first
implementation (numbers in the design record). Ternary weights and a 3.4× smaller cache, no measurable
loss, on a 12 GB card: that is the combination the project set out to find, at the
8B scale. Result files: [`results/cpca-qwen3-1.7b.json`](results/cpca-qwen3-1.7b.json),
[`results/cpca-bonsai-8b.json`](results/cpca-bonsai-8b.json),
[`results/cpca-bonsai-27b.json`](results/cpca-bonsai-27b.json).

This is a four-patch series against llama.cpp master, in
[`mscc/llamacpp/`](mscc/llamacpp/), touching the cache constructor, the attention
builder and the cache shift and nothing else: no new kernel, flash attention and the
quantisation kernels untouched, selected by an environment variable in this
prototype. A codebook for any GGUF comes from llama.cpp's own saved cache, no
HuggingFace weights needed:

```bash
python3 mscc/capture_gguf_kv.py --gguf model.gguf -o model.kvcb.npz     # writes model.cpca.gguf beside it
LLAMA_KV_CODEBOOK=model.cpca.gguf llama-server -m model.gguf -ctk q4_0 -ctv q4_0 -fa on
```

The design, the algebra and the milestone record are in
[`mscc/LLAMACPP-CPCA-DESIGN.md`](mscc/LLAMACPP-CPCA-DESIGN.md).

### The live path: attention over the codes, cache never decoded

The codec is linear, so its basis folds into the query and attention can read the
packed codes directly. [`lib/codespace.py`](lib/codespace.py) does this on any
HuggingFace model through transformers' attention interface: the document lives on the
GPU as bit-packed codes, the model's own cache holds only the question and what it
generates, and one softmax spans both. Measured against decode-then-attend on the same
items in the same run, twelve items per model at 1k context:

<!-- codespace-table -->
| model | same answer as decode-then-attend | correct: dense / decoded / code-space | KV memory | decode ms/token: dense → code-space |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | 12/12 | 8 / 8 / 8 | **3.96× smaller** | 24 → 106 |
| Qwen3-4B | 12/12 | 10 / 10 / 10 | 3.96× smaller | 47 → 159 |
| SmolLM2-1.7B | 11/12 | 8 / 7 / 7 | 3.96× smaller | 23 → 204 |
| OLMo2-1B | 12/12 | 8 / 7 / 7 | 3.96× smaller | 18 → 113 |
| BitNet-2B | 11/12 | 9 / 9 / 9 | 3.96× smaller | 544 → 612 |
<!-- /codespace-table -->

Attending over packed codes produced the **same answer as decode-then-attend on 58 of
60 items**, which is what the fold's exactness predicts and what
`python3 lib/codespace_selftest.py` proves on CPU in seconds. Both exceptions are the
same counting question, one that all three paths get wrong on both models: the
decoded path's f16 rounding of its reconstructed cache tipped a near-tie between two
wrong numbers, and the code-space path, which keeps float32 scores, sided with the
dense model each time. The memory figure is read from the tensors that were actually
resident, not computed from a formula.

**Two prices, both measured, neither hidden.** *Accuracy:* the live path needs per-head,
post-RoPE codes, and those compress worse than the storage codec's — 3.9× here against
15× at rest. At this rate the Qwen3-1.7B audit, standard profile, n=240, scores
**238/240 against the reference's 236/240**: no measurable loss at 3.9× live. (Two bits
per dimension, 7.8×, collapses at 4k context; the rate matters.) *Speed:* this is PyTorch unpacking bits with tensor ops
and looping over KV-head groups in Python, **4× to 9× slower per decoded token** on
the four ordinary models — slowest on SmolLM2, which has 32 KV heads and so 32 trips
round that loop per layer. (BitNet shows only 1.1× because, with its compiled kernel
disabled, its own eager weight path dominates every step.) A fused kernel is what turns
the memory saving into a speed saving, and none exists here yet. What is established is the part that had to come first: the
model answers correctly while its cache is a quarter the size. See
[CODE-SPACE-ATTENTION.md](mscc/CODE-SPACE-ATTENTION.md) for the derivation, the
withdrawn claim, and the full cost table. The result files behind the table are in
[`results/codespace/`](results/codespace/).

To reproduce on a model of your own (the live path needs its own codebook: per-head,
post-RoPE, at 4 bits per KV dimension, so `--unit-bits` is 4 × kv_heads × head_dim):

```bash
python3 -m mscc.cli kvfit --corpus CORPUS --model MODEL --unit-bits 4096 --per-head --postrope -o live.npz
python3 lib/codespace_test.py --model MODEL --codebook live.npz --n 12
```

A frame records the conditions it was produced under and **refuses rather than
degrades** — wrong model, wrong codebook, rate below the measured floor, unknown key
basis — and every refusal prints the measurement that justifies it. That matters
because a violated condition does not produce an error, it produces fluent confident
wrong text: forced past the guard, an under-rate frame answered *"the calibration
marker is the number 1234567890"* instead of `BRK-7742`.

There is a working agent-to-agent demo in [`demo/`](demo/handoff_server.py): a document
goes in, a frame comes out, a separate process answers questions from the frame alone,
and a deliberately poisoned run shows the guard refusing.

---

## The benchmark: what your KV setting actually costs

Every local-inference benchmark reports tokens per second and stops. This one reports
tokens per second **and what the speed cost you**, because they come apart:

- A configuration measured at **0.833 top-1 agreement** answered **0 of 12** planted
  questions correctly.
- A configuration **15× smaller** than its baseline ran **3.7× slower** than not
  compressing at all.

```bash
./auditor/reproduce.sh quick /path/to/model.gguf     # ~2 minutes
```

---

## What it found

Three models, one machine, one workload hash, full 1k/4k/16k ladder, 95% Wilson
intervals. Every quality figure is a delta against **that model's own uncompressed
reference**, measured in the same run.

| model | weights | `q8_0` KV (1.8×) | `q4_0` KV (3.2–3.5×) |
|---|---|---|---|
| Qwen3-1.7B | Q4_K_M | 236/240 — matches reference exactly | **209/240 · [0.822, 0.907] — distinguishably worse** |
| Qwen3-8B | Q4_K_M | 430/432 — matches reference exactly | 429/432 · [0.980, 0.998] — no measurable cost |
| Bonsai-8B | ternary Q2_0_g64 | 288/288 | 375/384 · [0.956, 0.988] — no measurable cost |

**`q8_0` KV is free.** On two of three models it matched its reference item for item.
If you run f16 KV today, this is the recommendation and it needs no qualification.

**`q4_0` KV costs something, and how much depends on model size.** Only the 1.7B
separates from its own reference. Qwen3-1.7B and Qwen3-8B are the same architecture
family at the same weight quant and the same cache compression — so it is size, not
weight quantisation. The ternary 8B agrees with the standard 8B, ruling that out too.

**Speed cannot see any of it.** Decode across f16 / q8_0 / q4_0 on the 1.7B is
113.4 / 105.3 / 103.3 tok/s — noise. And the failures are not blanks:

| planted | 1.7B under `q4_0` answered |
|---|---|
| `83-15-69` | `83-11-69` |
| `67-42-23` | `27-42-23` |
| `89-23-75` | `88-23-75` |

One digit, confidently wrong.

**The scales are real bytes.** Measured by memory slope, not computed from a formula:
`q4_0` is **3.24× smaller, not 4×**, because the per-block scale every quantised format
carries is charged. The naive figure overstates the saving by 23%.

---

## How it works

**Task success, not perplexity.** Every answer is **planted in the document** by the
generator, from invented names and invented numbers. A model that has never encountered
the subject matter scores identically, so the benchmark measures the optimisation and
not the model. Four tiers, escalating in what they demand of attention: single-span
retrieval, two-span linking, whole-span aggregation, and a real fact beside a plausible
decoy — because a degraded cache does not go blank, it goes *confidently adjacent*.

**Every quality number is a delta against the same machine's own reference arm,**
measured in the same run. That is what makes a laptop's row comparable to a
datacentre's without either being numerically identical to the other — which they are
not: on the same model and items, CUDA and a CPU build agreed on only **52%** of replies
byte-for-byte and flipped **8%** of verdicts, while their aggregate rates differed by
1 point.

**A task the reference arm cannot do is excluded and declared.** It measures the model,
not the method. The decision reads the interval, not the point estimate — a hard
threshold let one item decide whether a tier existed (43/48 on CUDA, 44/48 on CPU), so
two machines audited different tier sets and stopped being comparable.

**It adapts to the machine, never to the measurement.** The context ladder is probed
against free memory, a reserve is left for the rest of the system, and `-ngl` is not
forced. What adapts is *which* cells run; the items, scoring and reference rule inside a
cell never do.

**Every run reports its own detection floor.** At n=36 a quick run resolves a ~25%
shortfall — so it would *miss* `q4_0`'s 13% cost, and it says so in its own output.

---

## Enter your own method

The surface a backend needs is five methods: `start`, `stop`, `n_tokens`, `complete`,
`vram_bytes`. Implement `auditor/runner/backends.Backend` and your cache is measured by
the same reference rule, tier exclusion and intervals as everything else.

The first non-llama.cpp entrant is this repo's own codec — see below — because holding
your own method to the bar you built for everyone else is not optional.

---

## MSCC — a full-depth KV frame at 15×

`mscc/` is a codec that compresses a **full-depth KV cache for storage and transport**,
so a document read once can be handed to another process that never sees the text and
runs **zero layers** over it.

Audited by the benchmark above, same workload, same rules:

| arm | B/token | vs f16 | task success |
|---|---:|---:|---:|
| reference (uncompressed cache) | 114,688 | 1.00× | 236/240 |
| **`cpca1024`** | **7,616** | **15.06×** | **236/240** |

Identical hit counts at 15× compression, at n=240.

**Two caveats that are not footnotes.** The shipped CLI path compresses for storage and
transport and decodes back to full precision to use — it does **not** reduce live VRAM.
The live path above does, at 3.9× and in a slower prototype; see the section under the
codec heading. And the comparison above is cross-runtime (safetensors in PyTorch
against a GGUF in llama.cpp), so what compares is each arm's delta against its own
reference, which is weaker than a within-runtime comparison.

```bash
python3 -m mscc.cli kvfit    --corpus C --model M -o cb.npz
python3 -m mscc.cli kvencode DOC --model M --codebook cb.npz -o doc.kvf
python3 -m mscc.cli kvserve  --frame doc.kvf --model M --codebook cb.npz --ask "..."
```

A frame records the conditions it was produced under and **refuses rather than
degrades**: wrong model, wrong codebook, rate below the measured floor, unknown key
basis. Every refusal prints the measurement that justifies it.

---

## Requirements

- Python 3.11+ and a [llama.cpp](https://github.com/ggml-org/llama.cpp) build with
  `llama-server` (CUDA, Metal, Vulkan, ROCm or CPU — the CPU path is verified)
- A GGUF model
- For `mscc/`: PyTorch and `transformers`

```bash
./auditor/reproduce.sh workload        # fetch the public-domain haystack, generate tasks
./auditor/reproduce.sh selftest        # schema, scoring and validator tests, no GPU
./auditor/reproduce.sh quick MODEL     # ~2 minutes
./auditor/reproduce.sh audit MODEL     # the full ladder
```

When a run finishes it **asks** whether to share the result, so you can see how your
machine compares with others running the same model and settings. You can say yes,
always, or not this time. A non-interactive session always declines — silence is not
agreement — and the first upload from a machine prints the entire payload before
sending anything.

To stop being asked: `--never-share`.

See [`auditor/PRIVACY.md`](auditor/PRIVACY.md) for the closed field list and how to
delete a submission afterwards.

## Disclaimer — read before running this

**This software is provided "as is", without warranty of any kind.** You run it at your
own risk, and the authors and contributors accept **no responsibility or liability** for
anything that follows from running it or from using its results.

Specifically, and without limiting that:

- **It loads models and drives your GPU or CPU at sustained full load, for minutes to
  hours.** That is a thermal and power stress test as a side effect. Hardware failure,
  thermal throttling, instability, crashes, driver faults, data loss and voided
  warranties are your risk to accept. Check your cooling and power before a long run.
- **It downloads things** — a public-domain text, and whatever model you point it at.
  You are responsible for what you fetch and for complying with each model's own
  licence.
- **It can send data off your machine, but only if you ask it to.** Nothing is uploaded
  without `--upload`, and the first use prints the entire payload and asks. What is
  collected is listed in full in [`auditor/PRIVACY.md`](auditor/PRIVACY.md).
- **The results are measurements, not advice.** They come from a small number of models
  on single machines and are not authoritative. Decisions you make from them —
  production settings, hardware purchases, capacity planning — are yours. Verify
  anything that matters on your own hardware and workload.
- **Submitted results become public.** A submission includes your hardware model
  numbers, OS and driver versions. Read the field list first.

## Licence

MIT. See [`LICENSE`](LICENSE). The MIT text itself disclaims all warranties and
liability; the section above says what that means in practice for a benchmark.

The haystack is *Moby-Dick* (Herman Melville, 1851), public domain, fetched and
stripped of its Project Gutenberg header by `auditor/workload/fetch_haystack.py` and
pinned by hash.

## Reading further

- [`auditor/SPEC-v0.1.md`](auditor/SPEC-v0.1.md) — the protocol and its validity rules
- [`auditor/FIRST-RESULT.md`](auditor/FIRST-RESULT.md) — full tables, and the harness
  bugs found by testing, every one of which corrupted scores silently
- [`mscc/README.md`](mscc/README.md) — the frame format and its guard

Results are single-machine and single-run. Nothing here is authoritative; it is
reproducible, which is the part that matters.
