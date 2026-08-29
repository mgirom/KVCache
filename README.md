# KVCache — measure what your KV cache setting actually costs

Every local-inference benchmark reports tokens per second and stops. This one reports
tokens per second **and what the speed cost you**, because they come apart:

- A configuration measured at **0.833 top-1 agreement** answered **0 of 12** planted
  questions correctly.
- A configuration **15× smaller** than its baseline ran **3.7× slower** than not
  compressing at all.

Both are invisible under a tok/s-only benchmark. Both change the decision.

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

**Two caveats that are not footnotes.** It compresses for storage and transport and
decodes back to full precision to use — it does **not** reduce live VRAM, and is not a
drop-in for `-ctk q4_0`. And the comparison above is cross-runtime (safetensors in
PyTorch against a GGUF in llama.cpp), so what compares is each arm's delta against its
own reference, which is weaker than a within-runtime comparison.

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

Nothing is uploaded unless you ask, per run, and the tool prints the exact bytes first.
See [`auditor/PRIVACY.md`](auditor/PRIVACY.md) for the closed field list and how to
delete a submission.

## Licence

MIT. See [`LICENSE`](LICENSE).

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
