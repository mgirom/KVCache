---
doc: KV-AUDIT first results
date: 2026-08-29
status: three models, full three-rung ladder, confidence-aware tier exclusion
raw: ../results/auditor/{q3-1.7b-q4_k_m,qwen3-8b-q4_k_m,ternary-bonsai-8b-q2_0_g64}.json
workload: sha 3931af432845 (per_cell=16, 576 items)
---

# int4 KV quantisation is safe at 8B and measurably harmful at 1.7B

RTX A2000 12GB, llama.cpp b1-11cd988, CUDA. All three models ran the **same workload
hash** on the same machine; every reference arm ran in the same session as the arms it
baselines. Rates are pooled over the rungs that ran, with 95% Wilson intervals.

## Qwen3-1.7B (Q4_K_M)

| arm | KV B/token | vs f16 | task success | 95% CI |
|---|---:|---:|---:|---|
| **f16** (ref) | 115,712 | 1.00× | 236/240 · 0.983 | [0.958, 0.994] |
| **q8_0** | 64,366 | 1.80× | 236/240 · 0.983 | [0.958, 0.994] |
| **q4_0** | 35,694 | 3.24× | 209/240 · 0.871 | [0.822, 0.907] |

excluded: `t3_aggregate` at [1024, 4096, 16384]; `t4_distractor` at [4096, 16384, 1024]; `t2_link` at [16384]

## Qwen3-8B (Q4_K_M)

| arm | KV B/token | vs f16 | task success | 95% CI |
|---|---:|---:|---:|---|
| **f16** (ref) | 148,480 | 1.00× | 430/432 · 0.995 | [0.983, 0.999] |
| **q8_0** | 79,579 | 1.87× | 430/432 · 0.995 | [0.983, 0.999] |
| **q4_0** | 42,715 | 3.48× | 429/432 · 0.993 | [0.980, 0.998] |

excluded: `t3_aggregate` at [1024, 4096, 16384]

## Bonsai-8B (ternary Q2_0_g64)

| arm | KV B/token | vs f16 | task success | 95% CI |
|---|---:|---:|---:|---|
| **f16** (ref) | 148,480 | 1.00× | 380/384 · 0.990 | [0.974, 0.996] |
| **q8_0** | 79,579 | 1.87× | 383/384 · 0.997 | [0.985, 1.000] |
| **q4_0** | 42,715 | 3.48× | 375/384 · 0.977 | [0.956, 0.988] |

excluded: `t3_aggregate` at [1024, 4096, 16384]; `t4_distractor` at [16384]

## The three findings

**1. q8_0 KV is free.** On both models it tracks the uncompressed reference *exactly* —
the same hits and the same misses at every rung — at 1.8–1.9× less cache. If you are
running f16 KV today, this is the recommendation and it needs no qualification.

**2. q4_0 costs something, and what it costs depends on the model.** Only the 1.7B
separates from its own reference: [0.822, 0.907] against [0.958, 0.994]. The 8B does
not, at [0.980, 0.998].

**3. It is size, not weight quantisation.** Qwen3-1.7B and Qwen3-8B are the same
architecture family at the **same weight quant (Q4_K_M)** and the same ~3.2–3.5× cache
compression. Their q4_0 intervals — [0.822, 0.907] and [0.980, 0.998] — are nowhere
near overlapping. Bonsai-8B has **ternary** weights and the same KV geometry as
Qwen3-8B, and lands with the 8Bs at [0.956, 0.988], so the effect does not track weight
quantisation either.

The 1.7B's damage is present at **every** rung (82/96, 82/96, 45/48), so it is not a
short-context artifact and not a long-context one either.

## Speed cannot see any of it

Decode on the 1.7B across f16 / q8_0 / q4_0: **113.4 / 105.3 / 103.3 tok/s** — a spread
any benchmark would call noise, and in the direction that makes q4_0 look mildly slow
rather than mildly dangerous. A tok/s-only benchmark reports q4_0 as free on both
models. It is free on one.

The failures are not blanks. Qwen3-1.7B under q4_0:

| planted | answered |
|---|---|
| `83-15-69` | `83-11-69` |
| `67-42-23` | `27-42-23` |
| `35-49-25` | `35-49-22` |
| `89-23-75` | `88-23-75` |

One digit, confidently wrong.

## Why the 16k rung was worth its cost

It is where the cache is actually large, and it is the only rung that shows what
prefill costs: the 8B reference spends **16.8 seconds** on a 16k prefill, against 822 ms
at 1k. Any argument for caching or compressing KV lives in that number, and a ladder
that stops at 4k cannot show it.

It also produced the result that corrected the harness — see below.

## The scales are real bytes

Measured by memory slope across two context lengths, in whichever memory the cache
lands in, so the per-block scale every quantised format carries is charged:

| type | bits/element (1.7B) | naive claim | real ratio |
|---|---:|---:|---:|
| f16 | 16.14 | 16 | 1.00× |
| q8_0 | 8.96 | 8 | **1.80×**, not 2× |
| q4_0 | 4.96 | 4 | **3.24×**, not 4× |

q4_0 is 3.24× smaller, not 4×: the naive figure overstates the saving by 23%. Geometry
cross-check, read from the loader rather than assumed: Qwen3-8B at 36 × 2 × 8 × 128 × 2 B
predicts 147,456 B/token, and the slope measured 148,480 — 0.7% apart, consistent with
nvidia-smi's MiB granularity.

## Exclusion is per (tier, context), and the 16k rung is why

A tier the **reference arm** cannot do measures the model, not the optimisation. That
judgement is made per context, because capability is context-dependent:

| tier | 1.7B ref @1k | @4k | @16k | 8B ref |
|---|---|---|---|---|
| `t1_retrieve` | 48/48 keep | 48/48 keep | 48/48 keep | keep at all three |
| `t2_link` | 48/48 keep | 47/48 keep | **42/48 excluded** | keep at all three |
| `t4_distractor` | 43/48 (0.90) **excluded** | 33/48 **excluded** | **excluded** | keep at all three |
| `t3_aggregate` | 0/48 **excluded** | **excluded** | **excluded** | **excluded** |

So the 1.7B is audited on two tiers at 1k/4k and one at 16k; the 8B on three at all
three rungs. **This was found by the validator rejecting a real submission**, not by
inspection: the earlier logic pooled tier scores across contexts, kept `t2_link`
everywhere, and the per-rung check refused the result.

`t4_distractor` decaying 43 → 33 on the reference between 1k and 4k, with decoy grabs
rising 5 → 15, is the clearest single illustration of why the rule exists. That is the
model losing a discrimination with length, with no compression involved at all.

## Quality is not hardware-independent, and that broke the exclusion rule

Measured on 288 matched items, CUDA against a CPU-only llama-server build, same model
and same items:

| | |
|---|---|
| byte-identical reply | **151/288 = 0.524** |
| identical extracted answer | 254/288 = 0.882 |
| identical hit/miss verdict | 266/288 = 0.924 — **22 flipped** |
| aggregate f16 rate | CPU 0.719 · CUDA 0.708 — gap 0.010 |
| aggregate q4_0 rate | CPU 0.854 · CUDA 0.854 — gap 0.000 |

Half the replies differ textually and 8% of verdicts flip, while the aggregate is
stable to a percentage point: the flips are roughly symmetric and cancel. Different
kernels and reduction orders give bit-different logits; argmax is a discrete cutoff, so
a near-tie flips a token and the continuation diverges. The CPU answers `62-10-74`
where CUDA answers *"The maintenance access code for the Ashcombe signal tower is
62-10-74."*

**The consequence was worse than the variance.** Tier exclusion was a hard threshold at
0.9 on that noisy rate, so `t4_distractor` on the same model and items scored
43/48 = 0.8958 (excluded) on CUDA and 44/48 = 0.9167 (kept) on CPU. **One item decided
whether a tier existed**, two machines audited different tier sets, and their results
stopped being comparable — broken by exactly the small hardware differences a
distributed benchmark exists to capture.

A (tier, context) is now excluded only when the reference arm's 95% Wilson **upper
bound** is below the floor: confidently unusable, not merely under it. Stable across
the knife-edge — 43/48 and 44/48 both keep, 0/48 and 8/48 both exclude — and
deliberately conservative, since every score is a delta against that same reference
anyway.

## Five harness bugs found by testing, all of which corrupt scores silently

1. **Truncation scored as failure** — at `n_predict=24` a verbose preamble ate the
   budget before the answer landed (*"…is 25-3"*, correct and cut off).
2. **The first fix was too blunt** — rejecting every limit-stopped miss also rejected
   q4_0's real failures, since a degraded cache answers wrong then rambles. A miss is
   an artifact only when the reply is a strict **prefix** of the answer.
3. **Cache geometry hardcoded** to 28L × 8kv × 128d, silently wrong for any other model.
4. **The reference arm ran twice**, once to find excluded tiers and again on the
   survivors — a quarter of the job on an 8B, for answers already in hand.
5. **Backend mislabelled** — running the CPU build on a machine that has an NVIDIA card
   produced a submission recorded as `cuda`. Found by actually building and running a
   CPU-only server.
6. **The server under test was not checked to be ours.** A run bound to a default port
   another job was already serving, saw a healthy `/health`, and measured a different
   model on a different backend for a whole sweep. Nothing in the result said so; the
   only tell was prefill timings seven times too fast. The tool now claims a free port
   and refuses unless the server reports the model it launched. **A cross-backend
   result was published and retracted on the strength of this.**

## Sizing is part of the claim

The first version of this run used `per_cell=4` and supported **none** of its
comparisons: 1.7B q4_0 was [0.728, 0.928] against a reference [0.926, 1.000]. The
workload was resized to `per_cell=16` until the claims held, rather than the claims
being softened afterwards. Repeat runs cannot supply error bars here — greedy decoding
over a fixed workload reproduces itself exactly — so the uncertainty is binomial over
item sampling.

## What these results are not

- **Three models is three points.** The size reading is consistent across them and is
  not established by them. A 3B and a 14B would test it properly.
- **One machine, one run, one backend for the published numbers.** The CPU path is
  verified working end to end but has no published result.
- **`t4_distractor` is excluded on the 1.7B**, and it is the tier most sensitive to
  cache degradation. The 1.7B's measured cost is therefore taken on the tiers it can
  do, and is plausibly an *under*-estimate.
- **No 32k or beyond.** 16,384 is the top rung the ladder offers.
