---
doc: code-space attention — a live-memory path, and what is missing from it
date: 2026-08-29
status: algebra verified, codec measured, KERNEL NOT WRITTEN
raw: ../results/mscc-perhead-qwen3-1.7b.json, ../results/mscc-perhead2048-qwen3-1.7b.json
---

# Attention could read the codes directly. Here is the evidence, and the gap.

The shipped codec compresses a KV cache **for storage and transport**: it decodes back
to full precision before use, so live VRAM is unchanged. This note is about whether it
could avoid that, and what measuring said.

**Read the last section before quoting any number here.**

## The fold, which is exact

Attention needs `q · kᵢ` for every cached position. The codec is linear —
`kᵢ = (zᵢ Vᵀ)·s + μ` — so the basis moves to the query side:

```
q · kᵢ  =  zᵢ · (Vᵀ(s⊙q))  +  μ·q
               └──────────┘ once per step, not once per position
```

Attention then touches the **codes**, never a reconstructed vector. Verified against
reconstruct-then-attend on a real codebook: max difference **1.4e-04**, float32 noise.

The V side folds the same way: `out = Σᵢ pᵢvᵢ = ((Σᵢ pᵢzᵢ)Vᵀ)·s + μ`, accumulating in
code space and projecting once.

## CORRECTION: RoPE does not survive the fold on pre-RoPE codes

An earlier version of this note claimed the opposite, on the grounds that rotation is
orthogonal (`q · RoPEᵢ(k) = RoPE₋ᵢ(q) · k`). That identity is true and the conclusion
drawn from it was wrong. Rotating the query by `−i` gives a **different projected query
for every position `i`**, so the projection can no longer be computed once per step —
it costs a `[k × 128]` matvec per position, which is the same cost as decoding. The fold
saves nothing on pre-RoPE codes.

Measured rather than argued, on a real codebook and a real query:

| codes | fold error vs decode-then-attend |
|---|---:|
| pre-RoPE (what the note assumed) | **67.678 — wrong** |
| post-RoPE (keys as the cache holds them) | **0.000 — exact** |

**The fold requires post-RoPE codes.** That has two consequences, one bad and one
good. Bad: post-RoPE states compress worse — RoPE smears every key channel across the
document's rotation angles — so the live path pays a second compression penalty on top
of the per-head one below. Good: post-RoPE capture needs no architecture-specific hook,
so it runs on any HuggingFace model, not only ones with a `k_norm` to intercept.

The claim that pre-RoPE storage "happens to be the enabling condition" is withdrawn.
It is the disabling one.

## What the two requirements cost, measured

Every row is a delta against the same run's uncompressed reference on Qwen3-1.7B,
tiers the reference itself fails excluded per (tier, context). "quick" is 36 items per
rung and can only see gaps of roughly fifteen points or more; the standard profile
(240 per rung) is running on the 4-bit codebook and will replace those rows.

| codebook | basis | bits/dim | ctx 1k | ctx 4k | KV smaller by | profile |
|---|---|---:|---:|---:|---:|---|
| joint 1024 | pre-RoPE | 1 | 0.0 | 0.0 | **15.1x** | standard, n=240 |
| per-head 1024 | pre-RoPE | 1 | −95 pts | −100 pts | 15.1x | standard |
| per-head 2048 | pre-RoPE | 2 | 0.0 | 0.0 | **7.8x** | standard, n=240 |
| per-head 2048 | **post-RoPE** | 2 | −5.6 pts | **−27.8 pts** | 7.8x | quick, n=36 |
| per-head 4096 | **post-RoPE** | 4 | 0.0 | −8.3 pts | **3.9x** | quick, n=36 |

Read down the table and the price of going live is explicit. Storage-only, the codec
holds task success at 15x. Make the basis per-head so the fold is affordable: 7.8x.
Make the codes post-RoPE so the fold is exact: at the same 2 bits per dimension the
4k rung collapses, and it takes 4 bits per dimension -- 3.9x -- to get back to
something that might hold, with an 8-point gap at 4k that n=36 cannot resolve either
way. Only the last row can be attended over without decoding. Whether 3.9x live is
worth more than 15x at rest depends on what a deployment is short of: VRAM at
generation time, or bytes on the wire and disk.

The reason post-RoPE costs so much is not mysterious. Rotation mixes each key
channel pair by an angle that depends on position, so across a document a channel's
distribution is smeared over all angles; the per-channel standardisation and the PCA
both see a rounder, less compressible cloud. Pre-RoPE keys keep their anisotropy,
which is exactly what the codec exploits. A fold that worked on pre-RoPE codes would
recover the 7.8x row for the live path; the identity in the CORRECTION above says
why the plain version cannot, and nothing measured here says whether a cleverer one
could.


## The obstacle, which is real

The shipped codec fits **one basis across all 8 KV heads**, deliberately, to exploit
cross-head correlation. That is exactly what makes the fold unaffordable: a query head
needing only its own 128 dims must dot against the whole 897-component joint code.

| | compute per position per layer, K side |
|---|---:|
| standard f16 attention | 2,048 MACs |
| code-space, joint basis | 14,352 MACs — **7× worse** |
| code-space, per-head basis | 1,600 MACs — **0.78×** |

So the fold requires a per-head basis. Which costs compression.

## What that costs, measured

Audited by `auditor/`, same workload, same reference-arm rule, same intervals:

| codec | basis | vs f16 | task success | live-capable |
|---|---|---:|---:|---|
| reference — uncompressed | — | 1.00× | 236/240 | — |
| `cpca1024` **joint** (shipped) | all heads | 15.06× | **236/240** | no — 7× compute |
| `cpca1024` **per-head** | per head | 15.06× | **3/240** | yes, and destroyed |
| `cpca2048` **per-head** | per head | **7.76×** | **236/240** | **yes** |

**Three out of two hundred and forty.** Per-head reconstruction error rose only
1.5–1.9× against joint, which looked survivable and was not. That is the fourth time
in this project that a modest MSE increase has been a cliff in task terms, and the
reason the benchmark scores task success rather than a proxy. Had the idea been
published on the strength of the algebra, this is what would have been published.

At double the rate it is free: **236/240 at 7.76×**, the same count the joint codec
reaches at 15×, on a basis the fold can use. The cliff between the two is one doubling
of the rate.

## What is missing

**A kernel.** Every number above comes from decode-then-attend. Nobody has run
attention against packed codes. The memory-traffic argument —

| | bytes read per position, per (layer, K\|V) |
|---|---:|
| f16 cache | 2,048 |
| `cpca2048` codes | 256 — **8× less** |

— is arithmetic, and attention decode is memory-bound, so it *should* translate. But
unpacking variable-width codes inside a kernel is where this class of idea usually
dies, and until something measures it the honest claim is narrow:

> A KV codec 7.8× smaller with no measurable quality cost, whose structure permits
> attention to read the codes directly. **Kernel not yet written.**

Not "8× live memory savings". That would need a working kernel and a measurement.

**Also unestablished:** one model, one corpus, one machine. Whether the 7.76× cliff
sits in the same place on other architectures or GQA ratios is unknown.

## If you want to take this further

1. A PyTorch prototype of code-space attention — slow, but it would confirm the fold
   survives real GQA, softmax and RoPE end to end in generation, not just in a dot
   product.
2. Then a fused kernel, and the only measurement that settles it: tokens/sec and peak
   VRAM against an f16 cache at long context.
3. The per-head cliff on a second model, to see whether 7.76× is a property of the
   method or of Qwen3-1.7B.

## Prototype results: it runs, it is exact, it is slower

`lib/codespace.py` (`alphabet/scripts/codespace.py` in the research tree) implements
the fold as a transformers attention function. The harness runs three paths per item
on a real model: dense f16; decode-then-attend on the reconstructed cache; and
attention over packed codes with the cache never reconstructed. Twelve items per model,
1k context, 4 bits per KV dimension, chat-template prompting.

<!-- codespace-table -->
| model | same answer as decode-then-attend | correct: dense / decoded / code-space | KV memory | decode ms/token: dense → code-space |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | 12/12 | 8 / 8 / 8 | **3.96× smaller** | 24 → 106 |
| Qwen3-4B | 12/12 | 10 / 10 / 10 | 3.96× smaller | 47 → 159 |
| SmolLM2-1.7B | 11/12 | 8 / 7 / 7 | 3.96× smaller | 23 → 204 |
| OLMo2-1B | 12/12 | 8 / 7 / 7 | 3.96× smaller | 18 → 113 |
| BitNet-2B | 12/12 | 9 / 9 / 9 | 3.96× smaller | — |
<!-- /codespace-table -->

Three things the table shows and one it cannot. The fold is exact on real models, not
only in the identity: where whole generated sequences differed between the decoded and
code-space paths, the divergence was almost always after the answer, at a near-tie the
decoded path's f16 rounding resolved differently. The single answer-level exception
(SmolLM2, a counting item every path gets wrong) is the same mechanism landing on the
first token: dense said 10, decoded said 3, code-space said 10. The code-space path,
which keeps float32 scores, agreed with the dense model; the decoded path, which
rounds its reconstructed cache to f16, did not. The memory is real: the ratio is
bytes of resident packed tensors against the f16 cache they replace, and it is the
same on every architecture because it is a property of the rate. The hook is
architecture-neutral: Qwen3, Llama-family (SmolLM2), OLMo2 and BitNet needed no
per-model code, only a per-model codebook, because post-RoPE capture reads the cache as
the model leaves it. What the table cannot show is speed. Bit-unpacking with tensor
ops and a Python loop over KV-head groups is 4× to 9× slower per decoded token than
dense attention here, the factor growing with the number of KV heads (SmolLM2's 32 is
the worst case); the question prefill is roughly at par. That is the cost of having no
kernel, and it is the next thing to build, not a reason to doubt the rest.

The prototype's own self-test, `codespace_selftest.py`, runs on CPU in seconds: packing
is bit-exact against the codec's decode, both folds equal dotting with decoded states,
and on a tiny GQA Qwen3 the hook's logits match dense attention over the reconstructed
cache to float32 roundoff.
