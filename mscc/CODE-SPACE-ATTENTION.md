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
