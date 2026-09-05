# A compressed KV cache type for llama.cpp: design before code

**Goal.** A `cpca` cache type in llama.cpp so a GGUF model, including the ternary 8B
that already runs here, holds its KV cache as codes during generation. Measured by the
auditor with no runner changes, since the auditor already selects arms by cache type.

**The finding that makes this small.** llama.cpp already rotates its quantised KV
cache. Since PR #21038, any quantised K or V cache with a head dimension divisible by
64 is stored in a Hadamard-rotated basis: Q and K are multiplied by the same
orthogonal matrix before the cache write, V is rotated on write and the attention
output is un-rotated, and a cache shift un-rotates, re-applies RoPE and re-rotates.
That is the entire plumbing this design needed. What it lacks is the right rotation.
The codec's basis is a data-fitted orthogonal rotation with a per-channel scale and a
mean; supplied in place of the Hadamard, at the same `q4_0` bytes, it is the whole
change. The compression rate is llama.cpp's own `q4_0` rate, 3.56× against f16, and the
claim becomes a direct comparison: the `q4_0` cache that costs a 1.7B model 13 points
with a Hadamard rotation, measured with a fitted one.

## The algebra, in the kernel's terms

Per layer, per KV head `h`, the codec gives mean `μ[d]`, per-channel std `s[d]` and an
orthonormal basis `V[d×d]` (full rank: the real 4-bit codebook funds all 128
components almost evenly, so nothing is truncated). Two matrices do all the work:

| tensor | value | applied to | why |
|---|---|---|---|
| `k_rot` | `Vᵀ·diag(1/s)` | K before write, plus bias `−k_rot·μ` | standardise, rotate, centre; what q4_0 sees |
| `q_rot` | `Vᵀ·diag(s)` | Q before attention | `q_rot q · k_rot (k−μ) = q·k − q·μ`: exact, and the `q·μ` term is constant per query, so softmax is unchanged |
| `v_rot` | `V_vᵀ·diag(1/s_v)` | V before write, plus bias `−v_rot·μ_v` | same as K |
| `v_unrot` | `diag(s_v)·V_v` | attention output, plus `μ_v` | softmax sums to 1, so `Σp v = v_unrot(Σp v′) + μ_v` exactly |
| `k_unrot` | `diag(s)·V` | cache shift only, plus `μ` | recover K, re-RoPE, re-apply `k_rot` |

No bias component in the key, no truncation, no new kernel. Per-head matrices ride on
`ggml_mul_mat`'s dimension-2 broadcast, which maps query head `h` to KV head
`h / groups`, the same grouping the kernel already uses for `K·Q`. Elementwise biases
on the output are pre-expanded to `n_head` at export, because `ggml_add` broadcasts
by modulo, not division.

## Where it touches llama.cpp

1. **Codebook file.** One GGUF per model, tensors `blk.{il}.cpca_{q_rot,k_rot,k_bias,
   k_unrot,k_mean,v_rot,v_bias,v_unrot,v_mean}`, written by `mscc/export_cpca_gguf.py`
   from a fitted `.kvcb.npz`. Metadata records the model, head geometry, corpus digest
   and codebook hash so a mismatch refuses rather than degrades.
2. **Load** (`llama_kv_cache` constructor): if `LLAMA_KV_CODEBOOK` names a file and the
   cache is quantised, read it and keep the tensors in a backend buffer on the cache's
   device, the way weights are held. The prototype uses an environment variable so the
   public API is untouched; an upstream version would add a context parameter.
3. **Graph** (`build_attn`, KV variant): where `llama_mul_mat_hadamard` is applied to
   `q_cur`, `k_cur`, `v_cur` and the output, use the per-head fitted tensors when
   present, else the Hadamard as today.
4. **Shift** (`build_rope_shift`): the same substitution in the dequantise → un-rotate →
   RoPE → rotate → requantise sequence.

Not touched: allocation, views, `set_rows`, the flash-attention kernel, the
quantisation kernels, the server, the sampler, the auditor.

## What is given up, and why it is acceptable

- **Variable-width codes.** The PyTorch codec gives 6 bits to the first component and
  4 to the rest at this rate; `q4_0` gives 4 to all with a per-block scale that adapts
  per token. The emulated quick audit measures the difference before any C++ exists.
- **Attention-sink protection.** A cache type cannot mix rows. Every token is coded and
  the cost is measured.
- **The 15× at-rest rate.** Stays a PyTorch frame feature. This is the live path, and
  its rate is `q4_0`'s.

## Getting a codebook without a HuggingFace twin

The 8B ternary model has no HF form that fits this machine. llama.cpp can write its
cache to a state file layer by layer in the cache's own type; with an f16 cache and
the rotation disabled (`LLAMA_ATTN_ROT_DISABLE=1`) that is post-RoPE K and V for a
corpus, and the existing fitter produces the codebook. One reader, no model-specific
code.

## Order of work and pass conditions

1. PyTorch: `q4_0`-emulated full-rank codebook on Qwen3-1.7B, quick audit.
   **Pass:** holds at 1k and 4k where llama.cpp's Hadamard `q4_0` lost 13 points.
2. Exporter + loader. **Pass:** tensors read back on the device equal the npz.
3. Graph + shift. **Pass:** greedy answers identical to the PyTorch emulation on the
   auditor's items.
4. Audit: Qwen3-1.7B GGUF, `q4_0` with codebook against its own f16 reference,
   standard profile, beside the existing Hadamard `q4_0` result.
5. Codebook from a state file; audit the ternary 8B the same way.
6. Speed is already the kernel's; report it as measured.

## Status

| milestone | result |
|---|---|
| 1. emulated quick audit, q4_0 full-rank, Qwen3-1.7B | 36/36 vs 35/36 at 1k; 32/36 vs 34/36 at 4k; 3.5× |
| 2. exporter + loader | 252 tensors, 70.5 MiB on CUDA0, all export self-checks pass |
| 3. graph + shift, check on 12 items | codebook confirmed loaded; fitted q4_0 agrees with f16 on 10/12 answers, Hadamard q4_0 on 7/12; correct: f16 9, Hadamard 7, fitted 8, PyTorch emulation 8 |
| 4. codebook from a state file (Bonsai-8B, ternary, GGUF only) | 16k tokens captured in 14 s, fitted, exported, self-checks pass |
| 5. standard audit, Qwen3-1.7B GGUF, n=240 | reference 136/144, 95/96 · q8_0 141/144, 95/96 (1.80×) · **q4_0 Hadamard 120/144, 82/96 (3.24×)** · **q4_0 fitted 134/144, 90/96 (3.19×)** |
| 5b. Bonsai-8B (ternary, GGUF only, codebook from its state file), n=288 | reference 144/144, 143/144 · q4_0 Hadamard 143/144, 142/144 (3.48×) · q4_0 fitted 143/144, 144/144 (3.38×): q4_0 is already free at 8B and the fitted basis keeps it free |
| 5c. block types with the same codebook, Qwen3-1.7B, n=240 | q5_0 Hadamard 138/144, 96/96 · q5_0 fitted 140/144, 95/96 (2.94×) · iq4_nl Hadamard 104/144, 84/96 · **iq4_nl fitted 138/144, 93/96 (3.60×): 231/240 against the reference's 231/240** |
| 5d. Bonsai-27B (ternary Qwen3.5 hybrid, 7.6 GB, attention on 16 of 64 layers, 4 KV heads × 256), codebook from its state file, n=288 | reference 144/144, 142/144 · q4_0 Hadamard 142/144, 143/144 (3.14×) · **q4_0 fitted 143/144, 143/144 (3.27×)**: no measurable loss on the largest model this 12 GB card holds. iq4_nl not run here (kernel speed). Its `Q2_0` GGUF fails to load in this build; the `Q2_g64` file loads. The model opens answers with a blank line and an empty think block; the harness's empty-reply retry (SPEC) made it scorable. |
| 5e. Bonsai-8B, iq4_nl, n=288 | reference 144/144, 143/144 · iq4_nl Hadamard 143/144, 144/144 · iq4_nl fitted 143/144, 144/144 (3.57×): free either way at the higher rate, at the iq4_nl kernel's speed cost |
| 5f. 16k rung and f32 row (phase 1) | Bonsai-8B at 16k: reference 93/96 · q4_0 Hadamard 90/96 · q4_0 fitted 94/96 (3.38×). Bonsai-27B at 16k: reference 142/144 · q4_0 Hadamard 143/144 · q4_0 fitted 143/144 (3.27×, 15.5 tok/s). Qwen3-1.7B f32 cache: 140/144, 95/96 against f16 136/144, 95/96 at twice the bytes and half the speed: more precision than 16 bits buys nothing measurable. |
| 5g. Qwen3-1.7B, GGUF-fit fully-whitened q4_0 codebook, n=240 | reference 136/144, 95/96 · q4_0 Hadamard 120/144, 82/96 · **q4_0 fitted (whitened, fit on the served file's states) 134/144, 96/96 (3.23×)**: the 4k gap is closed; decode 83.6 vs 88.9 tok/s at 1k, 70.7 vs 71.8 at 4k |

The milestone-3 bar was "identical to the PyTorch path". It is not met literally and
cannot be: the GGUF holds Q4_K_M weights and the HF twin bf16, so the two arms
disagree on 3 of 12 answers for reasons upstream of the cache. The bar that matters is
the one the audit measures, and the 12-item check already points the same way the
emulation did: the fitted rotation tracks the dense model where the Hadamard does not.

Two implementation notes for whoever ports this. The cache's own ggml contexts may be
`no_alloc` (dummy buffers, filled later by the scheduler), so the codebook needs its own
always-allocated buffers. And `ggml_n_dims()` cannot distinguish `[d, n_head, 1]` from
`[d·n_head, 1]`, so the per-head multiply decides the layout from `ne[0]`.

**Reading milestone 5.** At the same bytes, the fitted basis takes q4_0's loss from 11
points to 1.4 at 1k context and from 13.5 to 5 at 4k. The 4k gap is real at n=96 but
small, and it is the number to attack next: the PyTorch codec at the same rate lost
nothing at n=240 with variable-width codes and four dense sink tokens, neither of which
a single ggml block type provides. The cheap experiments are the other block types
with the same codebook (q5_0 at 5.5 bits, iq4_nl at 4.5), since the rotation does not
depend on the quantiser.

## Speed, as measured by the same runs

| model | arm | decode tok/s at 1k | at 4k | prefill ms at 1k | at 4k |
|---|---|---:|---:|---:|---:|
| Qwen3-1.7B | f16 | 100.2 | 74.1 | 236 | 1117 |
| | q4_0 Hadamard | 87.1 | 68.9 | 282 | 1170 |
| | q4_0 fitted | 75.4 | 61.9 | 332 | 1399 |
| Bonsai-8B | f16 | 50.5 | 45.9 | 897 | 3571 |
| | q4_0 Hadamard | 46.9 | 37.4 | 891 | 3691 |
| | q4_0 fitted | 41.0 | 33.7 | 1012 | 4180 |

The fitted basis costs 10 to 13 percent of decode throughput and 13 to 20 percent of
prefill time against the Hadamard, on both models. The Hadamard multiply carries a
fast-path hint in ggml; the fitted one is a dense per-head matmul wrapped in two
permute-and-copy steps so that heads broadcast correctly. A fused per-head projection
kernel, or folding the rotation into the Q/K/V projection weights where the
architecture allows it, is where that time goes next.

**A speed caveat on the block types.** In this llama.cpp CUDA build only `q4_0` and
`q8_0` caches have a fast flash-attention path. `q5_0` and `iq4_nl` fall back to a slow
one: on the 1.7B at 4k context they prefill in 72 to 83 seconds against 1.1 for f16 and
decode at 13 to 17 tokens per second against 74, with or without the fitted basis. The
`iq4_nl` accuracy result stands; using it today means paying that kernel gap, which is
llama.cpp's to close, not the codebook's.
| Bonsai-27B | f16 | 17.0 | 16.9 | 3871 | 14774 |
| Bonsai-27B | q4_0 | 16.7 | 16.0 | 3879 | 14801 |
| Bonsai-27B | q4_0+cpca | 16.3 | 15.7 | 3956 | 15052 |

## Phase 1 closed out (2026-09-04)

Done without a decision from anyone: `--kv-codebook` and a context parameter (the
environment variable stays as a fallback); codebooks bound to model architecture,
name and geometry, refused on mismatch; f16 codebooks at 37 to 48 MB, checked finite on
load; the three codebooks published on the `codebooks` branch with a registry of
model hashes and measured rows; `tools/kvcache.py` to pull, serve and audit; a CI
build of the patched llama.cpp on Linux, macOS and Windows that also validates every
filed submission; `reproduce-llamacpp.sh`, verified from a clean clone (pinned commit, seven patches, built, flag present); a documents index. CI green on all three platforms. The 16k rung on both ternary models and the f32 row are in (5f). Still yours: the Pages source
switch and PR #1.

## Phase 2 (in progress, 2026-09-05)

**Speed.** The per-head multiply is now one `ggml_mul_mat_id` with constant index tables:
no permutes, no copies. Two things the operator's CUDA path needed that mixture models
never ask of it: matrices in f16 rather than f32, and never more slots than matrices,
so the query-side matrices are expanded to one per query head at load (16 × 128 × 128
× 2 bytes per layer). CPU was correct throughout; CUDA returned garbage until the
expansion. The 12-item check keeps the same counts and agreement as the copy-based
build. Measured at the standard profile: decode 80.9 against Hadamard's 85.2 tok/s at 1k (−5%, was −13%) and 66.2 against 68.9 at 4k (−4%, was −10%); prefill +9% and +6% (were +18% and +20%). Accuracy in that run 139/144 and 94/96 against 136/144 and 95/96, so the 4k gap read one point here against five before: part of the earlier five was item-level noise. The per-layer index tables were then made one shared pair per buffer, since sized by context they had counted as 1.8 KB per token of cache.

**The 4k gap.** Hypothesis: `q4_0` sets one scale per 32 codes and the rotated
components have very unequal spread, so the block holding the top component drowns
the small ones. Fix: whiten each component (divide by its spread), folding the inverse
into the query and output matrices, so the stored numbers change and the algebra does
not. On the real 1.7B states, key-side attention-score error against the true keys:

| layer | plain q4_0 | whitened q4_0 |
|---|---:|---:|
| 0 | 0.049 | 0.029 |
| 7 | 0.101 | 0.046 |
| 14 | 0.074 | 0.065 |
| 21 | 0.082 | 0.063 |
| 27 | 0.081 | 0.045 |

Values improve slightly; the four sink rows get 10 to 50% worse, since whitening
amplifies their small components. Partial whitening (a power of the spread below one)
is the obvious middle if the sink rows turn out to matter. Emulated quick audit and the
llama.cpp standard audit of the whitened codebook are queued.

**Fit on the states of the file you serve.** Removing a confound in the table above
changed its meaning. The "plain" codebook was fitted on the bf16 HuggingFace weights'
states; the whitened ones on the GGUF model's own saved cache. A plain codebook fitted
on the GGUF states scores 0.046 at layer 7 against the HF-fitted one's 0.101, and 0.050
against 0.081 at layer 27: most of the gain was the fit data, not the whitening, which
adds a further 5 to 8 percent on top (half-power slightly better than full). A Q4_K_M
model's keys are not its bf16 twin's keys. The capture tool fits on the served file by
construction; the earlier 1.7B rows were measured with the HF-fitted codebook, so
audits of the GGUF-fitted plain, fully whitened and half-whitened codebooks are queued
to see how much of the 4k gap this alone closes.
| Qwen3-1.7B | q4_0 fitted, fused multiply | 80.9 | 66.2 | 306 | 1257 |
