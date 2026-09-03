---
doc: KV-AUDIT PROTOCOL
version: 0.1.0-draft
date: 2026-08-27
status: draft — nothing has been run against this yet
---

# KV-Audit — protocol specification

A downloadable, cross-platform benchmark that answers one question:

> **You turned on a KV cache optimisation. What did it cost you?**

Every existing local-inference benchmark reports tokens per second. None of them
report what the speed was bought with. This one reports both, because this project
has the receipt that they decouple: a configuration measured at 0.833 top-1 agreement
answered **0 of 12** planted questions correctly, and a configuration 15x smaller than
its baseline ran **3.7x slower** than not using it at all.

## 0. The one rule everything else follows from

**Every quality number is a delta against the same model, unquantised, measured in the
same process on the same machine in the same run.**

Not against a published number. Not against another machine. Not against a different
model. The reference arm is mandatory and its result ships inside every submission.

This is what makes results comparable across a Mac laptop and an 8xH100 node without
requiring either to be deterministic with the other. It also means a submission that
omits the reference arm is not a low-quality result, it is an **invalid** one.

## 1. Two modes, and they are not the same product

| mode | what is pinned | what varies | what the number means |
|---|---|---|---|
| **`hardware`** | model, quant, method, tasks | the machine | "how fast is this box" — the Cinebench mode |
| **`method`** | model, tasks, the machine | the KV method under test | "what does this optimisation cost" — the auditor mode |

Quality axes are hardware-independent: the same model and method give the same recall
on an A2000 and an H100, within tolerance. Running them on ten thousand machines
yields the same number ten thousand times. Cost axes are hardware-dependent.

The product is the **join** of the two. Whether int4 KV is worth it depends on the
machine — on a 12 GB card memory pressure decides everything, on an 80 GB card nobody
would bother. That question cannot be answered today, and answering it requires
exactly the crowdsourced hardware data a downloadable benchmark collects.

## 1b. Adapt to the machine; never adapt the measurement

A benchmark meant to run on a laptop, a workstation and a server cannot assume any of
them. Two rules keep that from destroying comparability:

**What adapts: which rungs are attempted.** The tool probes free device and host memory
before it starts and runs the rungs that fit. Rungs that do not fit are recorded with
`ran: false` and a reason, never silently omitted. This is safe because rungs are
independently comparable — a laptop that reaches only 1,024 tokens produces a valid
result *at the 1,024 rung*, directly comparable to a server's 1,024 rung.

**What never adapts: anything inside a rung that runs.** The items, the burial depths,
the answer normalisation, the scoring rule, and the mandatory reference arm are the
workload. A workload that varies by machine measures nothing.

**Take as little of the machine as possible.** Two specific commitments:

- `-ngl` is **not** passed by default. Forcing "put every layer on the GPU and fail if
  you cannot" is fine on the machine a tool was written on and is a crash on a laptop,
  on a busy GPU, or on a CPU-only box. llama.cpp fits its own parameters to free device
  memory and keeps a reserve; the tool lets it, and records when a user overrides it.
- A memory **reserve** (default 1 GB, `--reserve-gb`) is excluded from the ladder
  calculation. A benchmark that consumes every available byte evicts whatever else the
  machine was doing; on a laptop that is the difference between a background task and
  an unusable computer.

**Measuring the cache without a GPU.** The size slope is taken over whichever memory
the cache lands in — device memory on CUDA, resident host memory on CPU and
unified-memory backends — and the result records which, in `_kv_measurement.measured_in`.
A slope that does not grow with context length in either memory is reported as
unmeasured rather than guessed.

The backend recorded in a submission comes from which binary ran, not from whether the
machine happens to have a GPU. Running a CPU build on a box with an NVIDIA card is a
CPU result, and labelling it otherwise is a mislabel a leaderboard would carry forever.

## 2. The pinned workload

A workload is a content-addressed bundle. Its `sha256` appears in every result, and
results from different workload hashes are never compared.

    workload/
      manifest.json      model pin, context ladder, tolerances, task file hash
      tasks.json         generated deterministically by gen_tasks.py, then frozen
      haystack.txt       the filler prose the tasks are planted into

**Model pin.** One model, one quant, identified by file sha256 — not by name. "Qwen3
1.7B Q4_K_M" is not a pin; a hash is. Cross-model comparison is out of scope for v0.1
because tokenisers differ, so identical text is a different amount of work and
tokens/sec stops meaning the same thing.

**Haystack licence.** Public domain only, pinned by hash. This is a standing rule in
this tree (2026-06-01): internal work may use any corpus, anything that ships publicly
is mirrored onto a neutral public-domain source first. Project Gutenberg prose is the
intended source. **No public workload ships on the Russell corpus.**

**Context ladder.** Candidates are 1,024 / 4,096 / 16,384 tokens, and which of them
are attempted is decided by section 1b's probe rather than by the workload. A machine
that cannot fit a rung records it as `skipped` with a reason; it never silently omits
it. A result covering fewer rungs is valid and comparable *at the rungs it ran*.

## 3. The tasks — simple to check, hard to survive

Four tiers. Every answer is **planted in the document**, so none of them test what the
model knows; a model that has never heard of the subject matter scores the same. Every
answer is checked by exact normalised string match — no judge model, no rubric, no
scoring drift between versions.

The design target is: **trivial at full precision, fragile under compression.** The
reference arm should score at or near 1.0 on every tier. If it does not, the task is
broken and gets cut, not kept as "hard".

| tier | what it stresses | example plant | question | answer |
|---|---|---|---|---|
| **T1 retrieve** | one span, one location | "The maintenance access code for the Dunraven lighthouse is 47-19-83." | what is the code | `47-19-83` |
| **T2 link** | two locations at once | "The Ferngate shipment weighed 6,412 kilograms." … 3,000 tokens later … "The Kelwick shipment weighed the same as the Ferngate shipment." | what did Kelwick weigh | `6,412` |
| **T3 aggregate** | the whole span, not one point | seven `Vessel logged:` lines scattered throughout | how many vessels were logged | `7` |
| **T4 distractor** | resisting the blur | "North gate code 11-22-33." … "South gate code 44-55-66." | the **south** gate code | `44-55-66` |

T2 deliberately carries **no arithmetic** ("the same as", not "twice"). Arithmetic
failures are model failures and would contaminate the signal.

T4 exists because it targets the observed failure mode directly: a degraded cache does
not go blank, it goes *confidently adjacent*. The mid-stack architecture in this repo
answered `1234567890` — fluent, formatted like a code, wrong.

**Burial depths:** 10% / 50% / 90% of context. **Items:** 4 per tier per depth per
context rung = 48 per rung, 144 for the full ladder. n is reported alongside every
rate; no rate is ever reported without it.

## 3b. Exclusion is per (tier, context), not per tier

A tier the reference arm cannot do measures the model, not the optimisation. That
judgement has to be made **per context length**, because capability is
context-dependent. Measured on Qwen3-1.7B with no compression involved at all:

| tier | reference @1k | @4k | @16k |
|---|---|---|---|
| `t2_link` | 48/48 | 47/48 | **42/48 — below floor** |
| `t4_distractor` | 43/48 | 33/48 | (decoy grabs rise 5 → 15) |

The model holds `t2_link` at 1k and 4k and loses it at 16k. Excluding it everywhere
would throw away two sound rungs; keeping it everywhere would charge every method for a
limit the model brought itself. So `excluded_tiers` maps a tier to the **contexts** it
is excluded at, and a rung reporting a tier excluded at that rung's context is invalid.

This was found by the validator rejecting a real submission, not by inspection.

## 3c. Profiles change how much runs, never how it is scored

A benchmark nobody finishes has no submissions, so there are two sizes:

| profile | what runs | wall clock (1.7B, this card) | what it can conclude |
|---|---|---|---|
| `quick` | one rung, 48 sampled items | **under 2 minutes** | cost figures; the pipeline works here |
| `standard` | the probed ladder, every item | 25 min – 3 h | quality, with intervals |

The items, depths, answer normalisation, scoring rule and mandatory reference arm are
identical in both. Only coverage differs, so the two remain comparable at the rungs and
tiers they share — the same rule as section 1b.

**What differs is statistical power, and it is reported rather than left to the reader.**
Every run prints the smallest shortfall its n could resolve:

| n | resolves a shortfall of |
|---:|---|
| 36 (quick) | ~25% or more |
| 96 | ~10% |
| 240 | ~6% |
| 432 | ~4% |

This matters concretely: `q4_0` costs about 13% on a 1.7B, so **a quick run would miss
it**. That is not a flaw in the quick profile, it is what a quick profile is; saying so
in its own output is the difference between a fast check and a misleading one.

**Sampled items must be the same items across arms.** When a limit is in force, the
arms answer exactly the items the reference answered on its surviving tiers — not an
independently drawn set of the same size. Striding a four-tier pool and a three-tier
pool by the same count selects different items, and the delta then compares an arm on
one set against a reference on another. That was a live bug, found by the quick profile
reporting n=36 for its reference and n=48 for its arms.

## 4. What gets measured

### Quality — hardware-independent, reported as deltas

- `task_success` per tier, per depth, per rung, with n
- `agreement` — per-position top-1 match with the reference arm over a held-out tail
- Both are reported. **Neither may be reported alone**, and a submission that reports
  agreement without task success is invalid. That rule exists because of the 0.833 /
  0-of-12 result: agreement is a proxy that certifies nothing.

### Cost — hardware-dependent, measured not declared

- `kv_bytes_per_token`, measured off the live cache, **with quantisation scales and
  any full-precision retained tokens charged**. A method that keeps per-token scales
  or a full-precision sink window pays for them here.
- `peak_vram_bytes` and `peak_host_bytes`, sampled during the run
- `prefill_ms`, `ttft_ms`, `decode_tok_per_s`, and — the one nobody publishes —
  `restore_ms`, the wall-clock to get a stored cache back into a usable state through
  the real serialisation path, not an idealised one.

`restore_ms` is mandatory for any method that stores or transports a cache. It is the
number that turned a 15x compression win in this repo into a 3.7x latency regression,
and omitting it is how that regression stayed invisible for a day.

## 5. Validity rules

A submission is rejected, not down-weighted, if any of these fail:

1. the reference arm is missing, or ran on a different machine, build, or session
2. `workload.sha256` does not match a published workload
3. quality is reported without task success
4. the reference arm scores below `manifest.reference_floor` on any tier that ran
   (the workload is broken on this stack; the result says nothing about the method)
5. declared `kv_bytes_per_token` differs from the measured value by more than 1%
6. a context rung is absent without a `skipped` record and a reason

## 5b. The server under test must be proved to be ours

A health check proves something is alive on a port. It does not prove it is the thing
the tool started, and the difference is not academic: a run here bound to a default
port that another job was already serving, saw a healthy response, and measured a
different model on a different backend for a full sweep. Nothing in the result said so.
The only tell was that the prefill timings were seven times too fast.

Two guards, both required:

- **Claim a free port.** The tool binds before it launches and walks upward until it
  finds one nothing is listening on. A fixed default port is a landmine on any machine
  that is not the developer's.
- **Check identity positively.** After the health check passes, the server's reported
  model must match the model that was launched, or the run refuses. Refusing is correct:
  a benchmark that measures a stranger and reports it as your result is worse than one
  that stops.

A submission produced without these guards cannot be trusted, which is why they are in
the protocol and not only in the implementation.

## 6. Integrity — what is and is not achievable

Anyone can POST a fabricated result. Full anti-cheat is not possible for a tool that
runs on the submitter's machine, and pretending otherwise would be worse than saying
so. What is achievable:

- signed release binaries; the build sha is recorded in the result
- a result hash over the canonical serialisation, so a row cannot be edited after
  submission without detection
- **plausibility screening**: does that decode rate make sense for that reported GPU
  and that model size, given everything else in the database? Outliers get flagged
  for review rather than silently trusted or silently dropped.
- reproduction: any flagged row can be re-run by anyone, because the workload is
  content-addressed and the model is pinned by hash

Results carry a `trust` field: `unverified` (default), `plausible` (passed screening),
`reproduced` (an independent submitter matched it within tolerance on comparable
hardware). The leaderboard shows the field. It does not hide it.

## 7. Telemetry and consent

Specs are collected because the join in section 1 is the entire product; they are not
collected because they are interesting.

- Upload is **opt-in per run**, never a default, and the tool is fully usable offline
  with `--no-upload`.
- The exact field list is printed before the first upload and lives in
  `PRIVACY.md`. It is closed: CPU/GPU model strings, core and memory counts, OS and
  driver versions, backend, build sha. No hostname, no username, no paths, no IP
  stored beyond rate-limiting, no serial numbers.
- A submission id is returned so a submitter can delete their own row later.

Retrofitting consent onto a database that already has rows in it is expensive and
sometimes impossible. This section exists in v0.1 for that reason.

## 8. Explicit non-goals for v0.1

- cross-model comparison (tokenisers differ; the number would not mean one thing)
- model quality benchmarking — this measures what an optimisation costs a model, and
  says nothing about whether the model is any good
- training or fine-tuning
- bit-exact cross-backend reproducibility — kernels differ; that is what the tolerance
  in the manifest and the mandatory same-machine reference arm are for

**Empty reply on a stop string (added 2026-09-04).** Generation stops server-side on a
blank line or a new "Question:" marker. A model that opens its answer with a blank line
or an empty `<think></think>` block (Qwen3.5 does both) would return nothing and score
a miss that has nothing to do with its cache. When a reply is empty and the stop
reason is a stop string, the runner retries that item once without server stops,
discards a leading think block, and applies the same stops client-side. The rule is
the same for every arm, and the item's record carries `retried_without_stops: true`
so the retry is visible in the results file.
