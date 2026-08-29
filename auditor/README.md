# KV-Audit — the spec, before any code

Cinebench for KV cache optimisations. Every local-inference benchmark today reports
tokens per second. This one reports tokens per second **and what the speed cost you**,
because this project has the receipt that they come apart:

- a configuration measured at **0.833 top-1 agreement** answered **0 of 12** planted
  questions correctly ([Gate 12](../findings/2026-08-26-alphabet/GATE-12-KVFRAME-RESULTS.md))
- a configuration **15× smaller** than its baseline ran **3.7× slower** than not using
  it at all ([timing correction](../results/gate12/timing.json))

Both would be invisible under a tok/s-only benchmark. Both change the decision.

## What is here

| file | what it is |
|---|---|
| [SPEC-v0.1.md](SPEC-v0.1.md) | the protocol: two modes, pinned workload, four axes, validity rules, integrity, consent |
| [result.schema.json](result.schema.json) | the submission contract, machine-checkable |
| [workload/gen_tasks.py](workload/gen_tasks.py) | deterministic task generator → `tasks.json`, then frozen and hashed |
| [validate.py](validate.py) | enforces the rules JSON Schema cannot express; `--selftest` proves it catches them |

Nothing here has been run against a model yet. It is deliberately spec-first: the
schema is the contract every implementation and every backend has to satisfy, and
writing it down is what exposes whether the four axes are actually well defined.

## The two rules everything hangs off

**1. Every quality number is a delta against the same model, unquantised, measured on
the same machine in the same run.** Not against a published number, not against another
machine. That is what makes a Mac laptop and an 8×H100 node comparable without either
being deterministic with the other. A submission without its reference arm is invalid,
not weak.

**2. No task may be answerable from the model's own knowledge.** Every answer is
planted in the document by the generator, from invented names and invented numbers. A
model that has never met the subject matter scores identically. What is left is the
only thing being measured: whether the optimisation still lets the model see its own
context.

## The task ladder

Trivial at full precision, fragile under compression. If the reference arm fails an
item, the item is broken and gets cut — difficulty comes from the compression, never
from the question.

| tier | stresses | breaks when |
|---|---|---|
| `t1_retrieve` | one span, one location | the cache blurs a single fact |
| `t2_link` | two spans, far apart, joined by identity (no arithmetic) | long-range attention thins |
| `t3_aggregate` | the whole span, not one point | coverage degrades anywhere |
| `t4_distractor` | the real fact beside a plausible near-twin | the cache goes *confidently adjacent* |

`t4` earns its place: a degraded cache does not go blank. The mid-stack architecture in
this repo answered `1234567890` — fluent, formatted exactly like a real access code,
and wrong.

## Try it

```bash
python3 auditor/workload/gen_tasks.py
```

```bash
python3 auditor/validate.py --selftest
```

## Status and what is deliberately missing

Draft. Open, in rough order:

- **the haystack.** Public-domain prose, pinned by hash. A standing rule in this tree
  (2026-06-01) is that anything shipping publicly is mirrored onto a neutral
  public-domain source first, so no public workload ships on the Russell corpus.
- **the runner.** Should be built on `llama.cpp`, not on the PyTorch harness in
  `alphabet/` — one binary, no Python for users to fight, and CUDA / Metal / Vulkan /
  ROCm / CPU backends already exist there. The PyTorch harness stays the research
  instrument; these are two different jobs.
- **determinism tolerances.** Different backends give different logits. The manifest
  needs per-axis tolerances decided in advance, not discovered afterwards.
- **the upload endpoint, and `PRIVACY.md`** with the closed field list. Consent is
  cheap now and sometimes impossible to retrofit.
- **a first real result**: int8 / int4 / int2 on two or three open models, published.
  That is what earns the right to ask anyone else to run this.
