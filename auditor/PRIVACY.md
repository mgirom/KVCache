---
doc: KV-AUDIT PRIVACY
version: 0.1.0-draft
date: 2026-08-29
---

# What is collected, when, and how to get rid of it

The short version: **the tool uploads nothing unless you say so, once, per run.** It is
fully usable with `--no-upload`, and everything it would have sent is written to a
local file first so you can read it before deciding.

## Why anything is collected at all

The benchmark's whole content is a join: *given this hardware, what does this KV
setting cost me?* Quality is measured against the machine's own reference arm; cost is
measured on the machine itself. Neither half answers the question alone, and the
hardware half cannot be inferred. That is the reason for the fields below, and the
limit of it — nothing is collected because it might be interesting later.

## The field list, in full

This list is **closed**. The tool refuses to upload a submission containing a key that
is not on it (`--strict-fields`, on by default), so adding a field is a visible change
to this document and not a quiet change to a payload.

**Machine**
| field | example | why |
|---|---|---|
| `os` | `linux` | backends and kernels differ by platform |
| `os_version` | `6.17.0-1032-oem` | kernel/driver interactions |
| `arch` | `x86_64` | ditto |
| `backend` | `cuda` | the single most important cost determinant |
| `backend_version` | `b1-11cd988` | comparability across llama.cpp builds |
| `cpu_model` | `13th Gen Intel(R) Core(TM) i9-13900` | CPU-backend throughput |
| `cpu_cores` | `32` | ditto |
| `ram_bytes` | `33339334656` | which context rungs were feasible |
| `gpu_model` | `NVIDIA RTX A2000 12GB` | the join's primary key, in practice |
| `gpu_vram_bytes` | `12884901888` | which rungs were feasible |
| `gpu_count` | `1` | ditto |
| `driver_version` | `595.84` | numerics and kernel selection change with drivers |

**Run**
| field | why |
|---|---|
| `run_id` | a fresh random 128-bit value per run. Not derived from anything about you or the machine, and not stable across runs. |
| `utc` | ordering results against driver and build changes |
| `tool.version`, `tool.build_sha` | which code produced the row |
| `workload.*` | which workload and model, by hash |
| measurements | the scores and timings — the point of the exercise |

## What is never collected

Not "not currently" — the uploader strips these and the field allowlist rejects them:

- hostname, username, or any account identifier
- filesystem paths (the model is identified by **hash**, never by where it lives)
- serial numbers, MAC addresses, UUIDs of hardware, or any stable machine identifier
- the contents of anything on your disk. The workload is a public-domain book and a
  generated task file, both shipped with the tool; your documents are never read.
- IP addresses beyond the transient use any web request makes. Not stored with the row.
- any telemetry outside a run you explicitly approved. There is no background reporting,
  no crash reporter, and no "check for updates" ping.

## Consent

**Nothing is sent without you agreeing to it, and the tool is fully useful offline.**

When a run finishes it **asks** whether to share, and says why: submissions are pooled
so you can see how your machine compares with others running the same model and
settings. Your options are *yes* (this run), *always*, or *not this time*. `--upload`
is a standing yes for scripted runs. A non-interactive session — a pipe, a cron job,
CI — always declines, because silence is not agreement.

**To stop being asked at all:** `python3 auditor/runner/submit.py --never-share`, or
`--never-share` on any run. It is a flag rather than a keystroke in the prompt so that
one mistyped character cannot permanently remove someone from the record — not to make
it hard to find, which is why it is documented here and in `--help`.

- The **first** upload from a machine prints the entire payload — the bytes, not a
  summary — and asks once. Later runs upload when the run finishes, without prompting.
- The answer is recorded in `~/.config/kv-audit/consent.json` **together with the
  version of the field list below**. Widening what is collected bumps that version and
  the tool asks again, so a broader collection can never inherit an older yes.
- `--forget-consent` revokes it. `--print-payload` shows what would be sent and exits
  without running anything.
- `--yes` skips the one-time question, for CI and unattended runs.

The earlier draft of this document promised opt-in *per run* with no remembered
setting. That was changed deliberately: requiring a click after every run is friction
that makes people either skip submitting or reach for `--yes` permanently, and neither
is better for them. What is preserved is the part that matters — an informed first yes,
tied to a specific field list, revocable, and never a default.

## Deleting your data

Each accepted upload returns a **deletion token**. Keeping it lets you remove that row
later without proving anything about who you are:

```
DELETE /api/v1/submission/<run_id>   with header  X-Deletion-Token: <token>
```

The token is stored locally in the run's output file. If you lose it, the row cannot be
attributed back to you by us either — which is the same property that makes the row
non-identifying in the first place. That is the trade, stated plainly rather than
discovered later.

## Legal basis, in the plain sense

If you are in a jurisdiction where this matters: the collected set is intended to be
non-personal. It describes a machine's model numbers, not a person, and carries no
stable identifier that links two runs from the same machine. The consent step exists
anyway, because "we decided it isn't personal data" is a judgement the person whose
machine it is should get to make too.

## If any of this changes

The field list is versioned with this document and the `schema_version` of a
submission. A submission produced under one version is never silently re-interpreted
under another. Widening the list requires a version bump, and the tool will prompt
again rather than reusing an earlier approval.
