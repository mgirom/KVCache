# submissions

One file per run: `submissions/YYYY/MM/<run_id>.json`.

Files, not a database, on purpose. The git history is tamper evidence nobody had to
build — you can see when a row appeared and that it has not been edited since — and
anyone can recompute the aggregates from the rows rather than trusting someone's SQL.

## Adding one

```bash
python3 auditor/runner/run.py --profile quick --model MODEL --arms q8_0,q4_0 \
    -o result.json --upload
```

`--upload` opens a pull request against this directory. Without it nothing is sent;
the tool is fully usable offline. The first upload from a machine prints the entire
payload and asks once — see [`../auditor/PRIVACY.md`](../auditor/PRIVACY.md) for the
closed field list and how to delete a submission afterwards.

## A note for maintainers

GitHub requires manual approval before Actions will run on a pull request from a
**first-time contributor**. So the first submission from a new person sits at
"action_required" until someone clicks *Approve and run workflows* on the PR. That is
GitHub's default and a sensible one -- it stops a stranger's PR from spending your
Actions minutes -- but it does mean a first submission needs a human before CI speaks.

## What CI checks

A pull request touching this directory is validated by the same `validate.py` the
submitter ran locally. It is **rejected**, not down-weighted, if the reference arm is
missing or below its floor, if quality is reported without task success, if declared
cache bytes disagree with measured by more than 1%, if a context rung is silently
omitted, or if the row's content no longer matches its own hash.

Being merged is not what makes a row trustworthy. Every row carries a `trust` field —
`unverified` until it passes plausibility screening, `reproduced` once an independent
machine matches it — and the leaderboard shows that field rather than hiding it.
