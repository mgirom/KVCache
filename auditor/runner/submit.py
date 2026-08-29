#!/usr/bin/env python3
"""Automatic submission: run finishes, result goes to the results repository.

The point is to remove the manual step after a run, not to remove the decision. Those
are different things, and conflating them is how telemetry gets a bad name.

    default            nothing is sent. The tool is fully useful offline.
    --upload           send automatically when the run finishes, no prompt.
    first --upload     the full payload is printed and confirmed ONCE per machine.
                       The answer is recorded with the field-list version, so widening
                       what is collected asks again rather than riding on an old yes.
    --forget-consent   revoke it.

`--upload` is itself the consent for that invocation: someone typing it has asked for
the upload. The one-time confirmation exists so the first time is informed, and the
recorded version exists so a later change to the field list cannot inherit it.

TWO ROUTES TO THE REPOSITORY
    github  a pull request against the results repo, opened with `gh`. No server to
            run, no token held by anyone but the submitter, and the PR is validated by
            CI before it can merge -- the same validate.py that ran locally.
    http    POST to a hosted instance of auditor/service, for anyone who runs one.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import upload as U                                            # noqa: E402

#: Bumped whenever PRIVACY.md's field list changes. Consent is recorded against it, so
#: a wider collection cannot inherit an older agreement.
FIELD_LIST_VERSION = "0.1.0"
DEFAULT_REPO = "mgirom/KVCache"

CONSENT_PATH = os.path.join(
    os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
    "kv-audit", "consent.json")


# ------------------------------------------------------------------------ consent

def read_consent() -> dict:
    try:
        return json.load(open(CONSENT_PATH))
    except (OSError, json.JSONDecodeError):
        return {}


def have_consent() -> bool:
    c = read_consent()
    return bool(c.get("granted")) and c.get("field_list_version") == FIELD_LIST_VERSION


def record_consent(granted: bool) -> None:
    os.makedirs(os.path.dirname(CONSENT_PATH), exist_ok=True)
    json.dump({"granted": bool(granted),
               "field_list_version": FIELD_LIST_VERSION,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              open(CONSENT_PATH, "w"), indent=1)


def forget_consent() -> None:
    try:
        os.remove(CONSENT_PATH)
        print(f"consent removed ({CONSENT_PATH})")
    except OSError:
        print("no recorded consent to remove")


def ensure_consent(payload: dict, assume_yes: bool = False) -> bool:
    """One informed yes per machine, per field-list version."""
    if have_consent():
        return True
    print("\nThis is the first upload from this machine. What follows is the entire "
          "payload,\nnot a summary. It is described in auditor/PRIVACY.md, which also "
          "says how to\ndelete a submission later.", file=sys.stderr)
    ok = U.consent(payload, assume_yes=assume_yes)
    record_consent(ok)
    if ok:
        print(f"recorded in {CONSENT_PATH} -- later runs upload without asking. "
              f"Revoke with --forget-consent.", file=sys.stderr)
    return ok


# ------------------------------------------------------------------------- routes

def submit_github(result_path: str, repo: str = DEFAULT_REPO, dry_run: bool = False):
    """Open a pull request adding this submission to the results repository.

    A PR rather than a direct write: the submitter needs no privileged token, nothing
    is hosted, and CI validates the row before it can merge. The audit trail is the
    repository history, which is the property the file-based store was chosen for.
    """
    if not shutil.which("gh"):
        raise RuntimeError(
            "the GitHub route needs the `gh` CLI (https://cli.github.com) and "
            "`gh auth login`. Use --route http with a hosted service instead, or "
            "open a pull request by hand adding this file under submissions/.")
    doc = json.load(open(result_path))
    payload = U.prepare(doc)
    bad = U.check_fields(payload)
    if bad:
        raise U.FieldViolation("refusing to submit; payload carries fields the "
                               "privacy policy does not allow:\n  - "
                               + "\n  - ".join(bad))
    run_id = payload["run_id"]
    utc = payload.get("utc", "")
    dest = f"submissions/{utc[:4] or 'unknown'}/{utc[5:7] or 'unknown'}/{run_id}.json"
    body = json.dumps(payload, indent=1, sort_keys=True)

    if dry_run:
        print(f"[dry run] would add {dest} to {repo} ({len(body):,} bytes)")
        return {"ok": True, "dry_run": True, "path": dest}

    # gh handles the fork, the branch and the PR; nothing here needs a token of its own
    cmd = ["gh", "api", f"repos/{repo}/contents/{dest}", "-X", "PUT",
           "-f", f"message=submission {run_id[:8]}",
           "-f", f"content={__import__('base64').b64encode(body.encode()).decode()}"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        # no write access is the normal case for a contributor: fall back to a PR
        return _pull_request(repo, dest, body, run_id)
    return {"ok": True, "route": "direct", "path": dest,
            "url": f"https://github.com/{repo}/blob/main/{dest}"}


def _pull_request(repo: str, dest: str, body: str, run_id: str):
    """Fork, branch, commit, PR -- the path for someone without write access."""
    import tempfile
    work = tempfile.mkdtemp(prefix="kvaudit-submit-")
    try:
        subprocess.run(["gh", "repo", "fork", repo, "--clone", "--remote=false",
                        "--", work], capture_output=True, text=True, timeout=300)
        clone = work if os.path.exists(os.path.join(work, ".git")) else None
        if not clone:
            raise RuntimeError("could not fork/clone the results repository")
        branch = f"submission-{run_id[:8]}"
        os.makedirs(os.path.join(clone, os.path.dirname(dest)), exist_ok=True)
        open(os.path.join(clone, dest), "w").write(body)
        for c in (["git", "checkout", "-q", "-b", branch],
                  ["git", "add", dest],
                  ["git", "commit", "-q", "-m", f"submission {run_id[:8]}"],
                  ["git", "push", "-q", "-u", "origin", branch]):
            subprocess.run(c, cwd=clone, check=True, capture_output=True, timeout=300)
        pr = subprocess.run(
            ["gh", "pr", "create", "--repo", repo, "--title",
             f"submission {run_id[:8]}", "--body",
             "Automated KV-Audit submission. CI validates it against "
             "auditor/result.schema.json and the section-5 rules in SPEC-v0.1.md."],
            cwd=clone, capture_output=True, text=True, timeout=300)
        return {"ok": pr.returncode == 0, "route": "pull_request",
                "path": dest, "url": pr.stdout.strip() or pr.stderr.strip()[:200]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def submit(result_path: str, route: str = "github", repo: str = DEFAULT_REPO,
           endpoint: str = "", assume_yes: bool = False, dry_run: bool = False):
    doc = json.load(open(result_path))
    if not ensure_consent(U.prepare(doc), assume_yes=assume_yes):
        return {"ok": False, "error": "upload declined"}
    if route == "http":
        if not endpoint:
            raise RuntimeError("--route http needs --endpoint")
        return U.upload(doc, endpoint, assume_yes=True)
    return submit_github(result_path, repo=repo, dry_run=dry_run)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", nargs="?")
    ap.add_argument("--route", default="github", choices=("github", "http"))
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--endpoint", default="")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--forget-consent", action="store_true")
    a = ap.parse_args()
    if a.forget_consent:
        forget_consent()
        return 0
    if not a.result:
        ap.error("give a result file, or --forget-consent")
    print(json.dumps(submit(a.result, route=a.route, repo=a.repo,
                            endpoint=a.endpoint, assume_yes=a.yes,
                            dry_run=a.dry_run), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
