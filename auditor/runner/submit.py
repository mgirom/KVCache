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

#: Remembered answer to the post-run offer. "ask" is the default: the question is worth
#: asking, and a benchmark that shares by default is a benchmark nobody trusts twice.
#:
#: PREF_NEVER is deliberately NOT offered as a keystroke in the prompt -- a single
#: mistyped character should not permanently remove someone from the record. It is
#: still reachable, via `--never-share`, and documented in PRIVACY.md. Removing the
#: option entirely would leave anyone who does not want to share being asked after
#: every run forever, which is nagware; keeping it one deliberate command away is the
#: compromise.
PREF_ASK, PREF_ALWAYS, PREF_NEVER = "ask", "always", "never"

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


def read_pref() -> str:
    return read_consent().get("share_pref", PREF_ASK)


def record_pref(pref: str) -> None:
    c = read_consent()
    c["share_pref"] = pref
    c.setdefault("field_list_version", FIELD_LIST_VERSION)
    c["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(CONSENT_PATH), exist_ok=True)
    json.dump(c, open(CONSENT_PATH, "w"), indent=1)


def offer_to_share(result_path: str, summary: str = "") -> str:
    """Ask, after the run, whether to share -- and say what sharing is for.

    The reason matters. "Send telemetry?" gets a no; "compare your machine against
    others running the same thing" is the actual point of a submission and is what the
    person gets back for it. Both are true here, so the question says the true one.

    Returns one of yes / always / no / never. Non-interactive sessions always decline:
    a benchmark must never take silence for agreement.
    """
    pref = read_pref()
    if pref == PREF_NEVER:
        return "never"
    if pref == PREF_ALWAYS:
        return "always"
    if not sys.stdin.isatty():
        return "no"

    print("\n" + "-" * 74, file=sys.stderr)
    print("Share this result?", file=sys.stderr)
    print("  Submissions are pooled so you can see how your machine compares with "
          "others\n  running the same model and settings -- which is the only way "
          "the question\n  'what does this cost on MY hardware' ever gets answered.",
          file=sys.stderr)
    if summary:
        print(f"\n{summary}", file=sys.stderr)
    print("\n  What is sent: the measurements above, plus your hardware and OS model "
          "numbers.\n  Never: hostname, username, file paths, IP, or anything from "
          "your disk.\n  Full list and how to delete a submission: auditor/PRIVACY.md",
          file=sys.stderr)
    print("-" * 74, file=sys.stderr)
    try:
        ans = input("  [y] yes   [a] always   [n] not this time : ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  not shared.", file=sys.stderr)
        return "no"
    if ans in ("a", "always"):
        record_pref(PREF_ALWAYS)
        print("  recorded -- later runs share without asking. "
              "Undo with --forget-consent.", file=sys.stderr)
        return "always"
    return "yes" if ans in ("y", "yes") else "no"


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


def _gh_identity() -> tuple[str, str]:
    """Name and email for the commit, taken from the authenticated GitHub account.

    Not from git config: a contributor may have none set, and a fresh clone inherits
    nothing. That is not hypothetical -- the first end-to-end test of this path failed
    at `git commit` with exit 128 on a machine with no global identity, which is the
    situation any first-time contributor is in. Deriving it from the account also makes
    the commit attribute to the person who submitted it.
    """
    try:
        out = subprocess.run(["gh", "api", "user", "--jq", "[.login, .id] | @tsv"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        login, uid = out.split("\t")
        return login, f"{uid}+{login}@users.noreply.github.com"
    except Exception:                                                  # noqa: BLE001
        return "kv-audit submitter", "kv-audit@users.noreply.github.com"


def _run(cmd, cwd, what):
    """Run a step and report what actually went wrong, not a bare CalledProcessError."""
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        detail = (p.stderr or p.stdout).strip().splitlines()
        raise RuntimeError(f"{what} failed: "
                           + (detail[-1][:200] if detail else f"exit {p.returncode}"))
    return p


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
        name, email = _gh_identity()
        os.makedirs(os.path.join(clone, os.path.dirname(dest)), exist_ok=True)
        open(os.path.join(clone, dest), "w").write(body)
        _run(["git", "checkout", "-q", "-b", branch], clone, "creating a branch")
        _run(["git", "add", dest], clone, "staging the submission")
        _run(["git", "-c", f"user.name={name}", "-c", f"user.email={email}",
              "commit", "-q", "-m", f"submission {run_id[:8]}"], clone, "committing")
        _run(["git", "push", "-q", "-u", "origin", branch], clone,
             "pushing to your fork")
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


def compare_with_peers(result_path: str, repo: str = DEFAULT_REPO, timeout: int = 20):
    """How this machine landed against everyone else running the same thing.

    This is what a submitter gets back for submitting, so it runs after a successful
    share rather than being promised and never delivered. Rows are matched on the same
    workload hash, the same model hash and the same arm -- anything looser would be
    comparing different measurements wearing the same units.
    """
    import urllib.error
    import urllib.request
    doc = json.load(open(result_path))
    wl = doc["workload"]["sha256"]
    mh = doc["workload"]["model"].get("sha256", "")
    api = f"https://api.github.com/repos/{repo}/contents/submissions"

    def fetch(url):
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "kv-audit"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    peers = []
    try:
        stack = [api]
        while stack and len(peers) < 200:
            for e in fetch(stack.pop()):
                if e["type"] == "dir":
                    stack.append(e["url"])
                elif e["name"].endswith(".json"):
                    try:
                        with urllib.request.urlopen(e["download_url"], timeout=timeout) as r:
                            peers.append(json.load(r))
                    except Exception:                                  # noqa: BLE001, PERF203
                        continue
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return ("No submissions in the repository yet -- yours will be the first "
                    "to compare against.")
        return f"(could not fetch peer results: HTTP {e.code})"
    except Exception as e:                                             # noqa: BLE001
        return f"(could not fetch peer results: {type(e).__name__})"

    def pooled(d, arm_name):
        for a in [d.get("reference", {})] + d.get("arms", []):
            if a.get("name") != arm_name:
                continue
            h = sum(r["quality"]["task_success"]["overall"]["hits"]
                    for r in a.get("rungs", []) if r.get("ran"))
            n = sum(r["quality"]["task_success"]["overall"]["n"]
                    for r in a.get("rungs", []) if r.get("ran"))
            tps = [r["cost"]["decode_tok_per_s"] for r in a.get("rungs", []) if r.get("ran")]
            return (h / n if n else None), (sorted(tps)[len(tps)//2] if tps else None)
        return None, None

    same = [p for p in peers
            if p.get("workload", {}).get("sha256") == wl
            and p.get("workload", {}).get("model", {}).get("sha256") == mh
            and p.get("run_id") != doc.get("run_id")]
    if not same:
        return ("You are the first submission for this model and workload. "
                "Later runs will compare against yours.")

    lines = [f"Compared with {len(same)} other submission(s) of the same model and "
             f"workload:", ""]
    for arm in doc.get("arms", []):
        mine_q, mine_s = pooled(doc, arm["name"])
        others = [pooled(p, arm["name"]) for p in same]
        oq = sorted(q for q, _ in others if q is not None)
        os_ = sorted(s for _, s in others if s is not None)
        if mine_q is None or not oq:
            continue
        faster = sum(1 for s in os_ if mine_s and s < mine_s)
        lines.append(
            f"  {arm['name']:<10} quality {mine_q:.3f}  vs peer median "
            f"{oq[len(oq)//2]:.3f}"
            + (f"   speed {mine_s:.1f} tok/s, faster than {faster}/{len(os_)}"
               if mine_s and os_ else ""))
    return "\n".join(lines)


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
    ap.add_argument("--forget-consent", action="store_true",
                    help="clear the remembered answer, so the offer is made again")
    ap.add_argument("--never-share", action="store_true",
                    help="stop being asked. Not offered as a keystroke in the prompt, "
                         "because one mistyped character should not remove you from "
                         "the record permanently.")
    a = ap.parse_args()
    if a.never_share:
        record_pref(PREF_NEVER)
        print(f"you will not be asked again ({CONSENT_PATH}). "
              f"Undo with --forget-consent.")
        return 0
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
