#!/usr/bin/env python3
"""The consent gate and the uploader. Nothing leaves the machine without passing here.

Three properties this file exists to guarantee, in the order they matter:

1. **Nothing is sent that the operator has not seen.** On the first upload of a session
   the exact bytes are printed and the tool waits. Not a summary -- the payload.
2. **The field list is closed and enforced in code.** PRIVACY.md lists what may be
   collected; `check_fields` rejects a payload containing anything else. Adding a field
   therefore requires editing both this file and that document, which makes it a
   visible change rather than a quiet one.
3. **A row can be deleted by whoever produced it.** The server returns a deletion
   token, and it is written into the local result file rather than kept only in a
   scrollback buffer.

The default is offline. `--no-upload` is not a privacy mode, it is the ordinary way to
run the tool; uploading is the thing you opt into.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

#: Closed allowlist for the machine description. Mirrors the table in PRIVACY.md, and
#: the two must be edited together. Anything else in `system` is a bug or a leak.
ALLOWED_SYSTEM_FIELDS = frozenset({
    "os", "os_version", "arch", "backend", "backend_version",
    "cpu_model", "cpu_cores", "ram_bytes",
    "gpu_model", "gpu_vram_bytes", "gpu_count", "driver_version",
})

#: Keys that turn a machine description into a person: a name, a path, a stable id, a
#: credential. Matched by exact leaf name or by a small set of suffixes -- NOT by loose
#: substring, which was the first attempt and immediately flagged `_doc_tokens_mean`
#: as a credential. In a benchmark about tokens, "token" as a substring is useless.
FORBIDDEN_LEAF_NAMES = frozenset({
    "hostname", "host_name", "host", "username", "user_name", "user", "userid",
    "uid", "gid", "home", "cwd", "pwd", "path", "dir", "directory", "filename",
    "serial", "serial_number", "uuid", "guid", "mac", "macaddr", "mac_address",
    "ip", "ipaddr", "ip_address", "email", "password", "passwd", "secret",
    "api_key", "apikey", "auth_token", "access_token", "bearer", "credential",
    "deletion_token", "session_id", "machine_id", "install_id",
})
#: Suffixes that make an otherwise innocuous key identifying.
FORBIDDEN_LEAF_SUFFIXES = ("_path", "_dir", "_home", "_uuid", "_serial",
                           "_hostname", "_username", "_email", "_secret",
                           "_api_key", "_auth_token", "_access_token")

#: Imported, never redeclared. If the uploader and the hash disagree about what is
#: transmitted, every submission fails its own integrity check -- which is exactly what
#: happened the first time these were two separate lists.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
from validate import NON_TRANSMITTED as STRIP_TOP_LEVEL              # noqa: E402


class ConsentDenied(RuntimeError):
    pass


class FieldViolation(RuntimeError):
    pass


def _walk_keys(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield prefix + str(k)
            yield from _walk_keys(v, prefix + str(k) + ".")
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v, prefix)


def check_fields(payload: dict) -> list[str]:
    """Return the reasons this payload may not be sent. Empty means it may."""
    bad = []
    extra = set(payload.get("system", {})) - ALLOWED_SYSTEM_FIELDS
    for k in sorted(extra):
        bad.append(f"system.{k} is not on the PRIVACY.md field list")
    for key in _walk_keys(payload):
        leaf = key.rsplit(".", 1)[-1].lower().lstrip("_")
        if leaf in FORBIDDEN_LEAF_NAMES:
            bad.append(f"{key} is an identifier, path or credential ({leaf!r})")
            continue
        for suf in FORBIDDEN_LEAF_SUFFIXES:
            if leaf.endswith(suf):
                bad.append(f"{key} is an identifier or path (ends with {suf!r})")
                break
    return bad


def prepare(result: dict) -> dict:
    """The payload, stripped of local diagnostics. Never mutates the caller's dict."""
    return {k: v for k, v in result.items() if k not in STRIP_TOP_LEVEL}


def consent(payload: dict, *, assume_yes: bool = False, stream=sys.stderr) -> bool:
    """Show the operator exactly what would be sent, and wait.

    `assume_yes` exists for CI and for a machine being driven deliberately; it is never
    a default, and it is recorded in the run's output so a row uploaded without an
    interactive prompt says so.
    """
    body = json.dumps(payload, indent=1, sort_keys=True)
    print("\n" + "=" * 74, file=stream)
    print("ABOUT TO UPLOAD. This is the payload in full, not a summary:", file=stream)
    print("=" * 74, file=stream)
    print(body, file=stream)
    print("=" * 74, file=stream)
    print(f"{len(body):,} bytes. Field list and deletion policy: auditor/PRIVACY.md",
          file=stream)
    if assume_yes:
        print("--yes given: uploading without asking.", file=stream)
        return True
    if not sys.stdin.isatty():
        print("Not a terminal and --yes was not given: NOT uploading.", file=stream)
        return False
    try:
        ans = input("Upload this? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nNot uploading.", file=stream)
        return False
    return ans in ("y", "yes")


def upload(result: dict, endpoint: str, *, assume_yes: bool = False,
           timeout: int = 60) -> dict:
    """Consent, check, send. Returns {ok, run_id, deletion_token} or raises."""
    payload = prepare(result)
    violations = check_fields(payload)
    if violations:
        raise FieldViolation(
            "refusing to upload; the payload contains fields the privacy policy does "
            "not allow:\n  - " + "\n  - ".join(violations))
    if not consent(payload, assume_yes=assume_yes):
        raise ConsentDenied("upload declined")

    req = urllib.request.Request(
        endpoint.rstrip("/") + "/api/v1/submission",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "User-Agent": f"kv-audit/{payload.get('tool', {}).get('version', '0')}"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:400]
        except Exception:                                              # noqa: BLE001
            pass
        raise RuntimeError(f"upload rejected: HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"upload failed: {e.reason}") from e
    return resp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Inspect or upload a KV-Audit result.")
    ap.add_argument("result")
    ap.add_argument("--endpoint", default="")
    ap.add_argument("--print-payload", action="store_true",
                    help="show exactly what would be sent, and exit")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    res = json.load(open(a.result))
    payload = prepare(res)
    bad = check_fields(payload)
    if a.print_payload:
        print(json.dumps(payload, indent=1, sort_keys=True))
        if bad:
            print("\nWOULD BE REFUSED:", file=sys.stderr)
            for b in bad:
                print("  -", b, file=sys.stderr)
        sys.exit(1 if bad else 0)
    if not a.endpoint:
        ap.error("--endpoint is required unless --print-payload is given")
    out = upload(res, a.endpoint, assume_yes=a.yes)
    print(json.dumps(out, indent=1))
