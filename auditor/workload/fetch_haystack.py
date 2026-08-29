#!/usr/bin/env python3
"""Fetch and normalise the public-domain haystack the tasks are planted into.

LICENCE, because it decides whether this can ship. A standing rule in this tree
(2026-06-01) is that anything published is mirrored onto a neutral public-domain
source first. Moby-Dick (1851) is public domain worldwide by age. Project Gutenberg
wraps its texts in a header and footer carrying the PG trademark and its own terms;
those are STRIPPED here, leaving only the public-domain work, which is what makes the
resulting file redistributable without PG's licence attached.

The output is pinned by sha256 in the workload manifest. If this script's
normalisation changes, the hash changes, and results from the two versions are never
compared -- which is the point.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import urllib.request

SOURCE = "https://www.gutenberg.org/files/2701/2701-0.txt"
TITLE = "Moby-Dick; or, The Whale — Herman Melville, 1851 (public domain)"

START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                   re.IGNORECASE)
END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                 re.IGNORECASE)

#: Where the work itself starts, past the title page, the transcriber's note and the
#: front matter. Two reasons to trim here rather than at the PG marker: the
#: transcriber's note is editorial matter added in the modern era, not part of the
#: 1851 work; and the front matter ("ETYMOLOGY", "EXTRACTS") is lists of quotations,
#: which is unlike the continuous prose a haystack is supposed to be.
CONTENT_START = "CHAPTER 1. Loomings."


def strip_boilerplate(raw: str) -> str:
    m = START.search(raw)
    if m:
        raw = raw[m.end():]
    m = END.search(raw)
    if m:
        raw = raw[:m.start()]
    return raw


def normalise(text: str) -> str:
    """Collapse the incidental formatting so the token stream is stable prose.

    Deliberately conservative: line endings, trailing spaces, and runs of blank
    lines. Nothing that changes words, because the haystack's job is to be ordinary
    prose that a planted sentence has to be found inside of.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=SOURCE)
    ap.add_argument("-o", "--out", default="auditor/workload/haystack.txt")
    ap.add_argument("--min-chars", type=int, default=800_000,
                    help="a 16k-token rung needs roughly 70k chars per document; the "
                         "floor is set well above that so items never wrap onto "
                         "themselves")
    a = ap.parse_args()

    if os.path.exists(a.out):
        cur = open(a.out, encoding="utf-8").read()
        print(f"exists: {a.out} ({len(cur):,} chars, "
              f"sha256 {hashlib.sha256(cur.encode()).hexdigest()})")
        return 0

    print(f"fetching {a.url}", flush=True)
    with urllib.request.urlopen(a.url, timeout=60) as r:
        raw = r.read().decode("utf-8", "replace")
    print(f"  {len(raw):,} chars raw")

    body = strip_boilerplate(raw)
    # last occurrence: the file carries a table of contents that repeats the heading
    i = body.rfind(CONTENT_START)
    if i < 0:
        print(f"REFUSED: content start {CONTENT_START!r} not found; the source layout "
              "changed and trimming cannot be verified", file=sys.stderr)
        return 2
    text = normalise(body[i:])
    if len(text) < a.min_chars:
        print(f"REFUSED: {len(text):,} chars is below the {a.min_chars:,} floor",
              file=sys.stderr)
        return 2
    # Hard gate. Any surviving mention means either the licence header is still
    # attached or the trim missed editorial matter, and in both cases the file is not
    # cleanly redistributable. Rejecting is correct; this fired once already.
    for marker in ("project gutenberg", "gutenberg.org", "transcriber"):
        if marker in text.lower():
            print(f"REFUSED: {marker!r} survives in the trimmed text; this is not "
                  "clean public-domain content", file=sys.stderr)
            return 2

    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    print(f"wrote {a.out}")
    print(f"  title   {TITLE}")
    print(f"  chars   {len(text):,}")
    print(f"  sha256  {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
