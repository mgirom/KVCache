#!/usr/bin/env python3
"""Render the store as one page. The page answers a question, not a scoreboard.

The useful query is not "who is fastest". It is: *I have this hardware and I run this
model -- what does this setting cost me?* So the page is a lookup grouped by
(hardware, model, context), and inside each group the arms are compared against that
machine's own reference. Comparing an arm on one machine to a reference on another is
the thing this whole design exists to prevent.
"""
from __future__ import annotations

import html
import json
import os
import sys

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
from server import pooled, stats                                   # noqa: E402


#: The page and the API must agree about what counts. They did not: the API excluded
#: flagged rows from aggregates while the page listed them inline next to good ones,
#: so a 99,999 tok/s row appeared beside a real 32 tok/s one with nothing but a small
#: word to tell them apart.
TRUST_ORDER = {"unverified": 0, "plausible": 1, "reproduced": 2}


def derive_trust(doc: dict, peers=None) -> str:
    """Trust is COMPUTED here, never read from the submission.

    Two reasons, and the second is the important one. A row that arrives as a pull
    request never passes through the service that would have assigned it, so a stored
    value is often simply absent. And a stored value is submitter-controlled: nothing
    stopped someone writing `"trust": "reproduced"` into their own file. Deriving it at
    render time from the plausibility screen makes it an assessment by whoever is
    displaying the data, which is the only thing it could honestly have been.
    """
    from store import screen
    return "unverified" if screen(doc, peers) else "plausible"


def _rows(docs, min_trust="plausible"):
    out, excluded = [], 0
    floor = TRUST_ORDER.get(min_trust, 1)
    for d in docs:
        if TRUST_ORDER.get(derive_trust(d, docs), 0) < floor:
            excluded += 1
            continue
        s, m = d.get("system", {}), d.get("workload", {}).get("model", {})
        rh, rn = pooled(d.get("reference", {}))
        for arm in d.get("arms", []):
            for r in arm.get("rungs", []):
                if not r.get("ran"):
                    continue
                ts = r["quality"]["task_success"]["overall"]
                ref_rung = next((x for x in d["reference"]["rungs"]
                                 if x.get("ran") and x["context"] == r["context"]), None)
                rts = ref_rung["quality"]["task_success"]["overall"] if ref_rung else None
                out.append({
                    "hw": s.get("gpu_model") or s.get("cpu_model", "?"),
                    "backend": s.get("backend", "?"),
                    "model": m.get("name", "?"), "ctx": r["context"],
                    "arm": arm["name"],
                    "rate": ts["hits"] / ts["n"] if ts["n"] else None,
                    "ref_rate": (rts["hits"] / rts["n"]) if rts and rts["n"] else None,
                    "n": ts["n"],
                    "bpt": r["cost"]["kv_bytes_per_token_measured"],
                    "ref_bpt": ref_rung["cost"]["kv_bytes_per_token_measured"] if ref_rung else 0,
                    "tps": r["cost"]["decode_tok_per_s"],
                    "trust": derive_trust(d, docs),
                })
    return out, excluded


def _headline(rows) -> str:
    """What the data says, before the tables say it in detail.

    A visitor arriving cold should not have to read nine grouped tables to find out
    whether a setting is safe. This is computed from the rows on the page -- never
    typed in -- so it cannot drift away from the evidence beneath it, and it says how
    many measurements are behind each line rather than stating a bare verdict.
    """
    by_method: dict = {}
    for r in rows:
        if r["rate"] is None or r["ref_rate"] is None:
            continue
        g = by_method.setdefault(r["arm"], {"deltas": [], "ratios": [], "models": set()})
        g["deltas"].append(r["rate"] - r["ref_rate"])
        if r["bpt"] and r["ref_bpt"]:
            g["ratios"].append(r["ref_bpt"] / r["bpt"])
        g["models"].add(r["model"])
    if not by_method:
        return ""
    out = ['<div class="head"><h2>What the submissions say so far</h2><ul>']
    for arm, g in sorted(by_method.items(), key=lambda kv: -len(kv[1]["deltas"])):
        d = sorted(g["deltas"])
        med = d[len(d) // 2]
        worst = min(d)
        ratio = (f'{sorted(g["ratios"])[len(g["ratios"]) // 2]:.1f}&times; smaller cache'
                 if g["ratios"] else "cache size not measured")
        # a median near zero with a bad worst case is the case worth naming: it is the
        # shape q4_0 actually has, and an average would hide it
        if med >= -0.005 and worst >= -0.02:
            verdict = '<span class="ok">no measurable cost</span>'
        elif med >= -0.02:
            verdict = (f'<span class="warn">no cost on most, but as much as '
                       f'{abs(worst):.0%} on one</span>')
        else:
            verdict = f'<span class="bad">costs {abs(med):.0%} of task success</span>'
        out.append(f'<li><code>{html.escape(arm)}</code> &mdash; {ratio}, {verdict}'
                   f' <span class="dim">({len(d)} measurement(s), '
                   f'{len(g["models"])} model(s))</span></li>')
    out.append('</ul><p class="dim">Each figure is a delta against that same '
               'machine\'s uncompressed reference. Worst case is shown alongside the '
               'median because a setting that is free on a large model and harmful on '
               'a small one has a fine average and is still a trap.</p></div>')
    return "".join(out)


def render(docs) -> str:
    st = stats_from(docs)
    rows, excluded = _rows(docs)
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["hw"], r["backend"], r["model"], r["ctx"]), []).append(r)

    body = []
    for (hw, backend, model, ctx), rs in sorted(groups.items()):
        body.append(f'<h3>{html.escape(model)} <span class="dim">on '
                    f'{html.escape(hw)} ({html.escape(backend)}), context '
                    f'{ctx:,}</span></h3>')
        body.append('<table><tr><th>method</th><th>KV B/token</th><th>vs f16</th>'
                    '<th>task success</th><th>vs this machine&#39;s reference</th>'
                    '<th>decode</th><th>trust</th></tr>')
        for r in sorted(rs, key=lambda x: -x["bpt"]):
            # a run with --skip-kv-measure has no cache figure; say so rather than
            # printing a zero that reads as "free"
            bpt = (f'{r["bpt"]:,.0f}' if r["bpt"]
                   else '<span class="dim">not measured</span>')
            ratio = (f'{r["ref_bpt"] / r["bpt"]:.2f}&times;'
                     if r["bpt"] and r["ref_bpt"] else "&mdash;")
            rate = f'{r["rate"]:.3f}' if r["rate"] is not None else "&mdash;"
            if r["rate"] is not None and r["ref_rate"] is not None:
                d = r["rate"] - r["ref_rate"]
                cls = "bad" if d < -0.02 else ("ok" if d >= -0.005 else "warn")
                delta = f'<span class="{cls}">{d:+.3f}</span>'
            else:
                delta = "&mdash;"
            body.append(
                f'<tr><td><code>{html.escape(r["arm"])}</code></td>'
                f'<td class="n">{bpt}</td><td class="n">{ratio}</td>'
                f'<td class="n">{rate} <span class="dim">n={r["n"]}</span></td>'
                f'<td class="n">{delta}</td>'
                f'<td class="n">{r["tps"]:.1f} tok/s</td>'
                f'<td class="dim">{html.escape(r["trust"])}</td></tr>')
        body.append("</table>")
    if not groups:
        body.append('<p class="dim">No submissions yet.</p>')

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KV-Audit results</title><style>
:root{{--bg:#fbfaf8;--fg:#1a1a18;--dim:#6b6862;--line:#e0ddd6;--card:#fff;
 --ok:#1a7f4b;--warn:#a86a1a;--bad:#b3261e}}
@media(prefers-color-scheme:dark){{:root{{--bg:#16161a;--fg:#e9e7e2;--dim:#9a968e;
 --line:#2e2e34;--card:#1d1d22;--ok:#5cc98d;--warn:#d69a45;--bad:#ef6b62}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);
 font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;padding:20px}}
.wrap{{max-width:900px;margin:0 auto}}h1{{font-size:1.4rem;margin:0 0 4px}}
h3{{font-size:1rem;margin:26px 0 8px}}.dim{{color:var(--dim);font-weight:400;font-size:.85rem}}
table{{width:100%;border-collapse:collapse;background:var(--card);
 border:1px solid var(--line);border-radius:8px;overflow:hidden;font-size:.9rem}}
th,td{{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line)}}
th{{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--dim)}}
tr:last-child td{{border-bottom:0}}.n{{text-align:right;font-variant-numeric:tabular-nums}}
code{{font-size:.88em}}.ok{{color:var(--ok)}}.warn{{color:var(--warn)}}.bad{{color:var(--bad);font-weight:600}}
.lede{{color:var(--dim);margin:0 0 18px}}
.head{{background:var(--card);border:1px solid var(--line);border-radius:10px;
 padding:14px 18px;margin:18px 0 6px}}
.head h2{{font-size:.78rem;text-transform:uppercase;letter-spacing:.09em;
 color:var(--dim);margin:0 0 10px}}
.head ul{{margin:0;padding-left:20px}}.head li{{margin:5px 0}}.note{{color:var(--dim);font-size:.85rem;
 border-top:1px solid var(--line);margin-top:30px;padding-top:14px}}
</style></head><body><div class="wrap">
<h1>KV-Audit results</h1>
<p class="lede">What a KV cache setting costs you, on hardware like yours.
Every quality figure is a delta against <b>that machine's own uncompressed
reference</b>, measured in the same run &mdash; which is what makes a laptop's row
comparable to a datacentre's.</p>
<p class="dim">{st['submissions']} submissions &middot;
{len(st['models'])} models &middot; {len(st['hardware'])} machines &middot;
{', '.join(html.escape(b) for b in st['backends'])}
{f"&middot; {excluded} row(s) held back by plausibility screening" if excluded else ""}</p>
{_headline(rows)}
{''.join(body)}
<p class="note">Task success is measured on planted facts a model cannot know in
advance, so it scores the optimisation and not the model. Rates carry their n; a
tier the reference arm could not do at a given context is excluded and declared.
<code>trust</code> is <code>unverified</code> until a row passes plausibility
screening and <code>reproduced</code> once an independent machine matches it.
Nobody's row is hidden for looking wrong.</p>
</div></body></html>"""


def stats_from(docs):
    models, hw, backends = set(), set(), set()
    for d in docs:
        models.add(d.get("workload", {}).get("model", {}).get("name", "?"))
        s = d.get("system", {})
        hw.add(s.get("gpu_model") or s.get("cpu_model", "?"))
        backends.add(s.get("backend", "?"))
    return {"submissions": len(docs), "models": sorted(models),
            "hardware": sorted(hw), "backends": sorted(backends)}


if __name__ == "__main__":
    import argparse
    from store import Store
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="submissions")
    ap.add_argument("-o", "--out", default="site/index.html")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write(render(list(Store(a.root).all())))
    print(f"wrote {a.out}")
