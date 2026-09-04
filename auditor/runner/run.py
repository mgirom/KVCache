#!/usr/bin/env python3
"""KV-Audit runner: drive llama.cpp across arms, score them, emit a valid submission.

One invocation = one machine, one workload, one mandatory reference arm, and one or
more arms under test. Every quality number is a delta against the reference arm
measured in this same run on this same machine, which is what makes a result from a
laptop comparable to one from a datacentre without either being bit-identical.

  python3 auditor/runner/run.py --model <gguf> --arms f16,q8_0,q4_0 \
      --contexts 1024,4096 -o results/auditor/run.json
"""
from __future__ import annotations

import argparse, hashlib, json, os, platform, subprocess, sys, time, uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, "auditor")]

from assemble import assemble, check, prompt_for                     # noqa: E402
from server import (LlamaServer, measure_kv_bytes_per_token,          # noqa: E402
                    gpu_process_vram_bytes, probe_model_meta)
import capability as CAP                                              # noqa: E402
import backends as BE                                                 # noqa: E402
import validate as V                                                 # noqa: E402

TIERS = ("t1_retrieve", "t2_link", "t3_aggregate", "t4_distractor")


def model_digest(path, chunk=8 << 20):
    """Pin a model by content, whether it is one GGUF file or a directory of weights.

    A directory is hashed by mscc.format.model_fingerprint, which digests config.json
    plus every weight file by content -- the same function the MSCC guard uses to
    decide whether a frame may be loaded, so a model has one identity across both
    tools rather than two that can disagree."""
    if os.path.isdir(path):
        sys.path.insert(0, ROOT)
        from mscc.format import model_fingerprint
        return model_fingerprint(path)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=15).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def system_info(cap):
    """The machine as it will be recorded. `cap` decides the backend, not the mere
    presence of a GPU: running the CPU build on a box that happens to have an NVIDIA
    card was reported as a CUDA result, which is a mislabel a leaderboard would carry
    forever."""
    gpu = sh("nvidia-smi --query-gpu=name,memory.total,driver_version "
             "--format=csv,noheader,nounits")
    info = {"os": cap["os"], "os_version": platform.release(), "arch": cap["arch"],
            "backend": cap["backend"],
            "cpu_model": (sh("lscpu | grep 'Model name' | head -1").split(":", 1)[-1]
                          .strip() or platform.processor()),
            "cpu_cores": os.cpu_count() or 1}
    try:
        info["ram_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        pass
    # GPU details only belong on a result that actually used the GPU
    if gpu and cap["backend"] in ("cuda", "rocm"):
        f = [p.strip() for p in gpu.splitlines()[0].split(",")]
        if len(f) >= 3:
            info.update(gpu_model=f[0], gpu_vram_bytes=int(float(f[1]) * 1024 * 1024),
                        driver_version=f[2],
                        gpu_count=len(gpu.strip().splitlines()))
    return info


#: A profile changes how much of the workload runs, and therefore what the result can
#: conclude. It must never change how a cell that runs is scored -- same items, same
#: normalisation, same reference-arm rule -- so profiles stay comparable at the rungs
#: and tiers they share. What differs is statistical power, and that is reported rather
#: than left for the reader to work out.
PROFILES = {
    "quick":    {"contexts": "1024", "limit": 48, "probe": (1024, 4096),
                 "why": "a few minutes: cost figures and a working-pipeline check"},
    "standard": {"contexts": "auto", "limit": 0, "probe": (2048, 16384),
                 "why": "the full ladder and item set"},
}


def detectable_gap(n: int, ref_rate: float = 0.98, z: float = 1.96) -> float:
    """Smallest shortfall from the reference this n could actually resolve.

    A quick run cannot settle a quality question and should say so in its own output
    rather than leaving someone to infer it from a wide interval. This walks down from
    the reference rate until the intervals separate, which is the same test the summary
    applies to the arms.
    """
    ref_lo = wilson(round(ref_rate * n), n)[0]
    for pct in range(0, 101):
        r = ref_rate - pct / 100.0
        if r < 0:
            break
        if wilson(round(r * n), n)[1] < ref_lo:
            return pct / 100.0
    return 1.0


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval on a proportion.

    Repeat runs would be the obvious way to get error bars and would be worthless
    here: decoding is greedy and the workload is fixed, so a re-run reproduces the
    same answers exactly. The uncertainty that actually exists is which items were
    drawn, and for a proportion that is a binomial interval. Wilson rather than
    normal-approximation because n is small and the rates sit near 1.0, where the
    normal interval runs past 100% and stops meaning anything.
    """
    if n <= 0:
        return (0.0, 0.0)
    ph = hits / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate(hits, n):
    lo, hi = wilson(int(hits), int(n))
    return {"hits": int(hits), "n": int(n), "rate": (hits / n) if n else 0.0,
            "ci95": [round(lo, 4), round(hi, 4)]}


def run_arm(make_backend, arm_name, contexts, items, haystack,
            n_predict, limit, verbose=True, tiers=None, skipped=None,
            only_ids=None):
    """Run one KV configuration across the context ladder. Returns the arm object."""
    rungs, details = [], []
    for ctx, why in sorted((skipped or {}).items()):
        rungs.append({"context": ctx, "ran": False, "skip_reason": why})
    for ctx in contexts:
        allowed = None if tiers is None else tiers.get(ctx, set())
        pool = [it for it in items if it["context"] == ctx
                and (allowed is None or it["tier"] in allowed)]
        if only_ids is not None:
            # Arms must answer the SAME items the reference answered, not an
            # independently sampled set of the same size. Striding a 4-tier pool and a
            # 3-tier pool by the same count picks different items, and the delta then
            # compares an arm on one set against a reference on another -- which is the
            # comparison this whole protocol exists to make impossible.
            pool = [it for it in pool if it["id"] in only_ids]
        elif limit and limit < len(pool):
            # stride, never head: items are generated tier-major, so [:N] would sample
            # one tier and report it as an overall score
            step = len(pool) / limit
            pool = [pool[int(i * step)] for i in range(limit)]
        if not pool:
            rungs.append({"context": ctx, "ran": False, "skip_reason": "user_skipped"})
            continue
        try:
            srv = make_backend(ctx).start()
        except Exception as e:                                        # noqa: BLE001
            reason = "insufficient_vram" if "alloc" in str(e).lower() else "unsupported"
            rungs.append({"context": ctx, "ran": False, "skip_reason": reason})
            if verbose:
                print(f"    ctx {ctx}: SKIPPED ({reason}) -- {e}", flush=True)
            continue

        live = TIERS if allowed is None else sorted(allowed)
        by_tier = {t: [0, 0] for t in live}
        by_depth, decoys, truncated = {}, [0, 0], [0, 0]
        prefill_ms, tok_s, peak = [], [], 0
        cpt_hint, doc_tokens, own_bpt = 4.2, [], None
        t0 = time.time()
        try:
            for k, it in enumerate(pool):
                doc, cpt = assemble(it, haystack, ctx, srv.n_tokens,
                                    offset_chars=(k * 977 + ctx) * 431,
                                    hint_cpt=cpt_hint)
                cpt_hint = cpt          # the search converges faster each item
                r = srv.complete(prompt_for(it, doc), n_predict=n_predict)
                reply = r.get("content", "")
                res = check(it, reply)
                # Counted as a harness artifact only when the budget ran out AND the
                # reply had produced a strict prefix of the right answer. Complete-but-
                # wrong, or a reply that never produced a number, are method failures.
                hit_limit = (r.get("stop_type") == "limit" and res["partial_answer"])
                tm = r.get("timings", {})
                doc_tokens.append(tm.get("prompt_n", 0))
                prefill_ms.append(tm.get("prompt_ms", 0.0))
                tok_s.append(tm.get("predicted_per_second", 0.0))
                by_tier[it["tier"]][0] += res["hit"]
                by_tier[it["tier"]][1] += 1
                d = str(it["depth"])
                by_depth.setdefault(d, [0, 0])
                by_depth[d][0] += res["hit"]
                by_depth[d][1] += 1
                truncated[0] += hit_limit and not res["hit"]
                truncated[1] += 1
                if "took_decoy" in res:
                    decoys[0] += res["took_decoy"]
                    decoys[1] += 1
                details.append({"ctx": ctx, "id": it["id"], "tier": it["tier"],
                                "depth": it["depth"], "hit": res["hit"],
                                "got": res["first"], "want": it["answer"],
                                "took_decoy": res.get("took_decoy"),
                                "stop_type": r.get("stop_type"),
                                "retried_without_stops": bool(r.get("retried_without_stops", False)),
                                "reply": reply[:200]})
                if k % 8 == 0:
                    peak = max(peak, srv.vram_bytes() or 0)
                if verbose and (k + 1) % 12 == 0:
                    hits = sum(v[0] for v in by_tier.values())
                    print(f"      {k+1}/{len(pool)}  hits {hits}", flush=True)
            peak = max(peak, srv.vram_bytes() or 0)
            # ask the backend for its cache size while it is still alive: stop() tears
            # the model down, and the cost dict is assembled after the finally block
            own_bpt = (srv.kv_bytes_per_token(ctx)
                       if hasattr(srv, "kv_bytes_per_token") else None)
        finally:
            srv.stop()
            time.sleep(1.5)

        hits = sum(v[0] for v in by_tier.values())
        n = sum(v[1] for v in by_tier.values())
        q = {"task_success": {
            "overall": rate(hits, n),
            "by_tier": {t: rate(*by_tier[t]) for t in live if by_tier[t][1]},
            "by_depth": {d: rate(*v) for d, v in sorted(by_depth.items())}}}
        cost = {"kv_bytes_per_token_measured": 0.0,       # filled in by the caller
                "prefill_ms": round(sum(prefill_ms) / len(prefill_ms), 2),
                "decode_tok_per_s": round(sum(tok_s) / len(tok_s), 2),
                "peak_vram_bytes": int(peak), "repeats": 1}
        # what the documents actually measured, so a rung that drifted off its
        # nominal size is visible rather than quietly mislabelled
        if doc_tokens:
            q["_doc_tokens_mean"] = round(sum(doc_tokens) / len(doc_tokens), 1)
        q["_truncated_misses"] = rate(truncated[0], truncated[1])
        # a backend that knows its own cache size exactly reports it; only the
        # llama.cpp arms need the memory-slope probe
        if own_bpt is not None:
            cost["kv_bytes_per_token_measured"] = round(own_bpt, 1)
        rungs.append({"context": ctx, "ran": True, "quality": q, "cost": cost})
        if verbose:
            print(f"    ctx {ctx}: {hits}/{n}  "
                  + "  ".join(f"{t[:2]}={by_tier[t][0]}/{by_tier[t][1]}"
                              for t in live if by_tier[t][1])
                  + f"  decoy={decoys[0]}/{decoys[1]}"
                  + f"  trunc={truncated[0]}"
                  + f"  prefill={cost['prefill_ms']:.0f}ms"
                  + f"  {cost['decode_tok_per_s']:.1f}tok/s"
                  + f"  [{time.time()-t0:.0f}s]", flush=True)
    return rungs, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.path.join(
        ROOT, "llama.cpp/build-cuda/bin/llama-server"))
    ap.add_argument("--model", default=os.path.join(
        ROOT, "models/qwen3-1.7b-fp/q3-1.7b-Q4_K_M.gguf"))
    ap.add_argument("--tasks", default=os.path.join(ROOT, "auditor/workload/tasks.json"))
    ap.add_argument("--haystack", default=os.path.join(ROOT, "auditor/workload/haystack.txt"))
    ap.add_argument("--backend", default="llamacpp", choices=("llamacpp", "mscc"),
                    help="what to measure. Any implementation of backends.Backend can "
                         "enter; 'mscc' audits this project's own codec by the same "
                         "rules as everything else.")
    ap.add_argument("--codebook", default="",
                    help="mscc backend: the fitted codebook for the arm's rate")
    ap.add_argument("--sink", type=int, default=4,
                    help="mscc backend: full-precision leading tokens")
    ap.add_argument("--arms", default="q8_0,q4_0",
                    help="KV cache types under test; the f16 reference is implicit")
    ap.add_argument("--profile", default="standard", choices=tuple(PROFILES),
                    help="how much of the workload to run. 'quick' takes a few "
                         "minutes and is honest about what it cannot conclude; "
                         "'standard' is the full ladder.")
    ap.add_argument("--contexts", default="",
                    help="Context ladder. 'auto' probes free memory and attempts only "
                         "the rungs that fit, recording the rest as skipped with a "
                         "reason. Pass an explicit list to override.")
    ap.add_argument("--ladder", default="1024,4096,16384",
                    help="candidate rungs that 'auto' chooses from")
    ap.add_argument("--reserve-gb", type=float, default=1.0,
                    help="memory left alone for the rest of the machine. A benchmark "
                         "that takes every byte is antisocial; on a laptop it is the "
                         "difference between a background task and an unusable box.")
    ap.add_argument("--ngl", type=int, default=None,
                    help="force GPU layers. Left unset, llama.cpp fits to free device "
                         "memory itself and keeps its own reserve -- which is almost "
                         "always what you want and is the only thing that works on a "
                         "machine unlike this one.")
    ap.add_argument("--limit", type=int, default=0,
                    help="items per rung, 0 = all. Sampled by striding, never by "
                         "taking the head, since items are generated tier-major.")
    ap.add_argument("--n-predict", type=int, default=64,
                    help="Generation budget. Must be generous: a verbose preamble "
                         "('The access code for the east gate at Marlow is ') can eat "
                         "the budget before the answer lands, and a truncated correct "
                         "answer scores as a miss -- charging the method for a harness "
                         "limit. Truncations are counted and surfaced.")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--reference-floor", type=float, default=0.9)
    ap.add_argument("--skip-kv-measure", action="store_true")
    ap.add_argument("--upload", action="store_true",
                    help="submit the result when the run finishes. Nothing is sent "
                         "without this. The first use on a machine prints the whole "
                         "payload and asks once; later runs go without prompting.")
    ap.add_argument("--repo", default="mgirom/KVCache",
                    help="results repository for --upload")
    ap.add_argument("--endpoint", default="",
                    help="hosted service instead of the repository, with --route http")
    ap.add_argument("--route", default="github", choices=("github", "http"))
    ap.add_argument("--never-share", action="store_true",
                    help="stop being asked to share, permanently")
    ap.add_argument("--yes", action="store_true",
                    help="skip the one-time confirmation. For CI and unattended runs; "
                         "the result records that it was not interactively confirmed.")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    prof = PROFILES[a.profile]
    # explicit flags always win over the profile; the profile only supplies defaults
    if not a.contexts:
        a.contexts = prof["contexts"]
    if not a.limit:
        a.limit = prof["limit"]
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    log_dir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "serverlogs")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    cap = CAP.detect(a.binary)
    print(CAP.describe(cap) + "\n", flush=True)

    # Only serialise on the GPU lock if this run will actually use the GPU. A CPU
    # backend queueing behind a CUDA job is a self-inflicted wait, and on a machine
    # with no GPU at all it is meaningless.
    if cap["backend"] in ("cuda", "rocm"):
        try:
            sys.path.insert(0, os.path.join(ROOT, "alphabet", "scripts"))
            import gpulock
            gpulock.acquire("kv-audit")
        except Exception:                                             # noqa: BLE001
            pass

    tasks = json.load(open(a.tasks))
    haystack = open(a.haystack, encoding="utf-8").read()
    wl_sha = hashlib.sha256(
        (tasks["sha256"] + hashlib.sha256(haystack.encode()).hexdigest()).encode()
    ).hexdigest()

    print(f"model     {os.path.basename(a.model)}")
    print(f"workload  {tasks['n_items']} items, sha {wl_sha[:16]}")
    print(f"arms      f16(ref) + {arms}\n", flush=True)

    def backend_for(name):
        """One arm name -> a factory taking a context length -> a Backend."""
        if a.backend == "mscc":
            bits = 0 if name in ("f16", "kv_exact") else int(name.replace("cpca", ""))
            cbp = a.codebook or os.path.join(
                ROOT, "mscc/accept/kv", f"book{'' if bits == 1024 else bits}.kvcb.npz")
            return lambda ctx: BE.MsccBackend(a.model, cbp, ctx, unit_bits=bits,
                                              sink=a.sink)
        # "q4_0+cpca": llama.cpp's q4_0 cache with the fitted rotation from --codebook
        base, plus_cpca = (name.split("+", 1) + [""])[:2]
        if plus_cpca and plus_cpca != "cpca":
            raise SystemExit(f"unknown arm modifier {plus_cpca!r} in {name!r}")
        if plus_cpca and not a.codebook:
            raise SystemExit(f"arm {name!r} needs --codebook (a .cpca.gguf)")
        return lambda ctx: BE.LlamaCppBackend(a.binary, a.model, ctx, ctk=base,
                                              ctv=base, port=a.port, ngl=a.ngl,
                                              log_dir=log_dir,
                                              codebook=a.codebook if plus_cpca else None)

    if a.backend == "llamacpp":
        model_meta = probe_model_meta(a.binary, a.model, port=a.port, log_dir=log_dir)
    else:
        b = backend_for("f16")(256)
        try:
            model_meta = b.start().model_meta()
        finally:
            b.stop()
    if model_meta:
        exp = (model_meta["n_layers"] * 2 * model_meta["n_kv_heads"]
               * model_meta["head_dim"] * 2)
        print(f"shape     {model_meta['arch']} {model_meta['n_layers']}L x "
              f"{model_meta['n_kv_heads']}kv x {model_meta['head_dim']}d "
              f"-> {exp:,} B/token at f16\n", flush=True)
    else:
        print("shape     UNREADABLE -- cache geometry will be recorded as 0\n",
              flush=True)

    kv_bytes, kv_detail = {}, {}
    if a.backend != "llamacpp":
        # The slope probe drives llama-server directly. For another backend the cache
        # size is declared by the codebook and charged by the frame, so it is recorded
        # from there rather than measured here -- and said so, not passed off as a
        # measurement.
        a.skip_kv_measure = True
    if not a.skip_kv_measure:
        print("measuring KV bytes/token by memory slope", flush=True)
        for t in ["f16"] + arms:
            # an arm name may carry "+cpca"; the server wants the base type and the
            # codebook by environment, and the probe must measure that same configuration
            base = t.split("+", 1)[0]
            env = {"LLAMA_KV_CODEBOOK": os.path.abspath(a.codebook)} if t.endswith("+cpca") else None
            bpt, det = measure_kv_bytes_per_token(
                a.binary, a.model, base, base, port=a.port, log_dir=log_dir,
                small=prof["probe"][0], large=prof["probe"][1], env=env)
            kv_bytes[t] = bpt
            kv_detail[t] = det
            print(f"  {t:>6}  {bpt:,.0f} B/token  (in {det.get('measured_in')})"
                  if bpt else f"  {t:>6}  unmeasured ({det.get('reason')})",
                  flush=True)
        print(flush=True)

    # --- the ladder adapts to the machine; the scoring inside a rung never does
    ladder = [int(x) for x in a.ladder.split(",")]
    skipped: dict[int, str] = {}
    if a.contexts.strip().lower() == "auto":
        model_bytes = os.path.getsize(a.model)
        worst = max((v for v in kv_bytes.values() if v), default=0) or \
            (model_meta.get("n_layers", 32) * 2
             * model_meta.get("n_kv_heads", 8)
             * model_meta.get("head_dim", 128) * 2)
        feas = CAP.feasible_contexts(cap, model_bytes, worst, ladder=ladder,
                                     reserve_bytes=int(a.reserve_gb * (1 << 30)))
        contexts = [c for c, ok, _ in feas if ok]
        skipped = {c: why for c, ok, why in feas if not ok}
        print("context ladder (auto, "
              f"{a.reserve_gb:.1f} GB left for the rest of the machine):", flush=True)
        for c, ok, why in feas:
            print(f"  {c:>6}  {'run' if ok else 'skip -- ' + why}", flush=True)
        print(flush=True)
        if not contexts:
            print("no rung fits in the available memory. Lower --reserve-gb, free "
                  "memory, or use a smaller model.", file=sys.stderr)
            return 2
    else:
        contexts = [int(x) for x in a.contexts.split(",")]
        print(f"context ladder (explicit): {contexts}\n", flush=True)

    all_details = []

    def requality(arm, live_by_ctx):
        """Rebuild an arm's scores over a subset of tiers, from the per-item record.

        The reference arm has to run every tier to discover which ones it cannot do.
        Re-running it afterwards on the survivors would be a second full pass for
        answers already in hand -- about a quarter of the whole job on the 8B. The
        per-item detail is already kept, so the surviving score is recomputed from it.
        """
        for r in arm["rungs"]:
            if not r.get("ran"):
                continue
            live_tiers = live_by_ctx.get(r["context"], set())
            rows = [d for d in all_details
                    if d["arm"] == arm["name"] and d["ctx"] == r["context"]
                    and d["tier"] in live_tiers]
            if not rows:
                r["ran"] = False
                r["skip_reason"] = "unsupported"
                r.pop("quality", None)
                r.pop("cost", None)
                continue
            bt, bd = {}, {}
            for d in rows:
                bt.setdefault(d["tier"], [0, 0])
                bt[d["tier"]][0] += d["hit"]
                bt[d["tier"]][1] += 1
                k = str(d["depth"])
                bd.setdefault(k, [0, 0])
                bd[k][0] += d["hit"]
                bd[k][1] += 1
            hits = sum(v[0] for v in bt.values())
            n = sum(v[1] for v in bt.values())
            r["quality"]["task_success"] = {
                "overall": rate(hits, n),
                "by_tier": {t: rate(*v) for t, v in sorted(bt.items())},
                "by_depth": {d: rate(*v) for d, v in sorted(bd.items())}}
        return arm

    def build(name, tiers=None, only_ids=None):
        """`tiers` is either None (everything) or {context: set_of_tiers}."""
        print(f"arm {name}", flush=True)
        rungs, det = run_arm(backend_for(name), name, contexts, tasks["items"],
                             haystack, a.n_predict, a.limit,
                             tiers=tiers, skipped=skipped, only_ids=only_ids)
        for r in rungs:
            if r.get("ran") and kv_bytes.get(name):
                r["cost"]["kv_bytes_per_token_measured"] = float(kv_bytes[name])
        for d in det:
            d["arm"] = name
        all_details.extend(det)
        # the label must say what actually ran: an MSCC arm stamped "llama.cpp" would
        # let a reader pool numbers the protocol says never to pool
        if a.backend == "mscc":
            fam = "none" if name in ("f16", "kv_exact") else "cpca"
            impl = ("mscc kv_exact (uncompressed handover)" if fam == "none"
                    else f"mscc {name} {os.path.basename(a.codebook)}")
        else:
            base = name.split("+", 1)[0]
            fam, impl = ("none" if name == "f16" else name), f"llama.cpp -ctk {base} -ctv {base}"
            if name.endswith("+cpca"):
                impl += f" LLAMA_KV_CODEBOOK={os.path.basename(a.codebook)}"
        return {"name": name, "method": {"family": fam, "impl": impl}, "rungs": rungs}

    # --- the reference arm decides which tiers this MODEL can be audited on.
    # A tier the reference fails is measuring the model, not the optimisation: at
    # 1.7B the whole-span aggregate tier scored 0/4 with no compression at all, and
    # counting it would have charged every method for a limit none of them caused.
    # Excluding it is recorded in the result, never silent, and a bigger model simply
    # keeps the tier.
    reference = build("f16")

    # Exclusion is decided per (tier, CONTEXT), not per tier. Capability is
    # context-dependent -- on the 1.7B the reference holds t2_link at 1k and 4k and
    # falls to 0.875 at 16k, and t4_distractor decays from 43/48 to 33/48 over the same
    # span. Excluding a tier everywhere because it fails at the longest rung would
    # throw away good short-context data; keeping it everywhere because it passes on
    # average would charge a method for a limit the model brought itself.
    excluded: dict[str, dict] = {}
    live_by_ctx: dict[int, set] = {}
    for r in reference["rungs"]:
        if not r.get("ran"):
            continue
        ctx = r["context"]
        live_by_ctx[ctx] = set()
        for t, v in r["quality"]["task_success"]["by_tier"].items():
            # Exclude only when we are CONFIDENT the reference is below the floor --
            # the upper bound of its interval, not the point estimate. A hard
            # threshold on a noisy rate is a knife-edge: the same model on the same
            # items scored 43/48 (0.8958, excluded) on CUDA and 44/48 (0.9167, kept)
            # on CPU, so ONE item decided whether a tier existed, and two machines
            # audited different tier sets and stopped being comparable. This rule is
            # deliberately conservative -- keep a tier unless it is clearly unusable --
            # because every score is a delta against this same reference anyway.
            if v["n"] and wilson(v["hits"], v["n"])[1] < a.reference_floor:
                e = excluded.setdefault(t, {"contexts": [], "reference": {},
                                            "reason": "reference arm below "
                                                      "reference_floor at this context"})
                e["contexts"].append(ctx)
                e["reference"][str(ctx)] = {"hits": v["hits"], "n": v["n"],
                                            "rate": round(v["hits"] / v["n"], 4)}
            else:
                live_by_ctx[ctx].add(t)
    if excluded:
        print("\nexcluded (reference arm could not do these at these lengths):",
              flush=True)
        for t, e in sorted(excluded.items()):
            det = "  ".join(f"{c}:{e['reference'][str(c)]['hits']}/"
                            f"{e['reference'][str(c)]['n']}" for c in e["contexts"])
            print(f"  {t:<16} at {e['contexts']}   {det}", flush=True)
        for c in sorted(live_by_ctx):
            print(f"  ctx {c:>6} audited on: "
                  f"{', '.join(sorted(live_by_ctx[c])) or '(none)'}", flush=True)
        print("  (reference rescored from its own item record, not re-run)\n",
              flush=True)
        reference = requality(reference, live_by_ctx)

    # every arm answers exactly the items the reference answered on its surviving
    # tiers -- matched sets, so the delta is a like-for-like comparison
    ref_ids = {d["id"] for d in all_details
               if d["arm"] == "f16" and d["tier"] in live_by_ctx.get(d["ctx"], set())}
    tested = [build(t, tiers=live_by_ctx, only_ids=ref_ids) for t in arms]

    doc = {
        "schema_version": "0.1.0", "run_id": uuid.uuid4().hex,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": {"name": "kv-audit", "version": "0.1.0",
                 "build_sha": sh(f"git -C {ROOT} rev-parse HEAD") or "0" * 40},
        "workload": {
            "id": "kvaudit-2026.1-draft", "sha256": wl_sha,
            "model": {"name": os.path.basename(a.model.rstrip("/")), "sha256": model_digest(a.model),
                      "n_layers": model_meta.get("n_layers", 0),
                      "n_kv_heads": model_meta.get("n_kv_heads", 0),
                      "head_dim": model_meta.get("head_dim", 0),
                      "kv_dtype": "f16"},
            "contexts": sorted(set(contexts) | set(skipped)),
            "profile": a.profile,
            "reference_floor": a.reference_floor,
            "excluded_tiers": excluded},
        "system": system_info(cap), "reference": reference, "arms": tested,
        "_capability": {k: v for k, v in cap.items() if k != "gpus"},
        "_kv_measurement": {k: {kk: vv for kk, vv in (d or {}).items()
                                if kk != "points"} for k, d in kv_detail.items()},
        "integrity": {},
    }
    doc["integrity"]["result_hash"] = V.canonical_hash(doc)

    json.dump(doc, open(a.out, "w"), indent=1)
    json.dump(all_details, open(a.out.replace(".json", ".items.json"), "w"), indent=1)

    errs = V.validate(doc)
    print("\n" + "=" * 72)
    if errs:
        print("SUBMISSION REJECTED by its own validator:")
        for e in errs:
            print("  -", e)
    else:
        print("submission is VALID")
    print("=" * 72)
    print(f"{'arm':<8} {'ctx':>6} {'B/token':>10} {'vs f16':>7} {'task':>9} "
          f"{'95% CI':>15} {'prefill':>9} {'tok/s':>7}")
    # take the reference cache size from the reference arm's own recorded cost, not
    # from the probe dict -- a backend that reports its size directly never fills that
    # dict, and the ratio column silently read 0.00x
    ref_b = next((r["cost"]["kv_bytes_per_token_measured"]
                  for r in reference["rungs"]
                  if r.get("ran") and r["cost"].get("kv_bytes_per_token_measured")),
                 kv_bytes.get("f16") or 0)
    for arm in [reference] + tested:
        for r in arm["rungs"]:
            if not r.get("ran"):
                print(f"{arm['name']:<8} {r['context']:>6}   skipped: {r['skip_reason']}")
                continue
            ts = r["quality"]["task_success"]["overall"]
            b = r["cost"]["kv_bytes_per_token_measured"]
            ci = ts.get("ci95", [0, 0])
            print(f"{arm['name']:<8} {r['context']:>6} {b:>10,.0f} "
                  f"{(ref_b / b if b else 0):>6.2f}x {ts['hits']:>4}/{ts['n']:<4} "
                  f"  [{ci[0]:.2f},{ci[1]:.2f}] "
                  f"{r['cost']['prefill_ms']:>8.0f}ms {r['cost']['decode_tok_per_s']:>7.1f}")
    # Say plainly whether the arms are distinguishable at this n. A table of point
    # estimates invites a ranking the sample size does not support.
    ref_n_total = sum(r["quality"]["task_success"]["overall"]["n"]
                      for r in reference["rungs"] if r.get("ran"))
    gap = detectable_gap(ref_n_total) if ref_n_total else 1.0
    print(f"\nstatistical power at n={ref_n_total}: this run can resolve a shortfall "
          f"of about {gap:.0%} or more.")
    if a.profile == "quick":
        print("  A quick run is for cost figures and for checking the pipeline works "
              "on this machine.\n  It cannot settle a quality question -- use "
              "--profile standard for that.")
    print("\nseparation vs the reference (pooled over rungs, 95% Wilson):")
    def pooled(arm):
        h = sum(r["quality"]["task_success"]["overall"]["hits"]
                for r in arm["rungs"] if r.get("ran"))
        n = sum(r["quality"]["task_success"]["overall"]["n"]
                for r in arm["rungs"] if r.get("ran"))
        return h, n
    rh, rn = pooled(reference)
    rlo, rhi = wilson(rh, rn)
    print(f"  {'f16 (ref)':<10} {rh:>3}/{rn:<3} [{rlo:.2f},{rhi:.2f}]")
    for arm in tested:
        h, n = pooled(arm)
        lo, hi = wilson(h, n)
        sep = "DISTINGUISHABLE from reference" if hi < rlo else \
              "not separated at this n"
        print(f"  {arm['name']:<10} {h:>3}/{n:<3} [{lo:.2f},{hi:.2f}]  {sep}")

    print(f"\nwrote {a.out}")

    # Ask, after the run, unless the answer is already known. --upload is a standing
    # yes; a non-interactive session is always a no.
    if a.never_share:
        try:
            import submit as S
            S.record_pref(S.PREF_NEVER)
            print("\nyou will not be asked to share again. "
                  "Undo with: python3 auditor/runner/submit.py --forget-consent")
        except Exception:                                             # noqa: BLE001
            pass
    want_upload = a.upload
    if not want_upload and not errs:
        try:
            import submit as S
            ans = S.offer_to_share(a.out)
            want_upload = ans in ("yes", "always")
        except Exception:                                             # noqa: BLE001
            want_upload = False

    if want_upload:
        if errs:
            print("\nNOT submitting: this result does not validate. A rejected row "
                  "helps nobody.", file=sys.stderr)
        else:
            try:
                import submit as S
                res = S.submit(a.out, route=a.route, repo=a.repo,
                               endpoint=a.endpoint, assume_yes=a.yes)
                print("\nsubmitted: " + json.dumps(res))
                if res.get("ok"):
                    print("\n" + S.compare_with_peers(a.out, repo=a.repo))
            except Exception as e:                                    # noqa: BLE001
                # a failed upload must never invalidate a good local run
                print(f"\nsubmission failed ({type(e).__name__}: {e}).\n"
                      f"The result is still at {a.out} and can be submitted later "
                      f"with:\n  python3 auditor/runner/submit.py {a.out}",
                      file=sys.stderr)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
