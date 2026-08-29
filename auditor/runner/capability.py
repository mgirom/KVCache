#!/usr/bin/env python3
"""What can this machine actually run, and how do we take as little of it as possible?

TWO PRINCIPLES, both of which the first version of this runner violated.

**Adapt to the machine, do not dictate to it.** The first runner passed `-ngl 99` --
"put every layer on the GPU, and fail if you cannot". That is fine on the box it was
written on and wrong everywhere else: a laptop with 6 GB of shared memory, a machine
whose GPU is busy, or a CPU-only server all get a crash instead of a result. llama.cpp
already fits its parameters to free device memory on its own and leaves a reserve; the
correct thing is to let it, and to record what it chose.

**Adapt WHICH rungs run, never HOW they are scored.** This is the line that keeps a
laptop's result comparable to a datacentre's. The context ladder is per-rung and rungs
are independently comparable, so a machine that can only reach 1,024 tokens produces a
valid submission covering the 1,024 rung. What must never adapt is the scoring, the
answer normalisation, or the items inside a rung that does run -- those are the
workload, and a workload that varies by machine measures nothing.

So: probe first, attempt what fits, and record every rung that did not run with the
reason it did not. `insufficient_vram` in a result is information, not a failure.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess


def _sh(cmd, timeout=15):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def host_memory_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    if platform.system() == "Darwin":
        out = _sh(["sysctl", "-n", "hw.memsize"])
        return int(out) if out.isdigit() else 0
    return 0


def free_host_bytes() -> int:
    """Available (not merely free) host memory, which is what we can actually use."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return int(host_memory_bytes() * 0.6)


def gpu_info() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _sh(["nvidia-smi",
               "--query-gpu=name,memory.total,memory.free,driver_version",
               "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.splitlines():
        f = [p.strip() for p in line.split(",")]
        if len(f) < 4:
            continue
        try:
            gpus.append({"name": f[0], "total_bytes": int(float(f[1])) * 1024 * 1024,
                         "free_bytes": int(float(f[2])) * 1024 * 1024,
                         "driver": f[3]})
        except ValueError:
            continue
    return gpus


def detect(binary: str) -> dict:
    """What this machine is, and which backend the given binary will use."""
    gpus = gpu_info()
    is_cuda = bool(gpus) and "build-cuda" in binary
    backend = "cuda" if is_cuda else ("metal" if platform.system() == "Darwin"
                                      else "cpu")
    cap = {
        "backend": backend,
        "os": {"Linux": "linux", "Windows": "windows",
               "Darwin": "macos"}.get(platform.system(), "linux"),
        "arch": "arm64" if platform.machine() in ("arm64", "aarch64") else "x86_64",
        "cpu_cores": os.cpu_count() or 1,
        "host_total_bytes": host_memory_bytes(),
        "host_free_bytes": free_host_bytes(),
        "gpus": gpus,
    }
    if backend == "cuda" and gpus:
        cap["device_free_bytes"] = gpus[0]["free_bytes"]
        cap["device_total_bytes"] = gpus[0]["total_bytes"]
    else:
        # unified or host memory: the cache lands in RAM
        cap["device_free_bytes"] = cap["host_free_bytes"]
        cap["device_total_bytes"] = cap["host_total_bytes"]
    return cap


def feasible_contexts(cap: dict, model_bytes: int, kv_bytes_per_token: float,
                      ladder=(1024, 4096, 16384), reserve_bytes: int = 1 << 30,
                      headroom_tokens: int = 512, overhead_factor: float = 1.25):
    """Which rungs of the ladder fit, and why the others do not.

    `reserve_bytes` is the point of this function. A benchmark that consumes every byte
    available is antisocial: it evicts whatever else the machine was doing, and on a
    laptop it is the difference between a background task and an unusable computer. One
    gigabyte is left alone by default, and the caller can raise it.

    Returns [(context, ok, reason_or_None), ...] -- never a bare list of what fits, so
    the skipped rungs can be recorded with their reason rather than vanishing.
    """
    budget = max(0, cap.get("device_free_bytes", 0) - reserve_bytes)
    out = []
    for ctx in ladder:
        need = model_bytes + (ctx + headroom_tokens) * kv_bytes_per_token
        need *= overhead_factor          # compute buffers, fragmentation, the runtime
        if need <= budget:
            out.append((ctx, True, None))
        else:
            out.append((ctx, False, "insufficient_vram" if cap["backend"] == "cuda"
                        else "insufficient_ram"))
    return out


def describe(cap: dict) -> str:
    gb = lambda b: f"{b / (1 << 30):.1f} GB"                       # noqa: E731
    lines = [f"backend   {cap['backend']}  ({cap['os']}/{cap['arch']}, "
             f"{cap['cpu_cores']} cores)"]
    if cap.get("gpus"):
        g = cap["gpus"][0]
        lines.append(f"gpu       {g['name']}  {gb(g['free_bytes'])} free of "
                     f"{gb(g['total_bytes'])}")
    lines.append(f"host ram  {gb(cap['host_free_bytes'])} available of "
                 f"{gb(cap['host_total_bytes'])}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    b = sys.argv[1] if len(sys.argv) > 1 else "llama.cpp/build-cuda/bin/llama-server"
    c = detect(b)
    print(describe(c))
    mb = os.path.getsize(sys.argv[2]) if len(sys.argv) > 2 else 1_282_439_040
    print("\nfeasible ladder for a 115,712 B/token cache:")
    for ctx, ok, why in feasible_contexts(c, mb, 115712):
        print(f"  {ctx:>6}  {'yes' if ok else 'no  (' + str(why) + ')'}")
