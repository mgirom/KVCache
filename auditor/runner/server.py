#!/usr/bin/env python3
"""llama-server lifecycle, and the measurement of what a KV cache actually costs.

WHY THE KV SIZE IS MEASURED BY SLOPE
------------------------------------
The obvious approach is to parse the server's startup log. On this build it prints
`KV buffer size = 0.00 MiB` -- the allocation is lazy, so the log is not a measurement
of anything. The next-most-obvious approach is to compute it from the config:
n_layers x 2 x n_kv_heads x head_dim x bits. That is a *declared* figure and it is
wrong in a specific, self-serving direction: it omits the per-block scales that every
quantised format carries. q4_0 stores 32 values as one f16 scale plus 32 nibbles --
4.5 bits per element, not 4 -- so the naive figure understates the real cost by 12%.

So it is measured by slope. Load the same model at two context lengths, read process
VRAM at each, and divide:

    bytes_per_token = (vram(C2) - vram(C1)) / (C2 - C1)

The subtraction cancels the model weights, the compute buffers, and the CUDA context,
all of which are constant in the context length. What is left is the cache and only
the cache, scales included, as actually allocated by the backend rather than as
described in a paper.

PORTABILITY. The slope is taken over whichever memory the cache actually lands in:
device memory on CUDA, resident host memory everywhere else. That covers CPU and
unified-memory backends without a special case, because on those the cache IS host
memory. Whichever of the two moves with context length is the one reported, and the
result records which was used, so nobody has to guess later.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


class ServerError(RuntimeError):
    pass


def free_port(preferred: int) -> int:
    """A port nothing else is listening on, starting from `preferred`.

    A fixed default port is a landmine on a machine that is not the developer's. Worse
    than a bind failure: if something is ALREADY serving there, a health poll succeeds
    and the run silently measures whatever that is.
    """
    import socket
    for port in range(preferred, preferred + 64):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
            sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sk.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise ServerError(f"no free port in {preferred}..{preferred + 63}")


def process_rss_bytes(pid: int) -> int | None:
    """Resident host memory for one process. The portable half of the pair."""
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:                                    # macOS / BSD
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out) * 1024 if out.isdigit() else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def gpu_process_vram_bytes(pid: int) -> int | None:
    """VRAM attributed to one process, in bytes. None if it cannot be determined."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid:
            try:
                return int(float(parts[1]) * 1024 * 1024)
            except ValueError:
                return None
    return None


class LlamaServer:
    """Start llama-server, wait for health, talk to it, stop it cleanly."""

    #: A rung's `context` is the DOCUMENT length. The server is given room for the
    #: question and the generated answer on top, otherwise a document sized exactly to
    #: the context leaves no space to answer in and every request 400s.
    HEADROOM = 512

    def __init__(self, binary, model, ctx, port=8099, ngl=None,
                 cache_type_k="f16", cache_type_v="f16", extra=(), log_dir=None,
                 env=None):
        self.binary, self.model, self.ctx, self.port = binary, model, ctx, port
        # extra environment for the server process (the cpca prototype selects its
        # codebook by LLAMA_KV_CODEBOOK); recorded so a result says what ran
        self.env = dict(env or {})
        self.ngl, self.ctk, self.ctv = ngl, cache_type_k, cache_type_v
        self.extra = list(extra)
        self.log_dir = log_dir
        self.proc = None
        self.log_path = None
        self.base = f"http://127.0.0.1:{port}"

    def cmd(self):
        c = [self.binary, "-m", self.model, "-c", str(self.ctx + self.HEADROOM),
             "-ctk", self.ctk, "-ctv", self.ctv,
             "--host", "127.0.0.1", "--port", str(self.port),
             "--no-webui", "-np", "1"]
        # -ngl is deliberately NOT passed by default. Forcing "-ngl 99" means "put
        # every layer on the GPU and fail if you cannot", which is fine on the machine
        # this was written on and a crash on a laptop, on a busy GPU, or on a CPU-only
        # box. Left alone, llama.cpp fits its parameters to free device memory and
        # keeps a reserve. Passing --ngl overrides that, and is recorded when it does.
        if self.ngl is not None:
            c += ["-ngl", str(self.ngl)]
        return c + list(self.extra)

    def start(self, timeout=180):
        # Claim a port nothing is on. Without this, a stale or unrelated server on the
        # default port answers /health, the run proceeds, and every number it produces
        # belongs to a different process. That happened here: a CPU run was silently
        # measured against a GPU server holding another model, and only the prefill
        # timings gave it away.
        self.port = free_port(self.port)
        self.base = f"http://127.0.0.1:{self.port}"
        tag = f"{self.ctk}_{self.ctv}{'+cpca' if self.env.get('LLAMA_KV_CODEBOOK') else ''}_c{self.ctx}"
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_path = os.path.join(self.log_dir, f"server_{tag}.log")
            fh = open(self.log_path, "w")
        else:
            fh = subprocess.DEVNULL
        penv = dict(os.environ); penv.update(self.env)
        self.proc = subprocess.Popen(self.cmd(), stdout=fh, stderr=subprocess.STDOUT, env=penv)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise ServerError(f"server exited rc={self.proc.returncode}; "
                                  f"see {self.log_path}")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=2) as r:
                    if json.load(r).get("status") == "ok":
                        self._assert_is_ours()
                        return self
            except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
                pass
            time.sleep(1.0)
        self.stop()
        raise ServerError(f"server did not become healthy in {timeout}s")

    def _assert_is_ours(self):
        """Positive identity check: the server answering must be serving OUR model.

        A health check proves something is alive, not that it is the thing we started.
        This compares the model path the server reports against the one we launched,
        and refuses rather than measuring a stranger.
        """
        try:
            with urllib.request.urlopen(self.base + "/props", timeout=5) as r:
                got = json.load(r).get("model_path", "")
        except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
            return                              # no /props: nothing to check against
        if got and os.path.basename(got) != os.path.basename(self.model):
            raise ServerError(
                f"port {self.port} is served by a DIFFERENT model "
                f"({os.path.basename(got)!r}, we launched "
                f"{os.path.basename(self.model)!r}). Refusing to measure it.")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=25)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.proc = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()

    # ------------------------------------------------------------------ endpoints
    def _post(self, path, body, timeout=600):
        req = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            # The server explains itself in the body. Swallowing it turns a one-line
            # "prompt is too long" into a stack trace that says nothing.
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:400]
            except Exception:                                          # noqa: BLE001
                pass
            raise ServerError(f"{path} -> HTTP {e.code}: {detail}") from e

    def tokenize(self, text):
        return self._post("/tokenize", {"content": text})["tokens"]

    def n_tokens(self, text):
        return len(self.tokenize(text))

    #: Stop at the first blank line or a new question. A reasoning-tuned model answers
    #: and then narrates -- Bonsai-8B replies "47-19-83\n\nOkay, so the user is asking
    #: for..." and burns the whole budget. The first-span scorer already handles that
    #: correctly, but generating the narration costs wall-clock on every item for
    #: nothing. Note the stops cannot fire at position 0 of a well-formed answer.
    STOPS = ["\n\n", "\nQuestion:", "\nQ:"]

    def complete(self, prompt, n_predict=24, timeout=600):
        """Greedy, and deliberately WITHOUT prompt caching.

        cache_prompt defaults to true in llama-server. Leaving it on would let item N
        reuse item N-1's shared prefix, so prefill timings would measure the cache
        hit rather than the work, and they would improve monotonically through the
        run. Every timing in this benchmark is a cold prefill.
        """
        return self._post("/completion", {
            "prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
            "top_k": 1, "seed": 0, "cache_prompt": False, "stream": False,
            "stop": list(self.STOPS),
        }, timeout=timeout)

    def model_meta(self) -> dict:
        """Read the model's real shape out of the loader log.

        The alternative was hardcoding 28 layers / 8 kv-heads / 128 head_dim, which is
        Qwen3-1.7B's shape and silently wrong for every other model. A KV benchmark
        that mislabels the cache geometry mislabels everything downstream of it.
        """
        meta = {}
        if not self.log_path or not os.path.exists(self.log_path):
            return meta
        pat = re.compile(r"([a-z0-9_]+)\.([a-z_.]+)\s+(?:u32|str|f32)\s+=\s+(\S+)")
        arch = None
        for line in open(self.log_path, errors="replace"):
            if "general.architecture" in line:
                m = re.search(r"general\.architecture\s+str\s+=\s+(\S+)", line)
                if m:
                    arch = m.group(1)
            if "llama_model_loader" not in line:
                continue
            m = pat.search(line)
            if m and arch and m.group(1) == arch:
                meta[m.group(2)] = m.group(3)
        if not arch:
            return meta
        def _i(k):
            try:
                return int(meta.get(k, 0))
            except ValueError:
                return 0
        n_head = _i("attention.head_count")
        emb = _i("embedding_length")
        head_dim = _i("attention.key_length") or (emb // n_head if n_head else 0)
        return {"arch": arch, "n_layers": _i("block_count"),
                "n_kv_heads": _i("attention.head_count_kv"),
                "head_dim": head_dim, "n_heads": n_head,
                "embedding_length": emb,
                "train_context": _i("context_length")}

    def memory_bytes(self) -> dict:
        """Both memories, so the caller can use whichever the cache lives in."""
        if not self.proc:
            return {"vram": None, "rss": None}
        return {"vram": gpu_process_vram_bytes(self.proc.pid),
                "rss": process_rss_bytes(self.proc.pid)}

    def vram_bytes(self):
        """Peak-tracking helper: device memory if there is any, else host."""
        m = self.memory_bytes()
        return m["vram"] if m["vram"] else m["rss"]


def probe_model_meta(binary, model, port=8099, log_dir=None):
    """One short high-verbosity load, purely to read the model's shape.

    The architecture keys only appear in the loader log, and only above the default
    verbosity. Raising verbosity for the benchmark servers themselves would bloat every
    log and put debug work on the timing path, so this is a separate ten-second probe.
    """
    import tempfile
    d = log_dir or tempfile.mkdtemp()
    srv = LlamaServer(binary, model, 256, port=port, log_dir=d, extra=["-lv", "4"])
    try:
        srv.start()
        return srv.model_meta()
    except ServerError:
        return {}
    finally:
        srv.stop()
        time.sleep(1.5)


def measure_kv_bytes_per_token(binary, model, ctk, ctv, small=2048, large=16384,
                               port=8099, settle=4.0, log_dir=None, env=None):
    """Slope method, over whichever memory the cache lands in.

    Returns (bytes_per_token, detail) or (None, detail). `detail["measured_in"]` says
    which memory was used, because on a CPU or unified-memory machine the answer is
    host RSS and a reader should not have to infer that.
    """
    pts = {"vram": {}, "rss": {}}
    for ctx in (small, large):
        srv = LlamaServer(binary, model, ctx, port=port, cache_type_k=ctk,
                          cache_type_v=ctv, log_dir=log_dir, env=env)
        try:
            srv.start()
            # the cache is allocated lazily on this build, so touch it before reading
            srv.complete("The", n_predict=1)
            time.sleep(settle)
            m = srv.memory_bytes()
            for k in ("vram", "rss"):
                if m[k] is not None:
                    pts[k][ctx] = m[k]
        finally:
            srv.stop()
            time.sleep(2.0)

    best = None
    for kind in ("vram", "rss"):
        d = pts[kind]
        if small in d and large in d:
            slope = (d[large] - d[small]) / (large - small)
            # a slope near zero means the cache is not in this memory; require the
            # measured growth to be at least a plausible fraction of a byte per token
            if slope > 8 and (best is None or slope > best[1]):
                best = (kind, slope)
    if best is None:
        return None, {"reason": "no memory grew with context length; cannot measure "
                                "the cache on this backend", "points": pts}
    return best[1], {"measured_in": best[0], "points": pts[best[0]],
                     "small": small, "large": large}
