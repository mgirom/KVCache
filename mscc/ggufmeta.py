"""GGUF metadata without building tensors: geometry and identity for any GGUF, including
ternary quant types gguf-py cannot reshape."""
import struct

def gguf_metadata(path):
    """Key/value metadata only, without building tensors -- the ternary GGUFs use quant
    types gguf-py's tensor reader cannot reshape, and geometry needs no tensors."""
    import struct
    f = open(path, "rb")
    rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]   # noqa: E731
    assert f.read(4) == b"GGUF", "not a GGUF file"
    ver = rd("I"); assert ver == 3, ver
    n_tensors, n_kv = rd("Q"), rd("Q")
    T = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    def rstr():
        n = rd("Q"); return f.read(n).decode("utf-8", "replace")
    def rval(t):
        if t == 8: return rstr()
        if t == 9:
            et, n = rd("I"), rd("Q")
            if et == 8: return [rstr() for _ in range(n)]
            sz = struct.calcsize("<" + T[et]); raw = f.read(sz * n)
            return list(struct.unpack("<" + T[et] * n, raw)) if n < 4096 else None   # skip huge arrays
        return rd(T[t])
    kv = {}
    for _ in range(n_kv):
        k = rstr(); t = rd("I"); kv[k] = rval(t)
    return kv


def model_geometry(path):
    kv = gguf_metadata(path)
    arch = kv["general.architecture"]
    g = lambda k, default=None: int(kv[f"{arch}.{k}"]) if f"{arch}.{k}" in kv else default   # noqa: E731
    n_head, n_layer = g("attention.head_count"), g("block_count")
    n_head_kv = g("attention.head_count_kv", n_head)
    hd = g("attention.key_length") or g("embedding_length") // n_head
    # hybrid models (Qwen3.5, Qwen3-Next): only some layers carry a KV cache. llama.cpp
    # reads an explicit recurrent-layer array if present, else attention sits on every
    # `full_attention_interval`-th layer, counting from 1 (src/models/qwen35.cpp).
    recr = kv.get(f"{arch}.attention.recurrent_layers")
    interval = g("full_attention_interval")
    if recr is not None:
        kv_layers = [i for i, r in enumerate(recr[:n_layer]) if not r]
    elif interval:
        kv_layers = [i for i in range(n_layer) if (i + 1) % interval == 0]
    else:
        kv_layers = list(range(n_layer))
    return dict(arch=arch, n_layer=n_layer, n_head=n_head, n_head_kv=n_head_kv, head_dim=hd,
                name=kv.get("general.name", ""), kv_layers=kv_layers)
