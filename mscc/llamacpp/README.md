# cpca for llama.cpp: the patch series

Applies to llama.cpp at the commit named in the first patch's header (the tree this
was developed on is the `b11cd988`-era master of late August 2026). Together the
patches add one thing: when `LLAMA_KV_CODEBOOK` names a codebook GGUF and the KV
cache is quantised, the cache stores keys and values in a data-fitted per-head basis
instead of the Hadamard basis llama.cpp uses since PR #21038. Everything else -- cache
layout, `set_rows`, flash attention, the quantisation kernels -- is untouched.

```bash
cd llama.cpp && git am /path/to/KVCache/mscc/llamacpp/*.patch
cmake --build build -j
LLAMA_KV_CODEBOOK=model.cpca.gguf ./build/bin/llama-server -m model.gguf -ctk q4_0 -ctv q4_0 -fa on
```

The codebook comes from `mscc/capture_gguf_kv.py` (any GGUF, no HF weights needed) or
from `kvfit --per-head --postrope --quant q4_0 --codes <head_dim>` plus
`mscc/export_cpca_gguf.py` when HF weights are available. The design, the algebra and
the measured results are in `mscc/LLAMACPP-CPCA-DESIGN.md`.

This is a prototype selected by environment variable. An upstream version would add a
context parameter and a `--kv-codebook` flag; the graph and cache changes are the same.
