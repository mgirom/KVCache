# codebooks

Fitted per-head rotations (cpca) for llama.cpp's quantised KV cache, one GGUF per model,
f16. Each file is bound to the model it was fitted on; llama.cpp refuses a mismatch.
The registry on `main` (`registry/models.json`) lists which model file and sha256 each
codebook belongs to and what was measured with it. Fetch with `tools/kvcache.py pull <id>`.
