#!/bin/bash
# Reproduce the llama.cpp live-path results from nothing.
#   ./reproduce-llamacpp.sh build                 -> ./llama.cpp/build/bin/llama-server with the cpca patches
#   ./reproduce-llamacpp.sh audit <id> MODEL.gguf  -> quick-profile audit of q4_0 vs q4_0+cpca on that model
# Requires: git, cmake, a C++ toolchain, python3 with requests; CUDA optional (GGML_CUDA=ON).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
COMMIT="$(python3 -c "import json; print(json.load(open('$HERE/registry/models.json'))['llama_cpp_commit'])")"
case "${1:-}" in
  build)
    [ -d "$HERE/llama.cpp" ] || git clone --quiet https://github.com/ggml-org/llama.cpp.git "$HERE/llama.cpp"
    cd "$HERE/llama.cpp" && git fetch --quiet && git checkout --quiet "$COMMIT"
    git config user.email reproduce@example.invalid && git config user.name reproduce
    git am --3way --quiet "$HERE"/mscc/llamacpp/*.patch || { echo "patches did not apply cleanly at $COMMIT"; exit 1; }
    cmake -B build ${GGML_CUDA:+-DGGML_CUDA=ON} -DLLAMA_CURL=OFF >/dev/null
    cmake --build build --config Release --target llama-server -j "$(nproc 2>/dev/null || echo 4)"
    bin="$(find build -name 'llama-server' -type f | head -1)"
    "$bin" --help 2>&1 | grep -q -- "--kv-codebook" && echo "built: $HERE/llama.cpp/$bin (set KVCACHE_LLAMA_SERVER to it)"
    ;;
  audit)
    id="${2:?model id}"; model="${3:?model.gguf}"
    export KVCACHE_LLAMA_SERVER="${KVCACHE_LLAMA_SERVER:-$(find "$HERE/llama.cpp/build" -name 'llama-server' -type f | head -1)}"
    python3 "$HERE/tools/kvcache.py" audit "$id" --model "$model" --profile "${PROFILE:-quick}"
    ;;
  *) echo "usage: $0 build | audit <id> MODEL.gguf"; exit 2;;
esac
