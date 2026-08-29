#!/usr/bin/env bash
# KV-Audit reproduction driver. Every published number in FIRST-RESULT.md comes from
# here; no argument lists what is available.
#
#   ./auditor/reproduce.sh workload    fetch the haystack, generate + hash the tasks
#   ./auditor/reproduce.sh quick MODEL  a two-minute run: cost figures, pipeline check
#   ./auditor/reproduce.sh selftest    schema, scoring and validator tests (no GPU)
#   ./auditor/reproduce.sh audit MODEL run one model across the arms
#   ./auditor/reproduce.sh all         everything, in order, on the models on disk
#
# Every GPU step takes /tmp/ternary-gpu.lock, so two runs launched at once queue
# rather than OOM each other. Runtimes are from an RTX A2000 12GB.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
PY=${PY:-/usr/bin/python3}
BIN=${KVAUDIT_BIN:-llama.cpp/build-cuda/bin/llama-server}
ARMS=${KVAUDIT_ARMS:-q8_0,q4_0}
CTX=${KVAUDIT_CTX:-1024,4096}
mkdir -p results/auditor logs

banner() { echo; echo "=== $* === $(date -u +%H:%M:%S)"; }

# Said here as well as in the README, because this is where someone starts a run that
# will hold their hardware at full load for a while.
disclaimer() {
  cat >&2 <<'EOD'
--------------------------------------------------------------------------------
NO WARRANTY. This drives your GPU/CPU at sustained full load for minutes to hours,
which is a thermal and power stress test as a side effect. You run it at your own
risk; the authors accept no liability for hardware damage, instability or data loss.
Results are measurements, not advice. Nothing is uploaded unless you pass --upload.
Details: README.md and auditor/PRIVACY.md
--------------------------------------------------------------------------------
EOD
}

g_workload() {
  banner "workload"
  $PY auditor/workload/fetch_haystack.py || return 1
  $PY auditor/workload/gen_tasks.py --per-cell 16
  echo "NOTE: per-cell 16 is not cosmetic. At per-cell 4 the headline comparison"
  echo "      did not separate at 95% -- q4_0 [0.728,0.928] against a reference"
  echo "      [0.926,1.000]. The workload is sized to the claims it has to support."
}

g_selftest() {
  banner "self-tests (no GPU)"
  $PY auditor/validate.py --selftest || return 1
  $PY - <<'EOF'
import sys; sys.path.insert(0, "auditor/runner")
from assemble import check
cases = [
    ({"answer": "47-19-83", "tier": "t1"}, "The code is 47-19-83.", True),
    ({"answer": "7", "tier": "t3"}, "I count 17 lines.", False),
    ({"answer": "6,412", "tier": "t2"}, "6412 kilograms", True),
    ({"answer": "25-38-90", "tier": "t1"}, "the code is 25-3", False),
]
bad = [c for c in cases if check(c[0], c[1])["hit"] != c[2]]
print(f"  scoring: {len(cases)-len(bad)}/{len(cases)} passed")
sys.exit(1 if bad else 0)
EOF
}

g_audit() {
  local model="$1"
  local name; name=$(basename "$model" .gguf | tr 'A-Z' 'a-z')
  banner "audit $name"
  [ -f "$model" ] || { echo "missing model: $model"; return 1; }
  $PY auditor/runner/run.py --binary "$BIN" --model "$model" \
      --arms "$ARMS" --contexts "$CTX" \
      -o "results/auditor/${name}.json" 2>&1 | tee "logs/kvaudit_${name}.log"
  return "${PIPESTATUS[0]}"
}

disclaimer
case "${1:-}" in
  workload) g_workload ;;
  selftest) g_selftest ;;
  audit)    g_audit "${2:?usage: reproduce.sh audit <model.gguf>}" ;;
  quick)
    m="${2:?usage: reproduce.sh quick <model.gguf>}"
    n=$(basename "$m" .gguf | tr 'A-Z' 'a-z')
    $PY auditor/runner/run.py --profile quick --binary "$BIN" --model "$m" \
        --arms "$ARMS" -o "results/auditor/quick-${n}.json"
    ;;
  all)
    g_workload && g_selftest || exit 1
    for m in models/qwen3-1.7b-fp/q3-1.7b-Q4_K_M.gguf \
             models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf \
             models/bonsai-8b/Ternary-Bonsai-8B-Q2_0_g64.gguf; do
      [ -f "$m" ] && g_audit "$m"
    done
    ;;
  *) sed -n '2,12p' "$0" ;;
esac
