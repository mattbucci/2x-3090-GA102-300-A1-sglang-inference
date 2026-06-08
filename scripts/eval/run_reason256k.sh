#!/bin/bash
# 256K REASONING-quality re-measure (RULER-style: multikey / vartrack / aggregate)
# at TRUE actual-token lengths up to ~262K. The prior quality256k JSONs topped out
# at ~137K ACTUAL tokens because the filler under-produced (3.8 vs the real ~6.9
# char/tok on the Qwen tokenizer); probe_256k_quality.py is now calibrated
# (approx==actual), so --lengths 262144 genuinely probes ~256K. Single-user
# (MAX_RUNNING=1, matching the qwen36 cuda-graph bs=1 capture), one model at a
# time (Rule 1: no concurrent serving). Mirrors run_v0512_fleet_eval.sh's launch
# primitives + env gotchas (PATH-pin python, CUDA_HOME, cd REPO, ENV_NAME/SGLANG_DIR).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
export ENV_NAME="${ENV_NAME:-sglang-v0512}"
export SGLANG_DIR="${SGLANG_DIR:-/data/sglang-rebase-v0512}"
export PATH="/home/letsrtfm/miniforge3/envs/$ENV_NAME/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/opt/cuda}"; export CUDA_PATH="${CUDA_PATH:-/opt/cuda}"
source "$REPO/scripts/common.sh" 2>/dev/null || true
activate_conda 2>/dev/null || true
cd "$REPO" || exit 1

# The four thinking MoE families whose KV pool genuinely fits 256K (657K-2.4M tok).
PRESETS="${PRESETS:-qwen36 qwen36-dense qwen36-ream qwen35-moe}"
LENGTHS="${LENGTHS:-1024,32768,65536,131072,200000,262144}"
PORT=23334
SERVER_TIMEOUT="${SERVER_TIMEOUT:-720}"
OUT=/tmp/reason256k; mkdir -p "$OUT"; rm -f "$OUT/done"; : > "$OUT/result.txt"
LOG_ROOT="/tmp/reason256k-logs"; mkdir -p "$LOG_ROOT"
log(){ echo "[reason256k $(date +%H:%M:%S)] $*" | tee -a "$OUT/result.txt"; }

stop_server(){ pkill -KILL -f "sglang.launch_server" 2>/dev/null || true; sleep 6; }
launch_server(){  # $1 preset, $2 logdir
  log "launching $1 @262144 (MAX_RUNNING=1)"
  MAX_RUNNING=1 nohup setsid bash "$REPO/scripts/launch.sh" "$1" \
    --context-length 262144 > "$2/server.log" 2>&1 < /dev/null &
  disown
}
wait_ready(){
  local end=$(( $(date +%s) + SERVER_TIMEOUT ))
  while [ "$(date +%s)" -lt "$end" ]; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:$PORT/health 2>/dev/null || echo 000)" = "200" ] \
      && { log "  server ready"; return 0; }
    sleep 12
  done
  log "  ERROR: server timeout"; tail -30 "$1/server.log"; return 1
}

log "presets: $PRESETS  lengths: $LENGTHS"
for P in $PRESETS; do
  LOG="$LOG_ROOT/$P"; mkdir -p "$LOG"
  log "=== $P ==="
  stop_server
  launch_server "$P" "$LOG"
  wait_ready "$LOG" || { stop_server; log "  SKIP $P (boot failed)"; continue; }
  log "  256K reasoning probe @ $LENGTHS"
  python "$REPO/scripts/eval/probe_256k_quality.py" --port $PORT --tag "$P" \
    --lengths "$LENGTHS" --out "benchmarks/quality/quality256k-$P-v0512.json" >> "$OUT/result.txt" 2>&1
  log "  probe rc=$? -> benchmarks/quality/quality256k-$P-v0512.json"
  stop_server
  log "=== $P done ==="
done
log "ALL DONE"
echo "done" > "$OUT/done"
