#!/bin/bash
# v0517_fleet_validate.sh — flip-gating validation for the sglang v0.5.17 rebase.
# Per preset: launch @256K on the STAGED v0.5.17 stack (env sglang-v0517,
# /data/sglang-rebase-v0517) -> full-N quality (MMLU 30 / HE 25 / LAB 8 /
# needle 1K..250K) -> capability probe (thinking/tool/vision/video/audio as
# applicable) -> stop. Receipts: benchmarks/quality/<preset>-v0517.json +
# cap-<preset>-v0517.json, comparable 1:1 against the *-v0516.json set.
#
# Order is risk-first for THIS flip: coder-30b-eval FIRST — it is CT-format
# MoE at TP=2, the exact case patch 030 protects, and v0.5.17's derive-shard
# rewrite turned the unpatched failure into a SILENT half-load (boots green,
# garbage quality) — MMLU/HE on this preset are the corruption tripwire.
# gemma4 second (017 regen: gelu beside upstream's new situ branch); qwen36
# third (kernel patches + DeltaNet); devstral (057/041/058 chain);
# nemotron3-omni exercises the 053 EVS re-anchor (video).
#
# Usage:
#   ./scripts/eval/v0517_fleet_validate.sh                # full fleet
#   PRESETS="gemma4-12b qwen36" ./scripts/eval/v0517_fleet_validate.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$HOME/AI/models}"

# Staged v0.5.17 stack (launch.sh honors these).
export ENV_NAME="${ENV_NAME:-sglang-v0517}"
export SGLANG_DIR="${SGLANG_DIR:-/data/sglang-rebase-v0517}"
export PATH="$HOME/miniforge3/envs/$ENV_NAME/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
export CUDA_PATH="${CUDA_PATH:-/opt/cuda}"
source "$REPO/scripts/common.sh" 2>/dev/null || true
cd "$REPO" || exit 1

# preset | think? (mc-budget selector)
FLEET=(
  "coder-30b-eval|no"
  "gemma4|yes"
  "qwen36|yes"
  "devstral|no"
  "qwen35-moe|yes"
  "gemma4-12b|yes"
  "gemma4-31b|yes"
  "gemma4-21b-reap|yes"
  "qwen36-ream|yes"
  "qwen36-dense|yes"
  "qwen3-ream|no"
  "nemotron3-omni|yes"
)

MMLU_N="${MMLU_N:-30}"
HE_N="${HE_N:-25}"
LAB_N="${LAB_N:-8}"
NEEDLE_LENGTHS="${NEEDLE_LENGTHS:-1024,16384,65536,131072,250000}"
MC_BUDGET_THINK="${MC_BUDGET_THINK:-2560}"
WORKERS="${WORKERS:-4}"
# First boot per preset JIT-compiles triton/tvm_ffi kernels on the fresh env.
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1800}"
PORT=23334
LOG_ROOT="/tmp/v0517-eval-logs"
mkdir -p "$LOG_ROOT"

log() { echo "[v0517-val $(date +%H:%M:%S)] $*"; }
stop_server() { pkill -KILL -f "sglang.launch_server" 2>/dev/null || true; sleep 6; }

launch_server() {  # $1 preset, $2 logdir
  log "launching $1 @256K (env $ENV_NAME)"
  MAX_RUNNING="${MAX_RUNNING:-6}" nohup setsid bash "$REPO/scripts/launch.sh" "$1" \
    --context-length 262144 > "$2/server.log" 2>&1 < /dev/null &
  disown
}

wait_ready() {
  local end=$(( $(date +%s) + SERVER_TIMEOUT ))
  while [ "$(date +%s)" -lt "$end" ]; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:$PORT/health 2>/dev/null || echo 000)" = "200" ] \
      && { log "  server ready"; return 0; }
    sleep 12
  done
  log "  ERROR: server timeout"; tail -30 "$1/server.log"; return 1
}

PRESETS_RUN="${PRESETS:-$(printf '%s\n' "${FLEET[@]}" | cut -d'|' -f1 | tr '\n' ' ')}"
log "fleet: $PRESETS_RUN"

for ENTRY in "${FLEET[@]}"; do
  IFS='|' read -r PRESET THINK <<< "$ENTRY"
  case " $PRESETS_RUN " in *" $PRESET "*) ;; *) continue ;; esac
  LOG="$LOG_ROOT/$PRESET"; mkdir -p "$LOG"
  [ "$THINK" = "yes" ] && MCB="$MC_BUDGET_THINK" || MCB=1024
  QJSON="benchmarks/quality/$PRESET-v0517.json"
  CJSON="benchmarks/quality/cap-$PRESET-v0517.json"
  if [ -s "$QJSON" ] && [ -s "$CJSON" ]; then log "=== $PRESET already done, skip ==="; continue; fi
  log "=== $PRESET (think=$THINK mc-budget=$MCB) ==="

  stop_server
  launch_server "$PRESET" "$LOG"
  wait_ready "$LOG" || { stop_server; log "  SKIP $PRESET (boot failed)"; continue; }

  log "  quality eval (mmlu=$MMLU_N he=$HE_N lab=$LAB_N needle=$NEEDLE_LENGTHS)"
  python "$REPO/scripts/eval/eval_and_chart.py" --run --port $PORT --tag "$PRESET-v0517" \
    --mmlu-samples "$MMLU_N" --humaneval-samples "$HE_N" --labbench-samples "$LAB_N" \
    --needle-lengths "$NEEDLE_LENGTHS" --mc-budget "$MCB" --workers "$WORKERS" \
    > "$LOG/quality.log" 2>&1
  log "    quality rc=$? -> $QJSON"

  log "  capability probe -> $CJSON"
  python "$REPO/scripts/eval/validate_capabilities.py" --port $PORT \
    --tag "$PRESET-v0517" --save "$CJSON" > "$LOG/caps.log" 2>&1
  log "    caps rc=$?"

  stop_server
  log "=== $PRESET done ==="
done

log "=== V0517 FLEET VALIDATION COMPLETE ==="
