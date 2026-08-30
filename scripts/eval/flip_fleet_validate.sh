#!/bin/bash
# flip_fleet_validate.sh — flip-gating validation for an SGLang stack rebase.
# Per preset: launch @256K on the STAGED stack (env sglang-$FLIP_TAG,
# /data/sglang-rebase-$FLIP_TAG) -> full-N quality (MMLU 30 / HE 25 / LAB 8 /
# needle 1K..250K) -> capability probe (thinking/tool/vision/video/audio as
# applicable) -> stop. Receipts: benchmarks/quality/<preset>-$FLIP_TAG.json +
# cap-<preset>-$FLIP_TAG.json, compared 1:1 against the previous stack's set by
# scripts/eval/compare_flip_receipts.py --old <prev> --new $FLIP_TAG.
#
# Generalized from the per-version validators at the v0.5.18 flip (2026-08-29).
# FLEET covers EVERY preset whose model is on disk (21 at the v0.5.18 flip; the
# deprecated `devstral-long` alias and the two presets with no local model,
# `coder-reap` / `qwen3-vl-moe`, are the only launch.sh entries left out).
# Entry format: preset|think|ctx|needle-lengths — ctx/needles default to
# 262144 / 1K..250K; 32K presets probe to 30K, qwen3-vl-32b to 120K.
# Order is risk-first for the CURRENT flip (edit per flip):
#   v0.5.18: coder-30b-eval FIRST — CT-format MoE at TP=2, the exact case
#   patch 030 protects (a silent half-load boots green with garbage quality:
#   MMLU/HE here are the corruption tripwire). nemotron3-omni second — it is
#   the only exerciser of the 053 EVS-video re-port (predicate moved to the
#   new mm_schedule.py). gemma4-31b third (production model; restored on the
#   new stack the moment the campaign is green). Then qwen36 (DeltaNet kernels
#   003/007/056), gemma4 (017 gelu beside situ), devstral (041/057/058 chain,
#   tokenizer routing), the rest of the receipted fleet, then the presets that
#   get their first receipts on this flip.
#
# Usage:
#   FLIP_TAG=v0518 ./scripts/eval/flip_fleet_validate.sh              # full fleet
#   FLIP_TAG=v0518 PRESETS="gemma4-12b qwen36" ./scripts/eval/flip_fleet_validate.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$HOME/AI/models}"

# Staged stack (launch.sh honors these).
FLIP_TAG="${FLIP_TAG:?set FLIP_TAG, e.g. v0518}"
export ENV_NAME="${ENV_NAME:-sglang-$FLIP_TAG}"
export SGLANG_DIR="${SGLANG_DIR:-/data/sglang-rebase-$FLIP_TAG}"
export PATH="$HOME/miniforge3/envs/$ENV_NAME/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
export CUDA_PATH="${CUDA_PATH:-/opt/cuda}"
source "$REPO/scripts/common.sh" 2>/dev/null || true
cd "$REPO" || exit 1

# preset | think? (mc-budget selector)
FLEET=(
  "coder-30b-eval|no"
  "nemotron3-omni|yes"
  "gemma4-31b|yes"
  "qwen36|yes"
  "gemma4|yes"
  "devstral|no"
  "qwen35-moe|yes"
  "gemma4-12b|yes"
  "gemma4-21b-reap|yes"
  "qwen36-ream|yes"
  "qwen36-dense|yes"
  "qwen3-ream|no"
  "qwen38|yes"
  "coder-30b|no"
  "coder-reap-25b|no"
  "coder-30b-ream|no"
  "qwen3-vl-32b|no|131072|1024,16384,65536,120000"
  "qwen36-vl-reap|yes"
  "qwen36-dense-ct|yes|32768|1024,16384,30000"
  "qwen35-dense|yes|32768|1024,16384,30000"
  "devstral-32k|no|32768|1024,16384,30000"
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
LOG_ROOT="/tmp/$FLIP_TAG-eval-logs"
mkdir -p "$LOG_ROOT"

log() { echo "[$FLIP_TAG-val $(date +%H:%M:%S)] $*"; }
stop_server() { pkill -KILL -f "sglang.launch_server" 2>/dev/null || true; sleep 6; }

launch_server() {  # $1 preset, $2 logdir, $3 context length
  log "launching $1 @$3 ctx (env $ENV_NAME)"
  MAX_RUNNING="${MAX_RUNNING:-6}" nohup setsid bash "$REPO/scripts/launch.sh" "$1" \
    --context-length "$3" > "$2/server.log" 2>&1 < /dev/null &
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

# Preflight: a broken model registry (e.g. a patch double-registering an arch
# after upstream adds its own) kills EVERY boot — catch it in 10 seconds
# instead of a 30-minute server timeout per preset (v0.5.17 flip lesson: the
# 035 EntryClass hunk vs upstream's new qwen3_5_text.py).
python -c "from sglang.srt.models.registry import ModelRegistry" \
  || { log "FATAL: model registry import failed — patched tree is unbootable"; exit 1; }

PRESETS_RUN="${PRESETS:-$(printf '%s\n' "${FLEET[@]}" | cut -d'|' -f1 | tr '\n' ' ')}"
log "fleet: $PRESETS_RUN"

for ENTRY in "${FLEET[@]}"; do
  IFS='|' read -r PRESET THINK CTX NEEDLES <<< "$ENTRY"
  CTX="${CTX:-262144}"; NEEDLES="${NEEDLES:-$NEEDLE_LENGTHS}"
  case " $PRESETS_RUN " in *" $PRESET "*) ;; *) continue ;; esac
  LOG="$LOG_ROOT/$PRESET"; mkdir -p "$LOG"
  [ "$THINK" = "yes" ] && MCB="$MC_BUDGET_THINK" || MCB=1024
  QJSON="benchmarks/quality/$PRESET-$FLIP_TAG.json"
  CJSON="benchmarks/quality/cap-$PRESET-$FLIP_TAG.json"
  if [ -s "$QJSON" ] && [ -s "$CJSON" ]; then log "=== $PRESET already done, skip ==="; continue; fi
  log "=== $PRESET (think=$THINK mc-budget=$MCB ctx=$CTX needles=$NEEDLES) ==="

  stop_server
  launch_server "$PRESET" "$LOG" "$CTX"
  wait_ready "$LOG" || { stop_server; log "  SKIP $PRESET (boot failed)"; continue; }

  log "  quality eval (mmlu=$MMLU_N he=$HE_N lab=$LAB_N needle=$NEEDLES)"
  python "$REPO/scripts/eval/eval_and_chart.py" --run --port $PORT --tag "$PRESET-$FLIP_TAG" \
    --mmlu-samples "$MMLU_N" --humaneval-samples "$HE_N" --labbench-samples "$LAB_N" \
    --needle-lengths "$NEEDLES" --mc-budget "$MCB" --workers "$WORKERS" \
    > "$LOG/quality.log" 2>&1
  log "    quality rc=$? -> $QJSON"

  log "  capability probe -> $CJSON"
  python "$REPO/scripts/eval/validate_capabilities.py" --port $PORT \
    --tag "$PRESET-$FLIP_TAG" --save "$CJSON" > "$LOG/caps.log" 2>&1
  log "    caps rc=$?"

  stop_server
  log "=== $PRESET done ==="
done

log "=== $FLIP_TAG FLEET VALIDATION COMPLETE ==="
