#!/bin/bash
# run_v0512_fleet_eval.sh — v0.5.12 fleet re-eval for the README Quality table +
# long-context tok/s charts + the 256K tool-reliability probe. One model served at
# a time (Rule 1). Per preset: launch@256K -> quality eval -> tok/s sweep -> 256K
# tool-use probe -> stop. Resumable (eval_and_chart caches; --skip-existing logic
# via on-disk receipts).
#
# Usage:
#   ./scripts/eval/run_v0512_fleet_eval.sh                 # full fleet
#   PRESETS="devstral" MMLU_N=5 HE_N=3 LAB_N=2 NEEDLE_LENGTHS=1024,16384 \
#     ./scripts/eval/run_v0512_fleet_eval.sh              # smoke
#
# Env overrides: PRESETS, MMLU_N, HE_N, LAB_N, NEEDLE_LENGTHS, TOOLUSE_LENGTHS,
#   MC_BUDGET_THINK, WORKERS, SERVER_TIMEOUT.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="${MODELS_DIR:-/home/letsrtfm/AI/models}"
# v0.5.12 serving tree + env (launch.sh honors these). Self-set so the script is
# robust to any launch mechanism (foreground, setsid, run_in_background).
export ENV_NAME="${ENV_NAME:-sglang-v0512}"
export SGLANG_DIR="${SGLANG_DIR:-/data/sglang-rebase-v0512}"
# Put the serving env's python on PATH FIRST. This is the set-u-safe way to pin
# the interpreter — do NOT `conda activate` here: its shell function references
# unbound vars ($PS1 etc.) which, under `set -u`, is a FATAL exit. The wrong python
# also breaks deps (Pixtral vision processor import -> vision presets crash at boot).
export PATH="/home/letsrtfm/miniforge3/envs/$ENV_NAME/bin:$PATH"
# tvm_ffi JIT-compiles some kernels on cold cache and needs CUDA_HOME; the minimal
# systemd-unit env lacks it (nvcc lives at /opt/cuda) -> server crashes at boot.
export CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
export CUDA_PATH="${CUDA_PATH:-/opt/cuda}"
# common.sh / activate_conda are set-u-safe and set LD_LIBRARY_PATH + PYTHONPATH.
source "$REPO/scripts/common.sh" 2>/dev/null || true
activate_conda 2>/dev/null || true
# Run from the repo root — eval_and_chart.py writes to the RELATIVE path
# benchmarks/quality; under a systemd unit the CWD is not the repo (-> PermissionError).
cd "$REPO" || exit 1

# preset | benchmark-slug | think? | model-path (relative to MODELS_DIR, for bench tokenizer)
FLEET=(
  "devstral|devstral-24b-awq|no|hf-mattbucci/Devstral-Small-2-24B-AWQ"
  "qwen3-ream|qwen3-30b-ream|no|Qwen3-30B-Instruct-2507-REAM-AWQ"
  "qwen36|qwen3.6-35b-a3b|yes|hf-mattbucci/Qwen3.6-35B-A3B-AWQ"
  "qwen36-ream|qwen3.6-ream|yes|hf-mattbucci/Qwen3.6-REAM-A3B-AWQ"
  "qwen36-dense|qwen3.6-27b|yes|hf-mattbucci/Qwen3.6-27B-AWQ"
  "gemma4-31b|gemma4-31b|yes|hf-mattbucci/gemma-4-31B-AWQ"
  "gemma4|gemma4-26b-awq|yes|hf-mattbucci/gemma-4-26B-AWQ"
)

# Chart libs (matplotlib) live in the `quant` env, not the serving env.
CHART_PY="${CHART_PY:-/home/letsrtfm/miniforge3/envs/quant/bin/python}"

MMLU_N="${MMLU_N:-30}"
HE_N="${HE_N:-20}"
LAB_N="${LAB_N:-15}"
NEEDLE_LENGTHS="${NEEDLE_LENGTHS:-1024,16384,65536,131072,250000}"
TOOLUSE_LENGTHS="${TOOLUSE_LENGTHS:-16384,65536,131072,196608,256000}"
MC_BUDGET_THINK="${MC_BUDGET_THINK:-2560}"
WORKERS="${WORKERS:-4}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-720}"
PORT=23334
LOG_ROOT="/tmp/v0512-eval-logs"
mkdir -p "$LOG_ROOT"

log() { echo "[v0512-eval $(date +%H:%M:%S)] $*"; }

stop_server() { pkill -KILL -f "sglang.launch_server" 2>/dev/null || true; sleep 6; }

launch_server() {  # $1 preset, $2 logdir
  log "launching $1 @256K"
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

# FRESH=1 clears prior quality JSONs so a fresh run doesn't RESUME from a
# stale/different-config cache (eval_and_chart.py resumes if the file exists).
# Leave FRESH unset to RESUME a crashed fleet (completed presets skip instantly).
if [ "${FRESH:-0}" = "1" ]; then
  for P in $PRESETS_RUN; do
    rm -f "benchmarks/quality/$P.json" "benchmarks/quality/tooluse256k-$P-v0512.json"
  done
  log "FRESH=1: cleared prior quality + probe JSONs for $PRESETS_RUN"
fi

for ENTRY in "${FLEET[@]}"; do
  IFS='|' read -r PRESET SLUG THINK MODELDIR <<< "$ENTRY"
  case " $PRESETS_RUN " in *" $PRESET "*) ;; *) continue ;; esac
  LOG="$LOG_ROOT/$PRESET"; mkdir -p "$LOG"
  MODELPATH="$MODELS_DIR/$MODELDIR"
  [ "$THINK" = "yes" ] && MCB="$MC_BUDGET_THINK" || MCB=1024
  log "=== $PRESET (slug=$SLUG think=$THINK mc-budget=$MCB) ==="

  stop_server
  launch_server "$PRESET" "$LOG"
  wait_ready "$LOG" || { stop_server; log "  SKIP $PRESET (boot failed)"; continue; }

  log "  quality eval (mmlu=$MMLU_N he=$HE_N lab=$LAB_N needle=$NEEDLE_LENGTHS)"
  python "$REPO/scripts/eval/eval_and_chart.py" --run --port $PORT --tag "$PRESET" \
    --mmlu-samples "$MMLU_N" --humaneval-samples "$HE_N" --labbench-samples "$LAB_N" \
    --needle-lengths "$NEEDLE_LENGTHS" --mc-budget "$MCB" --workers "$WORKERS" \
    > "$LOG/quality.log" 2>&1
  log "    quality rc=$? -> benchmarks/quality/$PRESET.json"

  log "  long-context tok/s sweep -> benchmarks/$SLUG/results.json"
  mkdir -p "$REPO/benchmarks/$SLUG"
  python "$REPO/scripts/bench/bench_long_context.py" --port $PORT \
    --name "$PRESET" --max-context "${BENCH_MAX_CTX:-262144}" \
    --output "$REPO/benchmarks/$SLUG/results.json" \
    --tokenizer "$MODELPATH" > "$LOG/tokps.log" 2>&1
  log "    tok/s rc=$?"

  log "  256K tool-use probe -> benchmarks/quality/tooluse256k-$PRESET-v0512.json"
  python "$REPO/scripts/eval/probe_256k_tooluse.py" --port $PORT --tag "$PRESET" \
    --lengths "$TOOLUSE_LENGTHS" \
    --out "$REPO/benchmarks/quality/tooluse256k-$PRESET-v0512.json" \
    > "$LOG/tooluse.log" 2>&1
  log "    probe rc=$?"

  stop_server
  log "=== $PRESET done ==="
done

log "regenerating charts (quant env for matplotlib)"
"$CHART_PY" "$REPO/scripts/bench/generate_charts.py"  > "$LOG_ROOT/charts.log" 2>&1
"$CHART_PY" "$REPO/scripts/eval/eval_and_chart.py" --chart >> "$LOG_ROOT/charts.log" 2>&1
log "=== FLEET EVAL COMPLETE ==="
