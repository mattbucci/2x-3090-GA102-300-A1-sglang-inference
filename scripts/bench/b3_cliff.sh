#!/bin/bash
# b3_cliff.sh — sprint B3: qwen3-ream 64K->128K decode cliff (Track C: -26%,
# TPOT 5.5->9.5 ms). Preset runs FlashInfer (no backend flag). Two arms:
#   fi     = preset as-is (control), fine context steps across the cliff
#   triton = --attention-backend triton (does the cliff move/disappear?)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="${MODELS_DIR:-/home/letsrtfm/AI/models}"
export ENV_NAME="${ENV_NAME:-sglang-v0512}"
export SGLANG_DIR="${SGLANG_DIR:-/data/sglang-rebase-v0512}"
export PATH="/home/letsrtfm/miniforge3/envs/$ENV_NAME/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
source "$REPO/scripts/common.sh" 2>/dev/null || true
activate_conda 2>/dev/null || true
cd "$REPO" || exit 1

PORT=23334
TOK="$MODELS_DIR/Qwen3-30B-Instruct-2507-REAM-AWQ"
OUT="$REPO/benchmarks/sprint-2026-06-kv-decode/B3-qwen3-ream"
LOGROOT="/tmp/swa-sweep-logs/B3"; mkdir -p "$OUT" "$LOGROOT"
CTXS="${CTXS:-49152 65536 81920 98304 114688 131072 147456}"

log(){ echo "[b3 $(date +%H:%M:%S)] $*"; }
stop_server(){ pkill -KILL -f "sglang.launch_server" 2>/dev/null || true; sleep 8; }

arm(){ # name extra_args
  local name="$1" extra="$2" RLOG="$LOGROOT/$name"; mkdir -p "$RLOG"
  log "===== arm $name ====="
  stop_server
  EXTRA_ARGS="$extra" nohup setsid bash "$REPO/scripts/launch.sh" qwen3-ream \
    --context-length 262144 > "$RLOG/server.log" 2>&1 < /dev/null &
  disown
  local end=$(( $(date +%s) + 900 )) ok=0
  while [ "$(date +%s)" -lt "$end" ]; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:$PORT/health 2>/dev/null || echo 000)" = "200" ] && { ok=1; break; }
    sleep 10
  done
  [ "$ok" = "1" ] || { log "boot FAILED"; tail -25 "$RLOG/server.log" > "$OUT/$name.bootfail.log"; stop_server; return 1; }
  log "  bench fine contexts: $CTXS"
  python "$REPO/scripts/bench/bench_long_context.py" --port $PORT \
    --name "qwen3-ream-$name" --contexts $CTXS \
    --output "$OUT/$name.json" --tokenizer "$TOK" > "$RLOG/bench.log" 2>&1
  log "  bench rc=$?"
  stop_server
}
arm fi ""
arm triton "--attention-backend triton"
log "done"
