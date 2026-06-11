#!/bin/bash
# deep_tooluse_fill.sh — fill the verified-retrieval-depth gaps for the chart:
# qwen36-dense / qwen36-ream / devstral were only ever probed to ~147K true (old
# ladder), so their real retrieval ceiling is unknown. Deep-probe them (to ~258K
# true, or to-pool for devstral) so the KV-capacity chart's depth bars are honest.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
export ENV_NAME="${ENV_NAME:-sglang-v0512}" SGLANG_DIR="${SGLANG_DIR:-/data/sglang-rebase-v0512}"
export PATH="/home/letsrtfm/miniforge3/envs/$ENV_NAME/bin:$PATH" CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
source "$REPO/scripts/common.sh" 2>/dev/null || true
cd "$REPO" || exit 1
PORT=23334
OUT="$REPO/benchmarks/sprint-2026-06-kv-decode/deep-tooluse-fill"; mkdir -p "$OUT"
log(){ echo "[fill $(date +%H:%M:%S)] $*"; }
stop(){ pkill -KILL -f "sglang.launch_server" 2>/dev/null || true; sleep 8; }

probe(){ # preset lengths
  local preset="$1"; local lengths="$2"; local rlog="/tmp/swa-sweep-logs/fill-$preset"; mkdir -p "$rlog"
  log "=== $preset ==="
  stop
  nohup setsid bash "$REPO/scripts/launch.sh" "$preset" --context-length 262144 \
    > "$rlog/server.log" 2>&1 < /dev/null & disown
  local end=$(( $(date +%s) + 900 )) ok=0
  while [ "$(date +%s)" -lt "$end" ]; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:$PORT/health 2>/dev/null||echo 000)" = "200" ] && { ok=1; break; }
    sleep 10
  done
  [ "$ok" = "1" ] || { log "$preset boot FAILED"; tail -25 "$rlog/server.log" > "$OUT/$preset.bootfail.log"; stop; return 1; }
  python "$REPO/scripts/eval/probe_256k_tooluse.py" --port $PORT --tag "$preset" \
    --lengths "$lengths" --out "$OUT/$preset.tooluse.json" > "$rlog/probe.log" 2>&1
  log "  $preset tooluse rc=$?"
  stop
}
probe qwen36-dense 28672,114688,229376,344064,448000
probe qwen36-ream  28672,114688,229376,344064,448000
probe devstral     28672,114688,229376,310000
log "done"
