#!/bin/bash
# swa_ratio_sweep.sh — Track A of the 2026-06 sprint: SWA sub-pool sizing vs
# 256K KV capacity on hybrid-SWA models (gemma4-12b / gemma4). See
# benchmarks/sprint-2026-06-kv-decode/LOG.md for hypotheses + decision rules.
#
# MODE=ladder  (default) boot-only: scrape KV pool sizes per ratio, ~5 min/run
# MODE=probe   boot + capabilities + 256K tool-use probe + decode curve
#
# Usage (detached, via sudo systemd-run — see CLAUDE.md):
#   PRESET=gemma4-12b MODE=ladder RATIOS="0.8 0.5 0.25 0.1 0.0625 0.03" \
#     scripts/bench/swa_ratio_sweep.sh
#   PRESET=gemma4-12b MODE=probe RATIOS="0.8 0.0625" scripts/bench/swa_ratio_sweep.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="${MODELS_DIR:-/home/letsrtfm/AI/models}"
export ENV_NAME="${ENV_NAME:-sglang-v0512}"
export SGLANG_DIR="${SGLANG_DIR:-/data/sglang-rebase-v0512}"
# set-u-safe interpreter pin (do NOT conda activate; see run_v0512_fleet_eval.sh)
export PATH="/home/letsrtfm/miniforge3/envs/$ENV_NAME/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
export CUDA_PATH="${CUDA_PATH:-/opt/cuda}"
source "$REPO/scripts/common.sh" 2>/dev/null || true
activate_conda 2>/dev/null || true
cd "$REPO" || exit 1

PRESET="${PRESET:-gemma4-12b}"
MODE="${MODE:-ladder}"
RATIOS="${RATIOS:-0.8 0.5 0.25 0.1 0.0625 0.03}"
CTX="${CTX:-262144}"
EXP="${EXP:-A1-$PRESET}"
TOOLUSE_LENGTHS="${TOOLUSE_LENGTHS:-16384,65536,131072,196608,256000}"
PORT=23334
SERVER_TIMEOUT="${SERVER_TIMEOUT:-900}"
OUT="$REPO/benchmarks/sprint-2026-06-kv-decode/$EXP"
LOGROOT="/tmp/swa-sweep-logs/$EXP"
mkdir -p "$OUT" "$LOGROOT"

# model path for bench tokenizer (matches fleet FLEET table)
case "$PRESET" in
  gemma4-12b) MODELPATH="$MODELS_DIR/gemma-4-12B-it-AWQ" ;;
  gemma4)     MODELPATH="$MODELS_DIR/hf-mattbucci/gemma-4-26B-AWQ" ;;
  *)          MODELPATH="" ;;
esac

log() { echo "[swa-sweep $(date +%H:%M:%S)] $*"; }
stop_server() { pkill -KILL -f "sglang.launch_server" 2>/dev/null || true; sleep 8; }

scrape_to_json() {  # $1 server.log  $2 out.json  $3 ratio  $4 boot_ok  $5 boot_secs
  python - "$1" "$2" "$3" "$4" "$5" <<'PY'
import json, re, sys
srvlog, out, ratio, boot_ok, boot_secs = sys.argv[1:6]
lines = []
try:
    raw = open(srvlog, errors="replace").read()
    pat = re.compile(r".*(max_total_num_tokens|swa|SWA|KV Cache|#token|mem_fraction|memory pool|avail mem).*", re.I)
    lines = [l.strip() for l in raw.splitlines() if pat.match(l)][-60:]
except FileNotFoundError:
    raw = ""
def grab(rx):
    m = None
    for m in re.finditer(rx, raw): pass
    return int(m.group(1)) if m else None
rec = {
    "ratio": float(ratio), "boot_ok": boot_ok == "1", "boot_secs": int(boot_secs),
    "max_total_num_tokens": grab(r"max_total_num_tokens[=:\s]+(\d+)"),
    "swa_max_total_num_tokens": grab(r"swa[_ ]max[_ ]total[_ ]num[_ ]tokens[=:\s]+(\d+)"),
    "pool_lines": lines,
}
json.dump(rec, open(out, "w"), indent=1)
print(f"  scraped: full={rec['max_total_num_tokens']} swa={rec['swa_max_total_num_tokens']}")
PY
}

log "sprint Track A: preset=$PRESET mode=$MODE ratios=[$RATIOS] ctx=$CTX -> $OUT"
for R in $RATIOS; do
  TAG="ratio-$R"; RLOG="$LOGROOT/$TAG"; mkdir -p "$RLOG"
  log "=== $TAG ==="
  stop_server
  T0=$(date +%s)
  MAX_RUNNING="${MAX_RUNNING:-6}" EXTRA_ARGS="--swa-full-tokens-ratio $R ${EXTRA_SWEEP_ARGS:-}" \
    nohup setsid bash "$REPO/scripts/launch.sh" "$PRESET" \
    --context-length "$CTX" > "$RLOG/server.log" 2>&1 < /dev/null &
  disown
  BOOT_OK=0
  END=$(( $(date +%s) + SERVER_TIMEOUT ))
  while [ "$(date +%s)" -lt "$END" ]; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:$PORT/health 2>/dev/null || echo 000)" = "200" ] \
      && { BOOT_OK=1; break; }
    # bail early if the launcher died
    pgrep -f "sglang.launch_server" >/dev/null 2>&1 || { sleep 5; pgrep -f "sglang.launch_server" >/dev/null 2>&1 || break; }
    sleep 10
  done
  BOOT_SECS=$(( $(date +%s) - T0 ))
  log "  boot_ok=$BOOT_OK in ${BOOT_SECS}s"
  scrape_to_json "$RLOG/server.log" "$OUT/${MODE}-${TAG}.json" "$R" "$BOOT_OK" "$BOOT_SECS"
  [ "$BOOT_OK" = "1" ] || { tail -25 "$RLOG/server.log" > "$OUT/${MODE}-${TAG}.bootfail.log"; stop_server; continue; }

  if [ "$MODE" = "probe" ]; then
    log "  capabilities battery"
    python "$REPO/scripts/eval/validate_capabilities.py" --port $PORT \
      > "$RLOG/caps.log" 2>&1
    CAPS_RC=$?
    cp "$RLOG/caps.log" "$OUT/probe-$TAG.caps.log"
    log "    caps rc=$CAPS_RC"
    log "  256K tool-use probe @[$TOOLUSE_LENGTHS]"
    python "$REPO/scripts/eval/probe_256k_tooluse.py" --port $PORT --tag "$PRESET-swa$R" \
      --lengths "$TOOLUSE_LENGTHS" \
      --out "$OUT/probe-$TAG.tooluse.json" > "$RLOG/tooluse.log" 2>&1
    log "    tooluse rc=$?"
    log "  decode curve (bench_long_context)"
    python "$REPO/scripts/bench/bench_long_context.py" --port $PORT \
      --name "$PRESET-swa$R" --max-context "$CTX" \
      --output "$OUT/probe-$TAG.decode.json" \
      ${MODELPATH:+--tokenizer "$MODELPATH"} > "$RLOG/decode.log" 2>&1
    log "    decode rc=$?"
  fi
  stop_server
done
log "sweep complete -> $OUT"
