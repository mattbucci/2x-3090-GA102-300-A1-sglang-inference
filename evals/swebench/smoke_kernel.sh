#!/bin/bash
# smoke_kernel.sh <preset> <quant_label>
#
# One-kernel smoke test: launch SGLang for $preset with the given quant
# override, probe /v1/chat/completions for a coherent reply, benchmark
# 5 short generations, kill the server, write result.json.
#
# quant_label semantics:
#   default     -> empty QUANT env (let launch.sh pick the preset's value)
#   awq_marlin  -> override QUANT=awq_marlin
#   <other>     -> override QUANT=<other>
#
# Result:
#   /tmp/smoke-kernel/<preset>/<quant_label>/{server.log, response.json, result.json}
#
# Uses port 23335 so it doesn't collide with the eval queue on 23334.
# A hard kill via setsid PGID ensures we don't leak GPU memory between runs.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$REPO_DIR/scripts/common.sh"
activate_conda 2>/dev/null || true

PRESET="${1:?usage: smoke_kernel.sh <preset> <quant_label>}"
QUANT_LABEL="${2:?usage: smoke_kernel.sh <preset> <quant_label>}"

PORT="${SMOKE_PORT:-23335}"
LOG_DIR="/tmp/smoke-kernel/$PRESET/$QUANT_LABEL"
mkdir -p "$LOG_DIR"

log() { echo "[smoke $PRESET/$QUANT_LABEL $(date +%H:%M:%S)] $*"; }

# Build the launch.sh environment override for this kernel
QUANT_ENV=""
case "$QUANT_LABEL" in
  default)    QUANT_ENV="" ;;  # let launch.sh keep its preset default
  awq_marlin) QUANT_ENV="QUANT=awq_marlin" ;;
  awq)        QUANT_ENV="QUANT=awq" ;;
  *)          QUANT_ENV="QUANT=$QUANT_LABEL" ;;
esac

# Hard cleanup of any stray server on $PORT before we start
pkill -KILL -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
sleep 2

log "launching ($QUANT_ENV) on port $PORT"
# setsid puts the launcher in its own process group so a single kill -PG cleans
# everything (python + helper procs).
env $QUANT_ENV setsid nohup bash "$REPO_DIR/scripts/launch.sh" "$PRESET" --port "$PORT" \
  > "$LOG_DIR/server.log" 2>&1 &
LAUNCH_PID=$!
echo $LAUNCH_PID > "$LOG_DIR/server.pid"
disown $LAUNCH_PID 2>/dev/null || true

cleanup() {
  log "cleanup"
  # Kill the entire process group (setsid means PGID==LAUNCH_PID)
  kill -KILL -- -"$LAUNCH_PID" 2>/dev/null || true
  pkill -KILL -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
  sleep 5
}
trap cleanup EXIT

# Wait for /health
DEADLINE=$(($(date +%s) + 720))
HEALTH=000
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  HEALTH=$(curl -s -o /dev/null -w "%{http_code}" -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
  [ "$HEALTH" = "200" ] && break
  sleep 10
done

if [ "$HEALTH" != "200" ]; then
  log "ERROR: server not healthy (last code=$HEALTH); tail of server.log:"
  tail -40 "$LOG_DIR/server.log"
  python3 - "$LOG_DIR/result.json" "$PRESET" "$QUANT_LABEL" <<'PY'
import json, sys
out = sys.argv[1]
json.dump({
    "preset": sys.argv[2],
    "quant_label": sys.argv[3],
    "ok": False,
    "reason": "server_not_healthy",
    "tokens_per_sec": 0,
}, open(out, "w"), indent=2)
PY
  exit 2
fi
log "server ready"

# 5 short generations -> average decode tok/s
python3 - "$LOG_DIR" "$PRESET" "$QUANT_LABEL" "$PORT" <<'PY'
import json, sys, time, urllib.request

log_dir, preset, quant_label, port = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"

def chat(prompt, max_tokens):
    body = json.dumps({
        "model": preset,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(endpoint, body, {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    dt = time.time() - t0
    return data, dt

# Sanity probe + record content
probe_data, probe_dt = chat("Write a Python function that reverses a list. Just the code, no explanation.", 200)
probe_content = probe_data.get("choices", [{}])[0].get("message", {}).get("content", "")
probe_tokens = probe_data.get("usage", {}).get("completion_tokens", 0)
ok_basic = bool(probe_tokens) and "def " in probe_content
print(f"[probe] tokens={probe_tokens} dt={probe_dt:.2f}s ok_basic={ok_basic}")
print(f"[probe] content[:200]={probe_content[:200]!r}")

# 5x decode-bench runs
prompts = [
    "Write a Python quicksort.",
    "Write a Go HTTP server that returns hello world.",
    "Write a SQL query that finds users who placed >5 orders last month.",
    "Write a regex that matches IPv4 addresses.",
    "Write a Rust function that computes Fibonacci numbers iteratively.",
]
total_tokens = 0
total_seconds = 0.0
for p in prompts:
    d, dt = chat(p, 128)
    t = d.get("usage", {}).get("completion_tokens", 0)
    total_tokens += t
    total_seconds += dt
    print(f"[bench] {t} tok / {dt:.2f}s = {t/dt:.1f} tok/s")

tps = total_tokens / total_seconds if total_seconds > 0 else 0.0

result = {
    "preset": preset,
    "quant_label": quant_label,
    "ok": ok_basic and tps > 0,
    "reason": None if (ok_basic and tps > 0) else ("bad_basic_response" if not ok_basic else "zero_tps"),
    "tokens_per_sec": round(tps, 2),
    "total_tokens": total_tokens,
    "total_seconds": round(total_seconds, 2),
    "probe_tokens": probe_tokens,
    "probe_content_head": probe_content[:300],
}
json.dump(result, open(f"{log_dir}/response.json", "w"), indent=2)
json.dump(result, open(f"{log_dir}/result.json", "w"), indent=2)
print(f"[result] tok/s={tps:.2f} ok={result['ok']}")
PY
RC=$?
log "done rc=$RC"
exit $RC
