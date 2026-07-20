#!/bin/bash
# serve_production.sh — stand up a preset as a persistent PRODUCTION endpoint on
# the SGLang-standard port 30000, distinct from the eval/bench harness port 23334
# (scripts/common.sh). The standalone probes (probe_codegen/vision/thinking.py)
# already default to 30000, so they hit this endpoint with no --port; the harness
# tooling (validate_capabilities, probe_256k_*, bench_long_context, bench_regression)
# stays on 23334, so an eval sweep never collides with the production server.
#
# Wraps scripts/launch.sh (bench-/serve-as-shipped discipline — never hand-assemble
# a serve command; that divergence is the b09882f lesson) and adds lifecycle:
#   - detached via setsid (PPID=1, survives session interrupt / harness restart)
#   - health-checked before declaring ready (cold TP=2 loads: patch 049 territory)
#   - PID + log tracked per-port under logs/production/ (gitignored)
#
# Usage:
#   scripts/serve_production.sh <preset>            # start (default action)
#   scripts/serve_production.sh start <preset>
#   scripts/serve_production.sh restart <preset>
#   scripts/serve_production.sh stop
#   scripts/serve_production.sh status
#   PORT=30001 scripts/serve_production.sh <preset> # override the production port
#   EXTRA='...' scripts/serve_production.sh gemma4-31b   # pass EXTRA_ARGS through
#                                                   #   (e.g. the 059 topk hook)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

PORT="${PORT:-30000}"
BASE_URL="http://127.0.0.1:$PORT"
RUN_DIR="$REPO_DIR/logs/production"
PID_FILE="$RUN_DIR/$PORT.pid"
LOG_FILE="$RUN_DIR/$PORT.log"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"   # 15 min: cold-cache TP=2 (patch 049)
mkdir -p "$RUN_DIR"

# The server process that actually owns the port (not the launch.sh wrapper).
server_pid() { pgrep -f "[s]glang.launch_server.*--port $PORT" | head -1; }
is_healthy() { curl -s -m 3 "$BASE_URL/health" >/dev/null 2>&1; }
served_model() {
    curl -s -m 5 "$BASE_URL/get_model_info" 2>/dev/null \
      | python3 -c "import json,sys;print(json.load(sys.stdin)['model_path'].split('/')[-1])" 2>/dev/null
}

do_status() {
    local sp; sp="$(server_pid)"
    if is_healthy; then
        echo "UP    port=$PORT  pid=${sp:-?}  model=$(served_model)  log=$LOG_FILE"
        return 0
    elif [ -n "$sp" ]; then
        echo "LOADING/UNHEALTHY  port=$PORT  pid=$sp  (see $LOG_FILE)"
        return 2
    else
        echo "DOWN  port=$PORT"
        return 1
    fi
}

do_stop() {
    local sp; sp="$(server_pid)"
    if [ -z "$sp" ] && ! is_healthy; then echo "already DOWN (port=$PORT)"; return 0; fi
    echo "stopping production server on port $PORT (pid ${sp:-?})..."
    pkill -f "[s]glang.launch_server.*--port $PORT" 2>/dev/null
    for _ in $(seq 1 24); do server_pid >/dev/null || break; sleep 2; done
    if server_pid >/dev/null; then
        echo "graceful stop timed out — sending SIGKILL"
        pkill -9 -f "[s]glang.launch_server.*--port $PORT" 2>/dev/null; sleep 3
    fi
    rm -f "$PID_FILE"
    echo "stopped."
}

do_start() {
    local preset="$1"
    [ -n "$preset" ] || { echo "ERROR: start needs a preset (e.g. gemma4-31b)"; exit 2; }

    if is_healthy; then
        echo "ERROR: a healthy server already owns port $PORT (model=$(served_model)). "
        echo "       Use 'restart $preset' or 'stop' first."
        exit 2
    fi
    # Guard the shared GPUs: refuse to stack onto a running eval/bench/other server.
    if pgrep -f "[s]glang.launch_server" | grep -q . ; then
        echo "ERROR: another sglang server is running (not on port $PORT). Refusing to"
        echo "       co-allocate the GPUs. Running servers:"
        pgrep -af "[s]glang.launch_server" | grep -oE '\--served-model-name [^ ]+|\--port [0-9]+' | paste - - | sed 's/^/         /'
        exit 2
    fi

    echo "starting production endpoint: preset=$preset port=$PORT ${EXTRA:+EXTRA_ARGS='$EXTRA'}"
    setsid bash -c "EXTRA_ARGS='${EXTRA:-}' '$REPO_DIR/scripts/launch.sh' '$preset' --port $PORT > '$LOG_FILE' 2>&1 & echo \$! > '$PID_FILE'; disown" </dev/null >/dev/null 2>&1 &
    disown
    sleep 3
    local lpid; lpid="$(cat "$PID_FILE" 2>/dev/null)"
    echo "launcher pid=$lpid (ppid=$(ps -p "${lpid:-0}" -o ppid= 2>/dev/null | tr -d ' ' || echo ?)); waiting for health (<=${HEALTH_TIMEOUT}s)..."

    local waited=0
    while [ "$waited" -lt "$HEALTH_TIMEOUT" ]; do
        if grep -q "fired up and ready" "$LOG_FILE" 2>/dev/null && is_healthy; then
            echo "READY after ~${waited}s"
            do_status
            return 0
        fi
        if ! kill -0 "$lpid" 2>/dev/null && ! server_pid >/dev/null; then
            echo "FAILED to start — launcher exited. Tail of $LOG_FILE:"
            grep -iE "error|Traceback|OutOfMemory|assert" "$LOG_FILE" | tail -8
            return 1
        fi
        sleep 5; waited=$((waited+5))
    done
    echo "TIMEOUT after ${HEALTH_TIMEOUT}s — server not healthy. See $LOG_FILE"
    return 1
}

# --- arg dispatch: action-first, or bareword-preset shorthand ---
ACTION="${1:-}"
case "$ACTION" in
    start)   do_start "${2:-}";;
    restart) do_stop; sleep 2; do_start "${2:-}";;
    stop)    do_stop;;
    status)  do_status;;
    "" )     echo "usage: serve_production.sh <preset> | start <preset> | restart <preset> | stop | status"; exit 2;;
    *)       do_start "$ACTION";;   # bareword = preset name -> start
esac
