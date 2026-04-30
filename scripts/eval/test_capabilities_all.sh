#!/bin/bash
# Quick capability validation across all AWQ models.
#
# Launches each model on TP=1 with 8K context (single-GPU testing while
# the second 3090 is offline), runs validate_capabilities.py, saves results
# to benchmarks/quality/capability_check.json. Tears down each server
# before starting the next.
#
# Usage:
#   ./scripts/eval/test_capabilities_all.sh                      # default model list
#   ./scripts/eval/test_capabilities_all.sh devstral coder-30b   # specific models
#
# Default skips video on text-only models. Override per-model in run_one().

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_DIR/scripts/common.sh"

PORT="${PORT:-23334}"
LOG_DIR="${LOG_DIR:-/tmp/capability-test-logs}"
mkdir -p "$LOG_DIR"
RESULTS="$REPO_DIR/benchmarks/quality/capability_check.json"
mkdir -p "$(dirname "$RESULTS")"

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(qwen3-ream coder-30b devstral)
fi

# Models without vision/video (skip those checks)
TEXT_ONLY=(coder-30b devstral qwen3-ream qwen35-moe qwen35)

is_text_only() {
    local m="$1"
    for t in "${TEXT_ONLY[@]}"; do
        [ "$m" = "$t" ] && return 0
    done
    return 1
}

wait_ready() {
    local pid="$1"
    local timeout="${2:-600}"
    local start; start=$(date +%s)
    while true; do
        if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            return 1
        fi
        local elapsed=$(($(date +%s) - start))
        if [ "$elapsed" -gt "$timeout" ]; then
            return 1
        fi
        sleep 2
    done
}

stop_server() {
    local pid="$1"
    [ -z "$pid" ] && return
    # Stop the SGLang process group (server spawns workers)
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    sleep 5
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    # Catch orphaned workers
    pkill -KILL -f "sglang.launch_server" 2>/dev/null || true
    sleep 5
}

run_one() {
    local model="$1"
    local logfile="$LOG_DIR/${model}.log"
    local pidfile="$LOG_DIR/${model}.pid"

    echo
    echo "=================================================="
    echo "Capability test: $model"
    echo "=================================================="

    # Launch server in its own session (setsid). 8K context fits any of our
    # AWQ MoE/Dense models on a single 24 GB 3090 with TP=1.
    setsid bash -c "
        cd '$REPO_DIR'
        source '$REPO_DIR/scripts/common.sh'
        activate_conda
        export CUDA_VISIBLE_DEVICES=0
        setup_nvidia_env
        export CUDA_VISIBLE_DEVICES=0
        '$REPO_DIR/scripts/launch.sh' '$model' --tp 1 --context-length 8192 --mem-fraction 0.85 \
            > '$logfile' 2>&1 &
        echo \$! > '$pidfile'
        wait
    " </dev/null >/dev/null 2>&1 &
    local launcher_pid=$!
    disown

    # The setsid-spawned process is the actual server; pidfile gets it
    sleep 5
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || echo "")"
    if [ -z "$pid" ]; then
        echo "FAIL: could not capture server PID for $model"
        kill -9 "$launcher_pid" 2>/dev/null || true
        return 1
    fi

    echo "Server PID $pid — waiting for ready (up to 10 min)..."
    if ! wait_ready "$pid" 600; then
        echo "FAIL: server didn't become ready. Last 40 log lines:"
        tail -40 "$logfile"
        stop_server "$pid"
        return 1
    fi

    # Skip vision/video for text-only models
    local extra_flags=()
    if is_text_only "$model"; then
        extra_flags+=(--skip-vision --skip-video)
    fi

    python "$REPO_DIR/scripts/eval/validate_capabilities.py" \
        --port "$PORT" --tag "$model" --save "$RESULTS" \
        "${extra_flags[@]}" || true

    stop_server "$pid"
    return 0
}

for model in "${MODELS[@]}"; do
    run_one "$model"
done

echo
echo "=================================================="
echo "All capability tests complete."
echo "Results: $RESULTS"
echo "=================================================="
[ -f "$RESULTS" ] && python3 -m json.tool "$RESULTS" || echo "(no results saved)"
