#!/bin/bash
# Sequential SWE-bench Lite bake-off: per (preset, scaffold) pair, launch the
# matching SGLang server at TP=2 / 256K / single-user, run docker_rollout.py
# for the scaffold, kill the server, score the predictions against the
# official Docker harness, and move on.
#
# Output layout (per the no-collision rule):
#
#     evals/swebench/runs/<preset>-<scaffold>-v2/
#         predictions.jsonl
#         predictions/<inst>.diff
#         logs/<inst>.log
#         scores-docker/
#         scores-docker-summary.json
#         meta.json
#
# Each phase is fully independent and idempotent — re-running with
# `--skip-existing` semantics (built into docker_rollout.py) resumes from
# wherever the last invocation stopped. Setting BAKEOFF_PHASES env var to
# a space-separated subset (e.g. `BAKEOFF_PHASES="p1 p3"`) limits which
# phases run.
#
# Usage:
#     ./evals/swebench/bake_off.sh                # all default phases
#     BAKEOFF_PHASES="p1" ./evals/swebench/bake_off.sh
#     BAKEOFF_TIMEOUT=2400 ./evals/swebench/bake_off.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$REPO_DIR/scripts/common.sh"
activate_conda

# --- Configuration ---------------------------------------------------------

# Phases as: <id>:<preset>:<scaffold>:<served_name>:<n_instances>:<comment>
# n_instances: 0 = full 300 (Lite); 10 = smoke. claw-code phases gated on
# the smoke result — review evals/swebench/runs/coder-30b-claw-code-smoke/
# before flipping `BAKEOFF_PHASES` to include the full claw runs.
PHASES=(
    "p1:coder-30b-eval:little-coder:coder-30b-eval:0:Coder-30B × little-coder (300)"
    "p2:coder-30b-eval:claw-code:coder-30b-eval:10:Coder-30B × claw-code SMOKE (10)"
    "p3:coder-reap-25b:opencode:coder-reap-25b:0:Coder-REAP-25B × opencode (300)"
    "p4:coder-reap-25b:little-coder:coder-reap-25b:0:Coder-REAP-25B × little-coder (300)"
    "p5:qwen36:opencode:qwen36:0:Qwen3.6-35B × opencode (300)"
    "p6:qwen36:little-coder:qwen36:0:Qwen3.6-35B × little-coder (300)"
    "p7:devstral-long:opencode:devstral-long:0:Devstral-24B × opencode (300)"
    "p8:devstral-long:little-coder:devstral-long:0:Devstral-24B × little-coder (300)"
    "p9:qwen3-ream:opencode:qwen3-ream:0:Qwen3-30B-REAM × opencode (300)"
    "p10:qwen3-ream:little-coder:qwen3-ream:0:Qwen3-30B-REAM × little-coder (300)"
    "p11:coder-30b-eval:claw-code:coder-30b-eval:0:Coder-30B × claw-code FULL (300, gated on p2)"
    "p12:coder-reap-25b:claw-code:coder-reap-25b:0:Coder-REAP-25B × claw-code (300, gated on p2)"
    "p13:qwen36:claw-code:qwen36:0:Qwen3.6-35B × claw-code (300, gated on p2)"
)

ENABLED="${BAKEOFF_PHASES:-p1 p2 p3 p4 p5 p6 p7 p8 p9 p10}"  # claw expansions gated separately
TIMEOUT="${BAKEOFF_TIMEOUT:-1800}"
LOG_DIR="${BAKEOFF_LOG_DIR:-/tmp/loop-bakeoff-logs}"
SERVER_LOG_DIR="$LOG_DIR/servers"
mkdir -p "$LOG_DIR" "$SERVER_LOG_DIR"

# --- Helpers ---------------------------------------------------------------

stop_server() {
    pkill -KILL -f "sglang.launch_server" 2>/dev/null || true
    pkill -KILL -f "scripts/launch.sh" 2>/dev/null || true
    sleep 4
    for _ in $(seq 1 20); do
        if ! curl -sf -o /dev/null "http://localhost:23334/health" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    sleep 3
}

launch_server() {
    local preset="$1"
    local logfile="$2"
    setsid bash -c "
        cd '$REPO_DIR'
        source '$REPO_DIR/scripts/common.sh'
        activate_conda
        exec '$REPO_DIR/scripts/launch.sh' '$preset' --tp 2 --context-length 262144 --max-running 1
    " > "$logfile" 2>&1 </dev/null &
    disown
}

wait_ready() {
    local timeout="$1"
    local start; start=$(date +%s)
    # The bash prelude (cd + source common.sh + activate_conda + exec
    # launch.sh) takes ~5-15s before `python -m sglang.launch_server` is
    # spawned. Without a boot grace the first pgrep below can race the
    # spawn and falsely report "server died" before the python process
    # even appears.
    local boot_grace=30
    while true; do
        local code
        code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:23334/health" 2>/dev/null || echo "000")
        if [ "$code" = "200" ]; then
            echo "  server ready in $(( $(date +%s) - start ))s"
            return 0
        fi
        if [ $(( $(date +%s) - start )) -gt "$timeout" ]; then
            echo "  TIMEOUT after ${timeout}s"
            return 1
        fi
        if [ $(( $(date +%s) - start )) -gt "$boot_grace" ]; then
            if ! pgrep -f "sglang.launch_server" > /dev/null && \
               ! pgrep -f "scripts/launch.sh" > /dev/null; then
                echo "  server died (post boot-grace; check log)"
                return 1
            fi
        fi
        sleep 6
    done
}

run_phase() {
    local pid="$1" preset="$2" scaffold="$3" served="$4" n_inst="$5" comment="$6"
    local out_dir="$REPO_DIR/evals/swebench/runs/${preset}-${scaffold}-v2"
    local server_log="$SERVER_LOG_DIR/${preset}-${scaffold}.log"
    local rollout_log="$LOG_DIR/${preset}-${scaffold}-rollout.log"
    local score_log="$LOG_DIR/${preset}-${scaffold}-score.log"

    echo
    echo "=================================================="
    echo "PHASE $pid: $comment"
    echo "  preset:   $preset"
    echo "  scaffold: $scaffold"
    echo "  out:      $out_dir"
    echo "=================================================="

    stop_server

    echo "  launching server..."
    launch_server "$preset" "$server_log"
    if ! wait_ready 1800; then
        tail -40 "$server_log"
        echo "PHASE $pid FAILED: server didn't become ready"
        stop_server
        return 1
    fi

    local n_flag=()
    if [ "$n_inst" -gt 0 ]; then
        n_flag=(--instances "$n_inst")
    fi

    echo "  running rollout (scaffold=$scaffold timeout=${TIMEOUT}s)..."
    python "$REPO_DIR/evals/swebench/docker_rollout.py" \
        --model "sglang/$served" \
        --scaffold "$scaffold" \
        --out "$out_dir" \
        --skip-existing \
        --timeout "$TIMEOUT" \
        "${n_flag[@]}" \
        > "$rollout_log" 2>&1
    local rc=$?
    echo "  rollout exited rc=$rc; last lines:"
    tail -8 "$rollout_log"

    stop_server

    if [ "$n_inst" -le 10 ] && [ "$n_inst" -gt 0 ]; then
        echo "  smoke-only — skipping score for now"
        return 0
    fi

    # Score in BACKGROUND, but flock-serialized across phases so only ONE
    # scorer runs at a time. The "1 rollout + 1 score concurrent" design
    # is preserved (Phase N+1's rollout still overlaps Phase N's score),
    # but Phase N+1's score blocks on the lock until Phase N's score
    # finishes — preventing the multi-scorer pile-up that OOM'd the box.
    #
    # Worker count: 8 (was 24). 24 sized peak memory at ~140 MB/container,
    # which ignored the docker pull+extract spike + GB-class pytest peaks
    # for sympy/matplotlib instances.
    #
    # Writes its own PID file so the chain can wait for all scores at the
    # end (or aggregate_bakeoff.py can detect in-progress runs by checking
    # whether scores-docker-summary.json exists).
    echo "  scoring against Docker harness (background, flock-serialized)..."
    setsid bash -c "
        flock -x '$LOG_DIR/score.lock' \
        python '$REPO_DIR/evals/swebench/score_docker.py' \
            --predictions '$out_dir/predictions.jsonl' \
            --max-workers '${BAKEOFF_SCORE_WORKERS:-8}' \
            > '$score_log' 2>&1
        echo \$? > '$out_dir/.score-rc'
    " </dev/null >/dev/null 2>&1 &
    disown
    local score_pid=$!
    echo "$score_pid" > "$out_dir/.score-pid"
    echo "  score PID: $score_pid (log: $score_log)"
}

# --- Main ------------------------------------------------------------------

echo "Bake-off starting at $(date)"
echo "Enabled phases: $ENABLED"
echo "Per-instance timeout: ${TIMEOUT}s"

# Sanity: claw binary present if any claw-code phase is enabled
if echo "$ENABLED" | grep -q -E "p2 |p2$|p11|p12|p13"; then
    if [ ! -x "$REPO_DIR/evals/swebench/docker/claw" ]; then
        echo "claw binary missing at evals/swebench/docker/claw — running build_claw.sh"
        bash "$REPO_DIR/scripts/build_claw.sh" || {
            echo "ERROR: claw build failed; either fix the build or remove claw phases from BAKEOFF_PHASES"
            exit 1
        }
    fi
fi

for phase in "${PHASES[@]}"; do
    IFS=':' read -r pid preset scaffold served n_inst comment <<< "$phase"
    if echo "$ENABLED" | grep -wq "$pid"; then
        run_phase "$pid" "$preset" "$scaffold" "$served" "$n_inst" "$comment"
    fi
done

stop_server

echo
echo "All rollouts dispatched. Waiting for in-flight scoring jobs to drain..."
# Each phase's score runs in background; collect them before exit.
while pgrep -f "swebench.harness.run_evaluation" > /dev/null; do
    sleep 30
done
echo "All scoring jobs drained at $(date)."

echo
echo "=================================================="
echo "Bake-off complete at $(date)"
echo "Per-phase artifacts:    evals/swebench/runs/<preset>-<scaffold>-v2/"
echo "Run aggregator:         python evals/swebench/aggregate_bakeoff.py"
echo "=================================================="
