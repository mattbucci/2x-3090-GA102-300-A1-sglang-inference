#!/bin/bash
# stop_queue.sh — cleanly stop the bake-off queue: wrapper, cycle, rollout
# drivers, serving process, and per-instance containers.
#
# Exists because doing this inline keeps self-matching: an interactive
# `pkill -f 'swebench.*rollout'` compound ALSO contains a real
# evals/swebench path plus the word rollout in its own cmdline and kills
# the invoking shell (bracket tricks don't help when the real path appears).
# A script file's cmdline is just "bash .../stop_queue.sh" — no self-match.
set -uo pipefail
LOG_ROOT="/tmp/run-model-cycle-logs"

WP=$(cat "$LOG_ROOT/loop-pid" 2>/dev/null || true)
[ -n "${WP:-}" ] && { kill -- -"$WP" 2>/dev/null || kill "$WP" 2>/dev/null; }

pkill -f 'run_all_cycles.sh' 2>/dev/null
pkill -f 'run_model_cycle.sh' 2>/dev/null
pkill -f 'docker_rollout.py' 2>/dev/null
pkill -f 'sglang.launch_server' 2>/dev/null
sleep 4
pkill -KILL -f 'sglang.launch_server' 2>/dev/null

if command -v docker >/dev/null; then
    sudo docker ps -q --filter "name=swebench" | xargs -r sudo docker rm -f 2>/dev/null
fi

sleep 2
if pgrep -f 'run_model_cycle.sh|run_all_cycles.sh' >/dev/null 2>&1; then
    echo "WARN: queue processes survived:" >&2
    pgrep -af 'run_model_cycle.sh|run_all_cycles.sh' >&2
    exit 1
fi
echo "queue stopped"
