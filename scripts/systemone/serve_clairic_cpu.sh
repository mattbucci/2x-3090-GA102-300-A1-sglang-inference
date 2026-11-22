#!/bin/bash
# Lifecycle for the Clairic Flash CPU System One server (port 30001 by default).
#   scripts/systemone/serve_clairic_cpu.sh start|stop|status|restart
# Env: conda env `clairic-cpu` (torch 2.13 CPU + transformers 5.10.2, see scripts/systemone/setup_clairic_cpu_env.sh).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${PORT:-30001}"
RUN="$REPO/logs/production"; PID="$RUN/$PORT.pid"; LOG="$RUN/$PORT.log"
mkdir -p "$RUN"
up(){ curl -s -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }
case "${1:-start}" in
  start)
    if up; then echo "already UP on :$PORT"; exit 0; fi
    source /home/letsrtfm/miniforge3/etc/profile.d/conda.sh; conda activate clairic-cpu
    # CPU only: hide the GPUs so the release runtime never touches CUDA.
    setsid env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 python "$REPO/scripts/systemone/serve_clairic_cpu.py" --port "$PORT" \
      > "$LOG" 2>&1 < /dev/null & echo $! > "$PID"
    for _ in $(seq 1 180); do up && { echo "UP    port=$PORT pid=$(cat "$PID") log=$LOG"; exit 0; }; kill -0 "$(cat "$PID")" 2>/dev/null || { echo "server exited; see $LOG"; exit 1; }; sleep 5; done
    echo "not healthy after 15 min; see $LOG"; exit 1 ;;
  stop)
    [ -f "$PID" ] && kill -TERM "$(cat "$PID")" 2>/dev/null; sleep 3; rm -f "$PID"; echo stopped ;;
  status)
    if up; then echo "UP    port=$PORT pid=$(cat "$PID" 2>/dev/null) model=$(curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')"; else echo "DOWN  port=$PORT"; fi ;;
  restart) "$0" stop; "$0" start ;;
esac
