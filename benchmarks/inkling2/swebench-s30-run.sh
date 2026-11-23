#!/bin/bash
# inkling2-mini omp sample: SWE-bench Lite first 30 ids, bare-served (upstream-main tree + Ampere port).
# One-off (the main queue stays PAUSED); pairs with gemma4-31b-omp-v5-spec30 (18/30, 8,925 s) for the production decision.
cd ~/AI/2x-3090-GA102-300-A1-sglang-inference || exit 1
L=/var/tmp/ink-s30/chain.log
pgrep -f '[s]glang.launch_server' >/dev/null && { echo "[$(date '+%F %T')] ABORT: a server is running" >> $L; echo "[$(date '+%F %T')] DONE" >> $L; exit 1; }
echo "[$(date '+%F %T')] START" >> $L
SCAFFOLDS=omp INSTANCES=30 RUN_TAG=v5-s30 SERVE_MODE=bare SERVER_TIMEOUT=1200 \
  LOG_DIR=/tmp/run-model-cycle-logs/inkling2-mini-s30 PAUSE_FILE=/var/tmp/ink-s30/NO-PAUSE \
  ./evals/swebench/run_model_cycle.sh inkling2-mini > /var/tmp/ink-s30/cycle.log 2>&1
echo "[$(date '+%F %T')] cycle rc=$?" >> $L
echo "[$(date '+%F %T')] DONE" >> $L
