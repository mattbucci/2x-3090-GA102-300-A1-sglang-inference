#!/bin/bash
# One-shot recovery: re-score qwen36-ream's three scaffolds whose inline
# score step silently failed on 2026-06-15 because /tmp/loop-bakeoff-logs
# (the flock dir) had been removed by /tmp cleanup mid-run — the cycle
# DONE'd rc=0 but the cells stayed stale (little-coder fake 0/40). Predictions
# are complete on disk; this just runs the Docker scorer over them.
# Ordered little-coder first (the thinking-ship payoff). Serialized via the
# same flock so it can't collide with the live queue's eventual scoring.
set -uo pipefail
REPO_DIR="/home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference"
source "$REPO_DIR/scripts/common.sh"
activate_conda
mkdir -p /tmp/loop-bakeoff-logs
cd "$REPO_DIR"
for SCAFFOLD in little-coder opencode claw-code; do
  OUT="$REPO_DIR/evals/swebench/runs/qwen36-ream-${SCAFFOLD}-v2"
  npred=$(wc -l < "$OUT/predictions.jsonl" 2>/dev/null || echo 0)
  echo "=== [$(date +%F\ %H:%M:%S)] scoring qwen36-ream $SCAFFOLD ($npred preds) ==="
  rm -f "$OUT/scores-docker-summary.json"
  rm -rf "$OUT/scores-docker"
  flock -x /tmp/loop-bakeoff-logs/score.lock \
    python -u "$REPO_DIR/evals/swebench/score_docker.py" \
      --predictions "$OUT/predictions.jsonl" \
      --max-workers 1 \
      --timeout 1800
  rc=$?
  if [ -f "$OUT/scores-docker-summary.json" ]; then
    python3 -c "import json;d=json.load(open('$OUT/scores-docker-summary.json'));print(f'=== RESULT $SCAFFOLD: {d[\"resolved\"]}/{d[\"total_predictions\"]} = {d[\"resolve_rate_pct\"]}%  (empty={d.get(\"empty_patch\",0)} err={d.get(\"error\",0)}) ===')"
  else
    echo "=== RESULT $SCAFFOLD: NO SUMMARY (rc=$rc) — scorer failed ==="
  fi
done
echo "=== ALL DONE $(date +%F\ %H:%M:%S) ==="
