#!/bin/bash
# One-shot recovery: re-score qwen36-ream cells whose inline score step silently
# failed on 2026-06-15 because /tmp/loop-bakeoff-logs (the flock dir) had been
# removed by /tmp cleanup mid-run — the cycle DONE'd rc=0 but the cell stayed
# stale (little-coder fake 0/40). Predictions are complete on disk; this just
# runs the Docker scorer over them.
#
# Only a cell with the COMPLETE 300 predictions AND a missing/stale score is a
# true "scoring failed" case worth re-running. As of 2026-06-15 that's ONLY
# little-coder (300 preds, no summary). opencode is already full (177/300) and
# claw-code has just 150 preds — a *rollout* gap that a re-score can't fix
# (needs a fresh rollout to reach 300), not a scoring gap. The guard below skips
# both so we don't churn ~15 h of docker I/O reproducing numbers we already have.
# Serialized via the bake-off flock so it can't collide with the live queue.
set -uo pipefail
REPO_DIR="/home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference"
source "$REPO_DIR/scripts/common.sh"
activate_conda
mkdir -p /tmp/loop-bakeoff-logs
cd "$REPO_DIR"
for SCAFFOLD in little-coder opencode claw-code; do
  OUT="$REPO_DIR/evals/swebench/runs/qwen36-ream-${SCAFFOLD}-v2"
  npred=$(wc -l < "$OUT/predictions.jsonl" 2>/dev/null || echo 0)
  # Skip partial-prediction cells — re-scoring can't add the missing rollouts.
  if [ "$npred" -lt 300 ]; then
    echo "=== SKIP qwen36-ream $SCAFFOLD: only $npred/300 preds (rollout gap, not a scoring gap) ==="
    continue
  fi
  # Skip cells that already have a valid full-300 summary.
  if [ -f "$OUT/scores-docker-summary.json" ] && \
     python3 -c "import json,sys;d=json.load(open('$OUT/scores-docker-summary.json'));sys.exit(0 if d.get('total_predictions',0)>=300 else 1)" 2>/dev/null; then
    echo "=== SKIP qwen36-ream $SCAFFOLD: already scored at 300 preds ==="
    continue
  fi
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
