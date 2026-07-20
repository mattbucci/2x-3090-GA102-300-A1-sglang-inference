#!/bin/bash
# 059 agentic-quality gate (R9700 decode_topk_agentic_ab.sh adopted, 2026-07-19):
# does --decode-topk-pages regress agentic coding on gemma4-31b? Same 6 SWE-bench
# Lite instances as the R9700 A/B (cross-stack comparability), opencode scaffold,
# Docker harness; the ONLY variable is the topk flag.
#   arm OFF  : bare gemma4-31b preset (graphs-on production default) — doubles as
#              the harness-health control for gemma4-31b (no bake-off cell yet).
#   arm TOPK : _ENV_GEMMA_TOPK="--decode-topk-pages 256 --decode-topk-page-size 64"
#              (budget 16,384 — engages once decode ctx > budget).
# PASS = TOPK resolved/applied/empty ~ OFF where topk engages. Then
# context_reliability_curve.py turns both cells into a depth-binned reliability
# ladder (invalid-tool-call rate + resolve rate by TRUE per-step context).
# Rule 2: rollout and score never overlap; server stopped before scoring.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
source scripts/common.sh; activate_conda; setup_nvidia_env
IDS="django__django-10914 mwaskom__seaborn-3010 pallets__flask-4992 psf__requests-3362 pydata__xarray-4094 pylint-dev__pylint-5859"
ROOT="$REPO/evals/swebench/runs"
mkdir -p /tmp/loop-bakeoff-logs

stop_server(){
  pkill -f '[s]glang.launch_server' 2>/dev/null || true
  for _ in $(seq 1 36); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${u:-9999}" -lt 1000 ] && break; sleep 5
  done
}

summarize(){
  echo "=== RESULT [$1] ==="
  python3 - "$2" "$1" <<'PYEOF'
import json, os, sys
d, lab = sys.argv[1], sys.argv[2]
p = os.path.join(d, "scores-docker-summary.json")
if os.path.exists(p):
    s = json.load(open(p))
    print(f"[{lab}] resolved={s.get('resolved')}/{s.get('total_predictions')} "
          f"empty={s.get('empty_patch')} error={s.get('error')}")
    for iid, v in sorted(s.get("per_instance", {}).items()):
        if v != "incomplete":
            print(f"  {iid:36s} {v}")
else:
    print(f"[{lab}] no summary at {p}")
PYEOF
}

run_arm(){  # $1=label $2=topk-env-value $3=outdir
  local LAB=$1 TOPKV=$2 OUT=$3; mkdir -p "$OUT"
  echo "=== [$LAB] serving gemma4-31b _ENV_GEMMA_TOPK='$TOPKV' $(date +%H:%M) ==="
  stop_server
  _ENV_GEMMA_TOPK="$TOPKV" ./scripts/launch.sh gemma4-31b > "$OUT/serve.log" 2>&1 &
  local ready=0
  for _ in $(seq 1 200); do
    grep -q "fired up and ready" "$OUT/serve.log" 2>/dev/null && { ready=1; break; }
    grep -qiE "OutOfMemory|core dumped|RuntimeError|AssertionError" "$OUT/serve.log" 2>/dev/null && break
    sleep 5
  done
  [ "$ready" = 1 ] || { echo "[$LAB] SERVE_FAILED"; stop_server; return 1; }
  echo "=== [$LAB] healthy; rollout 6 instances $(date +%H:%M) ==="
  python evals/swebench/docker_rollout.py --model "sglang/gemma4-31b" --scaffold opencode \
      --instance-ids $IDS --out "$OUT" --served-name gemma4-31b \
      --skip-existing > "$OUT/rollout.log" 2>&1 || true
  stop_server
  echo "=== [$LAB] scoring (docker, lock-serialized) $(date +%H:%M) ==="
  flock -x /tmp/loop-bakeoff-logs/score.lock \
    python -u evals/swebench/score_docker.py --predictions "$OUT/predictions.jsonl" \
      --max-workers 1 --timeout 1800 > "$OUT/score.log" 2>&1 || true
  summarize "$LAB" "$OUT"
}

echo "######## 059 agentic A/B  gemma4-31b  $(date) ########"
run_arm "OFF"  ""                                                    "$ROOT/topk-ab-gemma31b-off"
run_arm "TOPK" "--decode-topk-pages 256 --decode-topk-page-size 64"  "$ROOT/topk-ab-gemma31b-topk"
echo "######## A/B COMPLETE $(date) ########"
summarize "OFF"  "$ROOT/topk-ab-gemma31b-off"
summarize "TOPK" "$ROOT/topk-ab-gemma31b-topk"
echo "=== context-reliability ladder (depth-binned) ==="
python scripts/eval/context_reliability_curve.py \
  --cell "$ROOT/topk-ab-gemma31b-off" --cell "$ROOT/topk-ab-gemma31b-topk" \
  --out benchmarks/gemma-topk-port/context-reliability-ab-2026-07-19.json 2>&1 | tail -25 \
  || echo "(curve failed — inspect cells manually)"
echo "TOPK_AGENTIC_AB_DONE"
