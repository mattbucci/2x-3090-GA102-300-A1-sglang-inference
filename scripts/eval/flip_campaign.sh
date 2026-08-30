#!/bin/bash
# flip_campaign.sh — the whole GPU-side validation campaign for a stack flip,
# meant to run DETACHED (setsid) because it is multi-hour and the box has a
# ~9-17h kernel-BUG reboot cadence. Every phase is resumable: re-run the same
# command after a reboot and finished presets are skipped.
#
#   1. stop the production endpoint (:30000) — the campaign needs both GPUs
#   2. flip_fleet_validate.sh  — 12-preset full-N quality + capability probes
#                                on the STAGED stack (receipts *-$FLIP_TAG.json)
#   3. bench_regression.sh arm — the depth-verified throughput tripwire vs
#                                benchmarks/baselines.json (compare mode;
#                                BASELINE=save is a separate deliberate act).
#                                TRIPWIRE_PRESETS defaults to every preset that
#                                has a baseline (the 7 tripwire presets + qwen38)
#   4. compare_flip_receipts.py — side-by-side vs the previous stack, exit code
#                                = regression flag
# Production is NOT restarted here: the operator restarts it on the validated
# stack (scripts/serve_production.sh restart <preset>) after reading the receipts.
#
# Usage:  FLIP_TAG=v0518 PREV_TAG=v0517 scripts/eval/flip_campaign.sh
#         TRIPWIRE_PRESETS="qwen38" ...   # bench only these (resume / add-on)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
FLIP_TAG="${FLIP_TAG:?set FLIP_TAG, e.g. v0518}"
PREV_TAG="${PREV_TAG:?set PREV_TAG, e.g. v0517}"
export ENV_NAME="${ENV_NAME:-sglang-$FLIP_TAG}"
export SGLANG_DIR="${SGLANG_DIR:-/data/sglang-rebase-$FLIP_TAG}"
TRIPWIRE_PRESETS="${TRIPWIRE_PRESETS:-qwen36 qwen36-dense coder-30b qwen35-moe gemma4 nemotron3-omni devstral qwen38}"
LOG="/tmp/$FLIP_TAG-campaign.log"
ts() { date +%H:%M:%S; }
cd "$REPO" || exit 1
{
echo "[$(ts)] campaign $PREV_TAG -> $FLIP_TAG  env=$ENV_NAME tree=$SGLANG_DIR"
if [ "$("$REPO/scripts/serve_production.sh" status 2>/dev/null | awk '{print $1}')" = "UP" ]; then
  echo "[$(ts)] stopping production endpoint"; "$REPO/scripts/serve_production.sh" stop
fi
echo "[$(ts)] === phase 1: fleet quality + capability ==="
FLIP_TAG="$FLIP_TAG" "$REPO/scripts/eval/flip_fleet_validate.sh"; echo "[$(ts)] fleet rc=$?"
echo "[$(ts)] === phase 2: throughput tripwire ($TRIPWIRE_PRESETS; compare vs baselines.json) ==="
# shellcheck disable=SC2086
"$REPO/scripts/bench/bench_regression.sh" arm $TRIPWIRE_PRESETS; echo "[$(ts)] tripwire rc=$?"
echo "[$(ts)] === phase 3: receipts vs $PREV_TAG ==="
python3 "$REPO/scripts/eval/compare_flip_receipts.py" --old "$PREV_TAG" --new "$FLIP_TAG"; echo "[$(ts)] compare rc=$?"
echo "[$(ts)] === CAMPAIGN COMPLETE ==="
} >> "$LOG" 2>&1
