#!/bin/bash
# run_model_cycle.sh — full bakeoff eval cycle for a single SGLang preset.
#
# Sequence (Rule 2 enforced: no rollout + score concurrent):
#   0. Scaffold request audit (scaffold_request_audit.py: max thinking + output caps on the wire)
#   1. Launch SGLang server for $PRESET (serve_backend.sh: bare metal, or the
#      repo's OCI image when SERVE_MODE=docker / serve_mode.conf says docker)
#   2. Wait /health=200 (max 12 min); when a dcode lane is queued, prove one
#      /v1/responses function_call round-trip first (probe_responses_api.py)
#   3. For each scaffold in {opencode, opencode-dcp, little-coder, little-coder-rtk, prime, dcode}: full 300-inst rollout
#   4. Stop server
#   5. Audit each scaffold's predictions for infrastructure failures
#   6. If any infra failures: relaunch server, reroll just those instances, stop server
#   6b. Leak gate (audit_leakage.py --require-isolation): an exposed or unproven cell is never scored
#   7. Score each scaffold
#   7b. Benchmark-recall audit on opencode-family lanes (informational; <cell>/recall-audit.json)
#   8. Regenerate cell JSONs via aggregate_bakeoff.py
#   9. Print summary
#
# Total runtime per preset: ~6-18h depending on instance complexity.
# Output: evals/swebench/runs/<preset>-<scaffold>-$RUN_TAG/ for each scaffold
#         (v3 = network-isolated rollouts, 2026-09-19; v2 = the --network=host
#         era, exposure-confounded — see audit_leakage.py)
#         + benchmarks/quality/bakeoff-<preset>-<scaffold>.json
#
# Usage:
#   ./evals/swebench/run_model_cycle.sh <preset> [served_name]
#   served_name defaults to <preset>; only different if opencode.json maps
#   the preset to a different id under the sglang provider.
#
# Environment overrides:
#   SCAFFOLDS       space-separated list (default: "opencode opencode-dcp little-coder little-coder-rtk prime dcode")
#   INSTANCES       per-scaffold instance count (default: 0 = full 300)
#   TIMEOUT         per-instance rollout timeout in seconds (default: 1800)
#   LOG_DIR         where to write per-phase logs (default: /tmp/run-model-cycle-logs/<preset>)
#   SERVER_TIMEOUT  max seconds to wait for server /health=200 (default: 720)
#   SERVE_MODE      bare | docker — how the server is started (default: the
#                   tracked evals/swebench/serve_mode.conf). docker = the OCI
#                   image (SERVE_IMAGE, default sglang-cuda-3090:local) with a
#                   per-cycle API key every scaffold receives; see serve_backend.sh
#   RUN_TAG         run-dir suffix (default v3). Bump it when a harness defect
#                   invalidates every prior cell; aggregate_bakeoff.py keeps the
#                   highest version per (preset, scaffold)
#   PAUSE_FILE      while this path exists the cycle waits before touching the
#                   GPU (default /tmp/run-model-cycle-logs/PAUSE) — the hook for
#                   stack flips / image rebuilds at a cycle boundary

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$REPO_DIR/scripts/common.sh"
activate_conda 2>/dev/null || true
source "$SCRIPT_DIR/serve_backend.sh"

PRESET="${1:-}"
SERVED="${2:-$PRESET}"
if [ -z "$PRESET" ]; then
  echo "Usage: $0 <preset> [served_name]" >&2
  exit 1
fi

SCAFFOLDS="${SCAFFOLDS:-opencode opencode-dcp little-coder little-coder-rtk prime dcode}"  # claw-code retired 2026-08-30 (unmaintained, R9700 same day; historical cells remain); +prime/dcode (R9700 port) and the two A/B lanes opencode-dcp / little-coder-rtk 2026-08-31
INSTANCES="${INSTANCES:-0}"
TIMEOUT="${TIMEOUT:-1800}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-720}"
LOG_DIR="${LOG_DIR:-/tmp/run-model-cycle-logs/$PRESET}"
PAUSE_FILE="${PAUSE_FILE:-/tmp/run-model-cycle-logs/PAUSE}"
RUN_TAG="${RUN_TAG:-v3}"

mkdir -p "$LOG_DIR"
# Self-heal the score flock dir. run_all_cycles.sh creates /tmp/loop-bakeoff-logs
# only once at queue start, so /tmp cleanup mid-run (or bake_off.sh never having
# run) leaves the Phase-5 `flock /tmp/loop-bakeoff-logs/score.lock` opening a
# missing path -> flock errors, score_docker.py never runs, and the cell JSON
# silently keeps its stale content. Recreate it every cycle so scoring can't be
# skipped. (Root-caused 2026-06-15: qwen36-ream DONE'd rc=0 carrying a fake
# little-coder 0/40 cell after the dir had been cleaned away.)
mkdir -p /tmp/loop-bakeoff-logs
START=$(date +%s)

log() { echo "[$PRESET $(date +%H:%M:%S)] $*"; }

# --- boundary pause: a queue cycle starts only when nobody holds the GPU for
# a stack flip / image rebuild (touch $PAUSE_FILE before the previous cycle
# ends; rm it to release). Checked before anything runs.
if [ -e "$PAUSE_FILE" ]; then
  log "PAUSED: $PAUSE_FILE exists — waiting before starting the cycle"
  while [ -e "$PAUSE_FILE" ]; do sleep 60; done
  log "resuming: $PAUSE_FILE removed"
fi

# Resolve how the server is served (bare metal vs the OCI image); in docker
# mode this mints the per-cycle API key and checks the image is current.
serve_backend_init || { log "ERROR: serve backend init failed (SERVE_MODE=${SERVE_MODE:-?})"; exit 1; }
serve_receipt "$LOG_DIR/serve-backend.json"
log "serve mode: $SERVE_MODE${SWEBENCH_SERVE_IMAGE:+ (image $SWEBENCH_SERVE_IMAGE ${SWEBENCH_SERVE_IMAGE_ID#sha256:})}"

stop_server() {
  serve_stop "$PRESET"
}

launch_server() {
  # Reuse a healthy server already serving this preset (a cycle relaunched
  # after a harness-side fix; the preset's serving config is unchanged).
  # Launching a second one would fight for :23334 / VRAM. Phase 2's
  # stop_server still owns shutdown.
  if serve_reusable "$SERVED"; then
    log "reusing healthy server already serving $SERVED"
    return 0
  fi
  log "launching server ($SERVE_MODE)"
  serve_start "$PRESET" "$LOG_DIR/server.log" "$LOG_DIR/server.pid" || return 1
}

# Presets whose launch.sh entry omits QUANT (so the default vs awq_marlin
# choice is left implicit). For these we run a pre-cycle pair smoke test to
# verify both kernels produce coherent output and pick whichever decodes
# faster. Result is exported as QUANT for this cycle's launch_server.
needs_kernel_smoke() {
  case "$1" in
    qwen36-dense|gemma4) return 0 ;;
    *) return 1 ;;
  esac
}

run_kernel_smoke() {
  log "kernel smoke (default vs awq_marlin)"
  bash "$SCRIPT_DIR/smoke_kernel_pair.sh" "$PRESET" \
    > "$LOG_DIR/smoke.log" 2>&1
  local rc=$?
  local winner_env="/tmp/smoke-kernel/$PRESET/winner.env"
  if [ -f "$winner_env" ]; then
    # shellcheck disable=SC1090
    source "$winner_env"
    export QUANT
    log "smoke winner: QUANT=${QUANT:-<preset-default>} (rc=$rc)"
  else
    log "smoke produced no winner.env (rc=$rc); falling back to preset default"
  fi
}

wait_ready() {
  local end=$(($(date +%s) + $SERVER_TIMEOUT))
  while [ "$(date +%s)" -lt "$end" ]; do
    local code=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:23334/health 2>/dev/null || echo 000)
    [ "$code" = "200" ] && { log "server ready"; return 0; }
    sleep 12
  done
  log "ERROR: server timeout after ${SERVER_TIMEOUT}s"
  tail -40 "$LOG_DIR/server.log"
  return 1
}

# --- Phase 0: per-preset kernel smoke (only for presets that need it) ---
if needs_kernel_smoke "$PRESET"; then
  run_kernel_smoke
fi

# --- Phase 0.5: scaffold request audit (no GPU, no server) ---
# Proves what each lane will put on the wire before any of it costs GPU
# days: max thinking (no reasoning_effort / enable_thinking:false), output
# caps >= the harness floor, served model id. Runs the real
# build_scaffold_invocation() commands in a throwaway rollout container
# against a capture endpoint. Non-zero = a scaffold default drifted; the
# cycle stops rather than roll a mislabelled cell.
log "scaffold request audit ($SCAFFOLDS)"
python "$REPO_DIR/evals/swebench/scaffold_request_audit.py" \
  --served-name "$SERVED" --scaffolds "$SCAFFOLDS" \
  --receipt "$LOG_DIR/scaffold-audit.json" > "$LOG_DIR/scaffold-audit.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  log "ERROR: scaffold request audit failed (rc=$rc) — see $LOG_DIR/scaffold-audit.log"
  tail -12 "$LOG_DIR/scaffold-audit.log"
  exit 1
fi
log "scaffold request audit PASS"

# --- Phase 1: launch + rollouts ---
launch_server || { log "ERROR: server launch failed"; exit 1; }
wait_ready || { stop_server; exit 1; }

# --- Phase 1a: Responses-API gate (dcode lanes only) ---
# deepagents forces /v1/responses; v0.5.20's _validate_model 404s any request
# whose `model` is not the served name, and chat probes / the Phase-0 wire
# audit cannot see that (R9700 lost a full dcode lane to it in 10 s per
# instance). Prove one function_call round-trip on the live server first.
case " $SCAFFOLDS " in *" dcode "*)
  log "responses-api probe (dcode lane)"
  python "$REPO_DIR/scripts/eval/probe_responses_api.py" --port 23334 \
    ${SWEBENCH_API_KEY_FILE:+--api-key "$(_serve_api_key)"} \
    --json "$LOG_DIR/responses-probe.json" > "$LOG_DIR/responses-probe.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    log "ERROR: /v1/responses probe failed (rc=$rc) — see $LOG_DIR/responses-probe.log"
    tail -8 "$LOG_DIR/responses-probe.log"
    stop_server; exit 1
  fi
  log "responses-api probe PASS" ;;
esac

NEED_RESCORE=()  # cells that have predictions to score
NEED_RESCORE_AFTER_REROLL=()

for SCAFFOLD in $SCAFFOLDS; do
  OUT="$REPO_DIR/evals/swebench/runs/${PRESET}-${SCAFFOLD}-${RUN_TAG}"
  mkdir -p "$OUT"
  N_FLAG=()
  [ "$INSTANCES" -gt 0 ] && N_FLAG=(--instances "$INSTANCES")

  log "rollout $SCAFFOLD (out=$OUT instances=${INSTANCES:-300} timeout=$TIMEOUT)"
  cp -f "$LOG_DIR/serve-backend.json" "$OUT/serve-backend.json" 2>/dev/null || true
  python "$REPO_DIR/evals/swebench/docker_rollout.py" \
    --model "sglang/$PRESET" \
    --served-name "$SERVED" \
    --scaffold "$SCAFFOLD" \
    --out "$OUT" \
    --skip-existing \
    --timeout "$TIMEOUT" \
    --max-empty-streak 30 \
    "${N_FLAG[@]}" \
    > "$LOG_DIR/rollout-$SCAFFOLD.log" 2>&1
  rc=$?
  preds=$(wc -l < "$OUT/predictions.jsonl" 2>/dev/null || echo 0)
  log "rollout $SCAFFOLD rc=$rc preds=$preds"
  NEED_RESCORE+=("$SCAFFOLD")
done

# --- Phase 2: stop server before audit/reroll/score ---
stop_server

# --- Phase 3: audit ---
for SCAFFOLD in "${NEED_RESCORE[@]}"; do
  OUT="$REPO_DIR/evals/swebench/runs/${PRESET}-${SCAFFOLD}-${RUN_TAG}"
  log "audit $SCAFFOLD"
  python "$REPO_DIR/evals/swebench/audit_predictions.py" \
    --predictions "$OUT/predictions.jsonl" \
    --write-reroll-list "$LOG_DIR/reroll-list-$SCAFFOLD.txt" \
    > "$LOG_DIR/audit-$SCAFFOLD.log" 2>&1 || true
  n=$(wc -l < "$LOG_DIR/reroll-list-$SCAFFOLD.txt" 2>/dev/null || echo 0)
  log "audit $SCAFFOLD: $n infra-failure instances to reroll"
  [ "$n" -gt 0 ] && NEED_RESCORE_AFTER_REROLL+=("$SCAFFOLD")
done

# --- Phase 4: reroll if needed (single server-restart pass) ---
if [ "${#NEED_RESCORE_AFTER_REROLL[@]}" -gt 0 ]; then
  log "relaunching server for reroll"
  launch_server || log "ERROR: server launch failed on reroll"
  wait_ready || { stop_server; log "ERROR: server failed on reroll"; }

  for SCAFFOLD in "${NEED_RESCORE_AFTER_REROLL[@]}"; do
    OUT="$REPO_DIR/evals/swebench/runs/${PRESET}-${SCAFFOLD}-${RUN_TAG}"
    log "reroll $SCAFFOLD"
    python "$REPO_DIR/evals/swebench/reroll_infra_failures.py" \
      --cell "$OUT" \
      --model "sglang/$PRESET" \
      --served-name "$SERVED" \
      --scaffold "$SCAFFOLD" \
      --timeout "$TIMEOUT" \
      > "$LOG_DIR/reroll-$SCAFFOLD.log" 2>&1
    log "reroll $SCAFFOLD rc=$?"
  done
  stop_server
fi

# --- Phase 4.5: leak gate — a cell is scored only if no instance reached the
# answer (audit_leakage.py: 0 web UPSTREAM/SEARCH calls that did not fail) and
# every instance log carries both isolation proofs (network none + bridge,
# refs stripped). A failure here is a harness defect, never a model result:
# stop the cycle with the receipt at <cell>/leak-audit.json; the predictions
# stay and a relaunch --skip-existing resumes once the cause is fixed.
for SCAFFOLD in "${NEED_RESCORE[@]}"; do
  OUT="$REPO_DIR/evals/swebench/runs/${PRESET}-${SCAFFOLD}-${RUN_TAG}"
  log "leak audit $SCAFFOLD"
  python "$REPO_DIR/evals/swebench/audit_leakage.py" --run "$OUT" --require-isolation \
    > "$LOG_DIR/leak-audit-$SCAFFOLD.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    log "ERROR: leak audit FAILED for $SCAFFOLD (rc=$rc) — cell NOT scored; see $LOG_DIR/leak-audit-$SCAFFOLD.log"
    head -3 "$LOG_DIR/leak-audit-$SCAFFOLD.log"
    exit 1
  fi
  log "leak audit $SCAFFOLD: $(head -1 "$LOG_DIR/leak-audit-$SCAFFOLD.log" | sed 's/^[^:]*: //')"
done

# --- Phase 5: score each scaffold (Rule 2: server already stopped) ---
for SCAFFOLD in "${NEED_RESCORE[@]}"; do
  OUT="$REPO_DIR/evals/swebench/runs/${PRESET}-${SCAFFOLD}-${RUN_TAG}"
  log "score $SCAFFOLD"
  rm -f "$OUT/scores-docker-summary.json"
  rm -rf "$OUT/scores-docker"
  flock -x /tmp/loop-bakeoff-logs/score.lock \
    python "$REPO_DIR/evals/swebench/score_docker.py" \
      --predictions "$OUT/predictions.jsonl" \
      --max-workers 1 \
      --timeout "$TIMEOUT" \
      > "$LOG_DIR/score-$SCAFFOLD.log" 2>&1
  rc=$?
  if [ -f "$OUT/scores-docker-summary.json" ]; then
    python3 -c "
import json
d = json.load(open('$OUT/scores-docker-summary.json'))
print(f'  {\"$PRESET\":15s} x {\"$SCAFFOLD\":12s}: {d[\"resolved\"]}/{d[\"total_predictions\"]} = {d[\"resolve_rate_pct\"]}%  (unresolved={d[\"unresolved\"]} empty={d.get(\"empty_patch\",0)} err={d.get(\"error\",0)})')
"
  fi
done

# --- Phase 5.5: benchmark-recall audit (informational, never gates) ---
# How much of an opencode lane's reasoning went to recognising SWE-bench and
# trying to remember the gold patch (R9700 finding 2026-09-20; their wall hits
# were 15-30K-token recall thinks). Reads the per-instance session snapshots +
# the scores just written; receipt at <cell>/recall-audit.json. opencode-family
# lanes only -- the other scaffolds keep their own session formats.
for SCAFFOLD in "${NEED_RESCORE[@]}"; do
  case "$SCAFFOLD" in opencode*) ;; *) continue ;; esac
  OUT="$REPO_DIR/evals/swebench/runs/${PRESET}-${SCAFFOLD}-${RUN_TAG}"
  python "$REPO_DIR/evals/swebench/audit_benchmark_recall.py" --run "$OUT" \
    --out "$OUT/recall-audit.json" > "$LOG_DIR/recall-audit-$SCAFFOLD.log" 2>&1 \
    && log "recall audit $SCAFFOLD: $(grep -E '^  (sessions_with_recall|long_thinks_with_recall|wall_hits):' "$LOG_DIR/recall-audit-$SCAFFOLD.log" | tr -s ' \n' ' ')" \
    || log "recall audit $SCAFFOLD: $(head -1 "$LOG_DIR/recall-audit-$SCAFFOLD.log")"
done

# --- Phase 6: refresh cell JSONs ---
python "$REPO_DIR/evals/swebench/aggregate_bakeoff.py" \
  > "$LOG_DIR/aggregate.log" 2>&1
log "wrote cell JSONs"

DURATION=$(( $(date +%s) - START ))
log "=== cycle DONE in $((DURATION/3600))h $((DURATION/60 % 60))m ==="
