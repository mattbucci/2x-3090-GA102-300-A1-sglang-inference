#!/usr/bin/env bash
# Docker-mode parity smoke: prove the OCI serving image (serve_backend.sh
# SERVE_MODE=docker) can carry a bake-off cycle end to end BEFORE
# serve_mode.conf flips to `docker` and the queue relaunches on it.
#
# Per preset, in the order a real cycle would hit them:
#   1. scaffold request audit with the minted key on the wire (auth column;
#      no GPU) — the pi-0.83 "Bearer LLAMACPP_API_KEY" class
#   2. serve_start from the image (strict stale-image check, secure launch,
#      per-run API key) -> /health -> /v1/models lists the preset
#   3. bench_regression.sh <preset> compare vs baselines.json (bearer via
#      SGLANG_API_KEY; RUN_SUFFIX keeps the day's arm receipt intact) — the
#      container must be within the tripwire threshold of bare metal
#   4. probe_responses_api.py — /v1/responses function-call round-trip +
#      404 on an unknown model (the dcode lane's endpoint)
#   5. a SMOKE_INSTANCES-instance isolated rollout through the docker-served
#      server (network=none + bridge, key from SWEBENCH_API_KEY_FILE) and
#      audit_leakage.py --require-isolation on it; meta.json must say
#      serve_mode=docker network_mode=none
#   6. serve_stop + VRAM drain
#
# Usage: evals/swebench/smoke_docker_serving.sh <preset> [preset...]
# Env:   SMOKE_DIR        (default /tmp/docker-serve-smoke) logs, secrets, run dirs
#        SMOKE_SCAFFOLD   (default opencode) scaffold for the mini rollout
#        SMOKE_INSTANCES  (default 2)
#        SMOKE_SKIP_BENCH=1 / SMOKE_SKIP_ROLLOUT=1   shorten a re-run
#        SMOKE_FRESH_CACHE=1  drop the image's JIT cache volume first (do this
#                             once at a stack flip: triton/flashinfer caches
#                             from the previous torch/cu are dead weight)
#        SERVER_TIMEOUT   (default 1800 s — a fresh cache volume recompiles)
#        SERVE_* / SERVE_ALLOW_STALE_IMAGE as in serve_backend.sh
# Exit:  0 all presets PASS; 1 otherwise. Receipt: $SMOKE_DIR/<preset>/smoke-receipt.json
#
# Rule 2: refuses to start while a rollout / scoring container is running.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEBENCH_DIR="$SCRIPT_DIR"  # common.sh re-points SCRIPT_DIR at scripts/; this dir stays evals/swebench
REPO_DIR="$(cd "$SWEBENCH_DIR/../.." && pwd)"

[ $# -ge 1 ] || { echo "Usage: $0 <preset> [preset...]" >&2; exit 2; }

source "$REPO_DIR/scripts/common.sh"
activate_conda 2>/dev/null || true

export SERVE_MODE=docker
SMOKE_DIR="${SMOKE_DIR:-/tmp/docker-serve-smoke}"
SMOKE_SCAFFOLD="${SMOKE_SCAFFOLD:-opencode}"
SMOKE_INSTANCES="${SMOKE_INSTANCES:-2}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1800}"
export SERVE_SECRETS_DIR="${SERVE_SECRETS_DIR:-$SMOKE_DIR/secrets}"
mkdir -p "$SMOKE_DIR"

source "$SWEBENCH_DIR/serve_backend.sh"

log() { echo "[smoke $(date +%H:%M:%S)] $*"; }

# --- Rule 2 guard: one docker-heavy workload at a time ---
busy="$(docker ps --format '{{.Names}} {{.Image}}' 2>/dev/null | grep -E 'swebench-rollout|swebench-score|sweb\.eval' | head -3)"
if [ -n "$busy" ]; then
  echo "ERROR: rollout/scoring containers running — not starting a GPU smoke (Rule 2):" >&2
  echo "$busy" >&2
  exit 2
fi
if pgrep -f "run_all_cycles.sh" >/dev/null 2>&1; then
  echo "ERROR: run_all_cycles.sh is alive — stop the queue before the smoke" >&2
  exit 2
fi

serve_backend_init || { log "ERROR: serve backend init failed"; exit 1; }
log "image $SWEBENCH_SERVE_IMAGE (${SWEBENCH_SERVE_IMAGE_ID#sha256:}) key file $SWEBENCH_API_KEY_FILE"

if [ "${SMOKE_FRESH_CACHE:-0}" = "1" ]; then
  if docker volume inspect "$SERVE_CACHE_VOLUME" >/dev/null 2>&1; then
    log "dropping JIT cache volume $SERVE_CACHE_VOLUME (SMOKE_FRESH_CACHE=1)"
    docker volume rm "$SERVE_CACHE_VOLUME" >/dev/null || { log "ERROR: could not remove $SERVE_CACHE_VOLUME (in use?)"; exit 1; }
  fi
fi

wait_ready() {
  local log_file="$1" end=$(($(date +%s) + SERVER_TIMEOUT)) code
  while [ "$(date +%s)" -lt "$end" ]; do
    code=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:23334/health 2>/dev/null || echo 000)
    [ "$code" = "200" ] && return 0
    if [ "$(_serve_container_state "$SERVE_CONTAINER")" = "exited" ]; then
      log "ERROR: container exited during boot"; tail -30 "$log_file"; return 1
    fi
    sleep 10
  done
  log "ERROR: no /health after ${SERVER_TIMEOUT}s"; tail -40 "$log_file"; return 1
}

drain_vram() {
  local used
  for _ in $(seq 1 36); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    [ "${used:-9999}" -lt 1000 ] && return 0
    sleep 5
  done
  log "WARN: VRAM still ${used:-?} MiB after drain window"
  return 1
}

# receipt helpers: STEPS[name]=PASS|FAIL|SKIP, DETAIL[name]=text
declare -A STEPS DETAIL
mark() { STEPS[$1]="$2"; DETAIL[$1]="${3:-}"; log "$1: $2${3:+ — $3}"; }

write_receipt() {
  local preset="$1" out="$2" verdict="$3"
  python3 - "$out" "$preset" "$verdict" "$SWEBENCH_SERVE_IMAGE" "${SWEBENCH_SERVE_IMAGE_ID#sha256:}" \
    "$(for k in "${!STEPS[@]}"; do printf '%s\t%s\t%s\n' "$k" "${STEPS[$k]}" "${DETAIL[$k]}"; done)" <<'PY'
import json, sys, datetime
out, preset, verdict, image, image_id, rows = sys.argv[1:]
steps = {}
for line in rows.splitlines():
    if not line.strip(): continue
    k, s, d = (line.split("\t") + ["", ""])[:3]
    steps[k] = {"status": s, "detail": d}
json.dump({"preset": preset, "verdict": verdict, "serve_mode": "docker", "image": image, "image_id": image_id,
           "date": datetime.datetime.now().isoformat(timespec="seconds"), "steps": steps}, open(out, "w"), indent=2)
PY
}

smoke_preset() {
  local preset="$1" served="$1"
  local d="$SMOKE_DIR/$preset"; mkdir -p "$d"
  STEPS=(); DETAIL=()
  local fail=0 rc
  log "=== $preset ==="
  serve_receipt "$d/serve-backend.json"

  # 1. scaffold request audit (no GPU) — auth column must pass with the bakeoff key
  local audit_scaffolds="$SMOKE_SCAFFOLD"
  [[ " $audit_scaffolds " == *" dcode "* ]] || audit_scaffolds="$audit_scaffolds dcode"
  python "$SWEBENCH_DIR/scaffold_request_audit.py" --served-name "$served" --scaffolds "$audit_scaffolds" \
    --receipt "$d/scaffold-audit.json" > "$d/scaffold-audit.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    mark scaffold_audit PASS "$(python3 -c "import json;d=json.load(open('$d/scaffold-audit.json'));print('api_auth=%s '%d.get('api_auth')+' '.join(f\"{k}:auth={v.get('auth')}\" for k,v in d['verdicts'].items()))" 2>/dev/null)"
  else
    mark scaffold_audit FAIL "rc=$rc ($d/scaffold-audit.log)"; tail -8 "$d/scaffold-audit.log"; fail=1
  fi

  # 2. boot from the image
  local t0=$(date +%s)
  if ! serve_start "$preset" "$d/server.log" "$d/server.pid"; then
    mark boot FAIL "serve_start"; write_receipt "$preset" "$d/smoke-receipt.json" FAIL; return 1
  fi
  if ! wait_ready "$d/server.log"; then
    mark boot FAIL "no /health in ${SERVER_TIMEOUT}s"
    serve_stop "$preset"; drain_vram; write_receipt "$preset" "$d/smoke-receipt.json" FAIL; return 1
  fi
  local live; live="$(serve_models)"
  if [[ " $live " == *" $served "* ]]; then
    mark boot PASS "$(( $(date +%s) - t0 ))s to /health; /v1/models=[$live]"
  else
    mark boot FAIL "/v1/models=[$live] lacks $served"; fail=1
  fi
  # the key must be enforced: an unauthenticated /v1/models is a 401
  local anon; anon=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:23334/v1/models 2>/dev/null)
  if [ "$anon" = "401" ] || [ "$anon" = "403" ]; then mark api_auth PASS "anon /v1/models -> $anon"
  else mark api_auth FAIL "anon /v1/models -> $anon (expected 401)"; fail=1; fi

  # 3. tripwire compare against the bare-metal baseline
  if [ "${SMOKE_SKIP_BENCH:-0}" = "1" ]; then
    mark bench SKIP
  else
    SGLANG_API_KEY="$(_serve_api_key)" RUN_SUFFIX="docker-smoke" STACK_TAG="${STACK_TAG:-sglang-v0.5.20-docker}" \
      bash "$REPO_DIR/scripts/bench/bench_regression.sh" "$preset" > "$d/bench.log" 2>&1
    rc=$?
    local verdict_line; verdict_line="$(grep -E "^  $preset/|no baseline|^RESULT" "$d/bench.log" | sed 's/^ *//' | tr '\n' '|')"
    if [ $rc -eq 0 ]; then mark bench PASS "$verdict_line"; else mark bench FAIL "rc=$rc $verdict_line ($d/bench.log)"; fail=1; fi
  fi

  # 4. /v1/responses (dcode lane endpoint)
  python "$REPO_DIR/scripts/eval/probe_responses_api.py" --port 23334 --api-key "$(_serve_api_key)" \
    --json "$d/responses-probe.json" > "$d/responses-probe.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then mark responses_api PASS; else mark responses_api FAIL "rc=$rc ($d/responses-probe.log)"; tail -6 "$d/responses-probe.log"; fail=1; fi

  # 5. mini isolated rollout + leak audit
  if [ "${SMOKE_SKIP_ROLLOUT:-0}" = "1" ]; then
    mark rollout SKIP; mark leak_audit SKIP
  else
    local out="$SMOKE_DIR/runs/${preset}-${SMOKE_SCAFFOLD}-smoke"
    rm -rf "$out"; mkdir -p "$out"
    cp -f "$d/serve-backend.json" "$out/serve-backend.json"
    python "$SWEBENCH_DIR/docker_rollout.py" \
      --model "sglang/$preset" --served-name "$served" --scaffold "$SMOKE_SCAFFOLD" \
      --out "$out" --instances "$SMOKE_INSTANCES" --timeout 1800 --max-empty-streak 30 \
      > "$d/rollout-$SMOKE_SCAFFOLD.log" 2>&1
    rc=$?
    local preds nonempty meta_ok
    preds=$(wc -l < "$out/predictions.jsonl" 2>/dev/null || echo 0)
    nonempty=$(python3 -c "import json,sys; print(sum(1 for l in open('$out/predictions.jsonl') if json.loads(l).get('model_patch','').strip()))" 2>/dev/null || echo 0)
    # meta.json = {"runs": [per-run entries...], "scaffold", "model"}; the isolation
    # facts live on the latest run entry, not at the top level.
    meta_ok=$(python3 -c "import json; m=json.load(open('$out/meta.json'))['runs'][-1]; print('ok' if m.get('serve_mode')=='docker' and m.get('network_mode')=='none' and m.get('context_window',0)>=262144 and m.get('prompt_delivery')=='stdin-file' and m.get('mount_staging')=='neutral' and m.get('git_history')=='reinit-1-commit' else f\"serve_mode={m.get('serve_mode')} network_mode={m.get('network_mode')} context_window={m.get('context_window')} prompt_delivery={m.get('prompt_delivery')} mount_staging={m.get('mount_staging')} git_history={m.get('git_history')}\")" 2>/dev/null || echo "no meta.json")
    local budget trip
    # the `context budget:` line is in each instance's stderr section (logs/<iid>.log), not the lane log
    budget="$(cat "$out"/logs/*.log 2>/dev/null | grep -c 'context budget:')"; budget="${budget:-0}"
    trip="$(grep -c 'CONTEXT-BUDGET TRIPWIRE\|BRIDGE CHECK FAILED' "$d/rollout-$SMOKE_SCAFFOLD.log" 2>/dev/null)"; trip="${trip:-0}"
    if [ "$preds" -ge "$SMOKE_INSTANCES" ] && [ "$nonempty" -ge 1 ] && [ "$meta_ok" = "ok" ] && [ "$trip" = "0" ]; then
      mark rollout PASS "rc=$rc preds=$preds nonempty=$nonempty meta=$meta_ok budget_lines=$budget"
    else
      mark rollout FAIL "rc=$rc preds=$preds nonempty=$nonempty meta=$meta_ok tripwires=$trip ($d/rollout-$SMOKE_SCAFFOLD.log)"; fail=1
    fi
    python "$SWEBENCH_DIR/audit_leakage.py" --run "$out" --require-isolation > "$d/leak-audit.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then mark leak_audit PASS "$(head -1 "$d/leak-audit.log")"; else mark leak_audit FAIL "rc=$rc $(head -1 "$d/leak-audit.log")"; fail=1; fi
  fi

  # 6. stop + drain
  serve_stop "$preset"
  if drain_vram; then mark stop PASS; else mark stop FAIL "VRAM not released"; fail=1; fi

  local verdict=PASS; [ $fail -ne 0 ] && verdict=FAIL
  write_receipt "$preset" "$d/smoke-receipt.json" "$verdict"
  log "=== $preset: $verdict ($d/smoke-receipt.json) ==="
  [ $fail -eq 0 ]
}

overall=0
for preset in "$@"; do
  smoke_preset "$preset" || overall=1
done
echo ""
echo "docker-mode parity smoke: $([ $overall -eq 0 ] && echo PASS || echo FAIL) — receipts under $SMOKE_DIR/<preset>/smoke-receipt.json"
exit $overall
