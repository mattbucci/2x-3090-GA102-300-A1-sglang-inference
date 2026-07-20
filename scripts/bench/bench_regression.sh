#!/bin/bash
# Depth-verified throughput regression tripwire for the 2x RTX 3090 rig.
#
# Instrument: scripts/bench/bench_long_context.py (fleet invariants: single-user
# M=1, --random-range-ratio 1, server-verified actual_input_tokens, degenerate
# points flagged invalid/depth_shortfall). Three depths per preset: 1024, 32768,
# and "deep" (262144 self-capped by the live server's max_total_num_tokens +
# context_length — devstral lands ~196K; that IS its baseline depth).
#
# Baselines: benchmarks/baselines.json, fleet-standard SCHEMA v2 —
#   { "_meta": {schema:2, instrument, stack, hardware, output_tokens, saved},
#     "<preset>": { "1024": {tok_per_sec,tpot_ms,ttft_ms,actual_input_tokens},
#                   "32768": {...},
#                   "deep": {..., "label": <actual capped depth>} } }
# Gate: tok_per_sec drop >THRESHOLD% (default 10) at any depth => REGRESSION,
# exit 1. ttft_ms drift is reported WARN-only (prefill signal, no gate).
# Flagged points (invalid / depth_shortfall / actual <95% of label) are never
# saved and never compared — they are listed as NOT-COMPARED so a silently
# missing deep point can't masquerade as a PASS.
#
# Usage:
#   scripts/bench/bench_regression.sh <preset>          # bench live server (must
#                                                       #   be serving <preset>),
#                                                       #   compare vs baseline
#   BASELINE=save scripts/bench/bench_regression.sh <preset>   # save instead
#   scripts/bench/bench_regression.sh arm [preset...]   # serve each preset itself
#                                                       #   (launch.sh AS SHIPPED),
#                                                       #   bench, stop, next.
#                                                       #   Default list: the 7
#                                                       #   tripwire presets.
#   scripts/bench/bench_regression.sh check <preset> <run.json>  # compare/save
#                                                       #   from an existing run
#                                                       #   JSON (no GPU; used by
#                                                       #   the tripwire self-test)
#
# Ops rules:
#   - Pre-FLIP gate: compare-PASS on all 7 presets before any stack flip commit
#     (see patches/README.md rebase checklist).
#   - Re-run affected presets after any patch touching the serving hot path.
#   - BASELINE=save is a DELIBERATE act: only after a receipted WIN or a verified
#     flip — never automatically on a PASS. Save merges per-preset (idempotent;
#     a kernel-BUG reboot mid-arm resumes with completed presets skipped).
#   - Rule 1/2: arm mode refuses to start while calibration / docker rollout /
#     scoring is running. Bench the preset AS SHIPPED via launch.sh — never a
#     hand-assembled serve command (the b09882f lesson). The tokenizer is read
#     from the live server's /get_model_info model_path, so it cannot go stale.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
# common.sh redefines SCRIPT_DIR (burned the first arming run 2026-07-19: the
# bench path silently became scripts/ instead of scripts/bench/) — pin the
# instrument path before sourcing.
PYBENCH="$SCRIPT_DIR/bench_long_context.py"
source "$REPO_DIR/scripts/common.sh"
activate_conda
setup_nvidia_env
[ -f "$PYBENCH" ] || { echo "FATAL: instrument missing at $PYBENCH"; exit 2; }

BASELINES="${BASELINES:-$REPO_DIR/benchmarks/baselines.json}"
RUNS_DIR="$REPO_DIR/benchmarks/regression"
PORT="${PORT:-23334}"
BASE_URL="http://localhost:$PORT"
THRESHOLD="${THRESHOLD:-10}"
SAVE_BASELINE="${BASELINE:-}"
DATE="$(date +%F)"
STACK_TAG="${STACK_TAG:-sglang-v0.5.15}"

# One preset per distinct kernel/arch path (7): fused Qwen3.5/3.6-MoE AWQ-Marlin,
# dense Marlin, Qwen3Moe native-AWQ, DeltaNet hybrid, group-32 AWQ fallback path,
# Mamba2-hybrid moe_wna16 (the b09882f model), dense Mistral (tokenizer-backend
# canary family, patches 054/057).
TRIPWIRE_PRESETS=(qwen36 qwen36-dense coder-30b qwen35-moe gemma4 nemotron3-omni devstral)

mkdir -p "$RUNS_DIR"

server_served_name() {
    curl -s -m 5 "$BASE_URL/v1/models" | python3 -c \
        "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null
}

server_model_path() {
    curl -s -m 5 "$BASE_URL/get_model_info" | python3 -c \
        "import json,sys; print(json.load(sys.stdin)['model_path'])" 2>/dev/null
}

# process_run <preset> <run.json>  — schema-v2 compare (or save w/ BASELINE=save)
# from a bench_long_context.py output file. Returns 1 on REGRESSION.
process_run() {
    local preset="$1" run_json="$2"
    MODE="${SAVE_BASELINE:+save}" PRESET="$preset" RUN_JSON="$run_json" \
    BASELINES_PATH="$BASELINES" THRESHOLD_PCT="$THRESHOLD" STACK="$STACK_TAG" \
    python3 - <<'PYEOF'
import json, os, sys

preset = os.environ["PRESET"]
run_json = os.environ["RUN_JSON"]
baselines_path = os.environ["BASELINES_PATH"]
threshold = float(os.environ["THRESHOLD_PCT"])
mode = os.environ.get("MODE") or "compare"
stack = os.environ["STACK"]

payload = json.load(open(run_json))
sweep = payload.get("context_sweep", [])

# Map the sweep to schema-v2 depth keys: 1024, 32768, deep(=largest other point).
points, deep_candidates = {}, []
for e in sweep:
    if e.get("error"):
        continue
    rec = {k: e[k] for k in ("tok_per_sec", "tpot_ms", "ttft_ms", "actual_input_tokens") if k in e}
    rec["_flags"] = [f for f in ("invalid", "depth_shortfall") if e.get(f)]
    ctx = e["context"]
    if ctx == 1024:
        points["1024"] = rec
    elif ctx == 32768:
        points["32768"] = rec
    else:
        deep_candidates.append((ctx, rec))
if deep_candidates:
    ctx, rec = max(deep_candidates, key=lambda t: t[0])
    rec["label"] = ctx
    points["deep"] = rec

def label_of(key, rec):
    return rec.get("label") or int(key)

usable, refused = {}, []
for k, rec in points.items():
    if rec["_flags"]:
        refused.append((k, "+".join(rec["_flags"])))
    elif rec.get("actual_input_tokens", 0) < 0.95 * label_of(k, rec):
        refused.append((k, "actual<95%"))
    elif "tok_per_sec" not in rec:
        refused.append((k, "no tok_per_sec"))
    else:
        usable[k] = rec

for k, why in refused:
    print(f"  {preset}/{k}: NOT-COMPARED ({why}) — rerun this point")

baselines = {}
if os.path.exists(baselines_path):
    try:
        baselines = json.load(open(baselines_path))
    except Exception:
        pass

if mode == "save":
    if not usable:
        print(f"  {preset}: REFUSING save — no usable points")
        sys.exit(2)
    # Merge per-depth: a rerun whose deep point came back flagged must not
    # silently drop the previously-armed deep baseline.
    entry = dict(baselines.get(preset) or {})
    for k, rec in usable.items():
        entry[k] = {kk: vv for kk, vv in rec.items() if not kk.startswith("_")}
    kept = sorted(k for k in entry if k not in usable)
    if kept:
        print(f"  {preset}: kept prior baseline for {kept} (not usable in this run)")
    baselines[preset] = entry
    meta = baselines.setdefault("_meta", {})
    meta.update({
        "schema": 2,
        "instrument": "scripts/bench/bench_long_context.py",
        "stack": stack,
        "hardware": "2x RTX 3090 TP=2",
        "output_tokens": payload.get("output_tokens", 100),
        "saved": __import__("time").strftime("%Y-%m-%d"),
    })
    with open(baselines_path, "w") as f:
        json.dump(baselines, f, indent=2, sort_keys=True)
    depths = ", ".join(f"{k}={v['tok_per_sec']} tok/s" for k, v in sorted(usable.items()))
    print(f"  {preset}: baseline SAVED ({depths})")
    sys.exit(0)

base = baselines.get(preset)
if not base:
    print(f"  {preset}: no baseline (run with BASELINE=save to arm)")
    sys.exit(0)

failed = False
for k in ("1024", "32768", "deep"):
    b, c = base.get(k), usable.get(k)
    if not b:
        continue
    if not c:
        if not any(rk == k for rk, _ in refused):
            print(f"  {preset}/{k}: NOT-COMPARED (missing from run) — rerun this point")
        continue
    pct = (c["tok_per_sec"] - b["tok_per_sec"]) / b["tok_per_sec"] * 100
    status = "REGRESSION" if pct < -threshold else "ok"
    if pct < -threshold:
        failed = True
    lbl = f"{k}({label_of(k, c)})" if k == "deep" else k
    print(f"  {preset}/{lbl}: {b['tok_per_sec']} -> {c['tok_per_sec']} tok/s ({pct:+.1f}%) [{status}]")
    if b.get("ttft_ms") and c.get("ttft_ms"):
        tp = (c["ttft_ms"] - b["ttft_ms"]) / b["ttft_ms"] * 100
        if tp > threshold:
            print(f"  {preset}/{lbl}: WARN ttft {b['ttft_ms']:.0f} -> {c['ttft_ms']:.0f} ms ({tp:+.1f}%) — prefill drift (no gate)")

if failed:
    print(f"  *** {preset}: PERFORMANCE REGRESSION DETECTED ***")
    sys.exit(1)
print(f"  {preset}: within threshold.")
PYEOF
}

# bench_live <preset> — bench the already-running server (must serve <preset>).
bench_live() {
    local preset="$1"
    local served; served="$(server_served_name)"
    if [ -z "$served" ]; then
        echo "ERROR: no server on $BASE_URL"; return 2
    fi
    if [ "$served" != "$preset" ]; then
        echo "ERROR: server is serving '$served', not '$preset' — bench-as-shipped requires the matching preset"
        return 2
    fi
    local model_path; model_path="$(server_model_path)"
    if [ -z "$model_path" ]; then
        echo "ERROR: could not read model_path from /get_model_info (needed for --tokenizer)"; return 2
    fi
    local out="$RUNS_DIR/${preset}-${DATE}.json"
    echo "=== $preset (tokenizer: $model_path) ==="
    python "$PYBENCH" \
        --port "$PORT" --name "$preset" \
        --contexts 1024 32768 262144 \
        --output-tokens 100 \
        --tokenizer "$model_path" \
        --output "$out" || return 2
    process_run "$preset" "$out"
}

wait_health() {   # up to 30 min — cold-cache TP=2 loads (patch 049 territory)
    for _ in $(seq 1 360); do
        curl -s -m 2 "$BASE_URL/health" >/dev/null 2>&1 && return 0
        sleep 5
    done
    return 1
}

stop_server_and_drain() {
    pkill -f '[s]glang.launch_server' 2>/dev/null
    for _ in $(seq 1 36); do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
        [ "${used:-9999}" -lt 1000 ] && return 0
        sleep 5
    done
    echo "WARN: VRAM not drained after 180s"
    return 0
}

run_valid() {   # existing run JSON with >=1 clean point? (arm-resume skip test)
    python3 - "$1" <<'PYEOF'
import json, sys
try:
    p = json.load(open(sys.argv[1]))
    ok = any(("tok_per_sec" in e and not e.get("invalid") and not e.get("depth_shortfall"))
             for e in p.get("context_sweep", []))
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PYEOF
}

arm() {
    local presets=("$@")
    [ ${#presets[@]} -eq 0 ] && presets=("${TRIPWIRE_PRESETS[@]}")

    # Rule 1/2 preflight: no calibration, rollout, or scoring alongside serving.
    if pgrep -af 'run_model_cycle|run_all_cycles|docker_rollout|score_docker|quantize_' | grep -v grep | grep -q .; then
        echo "ERROR: eval/calibration workload running — Rule 1/2 forbids concurrent arm. Aborting."
        exit 2
    fi
    if curl -s -m 2 "$BASE_URL/health" >/dev/null 2>&1; then
        echo "ERROR: a server is already on $BASE_URL — stop it first (arm launches its own)."
        exit 2
    fi

    local overall=0
    for preset in "${presets[@]}"; do
        local out="$RUNS_DIR/${preset}-${DATE}.json"
        if [ -n "$SAVE_BASELINE" ] && [ -f "$out" ] && run_valid "$out"; then
            echo "=== $preset: run JSON exists for $DATE — processing without re-bench (resume) ==="
            process_run "$preset" "$out" || overall=1
            continue
        fi
        echo ""
        echo "=== arm: launching $preset ==="
        "$REPO_DIR/scripts/launch.sh" "$preset" > "$RUNS_DIR/serve-${preset}-${DATE}.log" 2>&1 &
        local lpid=$!
        if ! wait_health; then
            echo "ERROR: $preset did not become healthy in 30 min — see $RUNS_DIR/serve-${preset}-${DATE}.log"
            kill "$lpid" 2>/dev/null; stop_server_and_drain
            overall=1
            continue
        fi
        bench_live "$preset" || overall=1
        stop_server_and_drain
        wait "$lpid" 2>/dev/null
    done
    return $overall
}

case "${1:-}" in
    arm)
        shift
        echo "============================================"
        echo "Regression tripwire — ARM mode (mode: ${SAVE_BASELINE:+save}${SAVE_BASELINE:-compare}, threshold ${THRESHOLD}%)"
        echo "============================================"
        if arm "$@"; then
            echo ""; echo "RESULT: PASS"
        else
            echo ""; echo "RESULT: REGRESSION/FAILURE DETECTED"; exit 1
        fi
        ;;
    check)
        # internal/self-test: process an existing run JSON, no server needed
        [ $# -ge 3 ] || { echo "usage: bench_regression.sh check <preset> <run.json>"; exit 2; }
        process_run "$2" "$3"
        ;;
    "")
        echo "usage: bench_regression.sh <preset> | arm [preset...] | check <preset> <run.json>"
        echo "presets: ${TRIPWIRE_PRESETS[*]}"
        exit 2
        ;;
    *)
        preset="$1"
        echo "============================================"
        echo "Regression tripwire — $preset (mode: ${SAVE_BASELINE:+save}${SAVE_BASELINE:-compare}, threshold ${THRESHOLD}%)"
        echo "============================================"
        if bench_live "$preset"; then
            echo ""; echo "RESULT: PASS"
        else
            echo ""; echo "RESULT: REGRESSION/FAILURE DETECTED"; exit 1
        fi
        ;;
esac
