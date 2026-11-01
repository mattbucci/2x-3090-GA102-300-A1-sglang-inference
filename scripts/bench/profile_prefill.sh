#!/bin/bash
# profile_prefill.sh — torch-profiler capture of a FRESH prefill (cache flushed) at one or more prompt lengths on one
# preset, attributed per EXTEND step by trace_step_anatomy.py (exposed-time table: how much of a 32K / 131K prefill is
# Marlin GEMM vs attention vs the rest). Sizes the natural-text prompt against the served tokenizer like
# profile_decode_depth.sh. Companion to the decode profiles; the W4A8-prefill case rests on the GEMM share here.
#
#   PRESET=qwen38 LENGTHS="32768 131072" scripts/bench/profile_prefill.sh
#   -> benchmarks/profiling/<preset>-prefill-profile-<date>/<len>-rank<N>.txt + server.log; traces under /data/tmp
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"; cd "$REPO" || exit 1
source scripts/common.sh; activate_conda; setup_nvidia_env
PRESET="${PRESET:-qwen38}"; PORT="${PORT:-23334}"; LENGTHS="${LENGTHS:-32768 131072}"
# EXTEND_NEW=N profiles a cache-hit EXTEND instead: the prefix is prefilled un-profiled (and NOT flushed), then a
# request with N new tokens appended is profiled — the agentic tool-result turn. 0 = profile the fresh prefill.
EXTEND_NEW="${EXTEND_NEW:-0}"
HAYSTACK="${HAYSTACK:-$REPO/benchmarks/quality/haystack-platypus.txt}"
STAMP=$(date +%Y-%m-%d); TAG="${TAG:-$PRESET-$([ "${EXTEND_NEW:-0}" -gt 0 ] && echo extend${EXTEND_NEW} || echo prefill)-profile-$STAMP}"
OUT="$REPO/benchmarks/profiling/$TAG"; PROF_ROOT="${PROF_ROOT:-/data/tmp/profile-$TAG}"
mkdir -p "$OUT" "$PROF_ROOT"
log(){ echo "[profile $(date +%H:%M:%S)] $*"; }
health(){ curl -s -o /dev/null -w '%{http_code}' -m 10 "http://127.0.0.1:$PORT/health" 2>/dev/null; }
[ "$(health)" = 000 ] || { log "something already answers on :$PORT — refusing"; exit 1; }
pgrep -f 'sglang.launch_serve[r]' >/dev/null && { log "a launch_server process exists — refusing"; exit 1; }
log "boot $PRESET (profiler dir $PROF_ROOT)"
SGLANG_TORCH_PROFILER_DIR="$PROF_ROOT" PORT=$PORT setsid bash "$REPO/scripts/launch.sh" "$PRESET" > "$OUT/server.log" 2>&1 < /dev/null &
LAUNCH_PID=$!
end=$(( $(date +%s) + 1200 )); ok=0
while [ "$(date +%s)" -lt "$end" ]; do [ "$(health)" = 200 ] && { ok=1; break; }; sleep 10; done
[ "$ok" = 1 ] || { log "boot FAILED"; tail -20 "$OUT/server.log"; kill "$LAUNCH_PID" 2>/dev/null; exit 1; }
MODEL=$(curl -s "http://127.0.0.1:$PORT/v1/models" | python -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')
CTX_LEN=$(curl -s "http://127.0.0.1:$PORT/get_server_info" | python -c 'import sys,json; print(json.load(sys.stdin).get("context_length") or 262144)')
CPT=3.4
build(){ python - "$1" "$MODEL" "$2" "$HAYSTACK" "$CPT" <<'PY'
import sys, json
approx, model, out, hay, cpt = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
text = open(hay).read(); need = int(approx * cpt)
while len(text) < need: text += "\n\n" + text
json.dump({"model": model, "messages": [{"role": "user", "content": text[:need] + "\n\nReply with the single word OK."}],
           "max_tokens": 1, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, open(out, "w"))
PY
}
for L in $LENGTHS; do
  body="$PROF_ROOT/$L.body.json"; resp="$PROF_ROOT/$L.resp.json"; ptoks=""
  for attempt in 1 2 3 4; do
    build "$L" "$body"
    curl -s -X POST "http://127.0.0.1:$PORT/flush_cache" >/dev/null
    # one un-profiled fresh prefill first (kernel autotune / JIT warm), then the profiled one
    curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' -d @"$body" -o "$resp"
    ptoks=$(python -c 'import sys,json; j=json.load(open(sys.argv[1])); print(j["usage"]["prompt_tokens"] if "usage" in j else "")' "$resp")
    [ -n "$ptoks" ] && break
    got=$(grep -oE 'input \(([0-9]+) tokens\)|([0-9]+) tokens from the input' "$resp" | grep -oE '[0-9]+' | head -1)
    [ -n "$got" ] || { log "$L: rejected: $(head -c 200 "$resp")"; break; }
    CPT=$(python -c "import sys; print(round(float(sys.argv[1]) * float(sys.argv[2]) / float(sys.argv[3]) * 0.995, 4))" "$CPT" "$L" "$got")
    log "$L: tokenized to $got; chars/token -> $CPT (attempt $attempt)"
  done
  [ -n "$ptoks" ] || continue
  if [ "$EXTEND_NEW" -gt 0 ]; then
    # keep the prefix cached; append EXTEND_NEW new tokens (distinct text so the suffix itself is never a hit)
    python - "$body" "$EXTEND_NEW" <<'PY'
import sys, json
f, n = sys.argv[1], int(sys.argv[2]); b = json.load(open(f))
b["messages"][0]["content"] += "\n\n[turn] " + ("tool result line. " * max(1, n // 4))[: n * 4]
json.dump(b, open(f, "w"))
PY
    # warm the extend shape un-profiled (kernel JIT / plan) on a different suffix, then re-build the profiled one
    curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' -d @"$body" -o "$resp.warmext"
    python - "$body" "$EXTEND_NEW" <<'PY'
import sys, json
f, n = sys.argv[1], int(sys.argv[2]); b = json.load(open(f))
b["messages"][0]["content"] = b["messages"][0]["content"].rsplit("\n\n[turn] ", 1)[0] + "\n\n[turn] " + ("second result line. " * max(1, n // 4))[: n * 4]
json.dump(b, open(f, "w"))
PY
    log "$L: extend mode — $EXTEND_NEW new tokens on the cached prefix (shape warmed un-profiled: $(python -c 'import sys,json; j=json.load(open(sys.argv[1])); print(j.get("usage",{}).get("prompt_tokens"))' "$resp.warmext") prompt tokens)"
  else
    curl -s -X POST "http://127.0.0.1:$PORT/flush_cache" >/dev/null
  fi
  curl -s -X POST "http://127.0.0.1:$PORT/start_profile" -H 'content-type: application/json' -d '{}' >/dev/null
  t0=$(date +%s.%N)
  curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' -d @"$body" -o "$resp"
  t1=$(date +%s.%N)
  curl -s -X POST "http://127.0.0.1:$PORT/stop_profile" >/dev/null
  log "$L: $([ "$EXTEND_NEW" -gt 0 ] && echo "extend of $EXTEND_NEW on a $ptoks-token cached prefix" || echo "fresh prefill of $ptoks tokens"), wall $(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}') s (profiled)"
  sleep 25
  mkdir -p "$PROF_ROOT/$L"; rm -f "$PROF_ROOT/$L"/*.trace.json*
  find "$PROF_ROOT" -maxdepth 1 -type f -name '*.trace.json*' -exec mv {} "$PROF_ROOT/$L/" \;
  for tr in "$PROF_ROOT/$L"/*.trace.json*; do
    r=$(basename "$tr" | grep -oE 'TP-[0-9]+' | tr -d 'TP-'); r=${r:-0}
    { echo "# $PRESET fresh prefill — actual prompt_tokens=$ptoks, trace $(basename "$tr")"; python scripts/bench/trace_step_anatomy.py "$tr" --top 25 --skip 0; } > "$OUT/$L-rank$r.txt" 2>&1
  done
  grep -E 'EXTEND|step wall|GPU span|exposed' "$OUT/$L-rank0.txt" | head -5
done
log "stopping server"
kill "$LAUNCH_PID" 2>/dev/null; pkill -TERM -f 'sglang.launch_serve[r]' 2>/dev/null
for _ in $(seq 1 45); do pgrep -f 'sglang.launch_serve[r]|sglang::schedule[r]' >/dev/null || break; sleep 2; done
log "done -> $OUT"
