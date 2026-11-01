#!/bin/bash
# profile_decode_depth.sh — torch-profiler capture of single-user decode at a short and a deep context on one
# preset, attributed per step by trace_step_anatomy.py (CPU wall, GPU span, EXPOSED-time table, top kernels).
# One boot; each window first warms the prompt with a 1-token request (prefix lands in the radix cache) so the
# profiled 64-token request is decode-dominated; the prompt is natural text (the Open-Platypus haystack) so the
# server's actual prompt_tokens is recorded and no repetition attractor shapes the decode.
# Port of R9700's driver of the same name (their commit ca8c4aa); attribution uses our exposed-time tool.
#
#   PRESET=qwen38 DEEP_TOKENS=250000 scripts/bench/profile_decode_depth.sh
#   -> benchmarks/profiling/<preset>-decode-profile-<date>/{short,deep}-rank<N>.txt + server.log; traces under /data/tmp
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"; cd "$REPO" || exit 1
source scripts/common.sh; activate_conda; setup_nvidia_env
PRESET="${PRESET:-qwen38}"; PORT="${PORT:-23334}"
SHORT_TOKENS="${SHORT_TOKENS:-128}"; DEEP_TOKENS="${DEEP_TOKENS:-250000}"; DECODE_TOKENS="${DECODE_TOKENS:-64}"
WINDOWS="${WINDOWS:-short deep}"   # which windows to capture (deep alone re-runs into the same OUT dir)
CPT=3.4                             # chars per token, re-calibrated against the served tokenizer before the deep window
HAYSTACK="${HAYSTACK:-$REPO/benchmarks/quality/haystack-platypus.txt}"
STAMP=$(date +%Y-%m-%d); TAG="${TAG:-$PRESET-decode-profile-$STAMP}"
OUT="$REPO/benchmarks/profiling/$TAG"; PROF_ROOT="${PROF_ROOT:-/data/tmp/profile-$TAG}"
mkdir -p "$OUT" "$PROF_ROOT"
log(){ echo "[profile $(date +%H:%M:%S)] $*"; }
health(){ curl -s -o /dev/null -w '%{http_code}' -m 10 "http://127.0.0.1:$PORT/health" 2>/dev/null; }   # curl prints 000 itself on a refused connection
[ "$(health)" = 000 ] || { log "something already answers on :$PORT — refusing"; exit 1; }
pgrep -f 'sglang.launch_serve[r]' >/dev/null && { log "a launch_server process exists — refusing"; exit 1; }
log "boot $PRESET (profiler dir $PROF_ROOT)"
SGLANG_TORCH_PROFILER_DIR="$PROF_ROOT" PORT=$PORT setsid bash "$REPO/scripts/launch.sh" "$PRESET" > "$OUT/server.log" 2>&1 < /dev/null &
LAUNCH_PID=$!
end=$(( $(date +%s) + 1200 )); ok=0
while [ "$(date +%s)" -lt "$end" ]; do [ "$(health)" = 200 ] && { ok=1; break; }; sleep 10; done
[ "$ok" = 1 ] || { log "boot FAILED"; tail -20 "$OUT/server.log"; kill "$LAUNCH_PID" 2>/dev/null; exit 1; }
MODEL=$(curl -s "http://127.0.0.1:$PORT/v1/models" | python -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')
log "server ready, model=$MODEL"
profile_one(){  # <label> <approx prompt tokens>
  local label=$1; local approx=$2; local ptoks
  local bodyf="$PROF_ROOT/$label.body.json"; local warmf="$PROF_ROOT/$label.warm.json"; local respf="$PROF_ROOT/$label.resp.json"
  python - "$approx" "$MODEL" "$DECODE_TOKENS" "$bodyf" "$warmf" "$HAYSTACK" "$CPT" <<'PY'
import sys, json
approx, model, n, bodyf, warmf, hay, cpt = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6], float(sys.argv[7])
text = open(hay).read(); need = int(approx * cpt)
while len(text) < need: text += "\n\n" + text
prompt = text[:need] + "\n\nSummarize the passage above in one paragraph."
body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": n,
        "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
json.dump(body, open(bodyf, "w")); body["max_tokens"] = 1; json.dump(body, open(warmf, "w"))
PY
  # warm with the 1-token twin; on a window overflow the server names the true token count, so rescale and retry
  local attempt
  for attempt in 1 2 3 4; do
    curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' -d @"$warmf" -o "$PROF_ROOT/$label.warmresp.json"
    ptoks=$(python -c 'import sys,json; j=json.load(open(sys.argv[1])); print(j["usage"]["prompt_tokens"] if "usage" in j else "")' "$PROF_ROOT/$label.warmresp.json")
    [ -n "$ptoks" ] && break
    local got; got=$(grep -oE 'input \(([0-9]+) tokens\)' "$PROF_ROOT/$label.warmresp.json" | grep -oE '[0-9]+')
    [ -n "$got" ] || { log "$label: warm request rejected: $(head -c 300 "$PROF_ROOT/$label.warmresp.json")"; return 1; }
    CPT=$(python -c "import sys; print(round(float(sys.argv[1]) * float(sys.argv[2]) / float(sys.argv[3]) * 0.995, 4))" "$CPT" "$approx" "$got")
    log "$label: prompt tokenized to $got > window; chars/token -> $CPT, rebuilding (attempt $attempt)"
    python - "$approx" "$MODEL" "$DECODE_TOKENS" "$bodyf" "$warmf" "$HAYSTACK" "$CPT" <<'PY'
import sys, json
approx, model, n, bodyf, warmf, hay, cpt = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6], float(sys.argv[7])
text = open(hay).read(); need = int(approx * cpt)
while len(text) < need: text += "\n\n" + text
prompt = text[:need] + "\n\nSummarize the passage above in one paragraph."
body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": n,
        "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
json.dump(body, open(bodyf, "w")); body["max_tokens"] = 1; json.dump(body, open(warmf, "w"))
PY
  done
  [ -n "$ptoks" ] || { log "$label: could not fit the prompt into the window"; return 1; }
  log "$label: warm prefill done, actual prompt_tokens=$ptoks"
  curl -s -X POST "http://127.0.0.1:$PORT/start_profile" -H 'content-type: application/json' -d '{}' >/dev/null
  local t0; t0=$(date +%s.%N)
  curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' -d @"$bodyf" -o "$respf"
  local t1; t1=$(date +%s.%N)
  curl -s -X POST "http://127.0.0.1:$PORT/stop_profile" >/dev/null
  python - "$respf" <<'PY'
import sys, json; j = json.load(open(sys.argv[1])); u = j["usage"]
print(f"  usage prompt={u['prompt_tokens']} completion={u['completion_tokens']} finish={j['choices'][0]['finish_reason']}")
PY
  log "$label: profiled request wall $(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}') s"
  sleep 20
  mkdir -p "$PROF_ROOT/$label"; rm -f "$PROF_ROOT/$label"/*.trace.json*   # a failed earlier window leaves empty traces behind
  find "$PROF_ROOT" -maxdepth 1 -type f -name '*.trace.json*' -exec mv {} "$PROF_ROOT/$label/" \;
  local tr r
  for tr in "$PROF_ROOT/$label"/*.trace.json*; do
    r=$(basename "$tr" | grep -oE 'TP-[0-9]+' | tr -d 'TP-'); r=${r:-0}
    { echo "# $PRESET $label — actual prompt_tokens=$ptoks, $DECODE_TOKENS decode steps, trace $(basename "$tr")"; python scripts/bench/trace_step_anatomy.py "$tr" --top 25; } > "$OUT/$label-rank$r.txt" 2>&1
  done
  grep -E 'DECODE|step wall|GPU span|exposed' "$OUT/$label-rank0.txt" | head -6
}
calibrate_cpt(){  # one 1-token request of ~16K approx tokens -> chars/token of the haystack under the served tokenizer
  local f="$PROF_ROOT/cal.json" r="$PROF_ROOT/cal.resp.json"
  python - "$MODEL" "$f" "$HAYSTACK" <<'PY'
import sys, json
model, f, hay = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(hay).read()
while len(text) < 60000: text += "\n\n" + text
json.dump({"model": model, "messages": [{"role": "user", "content": text[:60000]}], "max_tokens": 1, "temperature": 0,
           "chat_template_kwargs": {"enable_thinking": False}}, open(f, "w"))
PY
  curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' -d @"$f" -o "$r"
  CPT=$(python -c 'import sys,json; j=json.load(open(sys.argv[1])); print(round(60000/(j["usage"]["prompt_tokens"]-40),3))' "$r")
  CTX_LEN=$(curl -s "http://127.0.0.1:$PORT/get_server_info" | python -c 'import sys,json; print(json.load(sys.stdin).get("context_length") or 262144)')
  local cap=$(( CTX_LEN - DECODE_TOKENS - 1024 ))
  [ "$DEEP_TOKENS" -gt "$cap" ] && { log "deep: $DEEP_TOKENS capped to $cap (context_length $CTX_LEN)"; DEEP_TOKENS=$cap; }
  log "calibrated chars/token=$CPT (served tokenizer), deep target $DEEP_TOKENS tokens"
}
for w in $WINDOWS; do
  case $w in
    short) profile_one short "$SHORT_TOKENS" ;;
    deep)  calibrate_cpt; profile_one deep "$DEEP_TOKENS" ;;
  esac
done
log "stopping server"
kill "$LAUNCH_PID" 2>/dev/null; pkill -TERM -f 'sglang.launch_serve[r]' 2>/dev/null
for _ in $(seq 1 45); do pgrep -f 'sglang.launch_serve[r]|sglang::schedule[r]' >/dev/null || break; sleep 2; done
log "done -> $OUT"
