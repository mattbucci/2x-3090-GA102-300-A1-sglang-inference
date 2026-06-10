#!/bin/bash
# b4_ngram.sh — sprint-2 B4: NGRAM speculative decoding trial on qwen36-dense.
# NGRAM = trie-based draft, NO draft model: zero extra VRAM, no draft-context
# cap — dodges both killers that closed EAGLE3/DFlash at 24 GB (2026-05-31).
# Arms: ctl (preset no-spec) vs ngram (--speculative-algorithm NGRAM).
# Probes per arm:
#   (a) bench_long_context @ {1K, 41K, 131K} filler  -> pessimistic floor
#   (b) code-rewrite probe (the agentic edit pattern: model re-emits given
#       code with small changes -> n-gram heaven) -> realistic ceiling
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="${MODELS_DIR:-/home/letsrtfm/AI/models}"
export ENV_NAME="${ENV_NAME:-sglang-v0512}"
export SGLANG_DIR="${SGLANG_DIR:-/data/sglang-rebase-v0512}"
export PATH="/home/letsrtfm/miniforge3/envs/$ENV_NAME/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
source "$REPO/scripts/common.sh" 2>/dev/null || true
cd "$REPO" || exit 1

PRESET="${PRESET:-qwen36-dense}"
PORT=23334
case "$PRESET" in
  qwen3-ream) TOK="$MODELS_DIR/Qwen3-30B-Instruct-2507-REAM-AWQ" ;;
  *)          TOK="$MODELS_DIR/hf-mattbucci/Qwen3.6-27B-AWQ" ;;
esac
OUT="$REPO/benchmarks/sprint-2026-06-kv-decode/B4-ngram-$PRESET"
LOGROOT="/tmp/swa-sweep-logs/B4-$PRESET"
mkdir -p "$OUT" "$LOGROOT"
log(){ echo "[b4 $(date +%H:%M:%S)] $*"; }
stop_server(){ pkill -KILL -f "sglang.launch_server" 2>/dev/null || true; sleep 8; }

code_probe(){ # $1 out.json — rewrite-style code gens, tok/s client-measured
  python - "$1" <<'PY'
import json, sys, time, urllib.request
SNIPPET = open("scripts/bench/bench_long_context.py").read()[:6000]
PROMPTS = [
  "Here is a Python file:\n```python\n" + SNIPPET + "\n```\nRewrite it verbatim but rename the variable `args` to `cli_args` everywhere. Output only the code.",
  "Here is a Python file:\n```python\n" + SNIPPET + "\n```\nAdd a docstring to every function, keeping all code identical otherwise. Output only the code.",
  "Write a Python CLI that wraps `nvidia-smi --query-gpu` with argparse flags for fields, format, and loop interval. ~120 lines.",
  "Here is a Python file:\n```python\n" + SNIPPET + "\n```\nProduce a unified diff that adds a --quiet flag suppressing all prints.",
]
rows=[]
for i,p in enumerate(PROMPTS):
    body=json.dumps({"model":"x","prompt":p,"max_tokens":700,"temperature":0.3}).encode()
    req=urllib.request.Request("http://127.0.0.1:23334/v1/completions", body, {"Content-Type":"application/json"})
    t0=time.time(); r=json.load(urllib.request.urlopen(req, timeout=600)); dt=time.time()-t0
    ct=r["usage"]["completion_tokens"]
    rows.append({"prompt":i,"completion_tokens":ct,"elapsed_s":round(dt,2),"tok_per_sec":round(ct/dt,1)})
    print(f"  probe {i}: {ct} tok in {dt:.1f}s = {ct/dt:.1f} tok/s")
agg=sum(r["completion_tokens"] for r in rows)/sum(r["elapsed_s"] for r in rows)
json.dump({"rows":rows,"agg_tok_per_sec":round(agg,1)}, open(sys.argv[1],"w"), indent=1)
print(f"  aggregate: {agg:.1f} tok/s")
PY
}

arm(){ # name extra
  local name="$1"
  local extra="$2"
  local RLOG="$LOGROOT/$name"
  mkdir -p "$RLOG"
  log "===== arm $name ====="
  stop_server
  EXTRA_ARGS="$extra" nohup setsid bash "$REPO/scripts/launch.sh" "$PRESET" \
    --context-length 262144 > "$RLOG/server.log" 2>&1 < /dev/null &
  disown
  local end=$(( $(date +%s) + 900 )) ok=0
  while [ "$(date +%s)" -lt "$end" ]; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:$PORT/health 2>/dev/null || echo 000)" = "200" ] && { ok=1; break; }
    sleep 10
  done
  [ "$ok" = "1" ] || { log "boot FAILED"; tail -30 "$RLOG/server.log" > "$OUT/$name.bootfail.log"; stop_server; return 1; }
  log "  code-rewrite probe"
  code_probe "$OUT/$name.code.json" >> "$RLOG/probe.log" 2>&1 && tail -2 "$RLOG/probe.log"
  log "  filler decode floor @ {1K, 41K, 131K}"
  python "$REPO/scripts/bench/bench_long_context.py" --port $PORT \
    --name "$PRESET-$name" --contexts 1024 40960 131072 \
    --output "$OUT/$name.decode.json" --tokenizer "$TOK" > "$RLOG/decode.log" 2>&1
  log "  decode rc=$?"
  grep -ioE "accept.{0,60}" "$RLOG/server.log" | tail -5 > "$OUT/$name.accept.log" || true
  stop_server
}
# Arms selectable via ARMS env. ngram on the DeltaNet hybrid needs the
# mamba extra_buffer strategy + SPEC_V2 (same combo as DFlash; radix-cache
# incompat with no_buffer otherwise — first run boot-failed on this).
ARMS="${ARMS:-ctl ngram}"
for A in $ARMS; do
  case "$A" in
    ctl)    arm ctl "" ;;
    ngram)  SGLANG_ENABLE_SPEC_V2=1 arm ngram "--speculative-algorithm NGRAM --mamba-scheduler-strategy extra_buffer" ;;
    fusion) arm fusion "--enable-flashinfer-allreduce-fusion" ;;
    ngram_plain) arm ngram "--speculative-algorithm NGRAM" ;;  # non-hybrid presets: no mamba flags
  esac
done
log "done"
