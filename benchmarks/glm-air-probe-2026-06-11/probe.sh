#!/bin/bash
# GLM-4.5-Air-REAP-82B-A12B-AWQ-4bit first-prefill probe on FlashInfer @ 2x3090.
# R9700 ask: does first-prefill serve clean on FlashInfer at 2x24GB? (isolates their
# ROCm MATH-SDPA prefill crash). 45.6GB weights on 48GB -> mem 0.97, graphs OFF,
# small ctx; fallback attempt at mem 0.985 / ctx 2048 if attempt 1 OOMs at load.
set -uo pipefail
export ENV_NAME="${ENV_NAME:-sglang-v0512}" SGLANG_DIR="${SGLANG_DIR:-/data/sglang-rebase-v0512}"
export PATH="/home/letsrtfm/miniforge3/envs/$ENV_NAME/bin:$PATH" CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
# raw launch_server bypasses launch.sh -> must apply setup_nvidia_env ourselves
# (TVM_FFI_GPU_BACKEND=cuda is load-bearing: ROCm leftovers on this box flip
# tvm_ffi's JIT to hip and kill any jit_kernel build — the a1/a2 bootfails)
source /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/scripts/common.sh 2>/dev/null || true
setup_nvidia_env 2>/dev/null || export TVM_FFI_GPU_BACKEND=cuda
MODEL=/data/models/hf-3rdparty/GLM-4.5-Air-REAP-82B-A12B-AWQ-4bit
PORT=23334
RLOG=/tmp/glm-probe; mkdir -p "$RLOG"
log(){ echo "[glm $(date +%H:%M:%S)] $*"; }

try_boot(){ # mem ctx tag
  local mem="$1" ctx="$2" tag="$3"
  pkill -KILL -f "sglang.launch_server" 2>/dev/null || true; sleep 8
  log "boot attempt $tag: mem=$mem ctx=$ctx (graphs off, chunked 1024)"
  nohup setsid python -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name glm45-air-reap \
    --tp 2 --port "$PORT" --host 0.0.0.0 \
    --context-length "$ctx" --mem-fraction-static "$mem" \
    --max-running-requests 1 --chunked-prefill-size 1024 \
    --disable-cuda-graph --skip-server-warmup --trust-remote-code \
    > "$RLOG/server-$tag.log" 2>&1 < /dev/null & disown
  local end=$(( $(date +%s) + 1200 ))
  while [ "$(date +%s)" -lt "$end" ]; do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:$PORT/health 2>/dev/null || echo 000)" = "200" ] && return 0
    if ! pgrep -f "sglang.launch_server" >/dev/null; then sleep 5; pgrep -f "sglang.launch_server" >/dev/null || return 1; fi
    sleep 10
  done
  return 1
}

probe(){
  log "prefill probe: ~3K-token prompt, 64 gen tokens"
  python - << 'PYEOF'
import json, time, urllib.request
filler = ("The quick brown fox jumps over the lazy dog. " * 420)  # ~3.7K chars*... aim ~3K tokens
body = {
    "model": "glm45-air-reap",
    "max_tokens": 64, "temperature": 0.3,
    "messages": [{"role": "user", "content": filler + "\n\nIn one word, what animal jumps over the dog above?"}],
}
t0 = time.time()
req = urllib.request.Request("http://127.0.0.1:23334/v1/chat/completions",
                             json.dumps(body).encode(), {"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    c = d["choices"][0]
    print(json.dumps({
        "ok": True, "elapsed_s": round(time.time()-t0, 1),
        "prompt_tokens": d.get("usage", {}).get("prompt_tokens"),
        "finish": c.get("finish_reason"),
        "text": (c["message"].get("content") or "")[:160],
    }))
except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)[:300], "elapsed_s": round(time.time()-t0, 1)}))
PYEOF
}

if try_boot 0.97 4096 a1; then
  log "BOOT OK (mem 0.97 / ctx 4096)"
  grep -E 'max_total_num_tokens|attention.backend|quantization|Marlin|marlin' "$RLOG/server-a1.log" | head -6
  probe
  probe
elif try_boot 0.985 2048 a2; then
  log "BOOT OK on fallback (mem 0.985 / ctx 2048)"
  grep -E 'max_total_num_tokens|attention.backend|quantization|Marlin|marlin' "$RLOG/server-a2.log" | head -6
  probe
else
  log "BOTH BOOTS FAILED — tails:"
  for t in a1 a2; do [ -f "$RLOG/server-$t.log" ] && { echo "--- $t:"; tail -20 "$RLOG/server-$t.log" | grep -vE '^\s*$' | tail -12; }; done
fi
pkill -KILL -f "sglang.launch_server" 2>/dev/null || true
log "probe done (server stopped)"
