#!/bin/bash
# Benchmark llama.cpp CUDA on 2x RTX 3090 (NVLink layer split)
# Matches SGLang bench parameters: 256 in / 256 out, conc sweep 1-32
#
# Usage:
#   ./scripts/bench/bench_llamacpp.sh <model.gguf> [label]
#   ./scripts/bench/bench_llamacpp.sh  # defaults to Devstral Q4_K_M

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LLAMA_DIR="$HOME/AI/llama.cpp"
LLAMA_BENCH="$LLAMA_DIR/build/bin/llama-bench"
LLAMA_SERVER="$LLAMA_DIR/build/bin/llama-server"
MODELS_DIR="$HOME/AI/models"
PORT=8080
CTX=131072  # 128K context

MODEL="${1:-$MODELS_DIR/Devstral-Small-2-24B-GGUF/Devstral-Small-2-24B-Q4_K_M.gguf}"
LABEL="${2:-$(basename "$MODEL" .gguf)}"

echo "=============================================="
echo "llama.cpp CUDA benchmark: $LABEL"
echo "Model: $MODEL"
echo "Context: $CTX tokens (128K)"
echo "GPUs: 2x RTX 3090, NVLink layer split"
echo "=============================================="

if [ ! -f "$LLAMA_BENCH" ]; then
    echo "ERROR: llama-bench not found at $LLAMA_BENCH"
    echo "Build llama.cpp: cd ~/AI/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j"
    exit 1
fi

# --- Phase 1: Raw kernel perf with llama-bench (2-GPU layer split) ---
echo ""
echo "=== Phase 1: llama-bench (raw performance, 2-GPU layer split) ==="
$LLAMA_BENCH \
    -m "$MODEL" \
    -ngl 99 \
    -sm layer \
    -ts 1,1 \
    -p 256 -n 256 \
    -r 3 \
    -o md 2>&1

echo ""
echo "=== Phase 1b: llama-bench with 4K prompt (prefill stress) ==="
$LLAMA_BENCH \
    -m "$MODEL" \
    -ngl 99 \
    -sm layer \
    -ts 1,1 \
    -p 4096 -n 256 \
    -r 3 \
    -o md 2>&1

# --- Phase 2: Server-based concurrent benchmark ---
echo ""
echo "=== Phase 2: Server concurrent throughput (256 in / 256 out) ==="

pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
sleep 2

echo "Starting llama-server: 2-GPU CUDA, ctx=$CTX, parallel=32, kv-cache=q8_0..."
$LLAMA_SERVER \
    -m "$MODEL" \
    --host 0.0.0.0 --port $PORT \
    -ngl 99 --split-mode layer \
    -ts 1,1 \
    -c $CTX \
    --parallel 32 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    > /dev/null 2>&1 &

# Wait for server
for i in $(seq 1 60); do
    curl -s "http://localhost:$PORT/health" > /dev/null 2>&1 && break
    [ "$i" -eq 60 ] && echo "ERROR: Server not ready" && exit 1
    sleep 2
done
echo "Server ready."

# Use sglang.bench_serving with vllm backend (OpenAI-compatible)
for CONC in 1 4 8 16 32; do
    NP=$((CONC * 4))
    echo ""
    echo "--- Concurrency $CONC ---"
    python -m sglang.bench_serving \
        --backend vllm \
        --base-url "http://localhost:$PORT" \
        --model "$LABEL" \
        --dataset-name random \
        --random-input 256 \
        --random-output 256 \
        --num-prompts $NP \
        --request-rate inf 2>&1 | grep -E "TPOT|throughput|TTFT|E2E"
done

pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
echo ""
echo "Done."
