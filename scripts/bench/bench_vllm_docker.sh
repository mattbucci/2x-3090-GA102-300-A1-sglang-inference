#!/bin/bash
# Benchmark vLLM Docker on 2x RTX 3090
# Matches SGLang bench parameters: 256 in / 256 out, conc sweep 1-32
#
# Usage:
#   ./scripts/bench/bench_vllm_docker.sh [hf_model_id] [label]
#   ./scripts/bench/bench_vllm_docker.sh  # defaults to Devstral from HuggingFace

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
PORT=8000
CONTAINER_NAME="vllm-bench"
VLLM_IMG="vllm/vllm-openai:latest"

MODEL="${1:-mistralai/Devstral-Small-2-24B-Instruct-2512}"
LABEL="${2:-$(basename "$MODEL")}"

echo "=============================================="
echo "vLLM Docker benchmark: $LABEL"
echo "Model: $MODEL"
echo "GPUs: 2x RTX 3090, TP=2"
echo "=============================================="

# Clean up old container
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Start vLLM server
echo "Starting vLLM Docker (TP=2, FP8 KV cache)..."
docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --shm-size 16g \
    -p $PORT:$PORT \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    "$VLLM_IMG" \
    --model "$MODEL" \
    --tensor-parallel-size 2 \
    --port $PORT \
    --trust-remote-code \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len 32768

# Wait for server
echo "Waiting for server..."
for i in $(seq 1 120); do
    curl -s "http://localhost:$PORT/health" > /dev/null 2>&1 && break
    [ "$i" -eq 120 ] && echo "ERROR: Server not ready" && docker logs "$CONTAINER_NAME" | tail -20 && exit 1
    sleep 2
done
echo "Server ready."

# Benchmark
for CONC in 1 4 8 16 32; do
    NP=$((CONC * 4))
    echo ""
    echo "--- Concurrency $CONC ---"
    python -m sglang.bench_serving \
        --backend vllm \
        --base-url "http://localhost:$PORT" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input 256 \
        --random-output 256 \
        --num-prompts $NP \
        --request-rate inf 2>&1 | grep -E "TPOT|throughput|TTFT|E2E"
done

# Cleanup
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
echo ""
echo "Done."
