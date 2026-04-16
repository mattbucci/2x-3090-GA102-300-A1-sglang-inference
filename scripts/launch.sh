#!/bin/bash
# Unified model launcher for SGLang on 2x RTX 3090 (48GB total)
#
# Usage:
#   ./scripts/launch.sh <model> [options]
#   ./scripts/launch.sh devstral
#   ./scripts/launch.sh coder-30b --context-length 16384
#   ./scripts/launch.sh gemma4 --port 8000
#   MODEL=/path/to/weights ./scripts/launch.sh devstral
#
# Models:
#   devstral       Devstral-24B AWQ (32K context, best all-round)
#   coder-30b      Qwen3-Coder-30B MoE AWQ (16K, best throughput)
#   gemma4         Gemma 4 26B MoE AWQ (4K)
#   gemma4-31b     Gemma 4 31B Dense AWQ (4K)
#   qwen35         Qwen3.5-27B DeltaNet AWQ (32K)
#
# Note: 80B+ models (coder-next, glm45-air) do NOT fit in 48GB VRAM.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# --- Defaults (overridden by model preset, then by CLI flags) ---
MODEL="${MODEL:-}"
TOKENIZER=""
QUANT="compressed-tensors"
DTYPE="float16"
CTX=32768
KV_DTYPE="fp8_e4m3"
MEM=0.85
MAX_RUNNING=32
CHUNKED=4096
DECODE_STEPS=4
CUDA_GRAPH=""
MAMBA_CACHE=""
CHAT_TEMPLATE=""
REASONING=""
OVERLAP=""
WARMUP=""
WATCHDOG=600
TP=2
EXTRA_ARGS=""

# --- Model presets (tuned for 48GB total VRAM) ---
apply_preset() {
    case "$1" in
        devstral)
            MODEL="${MODEL:-$MODELS_DIR/Devstral-24B-AWQ-4bit}"
            CTX=131072; MEM=0.85; MAX_RUNNING=1; CHUNKED=8192
            CUDA_GRAPH="--cuda-graph-max-bs 1"
            CHAT_TEMPLATE="--chat-template \$SCRIPT_DIR/devstral_chat_template.jinja"
            ;;
        devstral-32k)
            MODEL="${MODEL:-$MODELS_DIR/Devstral-24B-AWQ-4bit}"
            CTX=32768; MEM=0.90; MAX_RUNNING=64; CHUNKED=8192
            ;;
        coder-reap)
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-Coder-REAP-25B-A3B-W4A16}"
            QUANT="auto-round"
            CTX=131072; MEM=0.85; MAX_RUNNING=1; CHUNKED=8192
            CUDA_GRAPH="--cuda-graph-max-bs 1"
            ;;
        coder-30b)
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-Coder-30B-A3B-AWQ-4bit}"
            CTX=16384; MEM=0.85; MAX_RUNNING=32; CHUNKED=4096; DECODE_STEPS=8
            ;;
        gemma4)
            MODEL="${MODEL:-$MODELS_DIR/gemma-4-26B-A4B-it-AWQ-4bit}"
            CTX=4096; MAX_RUNNING=1; CHUNKED=2048
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            ;;
        gemma4-31b)
            MODEL="${MODEL:-$MODELS_DIR/gemma-4-31B-it-AWQ-4bit}"
            CTX=4096; MEM=0.85; MAX_RUNNING=1; CHUNKED=2048
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            ;;
        qwen3-vl-moe)
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-VL-30B-A3B-Instruct-AWQ-4bit}"
            CTX=16384; MEM=0.85; MAX_RUNNING=32; CHUNKED=4096; DECODE_STEPS=8
            ;;
        qwen3-vl-32b)
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-VL-32B-Instruct-AWQ-4bit}"
            CTX=16384; MEM=0.85; MAX_RUNNING=16; CHUNKED=4096
            ;;
        qwen35)
            MODEL="${MODEL:-$MODELS_DIR/Qwen3.5-27B-AWQ-4bit}"
            CTX=32768; MEM=0.80; MAX_RUNNING=8; CHUNKED=8192; DECODE_STEPS=32
            MAMBA_CACHE="--max-mamba-cache-size 8"
            CHAT_TEMPLATE="--chat-template \$MODEL/chat_template.jinja"
            REASONING="--reasoning-parser qwen3"
            ;;
        qwen35-moe)
            MODEL="${MODEL:-$MODELS_DIR/Qwen3.5-28B-A3B-REAP-CT}"
            QUANT="compressed-tensors"
            CTX=262144; MEM=0.80; MAX_RUNNING=4; CHUNKED=8192; DECODE_STEPS=4
            MAMBA_CACHE="--max-mamba-cache-size 4"
            REASONING="--reasoning-parser qwen3"
            CUDA_GRAPH="--disable-cuda-graph --disable-piecewise-cuda-graph"
            ;;
        *)
            echo "Unknown model: $1"
            echo "Run with -h for available models."
            exit 1
            ;;
    esac
}

# --- Parse arguments ---
PRESET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            head -17 "$0" | tail -16
            exit 0
            ;;
        --context-length) CTX="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --mem-fraction) MEM="$2"; shift 2 ;;
        --max-running) MAX_RUNNING="$2"; shift 2 ;;
        --decode-steps) DECODE_STEPS="$2"; shift 2 ;;
        --chunked-prefill) CHUNKED="$2"; shift 2 ;;
        --watchdog) WATCHDOG="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        -*)
            echo "Unknown option: $1"; exit 1 ;;
        *)
            if [[ -z "$PRESET" ]]; then
                PRESET="$1"; shift
            else
                echo "Unexpected argument: $1"; exit 1
            fi
            ;;
    esac
done

if [[ -z "$PRESET" ]]; then
    echo "Usage: $0 <model> [options]"
    echo "Run with -h for available models."
    exit 1
fi

apply_preset "$PRESET"

# Resolve chat template (deferred $MODEL expansion)
CHAT_TEMPLATE=$(eval echo "$CHAT_TEMPLATE")

# --- Setup environment ---
activate_conda
setup_nvidia_env

echo "=============================================="
echo "$PRESET — SGLang TP=$TP on RTX 3090"
echo "PyTorch $(python -c 'import torch; print(torch.__version__)')"
echo "Model:  $MODEL"
echo "Quant:  ${QUANT:-none}  Context: $CTX  Port: $PORT"
echo "=============================================="

# --- Build command ---
CMD=(python -m sglang.launch_server
    --model-path "$MODEL"
    --tensor-parallel-size "$TP"
    --dtype "$DTYPE"
    --kv-cache-dtype "$KV_DTYPE"
    --context-length "$CTX"
    --mem-fraction-static "$MEM"
    --max-running-requests "$MAX_RUNNING"
    --chunked-prefill-size "$CHUNKED"
    --num-continuous-decode-steps "$DECODE_STEPS"
    --trust-remote-code
    --watchdog-timeout "$WATCHDOG"
    --port "$PORT"
    --host 0.0.0.0
    --enable-metrics
    --disable-custom-all-reduce
)

[[ -n "$QUANT" ]] && CMD+=(--quantization "$QUANT")
[[ -n "$TOKENIZER" ]] && CMD+=($TOKENIZER)
[[ -n "$MAMBA_CACHE" ]] && CMD+=($MAMBA_CACHE)
[[ -n "$CHAT_TEMPLATE" ]] && CMD+=($CHAT_TEMPLATE)
[[ -n "$REASONING" ]] && CMD+=($REASONING)
[[ -n "$WARMUP" ]] && CMD+=($WARMUP)
[[ -n "$OVERLAP" ]] && CMD+=($OVERLAP)
[[ -n "$CUDA_GRAPH" ]] && CMD+=($CUDA_GRAPH)

exec "${CMD[@]}"
