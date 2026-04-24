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
#   qwen3-ream     Qwen3-30B REAM AWQ (262K, 197 tok/s, fastest)
#   devstral       Devstral-24B AWQ (131K, best all-round)
#   coder-30b      Qwen3-Coder-30B MoE AWQ (16K)
#   qwen35-moe     Qwen3.5-28B MoE REAP CT (262K, DeltaNet)
#   qwen35         Qwen3.5-27B DeltaNet AWQ (32K)
#   qwen36         Qwen3.6-35B-A3B GPTQ-Int4 (262K, DeltaNet + vision, thinking default)
#   gemma4         Gemma 4 26B MoE AWQ (4K)
#   gemma4-31b     Gemma 4 31B Dense AWQ (4K)
#   devstral-long  Devstral-24B AWQ at 217K KV ceiling (single-user long-context)
#
# Note: 80B+ models (coder-next, glm45-air) do NOT fit in 48GB VRAM.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# --- Defaults (overridden by model preset, then by CLI flags) ---
MODEL="${MODEL:-}"
TOKENIZER=""
QUANT="${QUANT:-compressed-tensors}"
DTYPE="float16"
CTX=32768
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"
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
EXTRA_ARGS="${EXTRA_ARGS:-}"

# --- Model presets (tuned for 48GB total VRAM) ---
apply_preset() {
    case "$1" in
        devstral)
            MODEL="${MODEL:-$MODELS_DIR/Devstral-24B-AWQ-Marlin}"
            QUANT="awq_marlin"
            CTX=131072; MEM=0.85; MAX_RUNNING=1; CHUNKED=8192
            CUDA_GRAPH="--cuda-graph-max-bs 1"
            CHAT_TEMPLATE="--chat-template \$SCRIPT_DIR/devstral_chat_template.jinja"
            ;;
        devstral-32k)
            MODEL="${MODEL:-$MODELS_DIR/Devstral-24B-AWQ-Marlin}"
            QUANT="awq_marlin"
            CTX=32768; MEM=0.90; MAX_RUNNING=64; CHUNKED=8192
            ;;
        devstral-long)
            # Single-user long-context preset: pushes KV ceiling from 131K (default)
            # to ~217K tokens at MEM=0.97 + no CUDA graph/overlap/radix cache.
            # Decode plateaus ~56 tok/s past 131K. Not for multi-user.
            MODEL="${MODEL:-$MODELS_DIR/Devstral-24B-AWQ-Marlin}"
            QUANT="awq_marlin"
            CTX=262144; MEM=0.97; MAX_RUNNING=1; CHUNKED=2048
            EXTRA_ARGS="${EXTRA_ARGS} --disable-cuda-graph --disable-overlap-schedule --disable-radix-cache"
            CHAT_TEMPLATE="--chat-template \$SCRIPT_DIR/devstral_chat_template.jinja"
            ;;
        coder-reap)
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-Coder-REAP-25B-A3B-W4A16}"
            QUANT="auto-round"
            CTX=131072; MEM=0.85; MAX_RUNNING=1; CHUNKED=8192
            CUDA_GRAPH="--cuda-graph-max-bs 1"
            ;;
        coder-30b)
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-Coder-30B-A3B-AWQ-Marlin}"
            QUANT="awq_marlin"
            CTX=16384; MEM=0.85; MAX_RUNNING=32; CHUNKED=4096; DECODE_STEPS=8
            ;;
        gemma4)
            MODEL="${MODEL:-$MODELS_DIR/gemma-4-26B-A4B-it-AWQ-4bit}"
            REASONING="--reasoning-parser gemma4"
            # Bumped CTX 4096 → 16384: validate_capabilities.check_thinking sends
            # max_tokens=4096 which overflows 4096 CTX with any prompt.
            CTX=16384; MAX_RUNNING=1; CHUNKED=4096
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            EXTRA_ARGS="${EXTRA_ARGS:-} --enable-multimodal"
            ;;
        gemma4-31b)
            MODEL="${MODEL:-$MODELS_DIR/gemma-4-31B-it-AWQ-4bit}"
            REASONING="--reasoning-parser gemma4"
            CTX=16384; MEM=0.85; MAX_RUNNING=1; CHUNKED=4096
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            EXTRA_ARGS="${EXTRA_ARGS:-} --enable-multimodal"
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
            MODEL="${MODEL:-$MODELS_DIR/Qwen3.5-28B-A3B-REAP-AWQ}"
            QUANT="awq_marlin"
            CTX=262144; MEM=0.80; MAX_RUNNING=4; CHUNKED=8192; DECODE_STEPS=4
            MAMBA_CACHE="--max-mamba-cache-size 4"
            REASONING="--reasoning-parser qwen3"
            CUDA_GRAPH="--disable-cuda-graph --disable-piecewise-cuda-graph"
            ;;
        qwen3-ream)
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-30B-Instruct-2507-REAM-AWQ}"
            QUANT="awq_marlin"
            CTX=262144; MEM=0.85; MAX_RUNNING=32; CHUNKED=4096; DECODE_STEPS=8
            REASONING="--reasoning-parser qwen3"
            ;;
        qwen36)
            # Qwen3.6-35B-A3B-GPTQ-Int4: 256-expert hybrid DeltaNet + gated attn,
            # 3B active, 262K native context, vision + thinking. Uses the same
            # SGLang Qwen3_5MoeForConditionalGeneration handler as Qwen3.5.
            # First-pass is text-only. Requires config flattening before launch:
            #   scripts/quantize/flatten_qwen36_config.py "$MODEL"
            # which promotes text_config.* to top level and sets
            # architectures=[Qwen3_5MoeForCausalLM]. Vision needs a separate
            # fix to the sglang loader that builds Qwen3VLMoeVisionConfig from
            # the dict — deferred until text path is validated.
            MODEL="${MODEL:-$MODELS_DIR/Qwen3.6-35B-A3B-GPTQ-Int4}"
            QUANT="${QUANT:-gptq_marlin}"
            CTX=262144; MEM=0.85; MAX_RUNNING=4; CHUNKED=4096; DECODE_STEPS=4
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
# CLI flags are collected into *_OVERRIDE vars first, then applied AFTER the
# preset so CLI always wins. Preset-only fields (MODEL, QUANT, CHAT_TEMPLATE,
# REASONING, CUDA_GRAPH, MAMBA_CACHE, WARMUP) are left to the preset.
PRESET=""
CTX_OVERRIDE=""
PORT_OVERRIDE=""
MEM_OVERRIDE=""
MAX_RUNNING_OVERRIDE=""
DECODE_STEPS_OVERRIDE=""
CHUNKED_OVERRIDE=""
WATCHDOG_OVERRIDE=""
TP_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            head -17 "$0" | tail -16
            exit 0
            ;;
        --context-length) CTX_OVERRIDE="$2"; shift 2 ;;
        --port) PORT_OVERRIDE="$2"; shift 2 ;;
        --mem-fraction) MEM_OVERRIDE="$2"; shift 2 ;;
        --max-running) MAX_RUNNING_OVERRIDE="$2"; shift 2 ;;
        --decode-steps) DECODE_STEPS_OVERRIDE="$2"; shift 2 ;;
        --chunked-prefill) CHUNKED_OVERRIDE="$2"; shift 2 ;;
        --watchdog) WATCHDOG_OVERRIDE="$2"; shift 2 ;;
        --tp) TP_OVERRIDE="$2"; shift 2 ;;
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

# Apply CLI overrides after preset so CLI always wins.
[[ -n "$CTX_OVERRIDE" ]] && CTX="$CTX_OVERRIDE"
[[ -n "$PORT_OVERRIDE" ]] && PORT="$PORT_OVERRIDE"
[[ -n "$MEM_OVERRIDE" ]] && MEM="$MEM_OVERRIDE"
[[ -n "$MAX_RUNNING_OVERRIDE" ]] && MAX_RUNNING="$MAX_RUNNING_OVERRIDE"
[[ -n "$DECODE_STEPS_OVERRIDE" ]] && DECODE_STEPS="$DECODE_STEPS_OVERRIDE"
[[ -n "$CHUNKED_OVERRIDE" ]] && CHUNKED="$CHUNKED_OVERRIDE"
[[ -n "$WATCHDOG_OVERRIDE" ]] && WATCHDOG="$WATCHDOG_OVERRIDE"
[[ -n "$TP_OVERRIDE" ]] && TP="$TP_OVERRIDE"

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
# EXTRA_ARGS lets callers append/override flags (e.g. --disable-cuda-graph,
# --enable-multimodal) without editing the script. Honor it from env.
[[ -n "${EXTRA_ARGS:-}" ]] && CMD+=(${EXTRA_ARGS})

exec "${CMD[@]}"
