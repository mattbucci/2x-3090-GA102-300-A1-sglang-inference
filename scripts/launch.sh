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
# QUANT default is set by the per-preset block below so presets that use
# `${QUANT:-X}` can honor an env override. Presets that hardcode `QUANT="X"`
# ignore env by design (most do — their weight format is fixed).
QUANT="${QUANT:-}"
DTYPE="${DTYPE:-float16}"
# Capture env-provided values BEFORE applying global defaults, so presets
# can do `CTX="${_ENV_CTX:-4096}"` and have the env override take precedence.
# Without this, the global default below (e.g. CTX=32768) would shadow the
# `:-` fallback inside the preset case-block. KV_DTYPE was already on this
# pattern; CTX/MEM/MAX_RUNNING/CHUNKED added 2026-05-01 after the
# qwen3-vl-32b preset edit revealed the gotcha.
_ENV_KV_DTYPE="${KV_DTYPE:-}"
_ENV_CTX="${CTX:-}"
_ENV_MEM="${MEM:-}"
_ENV_MAX_RUNNING="${MAX_RUNNING:-}"
_ENV_CHUNKED="${CHUNKED:-}"
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
            QUANT="${QUANT:-awq_marlin}"
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
            # Coder-REAP-25B-A3B-W4A16. On TP=1 / 24 GB the piecewise CUDA
            # graph capture (58 num-token shapes at CTX=131072 + CHUNKED=8192)
            # finishes cleanly but the detokenizer worker then hangs on the
            # very first prefill — /health stays 503 with "couldn't get a
            # response from detokenizer for last 20 seconds" forever. Adding
            # --skip-server-warmup alone doesn't fix it (the next real request
            # hits the same hang). --disable-piecewise-cuda-graph is the
            # working workaround (~5-10% TPOT cost at long context vs the
            # graph-on path). For TP=2 once the second 3090 returns this can
            # be removed and the headline 46 tok/s @ 131K should restore.
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-Coder-REAP-25B-A3B-W4A16}"
            QUANT="auto-round"
            CTX=131072; MEM=0.85; MAX_RUNNING=1; CHUNKED=8192
            CUDA_GRAPH="--cuda-graph-max-bs 1 --disable-piecewise-cuda-graph"
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            ;;
        coder-30b)
            # Repointed 2026-05-01 from local Apr-17 self-built AWQ-Marlin to the
            # CT→AWQ-Marlin conversion of `mattbucci/Qwen3-Coder-30B-A3B-AWQ`
            # (R9700-team's Apr-29 calibration). Same Marlin format, newer recipe.
            # Source: hf-mattbucci/Qwen3-Coder-30B-A3B-AWQ (CT format).
            # Conversion: scripts/quantize/convert_moe_ct_to_awq.py --group-size 128.
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/Qwen3-Coder-30B-A3B-AWQ-Marlin-from-CT}"
            QUANT="awq_marlin"
            CTX=16384; MEM=0.85; MAX_RUNNING=32; CHUNKED=4096; DECODE_STEPS=8
            ;;
        coder-30b-eval)
            # SWE-bench eval preset: 256K + single-batch CUDA graph, mirrors
            # `coder-reap-25b` so the two models can be benched apples-to-apples.
            # mattbucci/Qwen3-Coder-30B-A3B-AWQ ships compressed-tensors format,
            # not the native AWQ-Marlin layout the local AWQ-Marlin dir uses.
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/Qwen3-Coder-30B-A3B-AWQ}"
            QUANT="compressed-tensors"
            CTX=262144; MEM=0.90; MAX_RUNNING=1; CHUNKED=4096; DECODE_STEPS=8
            CUDA_GRAPH="--cuda-graph-max-bs 1"
            EXTRA_ARGS="${EXTRA_ARGS:-} --disable-piecewise-cuda-graph --tool-call-parser qwen3_coder"
            ;;
        coder-reap-25b)
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ}"
            QUANT="awq_marlin"
            CTX=262144; MEM=0.90; MAX_RUNNING=1; CHUNKED=4096; DECODE_STEPS=8
            CUDA_GRAPH="--cuda-graph-max-bs 1"
            EXTRA_ARGS="${EXTRA_ARGS:-} --disable-piecewise-cuda-graph --tool-call-parser qwen3_coder"
            ;;
        gemma4)
            # Gemma 4 26B MoE AWQ. Same head_dim=256 / Ampere FP8 incompat as
            # gemma4-31b (FlashInfer rejects head_dim=256, triton rejects FP8
            # E4M3 KV on sm_86) — same fix combo: triton attn + KV_DTYPE=auto
            # + disable-cuda-graph. Validator currently 1/4 (boots clean,
            # decode emits <pad> garbage); separate Gemma 4 MoE forward-path
            # bug investigation in patches/README.md "Open investigations".
            MODEL="${MODEL:-$MODELS_DIR/gemma-4-26B-A4B-it-AWQ-4bit}"
            REASONING="--reasoning-parser gemma4"
            KV_DTYPE="${_ENV_KV_DTYPE:-auto}"
            # Bumped CTX 4096 → 16384: validate_capabilities.check_thinking sends
            # max_tokens=4096 which overflows 4096 CTX with any prompt.
            CTX=16384; MAX_RUNNING=1; CHUNKED=4096
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            EXTRA_ARGS="${EXTRA_ARGS:-} --enable-multimodal --attention-backend triton --disable-cuda-graph --disable-piecewise-cuda-graph"
            ;;
        gemma4-31b)
            # Gemma 4 31B Dense AWQ AutoRound (head_dim=256). On 3090 sm_86 the
            # default FlashInfer + FP8 E4M3 KV combo fails twice:
            #   1. FlashInfer rejects head_dim=256 (NUM_MMA_D_QK=32 invalid in
            #      both BatchPrefillWithPaged- and BatchPrefillWithRagged-
            #      KVCacheDispatched), so even --disable-cuda-graph crashes on
            #      first prefill with "Unsupported max_mma_kv: 0".
            #   2. Triton attention works around head_dim but rejects FP8 E4M3
            #      KV on sm_86 (only fp8e4b15 / fp8e5 supported).
            # Combo --attention-backend triton + KV_DTYPE=auto (FP16) is the
            # working pair, validated 2026-05-01 via gemma4-31b-revalidate-Apr29
            # (basic+thinking PASS; vision hallucinates because checkpoint
            # registers as Gemma4ForCausalLM — separate metadata bug, not flags).
            MODEL="${MODEL:-$MODELS_DIR/gemma-4-31B-it-AWQ-4bit}"
            REASONING="--reasoning-parser gemma4"
            KV_DTYPE="${_ENV_KV_DTYPE:-auto}"
            CTX=16384; MEM=0.85; MAX_RUNNING=1; CHUNKED=4096
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            EXTRA_ARGS="${EXTRA_ARGS:-} --enable-multimodal --attention-backend triton --disable-cuda-graph --disable-piecewise-cuda-graph"
            ;;
        qwen3-vl-moe)
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-VL-30B-A3B-Instruct-AWQ-4bit}"
            CTX=16384; MEM=0.85; MAX_RUNNING=32; CHUNKED=4096; DECODE_STEPS=8
            ;;
        qwen3-vl-32b)
            # Qwen3-VL-32B-Instruct AWQ — 21 GB weights (5 shards). The prior
            # MEM=0.85 / MAX_RUNNING=16 / CTX=16384 defaults OOM cold at TP=1
            # on the KV-pool sizing step (post-weight-load: ~3 GB free, KV
            # pool wants 16 × 16384 × ~24 KB ≈ 6 GB). TP=1-viable defaults
            # below. For TP=2 once the second 3090 returns, override via env:
            # `CTX=150000 MEM=0.85 MAX_RUNNING=16 ./scripts/launch.sh qwen3-vl-32b`.
            MODEL="${MODEL:-$MODELS_DIR/Qwen3-VL-32B-Instruct-AWQ-4bit}"
            CTX="${_ENV_CTX:-4096}"
            MEM="${_ENV_MEM:-0.93}"
            MAX_RUNNING="${_ENV_MAX_RUNNING:-1}"
            CHUNKED="${_ENV_CHUNKED:-4096}"
            ;;
        qwen35)
            # Repointed 2026-05-01 (second time): now defaults to
            # mattbucci/Qwen3.6-27B-AWQ (R9700's `balanced_thinking_text` recal
            # 2026-05-01, 3/3 PASS basic+thinking+vision on Ampere TP=1 / 4K).
            # Prior default was mattbucci/Qwen3.5-27B-AWQ (Apr-29) which only
            # passed basic+thinking — vision regressed on the validator-patch.
            # Override with `MODEL=$MODELS_DIR/hf-mattbucci/Qwen3.5-27B-AWQ-4bit-calibrated`
            # if you want the older Qwen3.5 family for A/B testing.
            MODEL="${MODEL:-$MODELS_DIR/hf-mattbucci/Qwen3.6-27B-AWQ}"
            CTX=32768; MEM=0.80; MAX_RUNNING=8; CHUNKED=8192; DECODE_STEPS=32
            MAMBA_CACHE="--max-mamba-cache-size 8"
            CHAT_TEMPLATE="--chat-template \$MODEL/chat_template.jinja"
            REASONING="--reasoning-parser qwen3"
            ;;
        qwen35-moe)
            # Repointed 2026-05-02 from local Apr-14 REAP-AWQ (broken thinking
            # — Open-Platypus calibration stripped <think> tags) to the recal
            # `Qwen3.5-28B-A3B-REAP-AWQ-balanced-thinking-vision` (R9700-port
            # `balanced_thinking_vision` recipe + llava_instruct loader fix,
            # 18.22h CPU GPTQ then CT→AWQ). 3/3 PASS Ampere TP=1 / 8K:
            # basic finish=stop, thinking 3145 tok finish=stop, vision saw
            # red+circle+round. Cerebras's REAP variant (unlike the atbender
            # Qwen3.6-VL-REAP-26B that strips its tower) retained 333 visual
            # tensors so vision is functional after this recal.
            MODEL="${MODEL:-$MODELS_DIR/Qwen3.5-28B-A3B-REAP-AWQ-balanced-thinking-vision}"
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
            # Qwen3.6-35B-A3B AWQ-native (thinking + vision): 256-expert hybrid
            # DeltaNet + gated attn, 3B active, 262K native context. Default path
            # is the R9700-CT-upload converted to native AWQ via
            # scripts/quantize/convert_moe_ct_to_awq.py (BF16 fallback for the
            # [1, H] shared_expert_gate that's not AWQ-packable). Loads as
            # Qwen3_5MoeForConditionalGeneration with patch 019 applied.
            # Validator 4/4, ~33 tok/s short-ctx on 3090 TP=2.
            MODEL="${MODEL:-$MODELS_DIR/Qwen3.6-35B-A3B-AWQ-native-r9700-conv}"
            QUANT="${QUANT:-awq_marlin}"
            # Force bf16 KV: fp8_e4m3 KV produces garbage on this model via
            # Qwen3_5MoeForConditionalGeneration on Ampere. Env override
            # (KV_DTYPE=X on the command line) still wins.
            KV_DTYPE="${_ENV_KV_DTYPE:-auto}"
            CTX=262144; MEM=0.85; MAX_RUNNING=8; CHUNKED=8192; DECODE_STEPS=32
            MAMBA_CACHE="--max-mamba-cache-size 8"
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

# Friendly model name for OpenAI-compatible API consumers (eg. opencode).
# Defaults to the preset name; override with SERVED_NAME=foo.
SERVED_NAME="${SERVED_NAME:-$PRESET}"
[[ -n "$SERVED_NAME" ]] && CMD+=(--served-model-name "$SERVED_NAME")

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
