#!/bin/bash
# Homegrown REAP wrapper for Qwen3MoE / Qwen3_5MoE / Gemma4MoE / Nemotron-3.
#
# Mirrors the run_ream_qwen3moe.sh pattern: applies the unfused-experts
# monkey-patch, sets up GPU env, kicks off run_reap.py with sensible defaults.
#
# Pure pytorch + transformers; no vLLM dependency (unlike Cerebras's REAP tool
# at github.com/CerebrasResearch/reap which requires vllm==0.10 pinned and
# OOMs on R9700 with Coder-30B-class models due to single-GPU placement).
#
# See scripts/quantize/run_reap.py docstring for the algorithm + saliency
# formula details.
#
# Usage (3090):
#   REAP_ENV=quant ./scripts/quantize/run_reap.sh \
#       --model /data/models/Qwen3-Coder-30B-A3B-BF16 \
#       --save-path /data/models/Qwen3-Coder-30B-A3B-REAP-BF16 \
#       --keep-experts 96
#
# Default env (override via CUDA_VISIBLE_DEVICES, REAP_ENV):
#   CUDA_VISIBLE_DEVICES=0,1   (both 3090s needed for 30B-class BF16 bases)
#   REAP_ENV=quant             (3090 calibration env: transformers + accelerate + datasets)
#
# Ported from the R9700 stack (commit history there). Arch coverage today =
# plain Qwen3MoE only (the unfused-experts patch targets Qwen3MoeExperts). The
# fused Qwen3_5MoeExperts (Qwen3.5/3.6) and Gemma4/Nemotron layouts need the
# patch extended first — see README "MoE coverage matrix" backlog.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default env: the 3090 `quant` env has transformers + accelerate + datasets +
# llmcompressor. Override via REAP_ENV=<env_name>. Derive the conda base from
# CONDA_EXE (or fall back to ~/miniforge3) — no hardcoded operator home path.
REAP_ENV="${REAP_ENV:-quant}"
_CONDA_BASE="${CONDA_EXE%/bin/conda}"
_CONDA_BASE="${_CONDA_BASE:-$HOME/miniforge3}"
REAP_PYTHON="${REAP_PYTHON:-$_CONDA_BASE/envs/$REAP_ENV/bin/python}"

if [[ ! -x "$REAP_PYTHON" ]]; then
    echo "ERROR: REAP env '$REAP_ENV' python not found at $REAP_PYTHON" >&2
    echo "Override with REAP_ENV=<env_name> or REAP_PYTHON=<path>." >&2
    exit 1
fi

# Default to both GPUs (need ≥40GB combined for 30B-class Qwen3MoE; 62 GB+ BF16
# bases like Qwen3.6-35B-A3B spill to CPU offload on 48 GB total VRAM).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[run_reap] env=$REAP_ENV  python=$REAP_PYTHON"
echo "[run_reap] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[run_reap] forwarding args: $*"
echo ""

exec "$REAP_PYTHON" "$SCRIPT_DIR/run_reap.py" "$@"
