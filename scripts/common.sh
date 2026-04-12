#!/bin/bash
# Common configuration for 2x RTX 3090 SGLang inference
#
# Stock SGLang — no patches needed on NVIDIA.
# NCCL for multi-GPU communication.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# --- Conda ---
if [ -z "${CONDA_BASE:-}" ]; then
    if [ -n "${CONDA_EXE:-}" ]; then
        CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
    elif [ -d "$HOME/miniforge3" ]; then
        CONDA_BASE="$HOME/miniforge3"
    elif [ -d "$HOME/mambaforge" ]; then
        CONDA_BASE="$HOME/mambaforge"
    elif [ -d "$HOME/miniconda3" ]; then
        CONDA_BASE="$HOME/miniconda3"
    elif [ -d "$HOME/anaconda3" ]; then
        CONDA_BASE="$HOME/anaconda3"
    elif command -v conda &>/dev/null; then
        CONDA_BASE="$(conda info --base 2>/dev/null)"
    else
        echo "ERROR: Cannot find conda. Set CONDA_BASE=/path/to/conda"
        exit 1
    fi
fi
export CONDA_BASE

ENV_NAME="${ENV_NAME:-sglang}"
SGLANG_DIR="${SGLANG_DIR:-$REPO_DIR/components/sglang}"
MODELS_DIR="${MODELS_DIR:-$HOME/AI/models}"
PORT="${PORT:-23334}"
BASE_URL="http://localhost:${PORT}"

init_conda() {
    eval "$($CONDA_BASE/bin/conda shell.bash hook)"
}

activate_conda() {
    init_conda
    conda activate "$ENV_NAME"
}

# NVIDIA environment setup
setup_nvidia_env() {
    # GPU selection (both 3090s)
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

    # NCCL P2P over PCIe
    export NCCL_P2P_DISABLE=0
    export NCCL_SHM_DISABLE=0

    # NCCL debug — INFO so we can see transport selection
    export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
    export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,P2P}

    # PyTorch memory
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    export TOKENIZERS_PARALLELISM=false
    export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
    export PYTHONWARNINGS="ignore::UserWarning"
}
