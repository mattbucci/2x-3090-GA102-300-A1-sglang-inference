#!/bin/bash
# SGLang setup for 2x RTX 3090
#
# Stock SGLang — no patches needed on NVIDIA.
#
# Prerequisites:
#   - NVIDIA drivers + CUDA toolkit installed
#   - Miniforge3/Conda (auto-detected, or set CONDA_BASE)
#
# Usage:
#   ./scripts/setup.sh
#   ./scripts/setup.sh --skip-env   # Skip conda env creation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

SGLANG_REPO="https://github.com/sgl-project/sglang.git"
SGLANG_TAG="v0.5.10"

SKIP_ENV=false
for arg in "$@"; do
    case $arg in
        --skip-env) SKIP_ENV=true ;;
        -h|--help) head -12 "$0" | tail -10; exit 0 ;;
    esac
done

echo "=============================================="
echo "2x RTX 3090 Inference — Setup"
echo "=============================================="
echo "SGLang:  $SGLANG_TAG (stock, no patches)"
echo "Env:     $ENV_NAME"
echo "=============================================="

# Validate
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers."; exit 1
fi
if [ ! -f "$CONDA_BASE/bin/conda" ]; then
    echo "ERROR: Conda not found at $CONDA_BASE"; exit 1
fi

echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# -------------------------------------------------------------------
# Step 1: Clone SGLang
# -------------------------------------------------------------------
echo ""
if [ ! -d "$SGLANG_DIR" ] || [ ! -d "$SGLANG_DIR/.git" ]; then
    echo "[1/3] Cloning SGLang $SGLANG_TAG..."
    rm -rf "$SGLANG_DIR"
    mkdir -p "$(dirname "$SGLANG_DIR")"
    git clone --branch "$SGLANG_TAG" --depth 1 "$SGLANG_REPO" "$SGLANG_DIR"
else
    echo "[1/3] Using existing SGLang source at $SGLANG_DIR"
fi

# -------------------------------------------------------------------
# Step 2: Create conda environment + install packages
# -------------------------------------------------------------------
if [ "$SKIP_ENV" = false ]; then
    echo ""
    echo "[2/3] Creating conda environment: $ENV_NAME"

    init_conda
    conda deactivate 2>/dev/null || true
    if conda env list | grep -q "${ENV_NAME}"; then
        conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
    fi
    conda create -n "$ENV_NAME" python=3.12 -y
    conda activate "$ENV_NAME"

    echo "Installing SGLang from source (CUDA)..."
    cd "$SGLANG_DIR/python"
    pip install -e ".[srt]"

    echo "Upgrading transformers..."
    pip install --no-deps "transformers>=5.0" gguf
else
    echo "[2/3] Skipping conda env creation"
    init_conda
    conda activate "$ENV_NAME"
fi

# -------------------------------------------------------------------
# Step 3: Verify installation
# -------------------------------------------------------------------
echo ""
echo "[3/3] Verifying installation..."

python -c "
import torch
print(f'torch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Devices: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  Device {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)')
print(f'NCCL available: {torch.distributed.is_nccl_available()}')
import sglang
print(f'sglang {sglang.__version__}')
print()
print('All components verified!')
"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next: launch a model server:"
echo "  ./scripts/launch.sh devstral"
echo ""
