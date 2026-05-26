#!/bin/bash
# SGLang setup for 2x RTX 3090
#
# Clones SGLang v0.5.12 and applies the local patches in patches/*.patch
# (idempotent — git apply --check skips already-applied).
# Requires torch 2.11 + transformers 5.6 + flashinfer 0.6.11.post1 +
# compressed-tensors 0.15 (0.5.12 rebase 2026-05-26).
# NB: 0.5.12 folds the serving runtime into the base package (no [srt] extra)
# and adds a mandatory Rust gRPC ext that needs protoc; patch 037 drops that
# ext (we serve over HTTP) so `pip install -e .` works without protoc.
# See patches/README.md for per-patch narratives.
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
SGLANG_TAG="v0.5.12"

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
echo "SGLang:  $SGLANG_TAG + local patches"
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

# Apply local patches (idempotent — skips already-applied)
PATCH_DIR="$REPO_DIR/patches"
if [ -d "$PATCH_DIR" ] && ls "$PATCH_DIR"/*.patch &>/dev/null; then
    echo ""
    echo "Applying patches from $PATCH_DIR..."
    cd "$SGLANG_DIR"
    for p in "$PATCH_DIR"/*.patch; do
        pname="$(basename "$p")"
        if git apply --check "$p" 2>/dev/null; then
            git apply "$p"
            echo "  Applied: $pname"
        else
            echo "  Skipped (already applied or conflict): $pname"
        fi
    done
    cd "$REPO_DIR"
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
    pip install -e .   # 0.5.12 folded srt deps into base; patch 037 dropped the gRPC Rust ext (no protoc needed)

    echo "Upgrading transformers..."
    pip install --no-deps "transformers>=5.0" gguf

    # Eval/validator deps — pillow already comes in via SGLang's [srt] extras,
    # but imageio[ffmpeg] is needed for the validate_capabilities.py video
    # check (12-frame mp4 build via iio.imwrite). Without it the video step
    # silently skips with "no module named imageio" and you lose the modality.
    #
    # swebench is the official harness called by evals/swebench/score_docker.py
    # — without it the score step prints `ModuleNotFoundError: No module named
    # 'swebench'` and writes a 0/300 cell JSON, silently masking the real
    # numbers behind a "harness exited rc=1; trying to summarize anyway" line
    # in score-<scaffold>.log. We hit this on 2026-05-19 after a 25h cycle.
    echo "Installing eval/validator deps..."
    pip install "imageio[ffmpeg]" swebench
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
    print(f'  Device {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')
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
