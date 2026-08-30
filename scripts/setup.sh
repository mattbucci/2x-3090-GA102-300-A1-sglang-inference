#!/bin/bash
# SGLang setup for 2x RTX 3090
#
# Clones SGLang v0.5.18 and applies the local patches in patches/*.patch
# (idempotent — git apply --check skips already-applied). 28 patches; verified
# byte-identical to the live tree by the 3-gate pristine replay — scripted,
# scripts/test_patch_gates.sh (flipped from v0.5.17 2026-08-29; 24 applied
# clean, 004 + 053 re-ported to their moved anchors — model_config.py
# restructure, EVS predicate now in managers/mm_schedule.py; 061 added for the
# v0.5.18 gemma4-unified lm_head_is_tied boot crash).
# Requires transformers 5.12.1 + torch 2.13.0 + flashinfer 0.6.17 [cu13] +
# sglang-kernel 0.4.6.post1 + xgrammar 0.2.1 (env sglang-v0518 also has
# librosa + accelerate for the Parakeet audio path).
# NB: the serving runtime is in the base package (no [srt] extra) and upstream
# adds a mandatory Rust gRPC ext that needs protoc; patch 037 drops that
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
# Default stack = v0.5.18 (flipped 2026-08-29). The retained v0.5.17 tree
# (/data/sglang-rebase-v0517, env sglang-v0517) still serves via ENV_NAME/SGLANG_DIR
# overrides; to REBUILD an older stack from scratch, revert the flip commit (restores
# the v0.5.17 patch set + this tag) or override SGLANG_TAG + PATCH_DIR + ENV_NAME.
SGLANG_TAG="${SGLANG_TAG:-v0.5.18}"

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
# GPU_FREE_SETUP=1 is for container image builds: no GPU, driver, or nvidia-smi
# present at build time (the CUDA runtime is injected by the container toolkit
# at docker run). Bare-metal setups keep the hard driver requirement.
GPU_FREE_SETUP="${GPU_FREE_SETUP:-0}"
if [ "$GPU_FREE_SETUP" != "1" ] && ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers (or GPU_FREE_SETUP=1 for image builds)."; exit 1
fi
if [ ! -f "$CONDA_BASE/bin/conda" ]; then
    echo "ERROR: Conda not found at $CONDA_BASE"; exit 1
fi

if [ "$GPU_FREE_SETUP" != "1" ]; then
    echo ""
    echo "GPU info:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

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
    # Stale-workspace guard (R9700 cross-team 2026-06-10): applying v0.5.13
    # patches onto a checkout at a different tag "succeeds" as a wall of
    # silent "Skipped (conflict)" lines. Abort instead.
    _have_tag="$(git -C "$SGLANG_DIR" describe --tags --exact-match 2>/dev/null || echo unknown)"
    if [ "$_have_tag" != "$SGLANG_TAG" ]; then
        echo "ERROR: $SGLANG_DIR is at '$_have_tag', expected $SGLANG_TAG."
        echo "       Point SGLANG_DIR at a $SGLANG_TAG checkout (live tree:"
        echo "       /data/sglang-rebase-v0518) or remove the stale dir to re-clone."
        exit 1
    fi
fi
# Optional commit pin (image builds set it): a tag can be re-pointed upstream;
# the commit hash cannot.
if [ -n "${SGLANG_COMMIT:-}" ]; then
    _have_commit="$(git -C "$SGLANG_DIR" rev-parse HEAD)"
    if [ "$_have_commit" != "$SGLANG_COMMIT" ]; then
        echo "ERROR: $SGLANG_TAG resolves to $_have_commit, expected $SGLANG_COMMIT (tag moved upstream?)"
        exit 1
    fi
fi

# Apply local patches (idempotent — skips already-applied)
PATCH_DIR="${PATCH_DIR:-$REPO_DIR/patches}"
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
            # STRICT_PATCHES=1 (image builds, always-fresh clones): a skip can
            # only mean a broken chain — fail instead of hiding it. Bare-metal
            # reruns keep the idempotent skip (already-applied is normal there).
            if [ "${STRICT_PATCHES:-0}" = "1" ]; then
                echo "ERROR: patch does not apply on pristine $SGLANG_TAG: $pname"
                exit 1
            fi
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
    # SGLANG_BUILD_RUST_EXTS=none: v0.5.17 moved Rust-ext building into setup.py
    # with this native opt-out — it replaces retired patch 037 (we serve over
    # HTTP; the grpc/multimodal exts would need cargo+protoc, absent here).
    SGLANG_BUILD_RUST_EXTS=none pip install -e .

    # v0.5.18 hard-pins transformers==5.12.1 (fourth release on this pin — the
    # version the fleet is validated on). 5.12.1 ships gemma4_unified natively
    # but also routes Mistral checkpoints to the MistralCommonBackend tokenizer
    # (fixed by patch 057). Pin exactly — do NOT let it drift: newer tx changes
    # both of those paths under our feet.
    echo "Pinning transformers 5.12.1 + gguf..."
    pip install --no-deps "transformers==5.12.1" gguf

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
    # librosa + accelerate are SERVING deps for the nemotron3-omni Parakeet
    # audio path (the header note above documented them but this script never
    # installed them — the bare-metal envs got them by hand; fixed 2026-07-27
    # when the OCI image build made the gap load-bearing).
    echo "Installing eval/validator + audio-serving deps..."
    # swebench PINNED at 4.1.0: 5.0.2 is a breaking major and every historical
    # bake-off cell was scored on 4.1.0 — bumping mid-series breaks cell
    # comparability (R9700 finding 2026-08-30; their scorer carries the same pin).
    pip install "imageio[ffmpeg]" "swebench==4.1.0" librosa accelerate
else
    echo "[2/3] Skipping conda env creation"
    init_conda
    conda activate "$ENV_NAME"
fi

# -------------------------------------------------------------------
# Chat-template fix: accept the OpenAI `developer` role as `system`.
# Newer OpenAI-compat scaffolds (little-coder/pi-ai) send the system prompt
# with role `developer`; the Qwen3.5/3.6 templates raise on it -> 400 ->
# scaffold exits the rollout in ~3 s. Idempotent; re-run after any model
# re-download. (Root cause + receipts: patches/README.md.)
# -------------------------------------------------------------------
echo ""
echo "[2b/3] Patching chat templates (developer-role -> system)..."
python "$REPO_DIR/scripts/eval/patch_chat_templates_developer_role.py" || \
    echo "  (warning: chat-template patch step failed; little-coder rollouts may 400 on thinking presets)"
echo "[2c/3] Patching chat templates (OpenAI list-content -> text join)..."
python "$REPO_DIR/scripts/eval/patch_chat_templates_list_content.py" --scan || \
    echo "  (warning: list-content template patch failed; string-only templates blank structured content — agents run blind)"

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
