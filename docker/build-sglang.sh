#!/usr/bin/env bash
# Builder-only setup for the 2x3090 CUDA SGLang image. The build is GPU-free:
# every CUDA component arrives as a pinned pip wheel (torch cu13, flashinfer,
# sglang-kernel, nvidia-*), and the driver is injected at `docker run` by the
# NVIDIA container toolkit. Unlike the R9700 donor image there is no Rust
# toolchain: setup.sh installs with upstream's SGLANG_BUILD_RUST_EXTS=none
# opt-out (v0.5.17 setup.py auto-discovery; replaced retired patch 037), so
# `pip install -e .` needs neither cargo nor protoc.
# Adapted from the R9700 sister repo's docker/build-sglang.sh.
set -euo pipefail

# A truncated fetch makes apt report every repository as badly signed, which is
# indistinguishable from a real signing failure and kills the whole build. Retry
# network steps so one bad fetch does not fail CI.
retry() {
    local attempt
    for attempt in 1 2 3 4 5; do
        if "$@"; then
            return 0
        fi
        echo "retry: '$1' failed (attempt ${attempt}/5); retrying in $((attempt * 10))s" >&2
        sleep $((attempt * 10))
    done
    echo "retry: '$1' failed after 5 attempts" >&2
    return 1
}

download_verified() {
    local url=$1 output=$2 expected_sha256=$3
    if [[ ! "$expected_sha256" =~ ^[0-9a-f]{64}$ ]]; then
        echo "download_verified: invalid SHA-256 for $url" >&2
        return 2
    fi
    retry curl --proto '=https' --proto-redir '=https' --tlsv1.2 \
        --location --fail --silent --show-error "$url" -o "$output"
    printf '%s  %s\n' "$expected_sha256" "$output" | sha256sum --check --strict -
}

apt_update() {
    rm -rf /var/lib/apt/lists/*
    apt-get update
}

install_toolchain() {
    : "${MINIFORGE_VERSION:?MINIFORGE_VERSION is required}"
    : "${MINIFORGE_SHA256:?MINIFORGE_SHA256 is required}"

    retry apt_update
    retry apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential
    rm -rf /var/lib/apt/lists/*

    download_verified \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
        /tmp/miniforge.sh "$MINIFORGE_SHA256"
    bash /tmp/miniforge.sh -b -p "${CONDA_BASE}"
    rm /tmp/miniforge.sh
}

build_sglang() {
    # GPU_FREE_SETUP skips the nvidia-smi guard; STRICT_PATCHES turns any
    # patch skip into a hard failure (a fresh clone can only skip on a broken
    # chain); SGLANG_COMMIT pins the tag to an exact commit.
    GPU_FREE_SETUP=1 STRICT_PATCHES=1 \
        SGLANG_TAG="${SGLANG_TAG:?SGLANG_TAG is required}" \
        SGLANG_COMMIT="${SGLANG_COMMIT:?SGLANG_COMMIT is required}" \
        ./scripts/setup.sh
    # Compile-check every file the 27-patch chain touches on the serving hot
    # path — a mis-applied hunk that still passed `git apply` dies here, not at
    # first boot.
    "${CONDA_BASE}/bin/conda" run -n "${ENV_NAME}" python -m py_compile \
        "${SGLANG_DIR}/python/sglang/kernels/ops/mamba/causal_conv1d_triton.py" \
        "${SGLANG_DIR}/python/sglang/kernels/ops/attention/fla/fused_recurrent.py" \
        "${SGLANG_DIR}/python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py" \
        "${SGLANG_DIR}/python/sglang/kernels/ops/attention/decode_attention.py" \
        "${SGLANG_DIR}/python/sglang/kernels/ops/attention/extend_attention.py" \
        "${SGLANG_DIR}/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py" \
        "${SGLANG_DIR}/python/sglang/srt/layers/moe/moe_runner/marlin.py" \
        "${SGLANG_DIR}/python/sglang/srt/configs/model_config.py" \
        "${SGLANG_DIR}/python/sglang/srt/model_executor/model_runner_components/load_model_utils.py" \
        "${SGLANG_DIR}/python/sglang/srt/managers/mm_utils.py" \
        "${SGLANG_DIR}/python/sglang/srt/utils/hf_transformers/tokenizer.py" \
        "${SGLANG_DIR}/python/sglang/srt/utils/hf_transformers/common.py" \
        "${SGLANG_DIR}/python/sglang/srt/function_call/mistral_detector.py" \
        "${SGLANG_DIR}/python/sglang/srt/layers/attention/triton_backend.py" \
        "${SGLANG_DIR}/python/sglang/srt/server_args.py"
    # Patch-060 marker: the gemma4_unified config alias must be guarded on the
    # native transformers class or every gemma4_unified checkpoint crashes at
    # boot (v0.5.16-net-new).
    grep -q '\[3090 060\]' \
        "${SGLANG_DIR}/python/sglang/srt/utils/hf_transformers/common.py"
    # No Rust ext may have been BUILT (SGLANG_BUILD_RUST_EXTS=none in setup.sh;
    # the image ships no cargo/protoc, and none are needed on the HTTP path).
    if find "${SGLANG_DIR}/python/sglang" -name '_core*.so' | grep -q .; then
        echo "ERROR: a Rust extension module was built despite the opt-out" >&2
        exit 1
    fi
    "${CONDA_BASE}/bin/conda" run -n "${ENV_NAME}" python -c "import sglang; print(sglang.__version__)"
    "${CONDA_BASE}/bin/conda" clean -afy
    rm -rf "${SGLANG_DIR}/.git" "${REPO_DIR}/build"
    find "${REPO_DIR}" -type d -name __pycache__ -prune -exec rm -rf {} +
}

case "${1:-}" in
    install-toolchain)
        install_toolchain
        ;;
    build-sglang)
        build_sglang
        ;;
    *)
        echo "Usage: $0 {install-toolchain|build-sglang}" >&2
        exit 64
        ;;
esac
