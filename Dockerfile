# GPU-free build: every CUDA component is a pinned pip wheel (torch cu13,
# flashinfer, sglang-kernel, nvidia-*); the driver arrives at `docker run` via
# the NVIDIA container toolkit. Override build args with --build-arg; defaults
# are the supported v0.5.18 stack. Adapted from the R9700 sister repo's
# ROCm/gfx1201 image (same two-stage shape, no Rust toolchain — setup.sh
# builds with upstream's SGLANG_BUILD_RUST_EXTS=none opt-out).
ARG UBUNTU_BUILDER_IMAGE=docker.io/library/ubuntu:24.04@sha256:52df9b1ee71626e0088f7d400d5c6b5f7bb916f8f0c82b474289a4ece6cf3faf
# Runtime keeps the CUDA devel image: sglang JIT-compiles triton + tvm_ffi
# kernels at first boot and expects a full toolkit at CUDA_HOME (bare-metal
# parity: our hosts serve with CUDA_HOME=/opt/cuda).
ARG CUDA_RUNTIME_IMAGE=docker.io/nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c

FROM ${UBUNTU_BUILDER_IMAGE} AS builder
ARG MINIFORGE_VERSION=26.3.2-3
ARG MINIFORGE_SHA256=848194851a98903134187fbb4ab50efe87b003e0c0f808f97644b7524a62bf2c
ARG SGLANG_TAG=v0.5.18
ARG SGLANG_COMMIT=71de97b264b04dcd514cf904003028aefe9775c8
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONDA_BASE=/opt/conda ENV_NAME=sglang-v0518 \
    REPO_DIR=/opt/3090-inference SGLANG_DIR=/opt/3090-inference/components/sglang
COPY --chmod=0555 docker/build-sglang.sh /usr/local/bin/build-sglang
RUN MINIFORGE_VERSION="${MINIFORGE_VERSION}" MINIFORGE_SHA256="${MINIFORGE_SHA256}" \
    /usr/local/bin/build-sglang install-toolchain
WORKDIR ${REPO_DIR}
COPY scripts/ ${REPO_DIR}/scripts/
COPY patches/ ${REPO_DIR}/patches/
RUN SGLANG_TAG="${SGLANG_TAG}" SGLANG_COMMIT="${SGLANG_COMMIT}" \
    /usr/local/bin/build-sglang build-sglang

FROM ${CUDA_RUNTIME_IMAGE}
ARG APP_UID=10001
ARG APP_GID=10001
# torchcodec (video decode for the multimodal presets) dlopens the system
# libav* family at runtime; bare metal gets it from the host ffmpeg install,
# the container must ship it itself (without it, video capability silently
# degrades — the torchcodec import falls through its FFmpeg 8→7→6 probe chain).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
ENV HOME=/home/sglang XDG_CACHE_HOME=/home/sglang/.cache \
    CONDA_BASE=/opt/conda ENV_NAME=sglang-v0518 \
    REPO_DIR=/opt/3090-inference SGLANG_DIR=/opt/3090-inference/components/sglang \
    PATH=/opt/conda/envs/sglang-v0518/bin:/opt/conda/bin:${PATH} \
    CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda \
    TRITON_CACHE_DIR=/home/sglang/.cache/triton_3090 \
    MODELS_DIR=/models TOKENIZERS_PARALLELISM=false \
    SGLANG_SECURE_LAUNCH=1 SGLANG_TRUST_REMOTE_CODE=0 SGLANG_ENABLE_METRICS=0 \
    SGLANG_USE_PICKLE_IPC=0 SGLANG_MAX_QUEUED_REQUESTS=32 \
    NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 \
    NCCL_DEBUG=WARN \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /opt/3090-inference/components/sglang /opt/3090-inference/components/sglang
COPY --from=builder /opt/3090-inference/scripts/common.sh \
    /opt/3090-inference/scripts/launch.sh \
    /opt/3090-inference/scripts/
COPY --from=builder /opt/3090-inference/scripts/*.jinja /opt/3090-inference/scripts/
# sgl_kernel links libcuda.so.1 (the DRIVER library, injected only at
# `docker run --gpus`) at import time; the devel image's API stub stands in
# for it so the full import chain is verifiable GPU-free at build time.
# The import check runs as root with HOME=/home/sglang and flashinfer creates
# its workspace dir at import time — scrub the cache afterwards or the runtime
# user (10001) inherits a root-owned ~/.cache/flashinfer and dies at first JIT.
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
        "${CONDA_BASE}/bin/conda" run -n "${ENV_NAME}" python -c "import torch, sglang, sgl_kernel" \
    && rm /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && rm -rf /home/sglang/.cache /root/.cache /root/.triton
RUN groupadd --gid "${APP_GID}" sglang \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home \
        --home-dir /home/sglang --shell /usr/sbin/nologin sglang \
    && install -d -o "${APP_UID}" -g "${APP_GID}" -m 0700 \
        /home/sglang/.cache /home/sglang/.config
COPY --chmod=0555 docker/secure-launch.py /usr/local/libexec/sglang-cuda/secure-launch.py
COPY --chmod=0555 docker/entrypoint.sh /usr/local/bin/entrypoint.sh
WORKDIR ${REPO_DIR}
USER ${APP_UID}:${APP_GID}
EXPOSE 23334
STOPSIGNAL SIGTERM
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "sglang.launch_server", "--help"]
