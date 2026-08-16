#!/usr/bin/env bash
# Entrypoint for the 2x3090 CUDA SGLang image. Unlike the R9700 donor image
# there is no render-node selection step: NVIDIA device access is granted by
# the container toolkit (`docker run --gpus ...`), and CUDA_VISIBLE_DEVICES
# defaults to 0,1 in scripts/common.sh (override with -e CUDA_VISIBLE_DEVICES).
set -euo pipefail

umask 077
if (( $# == 0 )); then
    echo "Usage: docker run --gpus all IMAGE scripts/launch.sh <preset> [options]" >&2
    exit 64
fi
source /opt/conda/etc/profile.d/conda.sh
conda activate sglang-v0517
exec "$@"
