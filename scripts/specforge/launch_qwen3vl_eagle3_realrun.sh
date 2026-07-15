#!/bin/bash
# Qwen3-VL-32B EAGLE3 draft — REAL long-context run (R9700 deliverable #2).
# Mirrors launch_devstral_eagle3_realrun.sh: text-only AWQ target (extracted by
# extract_qwen3vl_text_only.py), 16K max-length on the longcode data, 2 epochs
# (the devstral sweep's best checkpoint depth: 1ep 2.79 / 2ep best / 3ep overfit),
# the 4 SpecForge 24GB memory fixes assumed applied in /data/specforge/SpecForge.
# Smoke-validated 2026-07-15 (loss 4.00->3.27 @15 steps, target via AwqMarlinLinear).
set -uo pipefail
cd /data/specforge/SpecForge
export PATH="/home/letsrtfm/miniforge3/envs/specforge-cuda/bin:$PATH"
export CUDA_HOME=/opt/cuda CUDA_PATH=/opt/cuda
DATA="${DATA:-cache/dataset/longcode_train.jsonl}"
CUDA_VISIBLE_DEVICES=0,1 SPECFORGE_TARGET_GPU0_GIB=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node 1 scripts/train_eagle3.py \
  --target-model-path /data/models/Qwen3-VL-32B-AWQ-textonly --target-model-backend hf \
  --draft-model-config configs/qwen3-32b-eagle3.json \
  --train-data-path "$DATA" \
  --output-dir outputs/qwen3vl32b-eagle3 --num-epochs 2 --batch-size 1 --tp-size 1 \
  --learning-rate 1e-4 --max-length 16384 --cache-dir cache --chat-template qwen3-instruct
