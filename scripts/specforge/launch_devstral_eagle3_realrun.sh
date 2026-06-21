#!/usr/bin/env bash
# Launch the REAL multi-day Devstral-24B EAGLE3 online training run on 2x3090.
#
# Layout (see scripts/specforge/eagle3_training_plan.md): target AWQ wholly on
# cuda:1 (~14GB) via SPECFORGE_TARGET_GPU0_GIB=1; draft + BF16 optimizer + TTT
# activations on cuda:0. The 16K fit-probe showed cuda:0 OOMs at ttt-length 7,
# so TTT is reduced (default 5 here) while KEEPING the 16K training context
# (the part the R9700 amendment says matters for long-context acceptance).
#
# Data: scripts/specforge/build_longctx_code_data.py packed OpenCodeInstruct into
# ~15.2K-token multi-turn code conversations (cache/dataset/longcode_train.jsonl,
# 2001 seqs). The trainer TRUNCATES to --max-length (does not pack), so long
# samples are mandatory.
#
# CACHE-BUST is load-bearing: SpecForge's processed_dataset cache key omits the
# chat template, so we clear it before every run after any template/data change.
set -euo pipefail

SF=/data/specforge/SpecForge
# Config chosen by empirical 2x24GB fit-probing (see eagle3_training_plan.md). Memory walls
# and fixes: (1) full-vocab fp32 teacher cast -> chunk over seq + bf16 + keep logits on cuda:1;
# (2) TTT MLP/logits -> gradient-checkpoint (attention can't: it mutates cache_hidden);
# (3) acceptance-rate softmax -> chunk over seq; (4) AdamW fp32 moments -> 8-bit (bitsandbytes,
# ~5GB saved). The remaining wall is the AdamW state alloc on the FIRST optimizer step (true
# peak is step 2+). With all four fixes, 16K/ttt4 is ~0.5GB short; 16K/ttt3 fits with ~3.4GB
# margin. We honor the R9700 amendment's >=16K TRAINING CONTEXT (the lever against long-context
# slowdown); ttt=3 is the speculative depth that fits at 16K on 2x24GB.
TTT="${TTT:-3}"
MAXLEN="${MAXLEN:-16384}"
EPOCHS="${EPOCHS:-10}"
DATA="${DATA:-cache/dataset/longcode_train.jsonl}"
OUT="${OUT:-outputs/devstral-eagle3}"
LOG="${LOG:-$SF/devstral_eagle3_realrun.log}"

source /home/letsrtfm/miniforge3/etc/profile.d/conda.sh
conda activate specforge-cuda
export HF_TOKEN=$(cat ~/.secrets/hf_token)
export CUDA_VISIBLE_DEVICES=0,1
export SPECFORGE_TARGET_GPU0_GIB=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SF"
echo "[launch] clearing stale processed-dataset cache (template-agnostic key)"
rm -rf cache/processed_dataset cache/vocab_mapping

echo "[launch] TTT=$TTT MAXLEN=$MAXLEN EPOCHS=$EPOCHS DATA=$DATA OUT=$OUT"
exec torchrun --standalone --nproc_per_node 1 scripts/train_eagle3.py \
  --target-model-path /data/models/Devstral-Small-2-24B-AWQ-textonly --target-model-backend hf \
  --draft-model-config configs/devstral-24b-eagle3.json --train-data-path "$DATA" \
  --output-dir "$OUT" --num-epochs "$EPOCHS" --batch-size 1 --tp-size 1 \
  --learning-rate 1e-4 --max-length "$MAXLEN" --ttt-length "$TTT" \
  --cache-dir cache --chat-template devstral \
  --save-interval 2000 --eval-interval 100000
