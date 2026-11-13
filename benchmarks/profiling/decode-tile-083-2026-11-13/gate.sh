#!/bin/bash
# Patch 083 parity gate: gemma4-31b + gemma4 at 64 keys x 8 warps, flint at 128 x 4
cd /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference
source scripts/common.sh; activate_conda; setup_nvidia_env
O=/var/tmp/gemma-attn; PORT=23334; BASE="http://127.0.0.1:$PORT"; D=2026-11-13; H=benchmarks/quality/haystack-platypus.txt
log(){ echo "[$(date '+%F %T')] $*" >> $O/gate.log; }
wait_health(){ for _ in $(seq 1 240); do curl -s -m 2 "$BASE/health" >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }
stop_drain(){ pkill -f '[s]glang.launch_server' 2>/dev/null; pkill -f '[s]glang::scheduler' 2>/dev/null; pkill -f '[s]glang::detokenizer' 2>/dev/null; for _ in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$u" -lt 1500 ] && return 0; sleep 3; done; }
step(){ n=$1; shift; timeout 5400 "$@" > $O/gate-$n.log 2>&1; log "STEP $n rc=$?"; }
gate(){ # preset blockN warps needle-lengths tripwire(0/1)
  local p=$1 bn=$2 w=$3 nl=$4 tw=$5
  stop_drain
  SGLANG_TRITON_DECODE_GROUPED_BLOCK_N=$bn SGLANG_TRITON_DECODE_GROUPED_NUM_WARPS=$w ./scripts/launch.sh $p > $O/serve-gate-$p.log 2>&1 &
  wait_health || { log "ERROR boot $p"; return; }
  step caps-$p python scripts/eval/validate_capabilities.py --port $PORT
  step needle-$p python scripts/eval/needle_band_probe.py --port $PORT --tag "$p-tile083-needle-$D" --lengths $nl --depths 0.1,0.5,0.9 --flush --max-tokens 2048 --temperature default --haystack $H
  step he-$p python -u scripts/eval/eval_and_chart.py --run --port $PORT --tag "$p-tile083-$D" --mmlu-samples 57 --humaneval-samples 30 --labbench-samples 0 --needle-lengths 0 --mc-budget 2560 --workers 1
  [ "$tw" = 1 ] && { RUN_SUFFIX=tile083 scripts/bench/bench_regression.sh $p > $O/gate-tripwire-$p.log 2>&1; log "STEP tripwire-$p rc=$?"; }
}
gate gemma4-31b 64 8 131072,240000 0
gate flint 128 4 131072,195000 1
gate gemma4 64 8 131072 1
stop_drain; log DONE
