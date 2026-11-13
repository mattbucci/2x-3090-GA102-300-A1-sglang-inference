#!/bin/bash
# Patch 083 gate 2: SMEM fallback on gemma4 26B, Flint off-arm MMLU/HE baseline, gemma4-31b re-check on the fallback code
cd /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference
source scripts/common.sh; activate_conda; setup_nvidia_env
O=/var/tmp/gemma-attn; PORT=23334; BASE="http://127.0.0.1:$PORT"; D=2026-11-13; H=benchmarks/quality/haystack-platypus.txt
log(){ echo "[$(date '+%F %T')] $*" >> $O/gate2.log; }
wait_health(){ for _ in $(seq 1 120); do curl -s -m 2 "$BASE/health" >/dev/null 2>&1 && return 0; pgrep -f '[s]glang.launch_server' >/dev/null || return 1; sleep 5; done; return 1; }
stop_drain(){ pkill -f '[s]glang.launch_server' 2>/dev/null; pkill -f '[s]glang::scheduler' 2>/dev/null; pkill -f '[s]glang::detokenizer' 2>/dev/null; for _ in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$u" -lt 1500 ] && return 0; sleep 3; done; }
step(){ n=$1; shift; timeout 5400 "$@" > $O/gate2-$n.log 2>&1; log "STEP $n rc=$?"; }
serve(){ local tag=$1 p=$2 bn=$3 w=$4; stop_drain; SGLANG_TRITON_DECODE_GROUPED_BLOCK_N=$bn SGLANG_TRITON_DECODE_GROUPED_NUM_WARPS=$w ./scripts/launch.sh $p > $O/serve-gate2-$tag.log 2>&1 & sleep 20; wait_health || { log "ERROR boot $tag"; return 1; }; }
client(){ local tag=$1 chars=$2; rm -f $O/flag; python /var/tmp/flint-overlap/client.py $BASE $chars $O/flag $O/client-$tag-$chars.json > $O/client-$tag-$chars.log 2>&1; log "CLIENT $tag $chars $(python3 -c "
import json;d=json.load(open('$O/client-$tag-$chars.json'));print(d['usage']['prompt_tokens'],d['usage']['completion_tokens'],round(d['tok_s'],2))" 2>&1 | tail -1)"; }
if serve g26 gemma4 64 8; then
  log "fallback lines: $(grep -c 'keeping the upstream 32 x 4 tile' $O/serve-gate2-g26.log)"
  step caps-g26 python scripts/eval/validate_capabilities.py --port $PORT
  RUN_SUFFIX=tile083 scripts/bench/bench_regression.sh gemma4 > $O/gate2-tripwire-g26.log 2>&1; log "STEP tripwire-g26 rc=$?"
  log "fallback lines after: $(grep -c 'keeping the upstream 32 x 4 tile' $O/serve-gate2-g26.log)"
fi
if serve foff flint 0 0; then
  step he-foff python -u scripts/eval/eval_and_chart.py --run --port $PORT --tag "flint-tile083-off-$D" --mmlu-samples 57 --humaneval-samples 30 --labbench-samples 0 --needle-lengths 0 --mc-budget 2560 --workers 1
fi
if serve g31 gemma4-31b 64 8; then client g31b 120000; client g31b 700000; fi
if serve g31off gemma4-31b 0 0; then client g31offb 700000; fi
stop_drain; log DONE
