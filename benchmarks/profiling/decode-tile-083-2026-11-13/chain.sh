#!/bin/bash
# Patch 083 served A/B: grouped decode stage-1 tile geometry, gemma4-31b + flint (plain decode)
cd /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference
source scripts/common.sh; activate_conda; setup_nvidia_env
O=/var/tmp/gemma-attn; PORT=23334; BASE="http://127.0.0.1:$PORT"
log(){ echo "[$(date '+%F %T')] $*" >> $O/chain.log; }
wait_health(){ for _ in $(seq 1 240); do curl -s -m 2 "$BASE/health" >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }
stop_drain(){ pkill -f '[s]glang.launch_server' 2>/dev/null; pkill -f '[s]glang::scheduler' 2>/dev/null; pkill -f '[s]glang::detokenizer' 2>/dev/null; for _ in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$u" -lt 1500 ] && return 0; sleep 3; done; }
run(){ # preset tag blockN warps deepchars
  local preset=$1 tag=$2 bn=$3 w=$4 deep=$5
  stop_drain
  SGLANG_TRITON_DECODE_GROUPED_BLOCK_N=$bn SGLANG_TRITON_DECODE_GROUPED_NUM_WARPS=$w ./scripts/launch.sh $preset > $O/serve-$tag.log 2>&1 &
  wait_health || { log "ERROR boot $tag"; return; }
  for chars in 120000 $deep; do
    rm -f $O/flag
    python /var/tmp/flint-overlap/client.py $BASE $chars $O/flag $O/client-$tag-$chars.json > $O/client-$tag-$chars.log 2>&1
    log "CLIENT $tag $chars $(python3 -c "
import json;d=json.load(open('$O/client-$tag-$chars.json'));print(d['usage']['prompt_tokens'],d['usage']['completion_tokens'],round(d['tok_s'],2),d['sha'],d['head_sha'])" 2>&1 | tail -1)"
  done
}
run gemma4-31b g-off 0 0 700000
run gemma4-31b g-n64w8 64 8 700000
run flint f-off 0 0 800000
run flint f-n128w4 128 4 800000
run flint f-n64w4 64 4 800000
run flint f-off2 0 0 800000
run flint f-n128w4b 128 4 800000
stop_drain; log DONE
