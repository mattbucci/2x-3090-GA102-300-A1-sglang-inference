#!/bin/bash
# Chart refresh 2026-10-31: every chart preset, fresh 10-point depth sweep at 300 W as shipped under the day's
# three levers (custom allreduce, INT8 lm_head, GDN conv state in the model dtype). One boot per preset.
# Before each overwrite the 10-25 results.json is copied to results-300w.json so the 300-vs-400 W chart keeps
# its same-boot pairing. Waits for the Flint tool probe; PAUSE file in $D holds between presets.
D=/var/tmp/chart-refresh-2026-10-31
REPO=/home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference
cd "$REPO" || exit 1
while kill -0 "$(cat /var/tmp/flint-tooluse/pid2)" 2>/dev/null; do sleep 20; done
source scripts/common.sh; activate_conda; setup_nvidia_env
LIST="flint|flint qwen38|qwen38 qwen36|qwen3.6-35b-a3b qwen36-ream|qwen3.6-ream qwen36-dense|qwen3.6-27b qwen35-moe|qwen3.5-28b-moe qwen3-ream|qwen3-30b-ream coder-30b|coder-30b-awq coder-reap-25b|coder-reap-25b devstral|devstral-24b-awq gemma4|gemma4-26b-awq gemma4-31b|gemma4-31b gemma4-12b|gemma4-12b gemma4-21b-reap|gemma4-21b-reap nemotron3-omni|nemotron3-omni qwen3-vl-32b|qwen3-vl-32b qwen35-dense|qwen35-27b-awq"
gpumax() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1; }
kill_server() { pkill -f '[s]glang.launch_server' 2>/dev/null; for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$u" -lt 1500 ] && return 0; sleep 2; done; pkill -KILL -f '[s]glang.launch_server' 2>/dev/null; sleep 10; }
for ent in $LIST; do
  while [ -f $D/PAUSE ]; do sleep 30; done
  P=${ent%%|*}; SLUG=${ent##*|}
  OUT=benchmarks/$SLUG/results.json
  [ -s "$OUT" ] && grep -qE '"timestamp": "2026-10-31 (1[7-9]|2[0-3])' "$OUT" && { echo "[$(date +%T)] $P already done, skip"; continue; }
  [ -s "$OUT" ] && [ ! -s benchmarks/$SLUG/results-300w.json ] && cp "$OUT" benchmarks/$SLUG/results-300w.json
  echo "[$(date +%T)] === $P ($SLUG) === gpu max $(gpumax) C"
  curl -sf -m 2 http://127.0.0.1:23334/health >/dev/null 2>&1 && { echo "  ERROR: a server already answers — aborting"; exit 1; }
  setsid bash -c "cd $REPO && PORT=23334 exec scripts/launch.sh $P > $D/$P.server.log 2>&1" </dev/null >/dev/null 2>&1 &
  ok=0; for i in $(seq 1 300); do curl -sf -m 3 http://127.0.0.1:23334/health >/dev/null 2>&1 && { ok=1; break; }; [ $i -gt 12 ] && ! pgrep -f "[s]glang.launch_server" >/dev/null && break; sleep 5; done
  [ $ok = 1 ] || { echo "  BOOT_FAILED $P"; kill_server; continue; }
  TOK=$(curl -sf http://127.0.0.1:23334/get_model_info | python3 -c "import json,sys; print(json.load(sys.stdin)['model_path'])")
  python scripts/bench/bench_long_context.py --name "$P" --tokenizer "$TOK" --output "$OUT" > $D/$P.300w.log 2>&1; echo "  300W rc=$? $(grep -cE 'tok/s' $D/$P.300w.log) points; deep: $(grep -E 'ctx=' $D/$P.300w.log | tail -1 | cut -c1-70)"
  kill_server
  echo "[$(date +%T)] done $P"
done
echo "CAMPAIGN_DONE $(date)"
