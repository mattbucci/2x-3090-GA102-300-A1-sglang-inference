#!/bin/bash
# Chart refresh 2026-10-25: every chart preset, fresh 10-point depth sweep at 300 W (ship cap)
# and again at 400 W, gpu-unlock on, CPU boost off, fan floor 100 %. One boot per preset.
D=/var/tmp/chart-refresh-2026-10-25
REPO=/home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference
cd "$REPO" || exit 1
source scripts/common.sh; activate_conda; setup_nvidia_env
echo "start" > $D/phase; python3 $D/sampler.py 2>>$D/sampler.err & SAMP=$!
# preset|slug  (slug = benchmarks/<slug>/results.json, generate_charts.py MODELS)
LIST="flint|flint qwen38|qwen38 qwen38-spec|qwen38 qwen36|qwen3.6-35b-a3b qwen36-ream|qwen3.6-ream qwen36-dense|qwen3.6-27b qwen35-moe|qwen3.5-28b-moe qwen3-ream|qwen3-30b-ream nemotron3-omni|nemotron3-omni devstral|devstral-24b-awq gemma4-31b|gemma4-31b gemma4|gemma4-26b-awq gemma4-21b-reap|gemma4-21b-reap gemma4-12b|gemma4-12b coder-30b|coder-30b-awq coder-reap-25b|coder-reap-25b qwen3-vl-32b|qwen3-vl-32b qwen35-dense|qwen35-27b-awq"
setpl() { sudo nvidia-smi -i 0 -pl $1 >/dev/null; sudo nvidia-smi -i 1 -pl $1 >/dev/null; }
dimm() { sensors -j 2>/dev/null | python3 -c 'import json,sys; s=json.load(sys.stdin); print(max(s["spd5118-i2c-12-50"]["temp1"]["temp1_input"], s["spd5118-i2c-12-52"]["temp1"]["temp1_input"]))'; }
gpumax() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1; }
guard() { for i in $(seq 1 60); do d=$(dimm); g=$(gpumax); awk -v d="$d" -v g="$g" 'BEGIN{exit !(d<53 && g<80)}' && return 0; echo "  guard: DIMM $d GPU $g — cooling 10 s"; sleep 10; done; echo "  guard: timed out, proceeding"; }
kill_server() { local pid=$1; kill $pid 2>/dev/null; for i in $(seq 1 60); do ps -p $pid >/dev/null || break; sleep 2; done; ps -p $pid >/dev/null && { kill -9 -- -$pid 2>/dev/null; kill -9 $pid 2>/dev/null; }
  for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$u" -lt 1500 ] && return 0; sleep 2; done; echo "  WARN: GPU memory not released"; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9; sleep 5; }
[ -f $D/PAUSE ] && echo "PAUSE present — waiting" ; while [ -f $D/PAUSE ]; do sleep 30; done
for ent in $LIST; do
  while [ -f $D/PAUSE ]; do sleep 30; done
  P=${ent%%|*}; SLUG=${ent##*|}; PRESET=$P; EXTRA=""
  [ "$P" = "qwen38-spec" ] && { PRESET=qwen38; EXTRA="SPEC_DECODE=1"; }
  OUT300=benchmarks/$SLUG/results.json; OUT400=benchmarks/$SLUG/results-400w.json
  [ "$P" = "qwen38-spec" ] && { OUT300=benchmarks/qwen38/results-spec.json; OUT400=benchmarks/qwen38/results-spec-400w.json; }
  [ -s "$OUT400" ] && grep -q '"timestamp": "2026-10-2[5-9]' "$OUT400" && { echo "[$(date +%T)] $P already done, skip"; continue; }
  echo "[$(date +%T)] === $P ($SLUG) ==="; echo "$P-boot" > $D/phase
  setpl 300
  setsid bash -c "cd $REPO && PORT=23334 $EXTRA exec scripts/launch.sh $PRESET > $D/$P.server.log 2>&1" </dev/null >/dev/null 2>&1 &
  sleep 5; SP=$(pgrep -f "sglang.launch_server" | head -n 1)
  ok=0; for i in $(seq 1 300); do curl -sf -m 3 http://127.0.0.1:23334/health >/dev/null 2>&1 && { ok=1; break; }; pgrep -f "sglang.launch_server" >/dev/null || break; sleep 5; done
  SP=$(pgrep -f "sglang.launch_server" | head -n 1)
  [ $ok = 1 ] || { echo "  BOOT_FAILED $P"; [ -n "$SP" ] && kill_server $SP; continue; }
  TOK=$(curl -sf http://127.0.0.1:23334/get_model_info | python3 -c "import json,sys; print(json.load(sys.stdin)['model_path'])")
  echo "  healthy; tokenizer $TOK"
  echo "$P-300w" > $D/phase
  python scripts/bench/bench_long_context.py --name "$P" --tokenizer "$TOK" --output "$OUT300" > $D/$P.300w.log 2>&1; echo "  300W rc=$? $(grep -c 'tok/s' $D/$P.300w.log) points"
  setpl 400; echo "$P-400w" > $D/phase; guard
  python scripts/bench/bench_long_context.py --name "$P" --tokenizer "$TOK" --output "$OUT400" > $D/$P.400w.log 2>&1; echo "  400W rc=$? $(grep -c 'tok/s' $D/$P.400w.log) points"
  setpl 300; echo "$P-teardown" > $D/phase
  kill_server $SP
  echo "[$(date +%T)] done $P"
done
setpl 300; echo STOP > $D/phase; wait $SAMP 2>/dev/null
echo "CAMPAIGN_DONE $(date)"
