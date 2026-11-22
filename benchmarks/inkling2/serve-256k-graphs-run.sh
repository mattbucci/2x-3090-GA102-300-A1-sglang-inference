#!/bin/bash
# First Inkling-2-Mini boot on 2x3090 (sglang-main + Ampere port). Stops gemma4-31b production first.
D=/var/tmp/inkling2/boot2; L=$D/run.log
cd /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference
echo "[$(date +%T)] stopping production :30000" >> $L
scripts/serve_production.sh stop >> $L 2>&1
for i in $(seq 1 60); do n=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$n" -lt 1500 ] && break; sleep 5; done
echo "[$(date +%T)] GPU mem max ${n} MiB" >> $L
source /home/letsrtfm/miniforge3/etc/profile.d/conda.sh; conda activate sglang-main
export PYTHONNOUSERSITE=1 SGLANG_DISABLE_REQUEST_LOGGING=1
python -m sglang.launch_server --model-path /data/models/hf-3rdparty/Inkling-2-Mini \
  --served-model-name inkling-2-mini --tp 2 --port 23400 --host 127.0.0.1 \
  --context-length 262144 --mem-fraction-static 0.80 --max-running-requests 1 \
  --attention-backend triton --disable-prefill-cuda-graph --cuda-graph-max-bs-decode 1 --trust-remote-code > $D/server.log 2>&1 &
SP=$!
echo "[$(date +%T)] server pid $SP" >> $L
for i in $(seq 1 180); do
  if curl -s -m 5 127.0.0.1:23400/health >/dev/null 2>&1 && curl -s -m 5 127.0.0.1:23400/v1/models | grep -q inkling; then echo "[$(date +%T)] HEALTHY" >> $L; break; fi
  kill -0 $SP 2>/dev/null || { echo "[$(date +%T)] server exited" >> $L; break; }
  sleep 10
done
python3 /var/tmp/inkling2/chat3.py >> $D/chat.log 2>&1
echo "[$(date +%T)] DONE" >> $L
