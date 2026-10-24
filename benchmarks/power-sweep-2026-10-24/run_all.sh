#!/bin/bash
D=/var/tmp/power-sweep-2026-10-24
REPO=/home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference
cd "$REPO" || exit 1
source scripts/common.sh; activate_conda; setup_nvidia_env
echo "boot-wait" > $D/phase
python3 $D/sampler.py 2>>$D/sampler.err &
SAMP=$!
for i in $(seq 1 120); do curl -sf -m 3 http://127.0.0.1:23334/health >/dev/null && break; sleep 5; done
curl -sf -m 3 http://127.0.0.1:23334/health >/dev/null || { echo "SERVER NOT HEALTHY"; echo STOP > $D/phase; exit 1; }
echo "server healthy $(date)"
python3 $D/sweep_serve.py; echo "serve rc=$? $(date)"
# pass B: tensor_rate at every level, server down
SP=$(cat $D/server.pid); kill $SP; for i in $(seq 1 60); do ps -p $SP >/dev/null || break; sleep 2; done
ps -p $SP >/dev/null && kill -9 $SP
for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$u" -lt 1500 ] && break; sleep 3; done
echo "server down $(date)"
: > $D/tensor-results.jsonl
for P in $(seq 200 10 400); do
  sudo nvidia-smi -i 0 -pl $P >/dev/null; sudo nvidia-smi -i 1 -pl $P >/dev/null
  echo "pl$P-tensor" > $D/phase; sleep 5
  python3 scripts/bench/tensor_rate.py 8192 | sed "s/^{/{\"pl_w\": $P, /" >> $D/tensor-results.jsonl
  echo "[$(date +%T)] tensor pl=$P done"
done
echo "restore" > $D/phase
sudo nvidia-smi -i 0 -pl 260 >/dev/null; sudo nvidia-smi -i 1 -pl 260 >/dev/null
echo STOP > $D/phase; wait $SAMP 2>/dev/null
echo "SWEEP_DONE $(date)"
