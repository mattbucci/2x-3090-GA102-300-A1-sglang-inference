#!/bin/bash
# Patch 083 gate 4: tripwire arms as shipped (launch.sh defaults) for gemma4-31b, flint, flint-spec
cd /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference
O=/var/tmp/gemma-attn
until grep -q DONE $O/gate3.log 2>/dev/null; do sleep 20; done
RUN_SUFFIX=tile083ship scripts/bench/bench_regression.sh arm gemma4-31b flint flint-spec > $O/gate4-arm.log 2>&1
echo "[$(date '+%F %T')] arm rc=$?" >> $O/gate4.log
echo DONE >> $O/gate4.log
