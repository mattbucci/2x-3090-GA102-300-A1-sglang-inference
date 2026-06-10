#!/bin/bash
# b1_arms.sh — sprint Track B1: gemma4 26B attention-backend × cuda-graph A/B.
# Control arm (triton + no-graph @ ratio 0.0625) = the A2 probe receipts (same
# day, same instrument): decode 34.1@1K → 31.2@256K, caps 5/5.
# Arms here (serial, one server at a time):
#   N  = torch_native, graphs OFF        (isolates backend cost)
#   G  = torch_native, graphs ON (bs=1)  (the hypothesis arm; MEM=0.78 for
#        graph+warmup headroom — qwen36 precedent: 0.85 OOMs at final init)
#   GT = triton, graphs ON               (diagnostic: capture WHY it fails)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
run_arm() { # name extra_sweep_args gemma_graph mem
  local name="$1" esa="$2" gg="$3" mem="$4"
  echo "[b1-arms $(date +%H:%M:%S)] ===== arm $name ====="
  # subshell + explicit export: ${var:+VAR=x} prefix-assignment does NOT work
  # (expansion happens after assignment parsing -> rc 127; arms G/GT no-oped
  # on the first run this way)
  (
    export PRESET=gemma4 MODE=probe RATIOS="0.0625" STAGES=caps,decode
    export EXP="B1-gemma4-26b-$name" EXTRA_SWEEP_ARGS="$esa"
    [ -n "$gg" ] && export _ENV_GEMMA_GRAPH="$gg"
    [ -n "$mem" ] && export MEM="$mem"
    "$REPO/scripts/bench/swa_ratio_sweep.sh"
  )
}
ARMS="${ARMS:-N G GT}"
case " $ARMS " in *" N "*)  run_arm N  "--attention-backend torch_native" "" "" ;; esac
case " $ARMS " in *" G "*)  run_arm G  "--attention-backend torch_native" "--cuda-graph-max-bs 1 --disable-piecewise-cuda-graph" "0.78" ;; esac
case " $ARMS " in *" GT "*) run_arm GT ""                                 "--cuda-graph-max-bs 1 --disable-piecewise-cuda-graph" "0.78" ;; esac
echo "[b1-arms $(date +%H:%M:%S)] all arms done"
