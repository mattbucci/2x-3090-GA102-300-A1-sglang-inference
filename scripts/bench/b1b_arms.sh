#!/bin/bash
# b1b_arms.sh — sprint B1b: graph-ON arms for gemma4-12b and gemma4-31b
# (26B receipts: graphs = 2.44x @1K / 1.31x @256K, caps 5/5, triton captures).
# Controls: 12B = A1 probe @0.0625 (41.3@1K -> 30.9@256K); 31B = fleet 06-07
# (33.4 @1K flat to its 16K-measured cap).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
GG="--cuda-graph-max-bs 1 --disable-piecewise-cuda-graph"
run() { # preset ratio exp
  echo "[b1b $(date +%H:%M:%S)] ===== $1 graphs-ON ====="
  (
    export PRESET="$1" MODE=probe RATIOS="$2" STAGES=caps,decode EXP="$3"
    export _ENV_GEMMA_GRAPH="$GG" MEM=0.78
    "$REPO/scripts/bench/swa_ratio_sweep.sh"
  )
}
run gemma4-12b 0.0625 B1-gemma4-12b-G
run gemma4-31b 0.8    B1-gemma4-31b-G
echo "[b1b $(date +%H:%M:%S)] done"
