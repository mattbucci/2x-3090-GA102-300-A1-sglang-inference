#!/bin/bash
# 31B graphs-arm decode re-run (tokenizer-map fix), then B3 cliff arms.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
(
  export PRESET=gemma4-31b MODE=probe RATIOS=0.8 STAGES=decode EXP=B1-gemma4-31b-G
  export _ENV_GEMMA_GRAPH="--cuda-graph-max-bs 1 --disable-piecewise-cuda-graph"
  "$SCRIPT_DIR/swa_ratio_sweep.sh"
)
"$SCRIPT_DIR/b3_cliff.sh"
