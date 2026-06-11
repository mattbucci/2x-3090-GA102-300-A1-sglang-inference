#!/bin/bash
# a4_push256k.sh — "can ALL models reach 256K?" boundary probes (boot-only):
#   31B @ mem 0.92 + tiny ratios  -> does the budget asymptote clear 262144?
#   devstral @ mem 0.92 / 0.90    -> dense FP8-KV; does +budget alone clear it?
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run(){ ( export PRESET="$1" MODE=ladder RATIOS="$2" EXP="$3"; export EXTRA_SWEEP_ARGS="$4"; "$SCRIPT_DIR/swa_ratio_sweep.sh" ); }
run gemma4-31b "0.05 0.04 0.02" A4-gemma4-31b-mem92 "--mem-fraction-static 0.92"
run devstral   "0.8"            A4-devstral-mem92   "--mem-fraction-static 0.92"
run devstral   "0.8"            A4-devstral-mem90   "--mem-fraction-static 0.90"
echo "[a4 $(date +%H:%M:%S)] done"
