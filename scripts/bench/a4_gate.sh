#!/bin/bash
# a4_gate.sh — gate the A4 budget bumps before wiring: 31B @ mem .92 (130K),
# devstral @ mem .90 (202K). Full battery + tooluse + decode each.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run(){ ( export PRESET="$1" MODE=probe RATIOS="$2" STAGES=caps,tooluse,decode EXP="$3"
        export EXTRA_SWEEP_ARGS="$4" TOOLUSE_LENGTHS="$5"
        "$SCRIPT_DIR/swa_ratio_sweep.sh" ); }
run gemma4-31b "0.1" A4-gate-gemma4-31b "--mem-fraction-static 0.92" "28672,114688,172032,215040"
run devstral   "0.8" A4-gate-devstral   "--mem-fraction-static 0.90" "28672,114688,229376,310000"
echo "[a4-gate $(date +%H:%M:%S)] done"
