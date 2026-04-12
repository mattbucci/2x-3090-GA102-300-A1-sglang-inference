#!/bin/bash
# Quick benchmark: single, 8-concurrent, 16-concurrent
# Records results to benchmarks.log
#
# Usage: ./scripts/bench/bench_quick.sh "patch description"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
RESULTS_FILE="$REPO_DIR/benchmarks.log"

source "$REPO_DIR/scripts/common.sh"
activate_conda

LABEL="${1:-unnamed}"
MODEL="${2:-auto}"
PORT="${3:-23334}"
BASE_URL="http://localhost:${PORT}"

# Auto-detect model
if [ "$MODEL" = "auto" ]; then
    MODEL=$(curl -s "$BASE_URL/v1/models" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
fi

# Wait for server
for i in $(seq 1 30); do
    curl -s "$BASE_URL/health" > /dev/null 2>&1 && break
    [ "$i" -eq 30 ] && echo "ERROR: Server not ready" && exit 1
    sleep 2
done

echo "=============================================" | tee -a "$RESULTS_FILE"
echo "BENCH: $LABEL  $(date '+%Y-%m-%d %H:%M')" | tee -a "$RESULTS_FILE"
echo "=============================================" | tee -a "$RESULTS_FILE"

for CONC in 1 8 16; do
    if [ "$CONC" -eq 1 ]; then
        NP=4; RR=1
    elif [ "$CONC" -eq 8 ]; then
        NP=32; RR=inf
    else
        NP=64; RR=inf
    fi

    RESULT=$(python -m sglang.bench_serving \
        --backend sglang \
        --base-url "$BASE_URL" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input 256 \
        --random-output 256 \
        --num-prompts $NP \
        --request-rate $RR 2>&1)

    TPOT=$(echo "$RESULT" | grep "Mean TPOT" | awk '{print $NF}')
    THROUGHPUT=$(echo "$RESULT" | grep "Output token throughput" | awk '{print $NF}')

    echo "  conc=$CONC: TPOT=${TPOT}ms  throughput=${THROUGHPUT} tok/s" | tee -a "$RESULTS_FILE"
done
echo "" | tee -a "$RESULTS_FILE"
