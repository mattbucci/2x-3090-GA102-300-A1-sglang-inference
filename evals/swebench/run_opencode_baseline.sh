#!/bin/bash
# run_opencode_baseline.sh — opencode-only sweep across the 7 remaining presets
# to confirm each model produces non-empty diffs on SWE-bench Lite.
#
# Why: instead of running the full 3-scaffold cycle per model (opencode +
# claw-code + little-coder, ~24-72h per model), do breadth-first opencode-only
# first so we catch broken cells (empty diffs, 400s, chat-template misalignments)
# in ~4-8h per model. Once all 7 score green on opencode, fan out to wider
# scaffolds with confidence.
#
# What this does:
#   - Sets QUEUE to the 7 presets we don't already have opencode-v2 for
#     (qwen36 already at 300/300 from the May-30 cycle stopped 2026-05-31)
#   - Sets SCAFFOLDS=opencode so run_model_cycle.sh runs only opencode rollout
#     + audit + score, skipping claw-code and little-coder
#   - Per-model: ~3-6h at current pace.
#
# Order surfaces failures fast: recently-touched cells (devstral + gemma4-31b
# with new chat templates) first; suspected non-tool-trained (qwen3-ream) next;
# then the proven coders.
#
# Rule 1 / 2 — Do NOT launch while score_docker.py is still scoring qwen36.
# Verify first:
#   tail /tmp/score-smoke-logs/qwen36-opencode.log | grep -E 'summary|resolved:'
#
# Detach pattern:
#   setsid bash -c './evals/swebench/run_opencode_baseline.sh' \
#       > /tmp/run-model-cycle-logs/queue-opencode-baseline.log 2>&1 &
#   disown
#   ps -p $! -o ppid=    # must print 1
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

export ENV_NAME="${ENV_NAME:-sglang-v0512}"
export SGLANG_DIR="${SGLANG_DIR:-/data/sglang-rebase-v0512}"

# Skip qwen36 — already at 300/300 opencode + scoring separately.
export QUEUE="${QUEUE:-devstral gemma4-31b qwen3-ream coder-30b coder-30b-eval coder-reap-25b qwen36-ream}"

# Single-scaffold per preset.
export SCAFFOLDS="opencode"

mkdir -p /tmp/run-model-cycle-logs
sg docker -c "./evals/swebench/run_all_cycles.sh"
RC=$?
echo "QUEUE_EXIT=$RC" >> /tmp/run-model-cycle-logs/queue-opencode-baseline.log
echo "opencode-baseline-done rc=$RC" > /tmp/run-model-cycle-logs/queue-opencode-baseline.done
