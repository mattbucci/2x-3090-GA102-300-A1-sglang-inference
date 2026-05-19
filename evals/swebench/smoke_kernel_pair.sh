#!/bin/bash
# smoke_kernel_pair.sh <preset>
#
# Runs smoke_kernel.sh for $preset twice — once with the preset's default
# QUANT and once with QUANT=awq_marlin — then compares result.json files
# and writes a winner.env containing the QUANT that should be exported to
# the eval cycle.
#
# Decision policy:
#   - awq_marlin wins if it is ok AND its tok/s > default's tok/s.
#   - Otherwise default wins (preserves preset's intent).
#   - If awq_marlin fails (server crash, gibberish), default always wins.
#   - If both fail, exit non-zero — eval cycle falls back to preset default
#     with a logged warning.
#
# Output:
#   /tmp/smoke-kernel/<preset>/winner.env         e.g. `QUANT=awq_marlin`
#   /tmp/smoke-kernel/<preset>/summary.json       both results + decision

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESET="${1:?usage: smoke_kernel_pair.sh <preset>}"

OUT_DIR="/tmp/smoke-kernel/$PRESET"
mkdir -p "$OUT_DIR"

log() { echo "[smoke-pair $PRESET $(date +%H:%M:%S)] $*"; }

# Run default first, then marlin. Sequential — both need TP=2.
log "kernel A: default"
bash "$SCRIPT_DIR/smoke_kernel.sh" "$PRESET" default || true

log "kernel B: awq_marlin"
bash "$SCRIPT_DIR/smoke_kernel.sh" "$PRESET" awq_marlin || true

python3 - "$OUT_DIR" "$PRESET" <<'PY'
import json, os, sys

out_dir, preset = sys.argv[1], sys.argv[2]

def load(label):
    path = f"{out_dir}/{label}/result.json"
    if not os.path.exists(path):
        return {"ok": False, "reason": "no_result_file", "tokens_per_sec": 0, "quant_label": label}
    return json.load(open(path))

d = load("default")
m = load("awq_marlin")

# Decision
winner = "default"
reason = ""
if m["ok"] and d["ok"]:
    if m["tokens_per_sec"] > d["tokens_per_sec"]:
        winner = "awq_marlin"
        reason = f"awq_marlin faster: {m['tokens_per_sec']} vs {d['tokens_per_sec']} tok/s"
    else:
        winner = "default"
        reason = f"default at-least-as-fast: {d['tokens_per_sec']} vs {m['tokens_per_sec']} tok/s"
elif m["ok"] and not d["ok"]:
    winner = "awq_marlin"
    reason = f"default failed ({d.get('reason')}), awq_marlin ok"
elif d["ok"] and not m["ok"]:
    winner = "default"
    reason = f"awq_marlin failed ({m.get('reason')}), default ok"
else:
    winner = "default"
    reason = f"both failed: default={d.get('reason')} awq_marlin={m.get('reason')}"

# winner.env semantics: empty QUANT means "let launch.sh use the preset default"
quant_export = "" if winner == "default" else winner

with open(f"{out_dir}/winner.env", "w") as f:
    if quant_export:
        f.write(f"QUANT={quant_export}\n")
    else:
        f.write(f"QUANT=\n")

summary = {
    "preset": preset,
    "winner": winner,
    "reason": reason,
    "default": d,
    "awq_marlin": m,
}
json.dump(summary, open(f"{out_dir}/summary.json", "w"), indent=2)
print(f"[decision] winner={winner}  reason={reason}")
print(f"[winner.env] QUANT={quant_export}")
PY

# Exit 0 only when at least one kernel succeeded.
if grep -q '"ok": true' "$OUT_DIR/default/result.json" 2>/dev/null \
|| grep -q '"ok": true' "$OUT_DIR/awq_marlin/result.json" 2>/dev/null; then
  log "decision written: $OUT_DIR/winner.env"
  exit 0
fi
log "WARNING: both smoke tests failed; falling back to preset default at eval time"
exit 1
