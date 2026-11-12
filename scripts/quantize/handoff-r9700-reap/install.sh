#!/bin/bash
# Copy the unified Qwen3_5Moe REAP tooling from the 3090 checkout into an R9700
# checkout, byte-identical, and verify against SHA256SUMS.
#   scripts/quantize/handoff-r9700-reap/install.sh <r9700-checkout>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/../../.." && pwd)"
DST="${1:?usage: install.sh <r9700-checkout>}"
[[ -d "$DST/ream-patches" && -d "$DST/scripts/quantize" ]] || { echo "ERROR: $DST is not an R9700 checkout" >&2; exit 1; }

(cd "$SRC" && sha256sum -c "$HERE/SHA256SUMS") || { echo "ERROR: source files differ from the Gate-A-green set" >&2; exit 1; }

# 3090 keeps the helper in patches/, R9700 in ream-patches/ — the scripts
# resolve whichever directory holds the file.
install -m 644 "$SRC/patches/qwen3_5moe_unfused_experts.py" "$DST/ream-patches/qwen3_5moe_unfused_experts.py"
install -m 644 "$SRC/scripts/quantize/run_reap.py" "$DST/scripts/quantize/run_reap.py"
install -m 644 "$SRC/scripts/quantize/test_qwen3_5moe_unfuse.py" "$DST/scripts/quantize/test_qwen3_5moe_unfuse.py"
install -m 755 "$HERE/gates.sh" "$DST/scripts/quantize/reap_gates.sh"

for f in patches/qwen3_5moe_unfused_experts.py:ream-patches/qwen3_5moe_unfused_experts.py \
         scripts/quantize/run_reap.py:scripts/quantize/run_reap.py \
         scripts/quantize/test_qwen3_5moe_unfuse.py:scripts/quantize/test_qwen3_5moe_unfuse.py; do
    cmp "$SRC/${f%%:*}" "$DST/${f##*:}"
done
echo "installed (byte-identical). Remaining manual edits: see $HERE/README.md (run_reap.sh REAP_ENV default, ream-patches/README.md row)."
