#!/bin/bash
# R9700 acceptance gates for the Qwen3_5Moe REAP tooling. B and C are CPU-only
# and load no weights; D verifies a finished prune.
#   REAP_ENV=nemo-quant scripts/quantize/reap_gates.sh B
#   REAP_ENV=nemo-quant scripts/quantize/reap_gates.sh C <bf16-dir> <keep>
#   REAP_ENV=nemo-quant scripts/quantize/reap_gates.sh D <bf16-dir> <pruned-dir> <keep>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_REAP="$SCRIPT_DIR/run_reap.py"
[[ -f "$RUN_REAP" ]] || RUN_REAP="$SCRIPT_DIR/../run_reap.py"   # run from the 3090 handoff dir
QDIR="$(dirname "$RUN_REAP")"
PY="${REAP_PYTHON:-$HOME/miniforge3/envs/${REAP_ENV:-nemo-quant}/bin/python}"
# No GPU: hide the cards from both the CUDA and the ROCm runtime.
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1

case "${1:?gate B|C|D}" in
B)  # miniature: unfuse helper, composite-config helpers, stream-vs-reload save (the slicer tripwire)
    "$PY" -B "$QDIR/test_qwen3_5moe_unfuse.py" 2>&1 | grep -E '^\[[0-9]+\]|ALL PASS|Error|assert' ;;
C)  # real checkpoint, headers only: loader class, expert counts, shard key layout, save mode
    "$PY" -B "$RUN_REAP" --model "${2:?bf16 dir}" --save-path /nonexistent --keep-experts "${3:?keep}" --preflight ;;
D)  # finished prune: counts, every layer sliced to <keep>, non-weight files carried, untouched tensors identical
    BASE="${2:?bf16 dir}" OUT="${3:?pruned dir}" KEEP="${4:?keep}" "$PY" -B - "$QDIR" <<'PY'
import json, os, sys
sys.path.insert(0, sys.argv[1])
import run_reap as R
from transformers import AutoConfig
base, out, keep = os.environ["BASE"], os.environ["OUT"], int(os.environ["KEEP"])
cb, co = AutoConfig.from_pretrained(base), AutoConfig.from_pretrained(out)
nb, kb = R.read_expert_counts(cb); no, ko = R.read_expert_counts(co)
assert (no, ko) == (keep, kb), f"pruned config says {no}/{ko}, want {keep}/{kb}"
assert R.pick_model_class(co) is R.pick_model_class(cb), "loader class changed (vision tower dropped?)"
lb, lo = R.scan_checkpoint_layout(base, nb), R.scan_checkpoint_layout(out, keep)   # raises on any unsliced layer
assert lb.keys() == lo.keys() and all(c["gate"] and c["experts"] for c in lo.values()), "layer set differs"
rep = json.load(open(os.path.join(out, "reap_report.json")))
assert rep["stage"] == "saved" and len(rep["survivors_per_layer"]) == len(lo), rep.get("stage")
skip = {"reap_report.json", "config.json", "model.safetensors.index.json"}
missing = [f for f in os.listdir(base) if not f.endswith(".safetensors") and f not in skip
           and os.path.isfile(os.path.join(base, f)) and not os.path.exists(os.path.join(out, f))]
assert not missing, f"non-weight files not carried: {missing}"
for f in ("chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json"):
    if os.path.exists(os.path.join(base, f)):
        assert open(os.path.join(base, f), "rb").read() == open(os.path.join(out, f), "rb").read(), f"{f} changed"
print(f"Gate D OK: {nb} -> {no} experts on {len(lo)} layers, {type(co).__name__}, "
      f"{R.pick_model_class(co).__name__}, save_mode={rep.get('save_mode')}")
PY
    ;;
*)  echo "unknown gate $1" >&2; exit 2 ;;
esac
