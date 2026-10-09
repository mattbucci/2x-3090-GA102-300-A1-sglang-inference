#!/usr/bin/env python3
"""classify_log under capture-at-wall (2026-10-08): rows rolled before the
wall-hit capture (no `patch_source`) or whose capture failed are harness
defects to re-roll; a clean tree at the wall stays model_timeout; a wall hit
with edits on disk is real_diff. Run: python3 evals/swebench/test_audit_predictions_wall_capture.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_predictions import classify_log  # noqa: E402

LOG = "# stdout\n{}\n# stderr\nisolation: network=none\n"
PS_TAIL = '{"type":"step_finish","part":{"reason":"tool-calls"}}'   # rfind landed on a `ps`-echoed marker
DIFF = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"

cases = [
    # (rc, patch, elapsed, patch_source, expected)
    (124, "", 1801.0, None, "infra_wall_capture_defect"),            # pre-capture wall, empty by construction
    (124, PS_TAIL, 1803.0, None, "infra_wall_capture_defect"),       # pre-capture wall, marker collision
    (124, DIFF, 1802.0, None, "real_diff"),                          # finished inside the wall, diff printed
    (124, "", 1805.0, "wall-exec-failed", "infra_wall_capture_defect"),
    (124, "", 1850.0, "wall-exec", "model_timeout"),                 # tree clean at the wall
    (124, DIFF, 1850.0, "wall-exec", "real_diff"),                   # edits on disk at the wall
    (124, DIFF, 1801.0, "stdout-fallback", "real_diff"),
    (0, "", 300.0, "stdout", "model_silent"),
    (0, DIFF, 300.0, "stdout", "real_diff"),
    (0, DIFF, 300.0, None, "real_diff"),
    (1, "", 12.0, "stdout", "infra_rollout_nonzero_rc"),
]
fails = []
for rc, patch, elapsed, src, want in cases:
    got, _ = classify_log(LOG.format(""), rc, patch, elapsed, src)
    if got != want:
        fails.append(f"rc={rc} src={src!r} patch_len={len(patch)} -> {got!r}, want {want!r}")
for f in fails:
    print("FAIL", f)
print(f"{len(cases) - len(fails)}/{len(cases)} ok")
sys.exit(1 if fails else 0)
