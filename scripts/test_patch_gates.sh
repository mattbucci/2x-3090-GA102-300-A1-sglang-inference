#!/bin/bash
# test_patch_gates.sh — the 3-gate patch-hygiene test from patches/README.md,
# scripted so every rebase / new-patch commit runs the same protocol instead of
# re-implementing it by hand (the by-hand version has been rebuilt at every
# rebase since v0.5.13).
#
# Gates:
#   (a) every patch in PATCH_DIR applies clean, in glob order, on a PRISTINE
#       $SGLANG_TAG worktree (no skips — a skip on pristine means a broken chain
#       that setup.sh's idempotent loop would hide);
#   (b) the patched pristine worktree is byte-identical to the live SGLANG_DIR
#       tree (catches live-tree edits that never made it into a patch);
#   (c) every patch FAILS `git apply --check` on the live (already-patched)
#       tree (rerun safety — a patch that still applies twice is mis-anchored,
#       the old-026 lookalike-block failure mode).
#
# Usage:
#   scripts/test_patch_gates.sh                        # gates for the default stack
#   SGLANG_TAG=v0.5.15 SGLANG_DIR=/data/sglang-rebase-v0515 \
#     PATCH_DIR=patches/v0.5.15 scripts/test_patch_gates.sh   # staged stack
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO/scripts/common.sh" 2>/dev/null || true

PATCH_DIR="${PATCH_DIR:-$REPO/patches}"
case "$PATCH_DIR" in /*) ;; *) PATCH_DIR="$REPO/$PATCH_DIR" ;; esac
[ -d "$PATCH_DIR" ] || { echo "FATAL: PATCH_DIR $PATCH_DIR missing"; exit 2; }
SGLANG_DIR="${SGLANG_DIR:?set SGLANG_DIR (live serving tree)}"
SGLANG_TAG="${SGLANG_TAG:-$(git -C "$SGLANG_DIR" describe --tags --abbrev=0 2>/dev/null)}"
[ -n "$SGLANG_TAG" ] || { echo "FATAL: cannot infer SGLANG_TAG from $SGLANG_DIR"; exit 2; }

PATCHES=("$PATCH_DIR"/*.patch)
N=${#PATCHES[@]}
WT="$(mktemp -d /tmp/patch-gates-XXXXXX)/wt"
FAIL=0
echo "== 3-gate patch test: $N patches from $PATCH_DIR vs $SGLANG_TAG =="

# --- gate (a): glob-order apply on pristine tag worktree ---
git -C "$SGLANG_DIR" worktree add -f "$WT" "$SGLANG_TAG" >/dev/null 2>&1 \
  || { echo "FATAL: cannot create worktree at $SGLANG_TAG (tag fetched?)"; exit 2; }
A_OK=0
for p in "${PATCHES[@]}"; do
  if git -C "$WT" apply "$p" 2>/dev/null; then A_OK=$((A_OK+1));
  else echo "  GATE-A FAIL (does not apply on pristine): $(basename "$p")"; FAIL=1; fi
done
echo "gate (a): $A_OK/$N apply clean on pristine $SGLANG_TAG"

# --- gate (b): byte-identity vs live tree ---
DIFFS=$(diff -rq "$WT/python/sglang" "$SGLANG_DIR/python/sglang" 2>&1 \
  | grep -v "_version.py\|egg-info\|__pycache__" || true)
if [ -n "$DIFFS" ]; then
  echo "gate (b): FAIL — live tree differs from pristine+patches:"; echo "$DIFFS" | head -10; FAIL=1
else
  echo "gate (b): live tree byte-identical to pristine+patches"
fi

# --- gate (c): rerun safety on the live tree ---
C_OK=0
for p in "${PATCHES[@]}"; do
  if git -C "$SGLANG_DIR" apply --check "$p" 2>/dev/null; then
    echo "  GATE-C FAIL (still applies on patched tree): $(basename "$p")"; FAIL=1
  else C_OK=$((C_OK+1)); fi
done
echo "gate (c): $C_OK/$N correctly fail on the patched tree"

git -C "$SGLANG_DIR" worktree remove --force "$WT" >/dev/null 2>&1
rmdir "$(dirname "$WT")" 2>/dev/null
[ "$FAIL" = 0 ] && echo "== ALL GATES PASS ==" || echo "== GATE FAILURES (see above) =="
exit $FAIL
