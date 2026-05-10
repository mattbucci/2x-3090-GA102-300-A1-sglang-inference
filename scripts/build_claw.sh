#!/bin/bash
# Build the claw-code Rust binary and stage it into evals/swebench/docker/
# for Dockerfile.rollout's COPY step.
#
# Why pre-build instead of building inside each rollout image:
#   - Rust toolchain + cargo build adds ~1.5 GB and ~5 min per image.
#   - With 300 instances × 5 models, the per-image cost dominates total time.
#   - The claw release binary is ~17 MB self-contained — COPY in is cheap.
#
# Re-run this when claw-code main moves; the rollout images won't pick up the
# new binary until they're rebuilt (Dockerfile.rollout ships claw as a
# regular file, no remote pull). For matrix runs that span days, build once
# at the start and treat that revision as the bake-off snapshot.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$REPO_DIR/components/claw-code"
DST="$REPO_DIR/evals/swebench/docker/claw"

if [ ! -d "$SRC_DIR" ]; then
    echo "Cloning ultraworkers/claw-code into $SRC_DIR..."
    git clone --depth 1 https://github.com/ultraworkers/claw-code.git "$SRC_DIR"
fi

cd "$SRC_DIR/rust"
echo "Building claw (cargo build --release --workspace)..."
cargo build --release --workspace

BIN="$SRC_DIR/rust/target/release/claw"
if [ ! -x "$BIN" ]; then
    echo "ERROR: build did not produce $BIN" >&2
    exit 1
fi

cp "$BIN" "$DST"
echo "Staged claw at $DST"
"$DST" --version
