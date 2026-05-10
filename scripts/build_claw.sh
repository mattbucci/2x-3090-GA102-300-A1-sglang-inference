#!/bin/bash
# Build the claw-code Rust binary inside a Debian Bullseye container so the
# resulting binary links against GLIBC 2.31 — old enough to run on every
# SWE-bench eval base image (Ubuntu 22.04 / GLIBC 2.35 and similar). A
# native build on the host (Arch / GLIBC 2.39) produces a binary that
# fails inside the rollout container with `version GLIBC_2.39 not found`.
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

# Pinned: rust:bullseye uses Debian 11 base (GLIBC 2.31). Newer Debian/Ubuntu
# images use GLIBC ≥ 2.34, which would re-introduce the cross-distro break.
RUST_IMAGE="${CLAW_BUILDER_IMAGE:-rust:bullseye}"

if [ ! -d "$SRC_DIR" ]; then
    echo "Cloning ultraworkers/claw-code into $SRC_DIR..."
    git clone --depth 1 https://github.com/ultraworkers/claw-code.git "$SRC_DIR"
fi

# Use a separate target dir so a stale host-side target/ (built against a
# different GLIBC) doesn't poison cargo's incremental cache.
BUILD_CACHE="${CLAW_BUILD_CACHE:-/tmp/claw-target}"
mkdir -p "$BUILD_CACHE"

echo "Building claw inside $RUST_IMAGE (target=$BUILD_CACHE)..."
docker run --rm \
    -v "$SRC_DIR/rust:/work:ro" \
    -v "$BUILD_CACHE:/build" \
    -e CARGO_TARGET_DIR=/build \
    -w /work \
    "$RUST_IMAGE" \
    cargo build --release --workspace

BIN="$BUILD_CACHE/release/claw"
if [ ! -x "$BIN" ]; then
    echo "ERROR: build did not produce $BIN" >&2
    exit 1
fi

cp "$BIN" "$DST"
echo "Staged claw at $DST"
"$DST" --version
