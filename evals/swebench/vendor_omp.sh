#!/usr/bin/env bash
# Produce evals/swebench/docker/omp-<ver>-linux-x64, the oh-my-pi (omp)
# standalone binary Dockerfile.rollout installs for the "omp" scaffold.
#
#   vendor_omp.sh            fetch the GitHub release asset (default)
#   vendor_omp.sh --mirror   fetch our HF mirror instead
#
# Why a vendored binary and not `npm install -g @oh-my-pi/pi-coding-agent`:
# the npm package declares `engines.bun >= 1.3.14` (no node entrypoint), and
# the project's own installer (omp.sh/install) pulls the same prebuilt binary
# from its build service. Fetching 255 MB at every per-instance image build
# is the prime-agent outage class again (upstream pruned a release mid-lane,
# 2026-10-09; npm ETARGET blip 2026-10-12) — so the asset is fetched ONCE,
# sha256-pinned here against the release's SHA256SUMS.txt, and bind-mounted
# into the image build. The binary is gitignored (255 MB). glibc floor is
# 2.17 (checked with objdump -T): every SWE-bench eval image qualifies.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VER="${OMP_VERSION:-$(sed -n 's/^ARG OMP_VERSION=\([^ ]*\).*/\1/p' "$HERE/docker/Dockerfile.rollout")}"
OUT="$HERE/docker/omp-${VER}-linux-x64"
RELEASE="https://github.com/can1357/oh-my-pi/releases/download/v${VER}/omp-linux-x64"
MIRROR="https://huggingface.co/datasets/mattbucci/swebench-scaffold-vendor/resolve/main/omp-${VER}-linux-x64"
declare -A SHA=(
  [18.12.2]=9513ef9f19d7991aac6ae0cb6077bcf16662092ddde861c002a91cedc6f47f9a
)

if [[ -z "${SHA[$VER]:-}" ]]; then
  echo "no pinned sha256 for omp $VER — add it from the release's SHA256SUMS.txt first" >&2; exit 1
fi
if [[ -f "$OUT" && "$(sha256sum "$OUT" | cut -d' ' -f1)" == "${SHA[$VER]}" ]]; then
  echo "$OUT  sha256=${SHA[$VER]} (already present)"; exit 0
fi
SRC="$RELEASE"; [[ "${1:-}" == "--mirror" ]] && SRC="$MIRROR"
curl -fL --retry 3 -o "$OUT.partial" "$SRC"
got="$(sha256sum "$OUT.partial" | cut -d' ' -f1)"
if [[ "$got" != "${SHA[$VER]}" ]]; then
  echo "sha256 mismatch for $VER: got $got want ${SHA[$VER]}" >&2; rm -f "$OUT.partial"; exit 1
fi
chmod 0755 "$OUT.partial" && mv "$OUT.partial" "$OUT"
echo "$OUT  sha256=$got"
