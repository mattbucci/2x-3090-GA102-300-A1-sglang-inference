#!/usr/bin/env bash
# Produce evals/swebench/docker/prime-agent-<ver>-vendor.tgz, the prime-agent
# installer footprint Dockerfile.rollout unpacks instead of running the
# installer (upstream pruned pre-0.10.0 releases from its bucket on
# 2026-10-09; 0.8.1 is no longer downloadable from them).
#
#   vendor_prime_agent.sh                 fetch the mirror (default)
#   vendor_prime_agent.sh --from-image X  re-capture from an image the
#                                         installer built (e.g. an old
#                                         swebench-rollout/<iid> image)
#
# The tarball is gitignored (198 MB). The capture is byte-for-byte what the
# installer left behind: the npm package under /opt/node/lib/node_modules,
# /opt/node/bin/prime-agent's target, the postinstall kernel venv + fd under
# /root/.prime, uv + its managed CPython under /root/.local and /root/.config.
# (The ~/.bashrc / ~/.profile `. "$HOME/.local/bin/env"` lines the installer
# appends are re-applied by the Dockerfile step.)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VER="${PRIME_AGENT_VERSION:-$(sed -n 's/^ARG PRIME_AGENT_VERSION=\([^ ]*\).*/\1/p' "$HERE/docker/Dockerfile.rollout")}"
OUT="$HERE/docker/prime-agent-${VER}-vendor.tgz"
MIRROR="https://huggingface.co/datasets/mattbucci/swebench-scaffold-vendor/resolve/main/prime-agent-${VER}-vendor.tgz"
declare -A SHA=(
  [0.8.1]=e0612c45fda6af13524ac6d6b32ddb09f97b107350b38b141d1746d8edde2242
)

if [[ "${1:-}" == "--from-image" ]]; then
  IMG="${2:?image}"
  NODE_REAL="$(docker run --rm --network=none --entrypoint readlink "$IMG" -f /opt/node)"
  docker run --rm --network=none --entrypoint tar "$IMG" -C / \
    --transform "s,^${NODE_REAL#/}/,opt/node/," -czf - \
    "${NODE_REAL#/}/lib/node_modules/prime-agent" root/.prime root/.local/share/uv \
    root/.local/bin root/.config/uv root/.config/fish > "$OUT.partial"
else
  curl -fL --retry 3 -o "$OUT.partial" "$MIRROR"
fi
got="$(sha256sum "$OUT.partial" | cut -d' ' -f1)"
if [[ -n "${SHA[$VER]:-}" && "$got" != "${SHA[$VER]}" ]]; then
  echo "sha256 mismatch for $VER: got $got want ${SHA[$VER]}" >&2; rm -f "$OUT.partial"; exit 1
fi
mv "$OUT.partial" "$OUT"
echo "$OUT  sha256=$got"
