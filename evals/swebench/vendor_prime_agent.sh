#!/usr/bin/env bash
# Produce evals/swebench/docker/prime-agent-<ver>-vendor.tgz: the prime-agent
# footprint Dockerfile.rollout unpacks (tar -C /) instead of running the
# upstream installer at every per-instance image build.
#
#   vendor_prime_agent.sh           fetch our HF mirror (default; sha-pinned)
#   vendor_prime_agent.sh --build   regenerate from the upstream release
#                                   tarball (docker + network, ~3 min) and
#                                   print the sha256 to pin below
#
# Why vendored: upstream distributes prime-agent only through a curl
# installer / release bucket (NOT on public npm) and pruned every pre-0.10.0
# release mid-campaign (2026-10-09: releases/v0.8.1 went 404, every
# per-instance build failed, the rolling lane wrote 29 crash rows). The
# tarball is gitignored (0.8.1: 198 MB; 0.10.0 / 0.10.1: ~170 MB).
#
# 0.10.0 (Rust rewrite, `stable` channel) layout inside the tarball:
#   opt/prime-agent/                 the upstream linux-x64 payload, verbatim
#                                    (binary + prime-agent-runtime/ + skills/)
#   root/.prime/agent/kernel-venv/   the Python kernel venv the binary would
#                                    otherwise bootstrap on first run (uv +
#                                    managed CPython 3.11 + prime-agent-runtime
#                                    + the 12 default packages + the Python
#                                    skills as editable installs)
#   root/.local/share/uv/python/     the uv-managed CPython the venv links to
#   root/.local/bin/uv, uvx          uv itself (unused once the venv is warm)
# The venv is built with the payload at /opt/prime-agent and HOME=/root
# because the kernel-venv records ABSOLUTE paths (.bootstrap-version pins
# every skill's package path; the editable .pth files point into
# /opt/prime-agent/skills/<x>; pyvenv.cfg points at the managed CPython) —
# a venv built elsewhere makes prime-agent re-run `uv pip install` at session
# start, which under the rollout's --network none costs ~30 s and disables
# every Python skill. A warm venv at the recorded paths makes zero uv
# invocations (verified host-side under `unshare -rn`, 2026-10-19).
# `--build` runs the bootstrap in a plain ubuntu:22.04 container (the eval
# images' glibc, 2.35; no system python, so uv's managed 3.11 is the only
# candidate). PyPI resolution is not byte-reproducible across builds — the
# pinned sha256 is the artifact we built, mirrored and ran the audit on.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VER="${PRIME_AGENT_VERSION:-$(sed -n 's/^ARG PRIME_AGENT_VERSION=\([^ ]*\).*/\1/p' "$HERE/docker/Dockerfile.rollout")}"
OUT="$HERE/docker/prime-agent-${VER}-vendor.tgz"
MIRROR_BASE="https://huggingface.co/datasets/mattbucci/swebench-scaffold-vendor/resolve/main"
MIRROR="$MIRROR_BASE/prime-agent-${VER}-vendor.tgz"
# Upstream release bucket (release-channel manifest: <base>/channels/stable).
UPSTREAM_BASE="${PRIME_AGENT_DOWNLOAD_BASE_URL:-https://pub-728493de92a943e2a9b2d17b4719f318.r2.dev}"
UPSTREAM="$UPSTREAM_BASE/releases/v${VER}/prime-agent-${VER}-linux-x64.tar.gz"
# The vendor tgz as built + mirrored (what the Dockerfile consumes).
declare -A SHA=(
  [0.8.1]=e0612c45fda6af13524ac6d6b32ddb09f97b107350b38b141d1746d8edde2242
  [0.10.0]=80094ff154063cb1c11d1abbaee6194022f91d16d3ee9ec5bc3b37363d41cb36
  [0.10.1]=8a67140f00cbed88f0783864d3f5f0369f887e2330595a580e651a040e06a78b
)
# The upstream linux-x64 release tarball (from the release's SHA256SUMS),
# the input of --build; also mirrored next to the vendor tgz.
declare -A SHA_UPSTREAM=(
  [0.10.0]=c16bd2af5e77b53f49b914a44742c4cf6a67c5ed5000041430b78dbd4c3bcbec
  [0.10.1]=21ec219183f4134dd23dce9572102e69456066fe374846525561c776d4d1945c
)

verify() {  # verify <file> <want>  — sha256 or die (and remove the file)
  local got; got="$(sha256sum "$1" | cut -d' ' -f1)"
  if [[ "$got" != "$2" ]]; then
    echo "sha256 mismatch for $1: got $got want $2" >&2; rm -f "$1"; exit 1
  fi
}

if [[ "${1:-}" == "--build" ]]; then
  [[ -n "${SHA_UPSTREAM[$VER]:-}" ]] || { echo "no upstream sha256 pinned for prime-agent $VER" >&2; exit 1; }
  WORK="$(mktemp -d "${TMPDIR:-/var/tmp}/prime-agent-vendor.XXXXXX")"
  mkdir -p "$WORK/in" "$WORK/out"
  IN="$WORK/in/prime-agent-${VER}-linux-x64.tar.gz"
  curl -fL --retry 3 -o "$IN" "$UPSTREAM" || curl -fL --retry 3 -o "$IN" "$MIRROR_BASE/prime-agent-${VER}-linux-x64.tar.gz"
  verify "$IN" "${SHA_UPSTREAM[$VER]}"
  cat > "$WORK/build.sh" <<'EOF'
set -euo pipefail
trap 'echo "build.sh: failed at line $LINENO" >&2' ERR
export HOME=/root DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq --no-install-recommends curl ca-certificates >/dev/null
mkdir -p /opt/prime-agent && tar -xzf "/vendor/in/prime-agent-${VER}-linux-x64.tar.gz" -C /opt/prime-agent
test "$(/opt/prime-agent/prime-agent --version 2>&1)" = "$VER"
# uv at /root/.local/bin (where the binary's ensure_uv looks); no rc edits.
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/root/.local/bin UV_NO_MODIFY_PATH=1 sh
export PATH=/root/.local/bin:$PATH UV_PYTHON_PREFERENCE=only-managed
# Install-time kernel preparation (what the upstream installer runs after
# extracting): uv python install 3.11, the venv, prime-agent-runtime from the
# payload sidecar, the default packages, the Python skills (editable).
/opt/prime-agent/prime-agent --prime-agent-bootstrap
# Second pass must be a no-op (nothing left to install): a warm venv is the
# whole point of this artifact.
/opt/prime-agent/prime-agent --prime-agent-bootstrap 2>&1 | tee /tmp/second-pass.log
grep -q '^kernel python: /root/.prime/agent/kernel-venv/bin/python' /tmp/second-pass.log
PY=/root/.prime/agent/kernel-venv/bin/python
"$PY" -c 'import rlm.repl, rlm.bash, rlm.mcp, rlm.harness, requests, httpx, yaml, tomli, dotenv, pandas, numpy, scipy, bs4, lxml, pydantic, tyro, dill'
# The install-time bootstrap records NO Python skills (.bootstrap-version
# python_skills: []); a SESSION resolves the kernel with the bundled skill set
# (edit, websearch, goal, …) and `uv pip install --editable`s whatever the
# record lacks — the first session of every rollout would do that, offline,
# and lose every skill. Warm it here: one headless session against a stub
# model server that answers with a single code-tool call (`ipython` through
# 0.10.0, `python_repl` since 0.10.1 — upstream #3400; PY_TOOL below) (the session
# then awaits kernel readiness, which installs the 9 skills and writes the
# record) and a final stop. The stub runs on the kernel venv's python.
cat > /tmp/stub_model.py <<'PYSTUB'
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
N = {"n": 0}
PY_TOOL = os.environ["PY_TOOL"]
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        b = json.dumps({"object": "list", "data": [{"id": "stub", "object": "model"}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        N["n"] += 1
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        def chunk(delta, fin=None):
            d = {"id": "stub", "object": "chat.completion.chunk", "created": int(time.time()), "model": "stub",
                 "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}
            if fin: d["usage"] = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            self.wfile.write(("data: " + json.dumps(d) + "\n\n").encode())
        if N["n"] == 1:
            chunk({"role": "assistant", "tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": PY_TOOL, "arguments": ""}}]})
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": json.dumps({"code": "import edit, websearch, goal; print('vendor-warm-ok')"})}}]})
            chunk({}, "tool_calls")
        else:
            chunk({"role": "assistant", "content": "OK"}); chunk({}, "stop")
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
HTTPServer(("127.0.0.1", 23334), H).serve_forever()
PYSTUB
"$PY" /tmp/stub_model.py & STUB=$!
mkdir -p /root/.prime/agent /tmp/warm && cd /tmp/warm
cat > /root/.prime/agent/models.json <<'J'
{"providers":{"stub":{"name":"stub","baseUrl":"http://127.0.0.1:23334/v1","api":"openai-completions","apiKey":"stub","compat":{"supportsDeveloperRole":false,"supportsReasoningEffort":false},"models":[{"id":"stub","reasoning":false,"contextWindow":262144,"maxTokens":32768}]}}}
J
sleep 1
echo "warm the kernel" | DO_NOT_TRACK=1 /opt/prime-agent/prime-agent --offline --model stub/stub -p 2>&1 | tee /tmp/warm.log
grep -q "^OK" /tmp/warm.log
kill "$STUB"; wait "$STUB" 2>/dev/null || true
# The recorded set is whatever a headless `-p` session resolves — exactly the
# shape every rollout runs, so parity with this warm session is the property.
# 0.10.0 recorded all 9 payload skills; 0.10.1 records 7 (agent_observe and
# rlm_heartbeat are daemon-only there: "daemonless sessions: subagents and
# wake-ups unavailable", upstream #3434). Gate: non-empty, a subset of the
# payload's Python skills, and the seven a coding session uses present.
"$PY" - <<'PYCHK'
import json, os, re
d = json.load(open("/root/.prime/agent/kernel-venv/.bootstrap-version"))
names = sorted(s["import_name"] for s in d["python_skills"])
payload = sorted(n.replace("-", "_") for n in os.listdir("/opt/prime-agent/skills")
                 if os.path.exists(f"/opt/prime-agent/skills/{n}/pyproject.toml"))
print("recorded python skills:", names)
print("payload python skills: ", payload, "not recorded:", sorted(set(payload) - set(names)))
core = {"agent_message", "attach_image", "compact", "edit", "goal", "refine", "websearch"}
assert names and set(names) <= set(payload), (names, payload)
assert core <= set(names), sorted(core - set(names))
assert all(s["package_path"].startswith("/opt/prime-agent/skills/") for s in d["python_skills"])
open("/tmp/recorded-skills.txt", "w").write(" ".join(names))
PYCHK
"$PY" -c "import $(tr ' ' ',' < /tmp/recorded-skills.txt)"
{
  echo "prime-agent ${VER} vendor footprint, built $(date -u +%FT%TZ) in ubuntu:22.04 by evals/swebench/vendor_prime_agent.sh --build"
  echo "upstream: prime-agent-${VER}-linux-x64.tar.gz sha256=${UPSTREAM_SHA}"
  echo "uv: $(uv --version)"; echo "kernel python: $("$PY" -c 'import sys; print(sys.version.split()[0], sys.executable)')"
  echo "--- uv pip list"; uv pip list --python "$PY"
} > /opt/prime-agent/VENDOR-MANIFEST.txt
cat /opt/prime-agent/VENDOR-MANIFEST.txt
# Prune: caches and anything per-user/per-session (the harness writes
# models.json + settings.json per run; sessions must start empty). The uv
# cache is not needed at runtime (editable installs point at the payload).
rm -rf /root/.cache /root/.local/share/uv/python/.temp /root/.local/bin/env /root/.local/bin/env.fish /tmp/warm
find /root/.prime/agent -mindepth 1 -maxdepth 1 ! -name kernel-venv -exec rm -rf {} +
ls -A /root/.prime/agent
cd /
tar --numeric-owner --owner=0 --group=0 -czf "/vendor/out/prime-agent-${VER}-vendor.tgz" \
    opt/prime-agent root/.prime/agent/kernel-venv root/.local/share/uv root/.local/bin
chown "$BUILD_UID" "/vendor/out/prime-agent-${VER}-vendor.tgz"
EOF
  # The model-facing code tool: `ipython` through 0.10.0, `python_repl` from 0.10.1 (#3400).
  PY_TOOL=ipython
  [[ "$(printf '%s\n' 0.10.1 "$VER" | sort -V | head -1)" == 0.10.1 ]] && PY_TOOL=python_repl
  docker run --rm -v "$WORK:/vendor" -e VER="$VER" -e UPSTREAM_SHA="${SHA_UPSTREAM[$VER]}" -e BUILD_UID="$(id -u)" -e PY_TOOL="$PY_TOOL" \
      ubuntu:22.04 bash /vendor/build.sh
  mv "$WORK/out/prime-agent-${VER}-vendor.tgz" "$OUT"
  rm -rf "$WORK"
  echo "$OUT  sha256=$(sha256sum "$OUT" | cut -d' ' -f1)  (pin this in SHA[$VER] and mirror it)"
  exit 0
fi

if [[ -z "${SHA[$VER]:-}" || "${SHA[$VER]}" == __* ]]; then
  echo "no pinned vendor sha256 for prime-agent $VER — run $0 --build, mirror the result, pin it" >&2; exit 1
fi
if [[ -f "$OUT" && "$(sha256sum "$OUT" | cut -d' ' -f1)" == "${SHA[$VER]}" ]]; then
  echo "$OUT  sha256=${SHA[$VER]} (already present)"; exit 0
fi
curl -fL --retry 3 -o "$OUT.partial" "$MIRROR"
verify "$OUT.partial" "${SHA[$VER]}"
mv "$OUT.partial" "$OUT"
echo "$OUT  sha256=${SHA[$VER]}"
