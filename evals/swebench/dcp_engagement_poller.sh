#!/usr/bin/env bash
# dcp_engagement_poller.sh — passive per-instance engagement receipts for the
# opencode-dcp bake-off lane.
#
# DCP (@tarquinen/opencode-dcp 3.x) prunes inside opencode's
# experimental.chat.messages.transform hook — nothing reaches the scaffold's
# stdout, so the per-instance rollout log cannot show whether pruning happened.
# The plugin persists $HOME/.local/share/opencode/storage/plugin/dcp/<session>.json
# ONLY after it has pruned (saveSessionState follows every tokensSaved update),
# with stats.totalPruneTokens. Containers are removed after diff capture, so we
# snapshot that file while the container is alive. Read-only docker exec; never
# touches the rollout.
#
# Output: $OUT/<instance_id>.json (last snapshot of the state file) and
#         $OUT/.seen/<instance_id> for every dcp-lane container observed, so
#         engagement rate = state files / seen.
set -u
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUT="${OUT:-$REPO_DIR/benchmarks/quality/dcp-engagement}"
INTERVAL="${INTERVAL:-30}"
mkdir -p "$OUT/.seen"
while true; do
  C=$(docker ps --format '{{.Names}}' | grep '^swebench-rollout-' | head -1)
  if [ -n "$C" ] && docker inspect -f '{{.Config.Env}}' "$C" 2>/dev/null | grep -q 'HOME=/opt/dcp-home'; then
    iid=${C#swebench-rollout-}; iid=${iid%-*}
    touch "$OUT/.seen/$iid"
    if docker exec "$C" bash -c 'cat /opt/dcp-home/.local/share/opencode/storage/plugin/dcp/*.json' > "$OUT/$iid.json.tmp" 2>/dev/null && [ -s "$OUT/$iid.json.tmp" ]; then
      mv "$OUT/$iid.json.tmp" "$OUT/$iid.json"
    else
      rm -f "$OUT/$iid.json.tmp"
    fi
  fi
  sleep "$INTERVAL"
done
