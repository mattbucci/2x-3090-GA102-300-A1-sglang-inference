#!/usr/bin/env bash
# rtk_engagement_poller.sh — passive per-instance engagement receipts for the
# little-coder-rtk bake-off lane (sibling of dcp_engagement_poller.sh).
#
# rtk rewrites bash tool calls inside pi (extension loaded via -e); the pi
# session jsonl records the PRE-mutation command, so the per-instance rollout
# log cannot show whether a rewrite executed. rtk itself keeps a savings ledger
# at $HOME/.local/share/rtk/history.db, appended on every EXECUTED `rtk <cmd>`
# (not on `rtk rewrite` lookups) — `rtk gain -f json` summarises it
# (total_commands, total_saved). Containers are removed after diff capture, so
# we snapshot while the container is alive. Read-only docker exec / cp; never
# touches the rollout.
#
# Output: $OUT/<instance_id>.json  (last `rtk gain -f json` snapshot)
#         $OUT/<instance_id>.db    (copy of history.db, per-command detail)
#         $OUT/.seen/<instance_id> for every rtk-lane container observed,
#         so engagement rate = snapshots with total_commands>0 / seen.
set -u
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUT="${OUT:-$REPO_DIR/benchmarks/quality/rtk-engagement}"
INTERVAL="${INTERVAL:-30}"
mkdir -p "$OUT/.seen"
while true; do
  C=$(docker ps --format '{{.Names}}' | grep '^swebench-rollout-' | head -1)
  if [ -n "$C" ] && docker inspect -f '{{.Config.Env}}' "$C" 2>/dev/null | grep -q 'HOME=/opt/rtk-home'; then
    iid=${C#swebench-rollout-}; iid=${iid%-*}
    touch "$OUT/.seen/$iid"
    if docker exec -e HOME=/opt/rtk-home -e RTK_TELEMETRY_DISABLED=1 "$C" rtk gain -f json > "$OUT/$iid.json.tmp" 2>/dev/null && [ -s "$OUT/$iid.json.tmp" ]; then
      mv "$OUT/$iid.json.tmp" "$OUT/$iid.json"
      docker cp "$C:/opt/rtk-home/.local/share/rtk/history.db" "$OUT/$iid.db.tmp" >/dev/null 2>&1 && mv "$OUT/$iid.db.tmp" "$OUT/$iid.db" || rm -f "$OUT/$iid.db.tmp"
    else
      rm -f "$OUT/$iid.json.tmp"
    fi
  fi
  sleep "$INTERVAL"
done
