#!/usr/bin/env bash
# rollout_image_janitor.sh — delete-after-use for per-instance rollout images.
#
# The 2026-08-31 six-scaffold rollout Dockerfile adds ~5.6 GB of UNIQUE layers per
# instance (dcode conda env + node 22 tarball + npm prefixes + scaffold homes) on
# top of each instance's swebench eval base — layers are per-parent, so nothing is
# shared across instances. 300 instances ≈ 1.7 TB: cannot persist on /.
#
# Rule: an instance's rollout image is deletable once its prediction exists in the
# lane currently rolling (= most recently modified runs/*/predictions.jsonl).
# Later lanes rebuild the image on demand — ensure_rollout_image() in
# docker_rollout.py already treats a missing tag as "build it" (this is the same
# path the post-scaffold-edit image nuke relies on). The image of the instance
# currently rolling is protected automatically: its container holds a reference,
# so `docker rmi` fails and we retry next pass.
# swebench/sweb.eval.* base images are never touched.
set -u
RUNS_DIR="${RUNS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/runs}"
INTERVAL="${INTERVAL:-600}"
while true; do
  lane=$(ls -t "$RUNS_DIR"/*/predictions.jsonl 2>/dev/null | head -1)
  if [ -n "${lane:-}" ]; then
    n=0
    while IFS= read -r iid; do
      [ -z "$iid" ] && continue
      docker image inspect "swebench-rollout/${iid}:latest" >/dev/null 2>&1 || continue
      docker rmi "swebench-rollout/${iid}:latest" >/dev/null 2>&1 && n=$((n+1))
    done < <(grep -o '"instance_id"[[:space:]]*:[[:space:]]*"[^"]*"' "$lane" | sed 's/.*"\([^"]*\)"$/\1/')
    # Per-instance build cache has no cross-instance reuse; idle cache is dead weight.
    # BUT prune only while the driver is in its model-rollout phase (a rollout
    # container is up) and no docker pull/build is in flight: containerd content
    # GC races an in-progress pull's ingest ("commit failed: rename ... no such
    # file", 2 lost instances on 2026-09-01).
    if [ -n "$(docker ps -q --filter name=swebench-rollout- 2>/dev/null)" ] && ! pgrep -f 'docker (pull|build) ' >/dev/null; then
      docker builder prune -af >/dev/null 2>&1
      docker image prune -f >/dev/null 2>&1  # dangling: old layers left by moved tags / refused rmi
    fi
    [ "$n" -gt 0 ] && echo "[janitor $(date '+%F %T')] lane=$(basename "$(dirname "$lane")") removed=${n} images; / free=$(df --output=avail -h / | tail -1 | tr -d ' ')"
  fi
  sleep "$INTERVAL"
done
