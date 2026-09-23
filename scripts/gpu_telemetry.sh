#!/usr/bin/env bash
# gpu_telemetry.sh — 30 s GPU telemetry to CSV so a server hang / watchdog kill
# has a temperature-power-clock history behind it (R9700 lost a card after a
# watchdog kill on 2026-09-22 with no telemetry to read; their port is sysfs,
# ours is nvidia-smi's query loop). Read-only; safe beside a running lane.
#
#   scripts/gpu_telemetry.sh            # foreground
#   setsid scripts/gpu_telemetry.sh </dev/null >/dev/null 2>&1 & disown   # detached
#
# Output: $GPU_TELEMETRY_DIR/nvidia-smi-<start>.csv (default /var/tmp/gpu-telemetry,
# outside the 10 d /tmp cleaner). One row per GPU per interval; throttle reasons
# are the bitmask (0x4 = SW power cap, the 260 W profile's normal state).
set -u
DIR="${GPU_TELEMETRY_DIR:-/var/tmp/gpu-telemetry}"
INTERVAL="${INTERVAL:-30}"
mkdir -p "$DIR"
OUT="$DIR/nvidia-smi-$(date +%Y%m%dT%H%M%S).csv"
echo $$ > "$DIR/pid"
exec nvidia-smi \
  --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used,pstate,clocks_throttle_reasons.active,fan.speed \
  --format=csv -l "$INTERVAL" > "$OUT"
