#!/bin/bash
# Boot-time cooling profile for 2x RTX 3090 on this rig.
#
# Run once per boot via gpu-cooling.service (Type=oneshot). Sets:
#   1. NVIDIA persistence mode on  (keeps the driver state warm so per-app
#      cold starts don't re-init the GPU and wipe the fan/power config)
#   2. Power limit 260 W per card (down from default 350 W). Picked to keep
#      DDR5 SPD sensors below ALARM HIGH (55 C) under sustained TP=2 bake-off
#      load; correlated with kernel-BUG hard resets on the prior chassis at
#      stock 350 W.
#   3. Corsair Commander Core XT case fans to 100% via `liquidctl`. The
#      device reports "(broken)" on the latest liquidctl but `set fanN speed`
#      still drives the duty; we ignore that label.
#   4. Seed each GPU fan to GPU_FAN_FLOOR_PCT (default 75%) via NVML
#      (nvmlDeviceSetFanSpeed_v2). This consumer-Ampere card has no hwmon
#      pwm endpoint under the open driver, so we drive fans through the
#      NVML API (requires root); the live curve is then maintained by
#      gpu-fan-curve.service.
#
# Tunables (via environment or unit-level Environment= lines):
#   POWER_LIMIT_W       default 260
#   GPU_FAN_FLOOR_PCT   default 75

set -u
LOG_TAG="gpu-cooling"
log() { logger -t "$LOG_TAG" -- "$*"; echo "[$LOG_TAG] $*"; }

POWER_LIMIT_W="${POWER_LIMIT_W:-260}"
GPU_FAN_FLOOR_PCT="${GPU_FAN_FLOOR_PCT:-75}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "ERROR: nvidia-smi not found"; exit 1
fi

log "enabling NVIDIA persistence mode"
nvidia-smi -pm 1 >/dev/null || log "persistence mode failed (continuing)"

log "setting power limit to ${POWER_LIMIT_W} W on every GPU"
for idx in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    if nvidia-smi -i "$idx" -pl "$POWER_LIMIT_W" >/dev/null 2>&1; then
        log "GPU $idx power limit -> ${POWER_LIMIT_W} W"
    else
        log "GPU $idx power limit failed (continuing)"
    fi
done

# Corsair Commander Core XT chassis fans. liquidctl labels the device as
# "(broken)" in newer releases but `set fanN speed` still applies. We push
# every available channel to 100% — over-cooling is fine here, the case
# is the heat sink for the GPUs.
if command -v liquidctl >/dev/null 2>&1; then
    if liquidctl list 2>/dev/null | grep -q "Corsair Commander"; then
        log "pushing Corsair Commander chassis fans to 100%"
        for ch in fan1 fan2 fan3 fan4 fan5 fan6; do
            liquidctl set "$ch" speed 100 >/dev/null 2>&1 || true
        done
    else
        log "Corsair Commander not detected via liquidctl (skipping case fans)"
    fi
else
    log "liquidctl not installed (skipping case fans)"
fi

# Seed GPU fans via NVML. The open NVIDIA driver does NOT expose pwm
# endpoints under /sys/class/hwmon for consumer Ampere; the supported path
# is the NVML SetFanSpeed_v2 API. Requires python-nvidia-ml-py system-wide
# (Arch package: extra/python-nvidia-ml-py).
log "seeding GPU fan floor at ${GPU_FAN_FLOOR_PCT}%"
PYTHON="${PYTHON:-/usr/bin/python3}"
if ! "$PYTHON" -c "import pynvml" 2>/dev/null; then
    log "WARN: pynvml not available on $PYTHON — install with: pacman -S python-nvidia-ml-py"
    log "WARN: fans will fall back to firmware auto curve until gpu-fan-curve.service starts"
else
    "$PYTHON" - "$GPU_FAN_FLOOR_PCT" <<'PYEOF' || log "WARN: NVML fan seed failed"
import sys, pynvml
target = int(sys.argv[1])
pynvml.nvmlInit()
n = pynvml.nvmlDeviceGetCount()
for i in range(n):
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    try:
        fan_count = pynvml.nvmlDeviceGetNumFans(h)
    except pynvml.NVMLError:
        fan_count = 0
    for f in range(fan_count):
        try:
            pynvml.nvmlDeviceSetFanSpeed_v2(h, f, target)
            print(f"GPU {i} fan {f} -> {target}%", flush=True)
        except pynvml.NVMLError as e:
            print(f"GPU {i} fan {f} set failed: {e}", flush=True)
PYEOF
fi

log "done"
exit 0
