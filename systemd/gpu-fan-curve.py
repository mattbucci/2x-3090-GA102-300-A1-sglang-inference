#!/usr/bin/python3
"""GPU fan-curve daemon for consumer Ampere on the open NVIDIA driver.

Polls each GPU temperature every POLL_SECS seconds and writes fan duty via
nvmlDeviceSetFanSpeed_v2. The curve (matches README spec):
    T <= RAMP_LOW_C   -> FLOOR_PCT (default 75%)
    T  > RAMP_HIGH_C  -> CEILING_PCT (default 100%)
    otherwise         -> linear ramp 75% -> 100% across RAMP_LOW_C..RAMP_HIGH_C

Drives every fan on every GPU to the same duty, picked from the *max*
temperature across all cards. Conservative: a hot spike on one card pulls
all fans up together.

The hwmon path used by the kernel `nvidia` driver does not exist on this
host (consumer Ampere + nvidia-open-dkms); NVML is the only working
fan-control surface. Requires python-nvidia-ml-py (Arch: extra/python-nvidia-ml-py).
"""
from __future__ import annotations

import os
import sys
import time
import signal
import syslog

try:
    import pynvml
except ImportError:
    syslog.openlog("gpu-fan-curve")
    syslog.syslog(syslog.LOG_ERR, "pynvml not installed; run: pacman -S python-nvidia-ml-py")
    print("ERROR: pynvml not installed; run: pacman -S python-nvidia-ml-py", file=sys.stderr)
    sys.exit(1)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


POLL_SECS = env_int("POLL_SECS", 4)
FLOOR_PCT = env_int("FLOOR_PCT", 75)
CEILING_PCT = env_int("CEILING_PCT", 100)
RAMP_LOW_C = env_int("RAMP_LOW_C", 60)
RAMP_HIGH_C = env_int("RAMP_HIGH_C", 80)


def duty_for_temp(t: int) -> int:
    if t <= RAMP_LOW_C:
        return FLOOR_PCT
    if t >= RAMP_HIGH_C:
        return CEILING_PCT
    span = RAMP_HIGH_C - RAMP_LOW_C
    rise = CEILING_PCT - FLOOR_PCT
    return FLOOR_PCT + (t - RAMP_LOW_C) * rise // span


def main() -> int:
    syslog.openlog("gpu-fan-curve")
    pynvml.nvmlInit()

    handles = []
    fan_counts = []
    n = pynvml.nvmlDeviceGetCount()
    for i in range(n):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        try:
            fc = pynvml.nvmlDeviceGetNumFans(h)
        except pynvml.NVMLError:
            fc = 0
        handles.append(h)
        fan_counts.append(fc)

    total_fans = sum(fan_counts)
    if total_fans == 0:
        syslog.syslog(syslog.LOG_ERR, "no NVML-controllable fans found")
        return 1

    syslog.syslog(
        syslog.LOG_INFO,
        f"managing {total_fans} fan(s) across {n} GPU(s); curve "
        f"{FLOOR_PCT}%/{RAMP_LOW_C}C .. {CEILING_PCT}%/{RAMP_HIGH_C}C; poll={POLL_SECS}s",
    )

    # Graceful shutdown — release fans to firmware auto control.
    def reset(*_args):
        for i, h in enumerate(handles):
            for f in range(fan_counts[i]):
                try:
                    pynvml.nvmlDeviceSetDefaultFanSpeed_v2(h, f)
                except pynvml.NVMLError:
                    pass
        syslog.syslog(syslog.LOG_INFO, "released fans to default; exiting")
        sys.exit(0)

    signal.signal(signal.SIGTERM, reset)
    signal.signal(signal.SIGINT, reset)

    last_duty = -1
    while True:
        max_t = 0
        for h in handles:
            try:
                t = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError:
                continue
            if t > max_t:
                max_t = t

        duty = duty_for_temp(max_t)
        if duty != last_duty:
            for i, h in enumerate(handles):
                for f in range(fan_counts[i]):
                    try:
                        pynvml.nvmlDeviceSetFanSpeed_v2(h, f, duty)
                    except pynvml.NVMLError as e:
                        syslog.syslog(syslog.LOG_WARNING, f"set GPU{i} fan{f} {duty}% failed: {e}")
            syslog.syslog(syslog.LOG_DEBUG, f"max_t={max_t}C duty={duty}%")
            last_duty = duty

        time.sleep(POLL_SECS)


if __name__ == "__main__":
    sys.exit(main() or 0)
