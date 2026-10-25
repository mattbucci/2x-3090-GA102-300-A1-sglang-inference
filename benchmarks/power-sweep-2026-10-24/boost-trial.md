# Why the DIMMs run hot, and the CPU-boost trial (2026-10-24, 17:12–18:13 PDT)

Setting: the `qwen36:opencode` bake-off lane rolling (docker-served SGLang, TP=2), gpu-unlock on,
GPU cap 280 W, 30 s samples of CPU Tctl (`k10temp`), DDR5 SPD hub temps (`spd5118` at 0x50/0x52,
alarm 55 °C), GPU temp/power/fan (`sustained-280w.csv`, `sustained-300w.csv`, `sustained-300w-b.csv`).
Decode step throughput from the lane's server log (`Decode batch … gen throughput`), batch 1,
compared in 8K-token depth buckets (`compare_windows.py`).

## Finding: the CPU heats the DIMMs, not the GPUs

With both GPUs idle (32 W, 37–41 °C) the CPU read 83–90 °C and the DIMMs 51 °C. In the power sweep
the DIMM temperature correlated equally with GPU 0 and GPU 1 (0.89 / 0.88) — case warmth, not one
card. The board is a mini-ITX MSI MPG B650I EDGE: the two DIMMs sit against the socket and share the
CPU cooler's leftover airflow. The CPU is hot because SGLang's two scheduler processes pin a core each
(measured 97 % and 71 % of a core with the GPUs at 0 % — zmq busy-poll when idle, CUDA stream
synchronisation when busy; the heavy-load picture is the same), and the Ryzen 9 7900 boosts them to
5.1 GHz. `--sleep-on-idle` (default off) would only lower the starting point between bursts.

## Trial steps, 15 min each, same lane

| step | CPU Tctl mean / max | DIMM mean / max / last | GPU 0 / GPU 1 max | GPU mean draw |
|---|---|---|---|---|
| baseline: boost on, fans 75 % floor, 280 W | 84.5 / 88.5 | 49.70 / 50.00 / 49.25 | 53 / 60 | 100 W |
| **boost off** (`/sys/devices/system/cpu/cpufreq/boost` = 0, cores 5.1 → 3.7 GHz) | **60.5 / 66.9** | 47.77 / 49.00 / 48.50 | 59 / 68 | 145 W |
| + GPU fan floor 100 % | 60.9 / 63.4 | 47.67 / 49.00 / 47.00 | 54 / 62 | 128 W |
| + GPU cap 300 W (light-load window, never at cap) | 61.2 / 63.1 | 47.52 / 48.00 / 46.75 | 56 / 65 | 104 W |

Decode cost of boost off, matched depth, batch 1 (`qwen36`, tok/s median, boost-on → boost-off):
0–8K 220.0 → 219.2 (−0.3 %), 8–16K 209.0 → 209.4 (+0.2 %), 16–24K 202.5 → 204.4 (+0.9 %),
24–32K 199.2 → 198.1 (−0.6 %), 32–40K 196.7 → 195.2 (−0.7 %). The scheduler's per-step work does not
need the boost clock.

## Decision

Boot profile (`systemd/gpu-cooling.sh`, `systemd/gpu-fan-curve.service`): CPU boost off, GPU fan floor
100 %, GPU cap 300 W. The DIMM margin to the 55 °C alarm went from ~4 °C to ~8 °C before any GPU load
is added; the earlier 300 W alarm crossing happened on top of an 85–90 °C CPU. Next levers if a long
heavy stretch still pushes the DIMMs: `scaling_max_freq` cap (3.0 GHz is the amd-pstate
lowest-nonlinear point) traded for a higher GPU cap, a BIOS eco PPT, and physically a fan across the
DIMMs plus a check of the CPU cooler and the three Commander fan channels that read 0 rpm.
