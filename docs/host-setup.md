# Host setup — 2× RTX 3090 on AM5 (NVLink, kernel, cooling, toolchain)

Everything host-side that the serving stack depends on. The top-level README keeps the hardware summary; this file has the load-bearing details and the reasons.

Tested hardware (current rig):

| Component | Spec |
|-----------|------|
| GPU | 2× NVIDIA RTX 3090 (24 GB each, 48 GB total) — NVLink bridge present; `nvidia-smi topo -m` reports `NV4` (~56 GB/s aggregate) |
| CPU | AMD Ryzen 9 7900 (12C/24T, Zen 4, AM5) |
| RAM | 64 GB DDR5-6000 (62 GB usable) |
| Motherboard | MSI MPG B650I EDGE WIFI (mini-ITX, AM5) |
| Storage | 2× 2 TB NVMe (`nvme0n1` = root, `nvme1n1` = `/data` models + caches) |
| Chassis fans | Corsair Commander Core XT (via `liquidctl`) |
| OS / Kernel | Arch (EndeavourOS) / `linux-zen-p2p` 6.18.zen1-1 (locally-built linux-zen + cosmetic `CONFIG_HSA_AMD_P2P=y`; pinned to 6.18 for stability) |
| NVIDIA driver / CUDA | `nvidia-open-dkms` 595.71.05 / CUDA 13.2 |
| NVLink | physical 4-link bridge installed between the two 3090s — `nvidia-smi nvlink --status` shows all 4 links at 14.06 GB/s (~56 GB/s aggregate) |

Both 3090s sit at PCIe Gen4 with the NVLink bridge; NCCL selects `P2P/IPC` transport (NVLink + peer-to-peer CUDA IPC) once everything below is in place.

### Why `NV4` reports — the load-bearing pieces

1. **Physical NVLink bridge installed** between the two 3090s. This is what produces the four 14.06 GB/s links (`nvidia-smi nvlink --status`). Without the bridge there is no `NV4` regardless of any software change.
2. **Two separate kernel boot args** in `/etc/kernel/cmdline` — both load-bearing for different failure modes:
   ```
   amd_iommu=on iommu=pt pcie_acs_override=downstream,multifunction pcie_ports=native pcie_ecrc=on
   ```
   - **`pcie_acs_override=downstream,multifunction`** — gives P2P traffic permission to traverse ACS-protected PCIe ports on this AM5 chipset. Without it, consumer-Ampere P2P is blocked at the chipset level and `nvidia-smi topo -m` reports `PHB`. Affects the routing decision.
   - **`iommu=pt`** — IOMMU passthrough mode (vs lazy DMA-translation default). Short-context TP=2 works either way; the wedge appears at long context. R9700 (sister stack, same mechanism on NCCL/RCCL) measured the failure cleanly: without `iommu=pt`, **131K-token decode collapses to 0.68 tok/s** with the NCCL log filling with channel-renegotiation churn (`178278 NCCL log lines`); with it, decode is healthy **16.83 tok/s** (`4 log lines`). NCCL prints `Missing iommu=pt … can lead to instability or hang` as the proximate warning. Affects how the kernel actually services the resulting DMAs.

   Backup of the pre-NVIDIA cmdline lives at `/etc/kernel/cmdline.bak.preNvidia`. Verify both args are live: `grep -oE "iommu=pt|pcie_acs_override=\S+" /proc/cmdline`.
3. **`nvidia-open-dkms`** (not `nvidia-open`) — DKMS rebuilds against installed headers every kernel bump. Modern open driver defaults `NVreg_DmaRemapPeerMmio=1`, which is what we want; nothing extra to set.

### Kernel choice

- `linux-zen`-family kernel, not stock `linux` — stock + open NVIDIA module hard-locked the host under sustained TP=2 / 256K load. The zen patchset eliminated the recurrence.
- Our actual install is `linux-zen-p2p` 6.18, locally built from the upstream Arch `linux-zen` PKGBUILD with one cosmetic change: `CONFIG_HSA_AMD_P2P=y` (an AMD-HSA driver flag we don't use — vestigial from earlier debugging). Stock `linux-zen` would serve the same role; the rename is historical. The rebuild script + rebuild path are in [`scripts/host-setup/rebuild_linux_zen_p2p.sh`](../scripts/host-setup/rebuild_linux_zen_p2p.sh).
- **Pin these in `pacman.conf`** so a routine `pacman -Syu` doesn't silently leapfrog `linux-headers` or `nvidia-open-dkms` and break the DKMS module-for-kernel pairing. Add to `/etc/pacman.conf`:
  ```
  IgnorePkg = linux-zen linux-zen-headers linux-zen-p2p linux-zen-p2p-headers linux-headers nvidia-open-dkms nvidia-utils cuda cuda-tools opencl-nvidia
  ```
  After that, updating any of those packages is a deliberate `pacman -S <pkg>` opt-in.

### Cooling and power profile (load-bearing)

Two systemd units hold a cooling profile required for multi-hour bake-off survival. DDR5 SPD sensors crossed `ALARM HIGH` (55 °C) under stock cooling + default 350 W per 3090, correlating with random heap corruption / kernel BUGs / hard resets. The profile stays in spec under sustained TP=2 inference.

| Unit | Action |
|------|--------|
| `gpu-cooling.service` | Boot oneshot. NVIDIA persistence mode, **260 W** power limit per 3090 (from 350 W), Corsair case fans to 100% via `liquidctl`, 75% GPU fan floor via NVML. |
| `gpu-fan-curve.service` | NVML daemon. Polls temp every 4 s. Fan duty 75% below 60 °C, linear to 100% by 80 °C. One hot card pulls all fans up. |

The fan curve runs through NVML, not hwmon — consumer Ampere on the open driver exposes no GPU `pwm*` under `/sys/class/hwmon`. NVML's `SetFanSpeed_v2` works as root. Scripts tracked under [`systemd/`](../systemd/):

```bash
sudo pacman -S --needed python-nvidia-ml-py
sudo install -m 0755 systemd/gpu-cooling.sh   /usr/local/bin/gpu-cooling.sh
sudo install -m 0755 systemd/gpu-fan-curve.py /usr/local/bin/gpu-fan-curve.py
sudo install -m 0644 systemd/gpu-cooling.service   /etc/systemd/system/
sudo install -m 0644 systemd/gpu-fan-curve.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now gpu-cooling.service gpu-fan-curve.service
```

Verify: `nvidia-smi --query-gpu=power.limit,fan.speed,temperature.gpu --format=csv` (expect 260 W, ~75% fans idle). 260 W picked to leave throughput headroom (a 256K coder-30b cycle pulls only ~245 W per card at steady-state decode) while cutting peak heat ~25%.

### Arch toolchain gotchas (host-side builds)

EndeavourOS ships a much newer toolchain than the docker eval images, so anything built **on the host** (kernels, Triton, pip packages with C extensions) hits issues docker hides. Surfaced by R9700 running an FP8 SWE-bench bake-off no-docker; relevant here for any host-side build:

- **gcc 16 defaults to C23** — `nullptr` is a reserved keyword and `-Wincompatible-pointer-types` / `-Wimplicit-*` are hard errors, so old C (astropy's bundled cfitsio, many scientific-Python C extensions) fails with `command '/usr/bin/cc' failed` or `expected identifier before 'nullptr'`. Build with `CFLAGS="-std=gnu17 -Wno-error=incompatible-pointer-types -Wno-error=implicit-function-declaration -Wno-error=implicit-int -Wno-error=int-conversion -Wno-error=return-mismatch"` (`-std=gnu17` is the load-bearing flag). The Docker eval images ship gcc <14 and sidestep this entirely.
- **Arch defaults `/tmp` to a RAM-backed tmpfs** (`df -h /tmp` — Type `tmpfs`, ~half of RAM). Bulk job I/O (repo clones, build dirs, per-instance venvs) fills it → `ENOSPC` → *silent* empty outputs, not a crash. Keep workdirs + `TMPDIR` on `/data` (`nvme1n1`).
- **Non-docker SWE-bench scoring deps** (only if you run the host-side scorer): `sudo pacman -S --needed gcc-fortran openblas lapack freetype2 libpng gsl fftw pkgconf`, and pre-install pyproject build-requires (`cython extension-helpers setuptools_scm oldest-supported-numpy meson-python pybind11`) since `PIP_NO_BUILD_ISOLATION=1` skips them.

Full host-side scaffold + toolchain notes (opencode + little-coder + claw-code, no docker): R9700 [`rules-for-agents.md`](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference/blob/main/rules-for-agents.md) "Host OS gotchas".
