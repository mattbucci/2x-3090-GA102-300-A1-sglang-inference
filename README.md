# NVIDIA Inference: SGLang on 2x RTX 3090

High-throughput LLM inference on 2× NVIDIA RTX 3090 (GA102-300-A1, Ampere). SGLang **v0.5.12** + 25 local patches, CUDA 13.2 / PyTorch cu130. This rig owns **all evals + AWQ/INT4 calibrations**; FP8 work lives with the [R9700 RDNA4 stack](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference).

## Fleet status (v0.5.12)

All presets pass their applicable capability checks. Full matrix: [`benchmarks/quality/fleet-capability-v0512-2026-05-26.md`](benchmarks/quality/fleet-capability-v0512-2026-05-26.md).

Passing 5/5 (basic+tool+thinking+vision+video): `qwen36`, `qwen35-moe`, `qwen36-dense`, `gemma4`, `gemma4-31b`, `qwen36-ream`. `qwen3-vl-32b` 4/4 (non-thinking VL). `devstral` 3/3 (basic+tool+vision). `qwen3-ream` basic (text-only generalist). Coder presets OK (text).

In-house AWQ rebuilds shipped under `mattbucci/*` (all preserve thinking/vision/tool where applicable):
- **`gemma-4-31B-AWQ`** — BF16→GPTQ→AWQ, vision tower kept FP16; replaces AutoRound (which hallucinated vision). 5/5 @ 256K.
- **`Devstral-Small-2-24B-AWQ`** — Devstral-2-2512 rebuild with function-calling calibration; fixes the community quant's broken tool-calling. 3/3.
- **`Qwen3.6-REAM-A3B-AWQ`** — coding-eval leader; vision tower grafted from qwen36 (was missing). 5/5.
- **`Qwen3-30B-Instruct-2507-REAM-AWQ`** — REAM 128→96 experts + AWQ from scratch. Fastest preset (107 tok/s @ 256K).
- **`Qwen3.6-35B-A3B-AWQ`** — native AWQ-Marlin rebuild (the prior `…-AWQ-CT…` was a misnamed GPTQ-CT that broke on v0.5.12's MoE loader). 5/5.

## Coding-eval bake-off (SWE-bench Lite, v2 Docker harness, 256K, single-user)

Best `(model, scaffold)` pair: `qwen36-ream` × **opencode** = **176/300 = 58.7%**.

| Preset | opencode | claw-code | little-coder |
|--------|:--------:|:---------:|:------------:|
| `qwen36-ream` (Qwen3.6-REAM-A3B-AWQ, thinking) | **176/300 = 58.7%** | 20/123 = 16.3% † | 0/10 = 0.0% † |
| `coder-30b-eval` (Qwen3-Coder-30B-A3B-AWQ CT) | 129/300 = 43.0% | 107/300 = 35.7% | 74/300 = 24.7% |
| `coder-reap-25b` (Cerebras Qwen3-Coder-REAP-25B-A3B-AWQ) | 125/300 = 41.7% | 122/300 = 40.7% | 101/300 = 33.7% |
| `coder-30b-ream` (Samsung SAIL Qwen3-Coder-30B-A3B-REAM-AWQ) | 116/300 = 38.7% | 109/300 = 36.3% | 73/300 = 24.3% |

† Thinking-mode models exhaust their `<think>` budget before committing a `tool_call` on tool-call-heavy scaffolds (claw/little-coder) — they belong on opencode. Coder-tuned models match claw's tool registry and score similarly on claw vs opencode. `coder-reap-25b` is the most-rounded preset; `qwen36-ream` wins when the scaffold matches (opencode).

Failure-mode analysis (over-edit signature, per-repo skew, oracle-ensemble ceiling of 49% across opencode∪claw, rollout self-clean), methodology, and per-cell receipts: [`patches/README.md`](patches/README.md) + [`benchmarks/quality/bakeoff-*.json`](benchmarks/quality/). Every preset's `--tool-call-parser` matches its chat-template tool format (see Known Issues).

## Speculative decoding (validated on R9700, porting to our AWQ stack)

R9700's spec-decode lane (2026-05-29) found EAGLE3 + DFlash both work against **INT4/AWQ targets** (the draft stays BF16; target quant is independent). Drafts are external — keep them full precision via `--speculative-draft-model-quantization unquant`, and pass `--speculative-attention-mode decode` (TP2 deadlock fix for `moe_wna16`).

| Target (our AWQ) | Draft | Algo | R9700 result |
|---|---|---|---|
| Coder-30B-A3B + REAP/REAM children | `lmsys/SGLang-EAGLE3-Qwen3-Coder-30B-A3B-Instruct-SpecForge` (parent-transfers to pruned children) | EAGLE3 | 97 tok/s (4.4×) @ 256K |
| Qwen3.6-35B-A3B + REAM | `z-lab/Qwen3.6-35B-A3B-DFlash` | DFLASH | 80 tok/s (3.7×) @ 256K |

Not applicable: gemma4 (no DFlash hook), AWQ's bundled MTP head is int4-dead so NEXTN/MTP is FP8-only. Local validation in progress — see `benchmarks/quality/specdec-*.json`.

## Known Issues (open)

- **`check_awq_scales.py` over-flags MoE structural sparsity.** For `Qwen/Qwen3.6-35B-A3B`, ~50-72% of layer-0 expert gate/up channels are already `7.8e-38` in the BF16 base (structural zeros); the fp16 AWQ scale faithfully flushes them to 0. A zero scale over a dead base channel is benign — only zero scales over *live* weights are a defect. qwen36 serves 5/5 despite 144 such flags.
- **`gemma-4-21b-REAP-AWQ-thinking-vision-v2` — DO NOT USE** (164 all-zero scale tensors from an empty `ignore` list). Shipping checkpoint is [`mattbucci/gemma-4-21B-REAP-AWQ`](https://huggingface.co/mattbucci/gemma-4-21B-REAP-AWQ) (v3b, regex `ignore`).
- **Qwen3-VL-30B MoE AWQ — SGLang loader broken.** `Qwen3VLMoeForConditionalGeneration` gibberish across 4 sources → upstream weight-mapping bug. Non-coder, low priority. Narrative in [`patches/README.md`](patches/README.md).
- **Qwen3.5-27B DeltaNet stuck at 32K** — DeltaNet TP replication forces 19 GB/GPU. Use `qwen3-ream` for long-context DeltaNet workloads.
- **60B+ models don't fit** — Coder-Next-REAM (35 GB), GLM-4.5-Air-REAP (43 GB) exceed 48 GB at MoE-AWQ.
- **Per-preset piecewise CUDA graph disables** — `coder-reap-25b` (cold-launch detokenizer hang), `qwen35-moe`/`qwen36` (DeltaNet+MoE+mamba_cache), `gemma4`/`gemma4-31b` (head_dim=256 + Ampere FP8 → triton-attn forced). Reasons in `launch.sh`.
- **Tool-call parser is per-preset and load-bearing.** SGLang only emits structured `tool_calls` when `--tool-call-parser <fmt>` matches the model's chat-template format. Mapping: Qwen3-Coder + Qwen3.5/3.6 (incl. VL-REAP/dense/MoE/REAM) → `qwen3_coder`; Qwen3-VL non-coder + Qwen3-30B-Instruct REAM → `qwen25`; Devstral → `mistral`; Gemma 4 → `gemma4`.

## Quick Start

```bash
./scripts/setup.sh                          # clone SGLang v0.5.12, apply patches, create conda env

# TP=2 / 256K presets (matrix standard):
./scripts/launch.sh qwen3-ream              # 262K @ 107 tok/s — REAM merged MoE, 96 experts
./scripts/launch.sh qwen36                  # Qwen3.6-35B-A3B MoE AWQ-Marlin — 256K, thinking+vision
./scripts/launch.sh qwen36-dense            # Qwen3.6-27B Dense AWQ — DeltaNet+attn
./scripts/launch.sh coder-30b               # Coder-30B-A3B MoE — peak throughput
./scripts/launch.sh coder-reap-25b          # Coder-REAP-25B MoE AWQ-Marlin — 256K @ 109 tok/s
./scripts/launch.sh qwen3-vl-32b            # Qwen3-VL-32B Dense — 131K @ TP=2
./scripts/launch.sh gemma4-31b              # Gemma 4 31B Dense AWQ (thinking+image+video, 256K)
./scripts/launch.sh devstral                # Devstral-Small-2-24B (tool+vision); devstral-long for 217K

python scripts/eval/validate_capabilities.py --port 23334    # auto-skips thinking/vision/video per preset
./scripts/eval/test_capabilities_all.sh                       # sweep across all AWQ presets
python scripts/bench/bench_long_context.py --port 23334 --name "Model" --contexts 1024 16384 131072 250000
```

Use `temperature >= 0.3` on Qwen3 family models — greedy decode at `temp=0` triggers a token-repetition loop.

## Prerequisites

Reference host (3090 rig as of 2026-05-17 motherboard swap):

| Component | Spec |
|-----------|------|
| GPU | 2× NVIDIA RTX 3090 (24 GB each, 48 GB total) — NVLink bridge present (`nvidia-smi topo -m` reports `NV4` ≈ 56 GB/s aggregate) |
| CPU | AMD Ryzen 9 7900 (12C/24T, max 5.48 GHz, Zen 4 AM5) |
| RAM | 2× 32 GB DDR5-6000 (64 GB total; 62 GB usable in OS) |
| Motherboard | MSI MPG B650I EDGE WIFI (MS-7D73, mini-ITX, AM5) |
| Storage | 2× Crucial P5 Plus 2 TB NVMe (`nvme0n1` = root, `nvme1n1` = `/data` models) |
| Chassis fans | Corsair Commander Core XT (controlled via `liquidctl`) |
| OS / Kernel | EndeavourOS (Arch, rolling) / `linux-zen-p2p` 6.18.0 (NVIDIA P2P-BAR1 patches) |
| NVIDIA driver | 595.71.05 (`nvidia-open-dkms`) / CUDA toolkit 13.2 |

Both 3090s sit at PCIe Gen4 with the NVLink bridge; NCCL selects `P2P/IPC` transport (NVLink + peer-to-peer CUDA IPC) once the kernel/driver combo below is in place.

### Kernel and driver

- `linux-zen`-family kernel (host runs `linux-zen-p2p` 6.18.0), not stock `linux` — the stock kernel + open NVIDIA module hard-locked the host under sustained TP=2 / 256K load. The zen patchset eliminated the recurrence; the `-p2p` variant re-enables consumer-Ampere PCIe-BAR1 P2P so `nvidia-smi topo -m` reports `NV4` instead of `PHB` on this AM5 platform.
- `nvidia-open-dkms` (not `nvidia-open`) — DKMS rebuilds the module for every kernel with headers installed.

### Cooling and power profile (load-bearing)

Two systemd units hold a cooling profile required for multi-hour bake-off survival. DDR5 SPD sensors crossed `ALARM HIGH` (55 °C) under stock cooling + default 350 W per 3090, correlating with random heap corruption / kernel BUGs / hard resets. The profile stays in spec under sustained TP=2 inference.

| Unit | Action |
|------|--------|
| `gpu-cooling.service` | Boot oneshot. NVIDIA persistence mode, **260 W** power limit per 3090 (from 350 W), Corsair case fans to 100% via `liquidctl`, 75% GPU fan floor via NVML. |
| `gpu-fan-curve.service` | NVML daemon. Polls temp every 4 s. Fan duty 75% below 60 °C, linear to 100% by 80 °C. One hot card pulls all fans up. |

The fan curve runs through NVML, not hwmon — consumer Ampere on the open driver exposes no GPU `pwm*` under `/sys/class/hwmon`. NVML's `SetFanSpeed_v2` works as root. Scripts tracked under [`systemd/`](systemd/):

```bash
sudo pacman -S --needed python-nvidia-ml-py
sudo install -m 0755 systemd/gpu-cooling.sh   /usr/local/bin/gpu-cooling.sh
sudo install -m 0755 systemd/gpu-fan-curve.py /usr/local/bin/gpu-fan-curve.py
sudo install -m 0644 systemd/gpu-cooling.service   /etc/systemd/system/
sudo install -m 0644 systemd/gpu-fan-curve.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now gpu-cooling.service gpu-fan-curve.service
```

Verify: `nvidia-smi --query-gpu=power.limit,fan.speed,temperature.gpu --format=csv` (expect 260 W, ~75% fans idle). 260 W picked to leave throughput headroom (a 256K coder-30b cycle pulls only ~245 W per card at steady-state decode) while cutting peak heat ~25%.

## Model Support

Single-user tok/s measured at the max-context value. All numbers are **fresh prefill** (radix cache disabled) unless noted.

| Model | Type | Max ctx | tok/s @max | TPOT | Launch | Status |
|-------|------|:-------:|:----------:|:----:|:------:|:-------|
| **Qwen3-30B REAM AWQ** | MoE (96 exp) | **262K** | **107** | 9.3 ms | `qwen3-ream` | In-house REAM rebuild 2026-05-29 (Instruct-2507 → REAM 128→96 → GPTQ → AWQ). basic PASS; text-only generalist (fastest preset). `mattbucci/Qwen3-30B-Instruct-2507-REAM-AWQ`. 183→107 tok/s @ 1K/250K. |
| **Qwen3.6-35B-A3B AWQ-Marlin** | DeltaNet+MoE (256 exp, VL) | **262K** | **31** | 32 ms | `qwen36` | Native AWQ-Marlin rebuild 2026-05-26 (replaces CT default that regressed on v0.5.12). **5/5 PASS**. `mattbucci/Qwen3.6-35B-A3B-AWQ`. |
| **Qwen3.6-27B AWQ** | Dense + DeltaNet | **131K** | **21** | 47 ms | `qwen36-dense` | R9700 self-cal at `mattbucci/Qwen3.6-27B-AWQ`. 4/4 PASS at TP=2. |
| **Qwen3-VL-32B Instruct** | Dense (VL) | **131K** | **40** | 25 ms | `qwen3-vl-32b` | R9700 self-cal at `mattbucci/Qwen3-VL-32B-AWQ`. 68→50→40 tok/s @ 1K/65K/131K, 3/3 PASS. |
| **Devstral-Small-2-24B AWQ** | Dense (VL) | 131K | 56 | 17.9 ms | `devstral` | In-house Devstral-2-2512 rebuild 2026-05-28 (FP8→BF16→GPTQ+tool-cal→AWQ). **3/3 PASS** — fixes community quant's broken tool-calling. `mattbucci/Devstral-Small-2-24B-AWQ`. `devstral-long` reaches 217K (50 tok/s, text-only path; vision OOMs at MEM=0.97). |
| Coder-REAP-30B AWQ-Marlin | MoE (96 exp) | **262K** | **109** | 9.2 ms | `coder-reap-25b` | R9700 in-house rebuild from upstream BF16 (96 exp/layer, GPTQ W4A16 + `moe_calibrate_all_experts`). |
| Coder-30B AWQ Marlin | MoE (128 exp) | 16K | **180** | 5.5 ms | `coder-30b` | 187→180 tok/s @ 1K/16K. Original AWQ-Marlin layout (vs `coder-30b-eval` = CT). |
| Qwen3.5-28B MoE REAP | DeltaNet+MoE (205 exp) | 262K | **30** | 32.7 ms | `qwen35-moe` | Decode flat 30.5 tok/s across 1K-250K, 4/4 PASS. R9700 cross-validated 4/4 on RDNA4. |
| **Gemma 4 31B Dense AWQ** | Dense (VL) | **256K** | ~22 | ~50 ms | `gemma4-31b` | In-house BF16→GPTQ→AWQ rebuild 2026-05-27 (LM INT4, vision FP16). **5/5 PASS** — replaces AutoRound (vision hallucinated). `mattbucci/gemma-4-31B-AWQ`. |
| Gemma 4 26B MoE | MoE (103 exp) | 16K | 22 | ~50 ms | `gemma4` | basic+thinking+content-aware vision. |
| Gemma 4 21B REAP AWQ | MoE (128 exp) | 4K | — | — | — | [`mattbucci/gemma-4-21B-REAP-AWQ`](https://huggingface.co/mattbucci/gemma-4-21B-REAP-AWQ). Audit clean. |

Per-model receipts in `benchmarks/quality/*-rebuild-v0512.json`.

### VRAM context limits (KV dtype varies, TP=2, 48 GB total)

| Model | Wt/GPU | KV/token | Max context |
|-------|:------:|:--------:|:-----------:|
| Qwen3-30B REAM AWQ | 6.2 GB | 36 KB | 262K |
| Qwen3.5-28B MoE REAP CT | 8.1 GB | 5 KB | 262K |
| Qwen3.6-35B-A3B AWQ-native | 9.87 GB | ~8 KB hybrid | 262K |
| Coder-30B AWQ | 8.0 GB | 36 KB | 262K |
| Devstral-24B AWQ (long preset) | 7.0 GB | 80 KB | **217K** (3090 ceiling for 24B dense @ TP=2) |
| Coder-REAP-25B W4A16 | 6.5 GB | 72 KB | 131K |
| Qwen3.5-27B AWQ | 19.0 GB | 24 KB | 32K (weights replicated for DeltaNet TP) |

## Benchmarks

Per-model long-context sweeps in `benchmarks/<model>/`. Quality:

| Model | MMLU | HumanEval | LAB-Bench | Needle |
|-------|:----:|:---------:|:---------:|:------:|
| `coder-30b` | **91.2%** | 96.7% | 33.3% | ✓ to 4K |
| `qwen3-vl-32b` | **91.2%** | 83.3% | **39.8%** | ✓ to 4K |
| `coder-reap-25b` | 77.2% | 96.7% | (n/a) | ✓ to 65K |

Methodology: MMLU (1 q/subject × 57), HumanEval pass@1 (30), [LAB-Bench](https://github.com/Future-House/LAB-Bench) (7×50), needle (1K→65K). Receipts in `benchmarks/quality/*-v0511.json`. SWE-bench Lite rates in the bake-off table above. **TODO:** RULER, LongBench Pro, LiveCodeBench.

## Setup

```bash
./scripts/setup.sh
# or manually:
cd components/sglang && git checkout v0.5.12
for p in ../../patches/*.patch; do git apply "$p"; done
cd python && pip install -e .
```

| Component | Version |
|-----------|---------|
| SGLang | v0.5.12 + 25 local patches |
| PyTorch | 2.11.0 + cu130 |
| CUDA | 13.2 driver (595.71.05) / cu130 wheel |
| transformers | 5.6.0 (v0.5.12 pin) |
| FlashInfer | 0.6.11.post1 |
| compressed-tensors | 0.15.0.1 (serving env); 0.15.1.dev (`quant` calibration env) |

The serving tree lives at `/data/sglang-rebase-v0512` (env `sglang-v0512`); launch with `ENV_NAME`/`SGLANG_DIR` overrides. Calibration uses the separate `quant` env.

## Patches

25 patches (`ls patches/*.patch | wc -l`) targeting SGLang v0.5.12. Notable:
- **002** Qwen3-Next AWQ weight_loader fix (port from R9700)
- **028** Gemma 4 MM per-expert AWQ loader (cross-stack with R9700)
- **030** fused_moe_triton presharded-w2 detection (CT MoE at TP≥2)
- **031** Qwen3.5/3.6 DeltaNet AWQ weight_loader
- **034** sampler ±Inf detection (port from R9700)
- **039/040** Gemma4 dense loader — `num_experts` fallback + top-level `Gemma4Config` head-dim remap (gemma4-31b)

Per-patch narratives + closed-item history in [`patches/README.md`](patches/README.md).

## Quantization

Self-calibrated models use the `quant` conda env:

```bash
conda activate quant
CUDA_VISIBLE_DEVICES="" python -u scripts/quantize/quantize_qwen36_27b_thinking_vision.py   # 27B template
python scripts/quantize/convert_moe_ct_to_awq.py <ct_src> <awq_dst>                          # MoE CT→native AWQ
```

`scripts/quantize/calibration_datasets.py` builds capability-preserving recipes (`thinking_vision` / `code_vision` / `code_vision_tools` / `balanced_thinking_vision` …) from AM-Thinking-v1, NuminaMath-CoT, LLaVA-Instruct, Hermes-function-calling, UltraChat, python-instruct. REAM/REAP expert compression in [`REAM.md`](scripts/quantize/REAM.md). See [rules-for-agents.md](rules-for-agents.md). Launch detached calibrations with `conda activate <env>` + `python -u` (not `conda run`, which buffers all output).

## Sister teams

- **[R9700 (RDNA4, ROCm)](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference)** — FP8 calibration owner (native gfx1201 FP8); RDNA4 serving stack. Originated the FP32-softmax patch 011 and the CT→native AWQ converter; shipped the EAGLE3/DFlash spec-decode recipes we're porting. Both stacks publish under `mattbucci/*` with format suffixes.
- **[M4 (Apple Silicon, MLX)](https://github.com/mattbucci/m4-sglang-inference)** — MLX bridge; cross-checks chat-template + multimodal plumbing.

## Repo layout

```
patches/                  # SGLang v0.5.12 patches — narratives in patches/README.md
benchmarks/               # per-model JSON; quality/ = MMLU/HumanEval/LAB-Bench/Needle + capability matrix
scripts/
  launch.sh / common.sh / setup.sh
  bench/ eval/ quantize/ test/
components/sglang/        # SGLang v0.5.12 + patches (serving tree at /data/sglang-rebase-v0512)
systemd/                  # cooling profile units
```
