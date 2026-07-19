# Rules for AI Agents — 2x RTX 3090

## Inference Engine

All inference MUST use SGLang. No vLLM, no llama.cpp unless explicitly for comparison benchmarks.

## Hardware
- 2x NVIDIA RTX 3090 (GA102-300-A1, Ampere, sm_86)
- 24 GB GDDR6X each, 48 GB total
- NCCL over PCIe for TP=2

### Host filesystem layout
Two 2 TB NVMe drives (Crucial P5 Plus, `CT2000P5PSSD8`), mirroring the fleet standard:

| Drive | Partition | Mount | Holds |
|---|---|---|---|
| **nvme0n1** (OS) | p1 vfat / p2 ext4 / p3 swap | `/efi` · `/` · swap | OS + `/var/lib/docker` (SWE-bench rollout images — the drive that fills) |
| **nvme1n1** (data) | p1 ext4, `noatime`, whole-disk | `/data` | `/data/models`, `/data/cache/huggingface`, `/data/sglang-rebase-v05*` |

- **Models:** `/data/models`, reached via the `~/AI/models → /data/models` symlink. That link *is* the fleet
  `MODELS_DIR=$HOME/AI/models` convention (identical default across all three rigs' `common.sh`). **Keep it a
  symlink** — root-free, portable, and safer than a bind mount (`rm -rf ~/AI/models` removes only the link, not
  the store). The top-level dir symlink is a convenience, not a footgun.
- **HF cache:** `HF_HOME=/data/cache/huggingface`; `~/.cache/huggingface` also symlinks there. Snapshot links
  under `/data/models/*` and `/data/models/hf-mattbucci/*` resolve into `…/hub/models--*/snapshots/*` — resolvable,
  benign.
- **The one hard rule (the trap that cost the fleet 16h):** a model directory name must never lie about its
  content. A `*-BF16` name must resolve to a *genuine* BF16 base (real dir, or a symlink to real BF16) — **never**
  to AWQ/GPTQ/CT/Int4 weights, because `run_all_calibrations.sh` feeds `$MODELS_DIR/<name>-BF16` as a calibration
  base and a quantized target silently double-quantizes. Names that *advertise* a quant stage (`…-BF16-v1-AWQ-CT`)
  are honest and fine. `scripts/maint/models_manifest.py` audits every entry per host (flags `TRAP` /
  `COLLISION` / `DANGLING`) and the calibration entry points abort loudly on a quantized BF16 base — see
  `experiments/04-models-manifest-bf16-symlink-defuse.md`.
- **Disk headroom is per-host, not a shared convention** — don't copy another rig's purge list. Check `df -h /data`
  (models) and `df -h /` (docker images) on *this* box; `scripts/maint/disk_hygiene.sh report` classifies both.

## Server Launch

```bash
source scripts/common.sh
activate_conda
setup_nvidia_env
./scripts/launch.sh <preset>
```

Always source `common.sh` + `activate_conda` + `setup_nvidia_env` before launching.
**`scripts/launch.sh` presets are the single source of truth for per-model flags** — read
the preset, don't restate flags here (duplication is what made earlier versions of this
doc rot).

### CUDA graphs: ON — never disable without an A/B receipt
- Graphs run **ON by default**: the launch.sh default is `CUDA_GRAPH=""` (launch.sh:66 —
  no flag appended, SGLang's own graphs-on default applies). Several presets additionally
  pin `--cuda-graph-max-bs 1` for single-user bs=1 capture.
- `qwen36-vl-reap` (launch.sh:444) is the ONE preset that disables graphs, via
  `--disable-cuda-graph --disable-piecewise-cuda-graph`.
- Receipt for why graphs stay on: enabling them took qwen36 single-user decode
  **31 → 129 tok/s @262K (4.15×, TPOT 32.1 → 7.8 ms)** with 5/5 capabilities under
  graph replay (launch.sh:589-599 preset comment, 2026-06-07).
- `--disable-custom-all-reduce` is a SEPARATE flag and a launch.sh **global default**
  (launch.sh:808): custom AR breaks graph capture on sm_86 TP=2. Retest only via
  `ENABLE_CUSTOM_AR=1`. Receipt: `benchmarks/allreduce-accel-null-2026-06-15.md`.

### Quantization flags
- `--quantization awq` for our self-calibrated native-AWQ checkpoints (auto-promotes to
  `awq_marlin`); some presets pin `awq_marlin` or `moe_wna16` explicitly — the preset knows.
- `--quantization compressed-tensors` only for our own CT-format checkpoints
  (e.g. the `qwen36-dense-ct` preset, launch.sh:482).

## Context & VRAM

Matrix standard: **TP=2, `--context-length 262144`, MAX_RUNNING=1** (CLAUDE.md, Current
Hardware State). 17 presets serve 262144; the rest are capped by KV pool or model
card, not by preference. Per-model **server-verified** depth caps and decode tok/s live in
the README section "Performance — single-user decode at 256K" — copy caps from there, not
from memory. Two examples of the framing:
- `devstral`: 202K real KV pool (42 tok/s @196K) — full-attention-bound.
- `qwen3-vl-32b`: **131K model-card cap** (35 tok/s @127K measured — 127K is the measured
  decode depth, not the cap).

The one hard rule that survives from the old table: **80B+ models do NOT fit in 48 GB.**

## Quantization Pipeline

All ships use **AWQ 4-bit** (native AWQ format). The pipeline:

```
upstream BF16 → GPTQ calibration (llmcompressor) → compressed-tensors → CT→AWQ conversion → native AWQ
```

### Division of labor — calibrations do NOT run on this box
Since 2026-05-19, **all calibrations run on the separate same-repo calibration device**;
this eval box only `git pull --rebase`s to pick up its commits. **Rule 1: never run a
calibration concurrently with serving/eval on the same box** (host OOM + crash receipt in
CLAUDE.md). The pipeline notes below are reference for the calibration device.

### Own-builds only — no community quants as ships or bases
Every `mattbucci/*-AWQ` ship is calibrated end-to-end from the upstream BF16 base
(prune-ourselves rule, README Direction). Pre-quantized 3rd-party AWQ uploads are
reference points only — never ships, never calibration bases.

### Conda env split (calibration device)
**llmcompressor MUST run in the `quant` env, never the serving env** — conflicting
transformers/compressed-tensors/torch pins break both. The serving env
(version-suffixed, e.g. `sglang-v0515`, resolved by `common.sh`) is for inference only.

### DeltaNet/Mamba/SSM layers — DO NOT quantize to INT4
Models with recurrent state accumulate quantization error: `S(t) = gating * S(t-1) + delta`.
- Qwen3.5/3.6 DeltaNet: exclude ONLY `in_proj_a$` / `in_proj_b$` (narrow regex ignore) —
  broad `linear_attn` excludes break the loader; see CLAUDE.md Calibration recipe specifics.
- Community AWQ checkpoints that quantize these layers produce garbage output or Triton
  kernel dtype mismatches (bf16 vs fp16 in conv_state branches).

### MoE calibration — CRITICAL
Standard GPTQ fails for MoE due to expert routing imbalance:
- Use **at least 512 calibration samples** with sequence length ≥1024.
- Verify all experts receive calibration data — check scales for inf/nan/zero.
- For fused expert Parameters: unfuse to per-expert nn.Linear before calibration
  (`patches/qwen3_5moe_unfused_experts.py` pattern).

### Scale audit gate — run after EVERY CT→AWQ conversion
Run `scripts/eval/check_awq_scales.py` on every converted checkpoint before ship:
it scans every `*.scales` / `*.weight_scale` tensor for all-zero / NaN / Inf / extreme
values. `validate_capabilities.py` cannot catch silent zero-scales (model boots, NaN
logits masked). **Pass `--base <bf16_base_dir>` for MoE ships** so the dead-channel
comparator downgrades benign structural-sparsity zeros while still flagging live-block
defects. Non-zero exit = do NOT ship. (CLAUDE.md Critical Rules; the forensic method that
caught the 16h Gemma v3 loss.)

### Chat templates — ALWAYS verify
SGLang reads chat templates from the tokenizer, NOT from standalone jinja files.
Many HuggingFace models ship `chat_template.jinja` as a separate file that SGLang ignores.

**After downloading or calibrating any model, verify:**
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("path/to/model", trust_remote_code=True)
assert tok.chat_template is not None, "Missing chat template!"
```

If `chat_template` is None:
1. Check for `chat_template.jinja` in the model directory
2. Embed its contents into `tokenizer_config.json` as the `chat_template` field
3. Or pass `--chat-template path/to/template.jinja` to SGLang launch

Without a chat template, SGLang falls back to a generic format that produces
wrong outputs (no system prompt handling, wrong special tokens, etc.). Also verify the
template handles list-content and the roles your scaffold sends (see
`scripts/eval/patch_chat_templates_list_content.py` / `patch_chat_templates_developer_role.py`,
both wired into setup.sh — each guards against a receipted silent-blindness failure).

### AWQ checkpoint format
- Marlin requires: output dim divisible by 64, input dim divisible by 128
- Layers that don't meet alignment fall back to torch dequant (upstream since v0.5.11)
- Expert naming: SGLang loader expects `experts.{id}.{proj}.{suffix}` (expert-first).
  llmcompressor's CT save format puts proj first as `experts.{proj}.{id}.{suffix}` —
  the CT→AWQ converters normalize to expert-first via `_normalize_expert_key()`.
  Mismatch silently drops every per-expert key (21B-REAP-v3 saga, commit `839e44b`).
- quant_method: "awq", version: "gemm", zero_point: true, group_size: 128
  (the Gemma 4 family ships group-32 — the receipted cause of its Marlin fallback;
  README Tooling / Performance narrative)

## Benchmarking

- Primary target: **single-user (M=1) decode at TRUE depth** (CLAUDE.md Optimization
  Target). Multi-user throughput is secondary — never sacrifice M=1 latency for it.
- Decode tok/s comes from **server-log gen-throughput or server-verified
  `actual_input_tokens`** — never client-side TPOT (client TPOT under-measures), and
  never `bench_serving --dataset-name random` without `--random-range-ratio 1`
  (the default draws prompt lengths uniform in [1, N]; receipt:
  `benchmarks/bench-depth-bug-2026-07-14.md`).
- Instruments: `scripts/bench/bench_long_context.py` (depth sweep, server-verified) and
  `scripts/eval/run_v0512_fleet_eval.sh` (quality + tok/s + 256K probes; name is
  historical, harness is current).
- Receipts land under `benchmarks/`; regenerate charts with
  `python scripts/bench/generate_charts.py`.
- `scripts/bench/bench_regression.sh` exists but `benchmarks/baselines.json` is not yet
  armed — do not claim a regression tripwire until the fleet-audit item arms it.
