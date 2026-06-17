# Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ — does NOT serve on 2×3090 (2026-06-16)

**Ship under test:** `mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ` (calib-device build,
commit `9edacecf`, pushed 2026-06-16 16:33 PDT). Downloaded clean (22 GB, 34 shards).
**Verdict: BLOCKED on serving — needs a re-build (preferred) or a serving patch.** Capabilities
could NOT be validated because the model never finished loading on any int4 kernel / TP config.

## Pre-serve gates that PASSED
- **Quant pattern correct.** Safetensors index: Mamba layers (0,2,…) qweight=0, attention layer (5)
  qweight=0, vision/radio + audio (Parakeet) encoder towers qweight=0 — i.e. only the MoE/MLP layers
  are INT4 (5934 qweight), Mamba/attn/towers stay BF16, exactly per the plan.
- **`check_awq_scales.py`: 5934 scales + 5934 qweight, 0 flagged.** No zero-scale/NaN/Inf disaster.
- **Arch supported:** `NemotronH_Nano_Omni_Reasoning_V3` is a registered SGLang EntryClass; parsers
  `--reasoning-parser nemotron_3` + `--tool-call-parser qwen3_coder` correct; librosa installed in the
  `sglang` env (was only in `sglang-v0512`) so the Parakeet audio extractor imports.

## ROOT CAUSE — nonstandard dual zero-point in the checkpoint
**Every quantized tensor carries BOTH `.qzeros` (standard AWQ) AND an extra `.weight_zero_point`
(compressed-tensors naming) — 5934 of each.** A standard AWQ checkpoint has only
`qweight`/`qzeros`/`scales`. The extra `.weight_zero_point` is CT residue left by the GPTQ→CT→AWQ
conversion. SGLang's MoE loader (`nemotron_h.py` load_weights → `expert_params_mapping`) maps the
checkpoint's per-expert `experts.{i}.{up,down}_proj.weight_zero_point` to a fused FusedMoE param
`experts.w2_weight_zero_point`, which the moe_wna16 FusedMoE module never creates →
`KeyError: model.layers.1.mixer.experts.w2_weight_zero_point`.

## Full failure map (5 attempts, TP/kernel matrix)
| kernel / config | result |
|---|---|
| `awq_marlin` TP=2 (auto-detected default) | `ValueError: output_size_per_partition=5152 not divisible by min_thread_n=64` — a quantized dim (full 10304) doesn't split into 64-multiples across TP=2 (5152/rank) |
| `awq_marlin` TP=1 | **OOM** — 22.6 GB weights on one card (no fit on 24 GB with any KV) |
| `awq` (GEMM) TP=2 | **OOM** — 22.67 GB/card; the MoE experts do **not** TP-shard under this path (replicated per rank) |
| `moe_wna16` TP=2 | `KeyError: …experts.w2_weight_zero_point` (the dual-zero-point mismatch) |
| `awq_marlin` + `--cpu-offload-gb 8` TP=1 | same `KeyError: …w2_weight_zero_point` (offload routes MoE through the wna16 path) |

Two compounding problems beyond the format: (a) the int4 MoE does not TP-shard under the awq/marlin
paths for `nemotron_h` (≈22.6 GB weights needed per card regardless of TP — won't fit 24 GB), and
(b) the marlin TP=2 path also hits a 5152 (=10304/2) shape that isn't 64-divisible.

## Recommended fix (calibration device — its lane)
Re-emit a **standard AWQ** checkpoint: drop the redundant `.weight_zero_point` tensors (keep only
`qweight`/`qzeros`/`scales`), so SGLang's `moe_wna16`/`awq_marlin` MoE path loads it **packed +
TP-sharded** (which should also resolve the 22.6 GB/card memory wall — packed int4 MoE shards to
fit). Also make the offending dense/expert dim TP=2-Marlin-friendly (pad to a 128-multiple, or keep
that layer BF16) so `awq_marlin` TP=2 works. R9700 already serves the **FP8** variant at 256K (FP8
sidesteps all three int4 issues).

## Serving-side alternative (if a re-build isn't wanted)
A targeted SGLang patch to (1) ignore the extra `.weight_zero_point` tensors during load for this
arch and (2) handle the TP=2 shape — but this works around a malformed checkpoint and doesn't, by
itself, guarantee the MoE TP-shards. Re-build is cleaner.

## Serving recipe (ready for when a fixed checkpoint lands)
Preset `nemotron3-omni` is wired in `launch.sh` (`--reasoning-parser nemotron_3 --tool-call-parser
qwen3_coder --trust-remote-code --enable-multimodal`, mamba cache, chat_template). For caps-validation
once it loads: TP=1 small-ctx if it fits, else TP=2; then `validate_capabilities.py` (basic + thinking
+ image + video + tool) + an audio probe (librosa installed).
