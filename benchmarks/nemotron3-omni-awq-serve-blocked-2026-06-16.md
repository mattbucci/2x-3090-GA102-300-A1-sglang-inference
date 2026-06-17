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

---

## RE-CHECK — revision `4c98711` (2026-06-16, calib-device rebuild)

The calib device re-uploaded (commit `4c98711`, was `9edacecf`). **The dual-zero-point format defect
is FIXED**: `.weight_zero_point` tensors are gone (0, was 5934); quantized tensors are clean standard
AWQ (`qweight`/`qzeros`/`scales`). `check_awq_scales` clean. But it **still does not serve** — two
further blockers, now precisely diagnosed:

### Blocker A — `modules_to_not_convert` is empty (config defect, calib must fix)
The 10304-dim that broke awq_marlin TP=2 (5152/rank) is `language_model.backbone.layers.N.mixer.in_proj`
— the **Mamba2 in_proj**, which is BF16 (`.weight`, not `.qweight`). SGLang applies the awq quant
*method* per-module based on `quantization_config.modules_to_not_convert`, which is **`[]`** — so it
tries to awq-wrap the BF16 Mamba/attention/vision/audio Linears → Marlin shape fail on the Mamba
in_proj. The tensors are correct; only the config's exclusion list is empty (the calib quantize script
didn't write its ignore list to the output config). **Fix (config-only, no re-quant): populate**
`quantization_config.modules_to_not_convert` (and `ignore`) — verified to get past blocker A:
`["sound_encoder","sound_projection","mlp1","vision_model","in_proj","out_proj","conv1d","q_proj","k_proj","v_proj","o_proj","gate","lm_head","embeddings"]`
(leaves exactly `mixer.experts.N.{up,down}_proj` + `mixer.shared_experts.{up,down}_proj` quantized).

### Blocker B — non-gated, 1856-intermediate MoE experts load on NO SGLang int4 MoE kernel
Even with blocker A fixed, the experts fail on every int4 MoE path (NemotronH MoE is **non-gated**
squared-ReLU, `moe_intermediate_size=1856`):
- **awq_marlin**: expert down_proj `input_size_per_partition=1856` ∤ `min_thread_k=128` (1856=128×14.5) — a hard Marlin constraint, TP-independent.
- **moe_wna16**: `RuntimeError: start(1856)+length(1856) exceeds dimension size(1856)` — moe_wna16 create_weights hardcodes a **gated** fused gate_up (`2*intermediate`); NemotronH has only `up_proj` (no gate), so the loader writes the up half at offset 1856 into an 1856-dim slot. (`moe_wna16.py` create_weights line ~277/302 + loader line ~510.)
- **awq (gemm) / cpu-offload**: `KeyError: experts.w2_qweight` — expects fused w13/w2 expert naming; checkpoint has per-expert `up_proj`/`down_proj`.

### Net + recommended path
Serving this int4 ship on our SGLang needs BOTH: (1) calib populates `modules_to_not_convert`
(necessary), AND (2) the expert MoE loads — for which the cleanest options are, in order:
1. **Serving patch: add non-gated-MoE support to `moe_wna16`** (create gate_up at `1×intermediate`
   when the model is non-gated, and load `up_proj` at offset 0). General fix; unblocks this + any
   future non-gated AWQ MoE. Not Marlin → the 1856 shape is fine. (3090 serving lane.)
2. **Re-build with expert `moe_intermediate_size` padded to a 128-multiple** (1856→1920 or 2048) so
   `awq_marlin` works (fast, packed, sharded) — bigger calib change, changes the tensors.
3. **Serve FP8** (R9700's path) — FP8 W8A16 avoids the int4 Marlin/wna16 shape+gating constraints.

Capabilities (thinking/image/video/audio/tool) remain **unvalidated** — gated on the model loading.

---

## RE-TEST on SGLang v0.5.13.post1 (2026-06-16, after the upgrade)

Per user direction, upgraded the stack to **v0.5.13.post1** (new tree `/data/sglang-rebase-v0513` + env
`sglang-v0513`; 16 of 26 patches apply clean, 6 upstreamed drops, 4 Gemma/Cohere regenerates pending —
none of the 4 affect Nemotron). Retested with the `modules_to_not_convert` config fix applied.

**Still BLOCKED — the upgrade does not unblock the int4 MoE.** Root cause refined:
- `moe_wna16` in v0.5.13 (and current upstream main) **still asserts `activation == "silu"` (line ~390) and
  builds the gated `2 * intermediate` fused gate_up** — i.e. it has **no non-gated (relu2) support at all**.
  NemotronH's non-gated experts → the same `start(1856)+length(1856) exceeds 1856` overflow as on 0.5.11/0.5.12.
- SGLang issue **#21149's non-gated fix landed in the Marlin/CUTLASS/Triton MoE *runners*, NOT in
  `moe_wna16`.** So `awq_marlin` *would* now handle non-gated activation — but it can't reach that runner:
  the expert `down_proj` input (`moe_intermediate_size=1856`) fails Marlin's `min_thread_k=128`
  divisibility check at create-weights time (1856 = 128×14.5), TP-independent.

**So: not our implementation** (we never patched moe_wna16/marlin gating; `moe_wna16` is byte-identical
0.5.11→0.5.13), and **not fixed by the upgrade** (the gap is moe_wna16-non-gated + the Marlin int4 expert
shape, both unaddressed by #21149's runner-activation fix).

### Cleanest paths to actually serve it (int4, our stack)
1. **Re-build with `moe_intermediate_size` padded to a 128-multiple** (1856 → 1920 = 128×15, or 2048).
   Then `awq_marlin` works on v0.5.13 — shape passes AND #21149 gives the marlin runner non-gated relu2.
   This is the single cleanest unblock (a calib change; the experts are the only quantized weights).
2. **Serving patch: add non-gated (relu2) support to `moe_wna16`** (relax the silu assert + build/load a
   `1×intermediate` gate_up for non-gated). General fix but framework-level (FusedMoE w13 is gated-shaped).
3. **FP8** (R9700's path) — sidesteps both int4 issues.
