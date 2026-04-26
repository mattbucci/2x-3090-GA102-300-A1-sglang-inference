# Patches, Fixes, and Historical Findings — 3090

This file collects the details of **what was fixed and why** — per-patch narratives, root-cause notes, and cross-team learnings. The top-level `README.md` keeps only current state; once an issue is closed with a patch, the narrative lives here.

Patches apply in numeric order against SGLang v0.5.10. `scripts/setup.sh` applies every `*.patch` in this directory idempotently.

---

## Per-patch notes

### 001 — upstream-sync (~3,000 LOC)
Cherry-picks from SGLang main that we need for supported architectures: Gemma 4, Qwen3.5 / Qwen3-Next, Triton attention updates, pool_configurator.

### 002 — nvidia-model-fixes (923 LOC)
Marlin shape fallback, DeltaNet TP replication, Gemma4 config fixes.

### 003 — deltanet-triton-dtype-fix (51 LOC)
DeltaNet `conv_state` bf16/fp16 cast fix.

### 004 — gemma4-causal-lm-fix (19 LOC)
CausalLM architectures are text-only even if the config class exposes `vision_config` / `audio_config` as class defaults (Gemma4ForCausalLM uses Gemma4Config). Fix: gate `is_multimodal` on `is_causal_lm_only`.

### 005 — ampere-fp8-triton-fallback (59 LOC)
FP8 KV cache on sm_86. sm_86 Triton emits `fp8e4nv` which is not supported on Ampere; route through a PyTorch fallback. FlashInfer handles FP8 KV for `head_dim ≤ 256`.

### 006 — awq-bf16-activation-support (15 LOC)
BF16 activations with AWQ dequant. Needed for Gemma 4 (hidden_size=5376 → FP16 overflow at layer 2). Marlin kernels accumulate in FP32 so BF16 activations are safe.

### 007 — ampere-deltanet-kernel-tuning (48 LOC)
DeltaNet BV=64 tuning for sm_86. Default `BV=32, num_warps=1` under-utilizes the RTX 3090. Sweep found BV=64 gives **1.57x**:

| Config      | BV  | ms/layer | Speedup |
|-------------|:---:|:--------:|:-------:|
| baseline    | 32  | 0.018    | 1.00x   |
| **BV64-w1** | 64  | 0.011    | **1.57x** |

### 008 — awq-moe-wna16-fallback (64 LOC)
`awq_marlin_moe_repack` doubles peak memory during repacking (old + new tensors coexist). For 128-expert MoE models this adds ~7 GB peak/GPU. Env var `SGLANG_FORCE_MOE_WNA16=1` bypasses Marlin repack and routes to [MoeWNA16 Triton kernels](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/moe_wna16.py). Only works with compressed-tensors format; AWQ packing ≠ WNA16 packing.

### 009 — qwen35-moe-causalLM
Qwen3.5 MoE text-only CausalLM wrapper. Adds logits processor + mrope handling; `setup.sh` auto-patches on apply. Required for the REAP-28B path.

### 011 — triton-attention-fp32 (R9700 backport)
**Root cause:** Triton attention kernels accumulate `e_max`/`e_sum`/`re_scale` in BF16 and `tl.dot()` is called without `out_dtype=tl.float32`. Causes 15% mean error vs FP32 after 128 KV tokens; compounds catastrophically over 60 layers (Gemma 31B). Both RDNA4 and Blackwell SM12.x hit this; Ampere/Hopper tolerate it. **Fix:** FP32 casts throughout the decode + extend kernels. We use FlashInfer (FP32 internally) for most models, so this only bites on models forced onto Triton attention (Gemma 4 if `head_dim=512` workaround ever lands; Qwen3-VL-32B Dense with 64 layers is also a candidate).

### 012 — sliding-window-decode-fix (R9700 backport)
`window_kv_offsets` was captured then discarded at `triton_backend.py:278` — SWA decode computed on full pool indices instead of the window slice.

### 014 — gemma4-reasoning-parser (40 LOC, R9700 backport)
`Gemma4Detector` cherry-picked from unreleased upstream SGLang PR [#21952](https://github.com/sgl-project/sglang/pull/21952). Enables `--reasoning-parser gemma4` for `<|channel>thought` / `<channel|>` markers. Wired into the `gemma4` and `gemma4-31b` launch presets.

### 015 — ct-wna16-dequant-layout-fix
The `CompressedTensorsWNA16` dequant fallback (the path Marlin rejects) assumed `[in//pack, out]` layout and `[num_groups, out]` scales. Real CT layout is `[out, in//pack]` and `[out, num_groups]`. On TP-sharded RowParallel layers this silently produced garbage shapes (Gemma 4 down_proj: in=2112 → 1056/GPU → 33 groups, so scales came out `[out, 33]` but were multiplied as if `[1056, out]`). Rewrote the fallback to keep the native `[out, in]` orientation and transpose once at the final matmul.

### 016 — ct-moe-gelu-triton-route (47 LOC)
Gemma 4 MoE uses gelu activation. `CompressedTensorsWNA16MoE` (Marlin MoE path) asserts silu-only. Sibling `CompressedTensorsWNA16TritonMoE` uses the Triton fused-MoE runner which handles both activations (`moe_runner/triton.py` lines 215/231). Two changes:
- `compressed_tensors.py`: env var `SGLANG_FORCE_CT_MOE_TRITON=1` routes CT MoE to the Triton scheme on CUDA (default path on HIP).
- `compressed_tensors_wNa16_moe.py`: relax SiLU-only assertion in the Triton scheme to `silu | gelu`.

Launches Gemma 4 26B MoE on 3090. Generation quality is still degraded (see Known Issues / top-level README).

### 017 — moe-wna16-gelu-activation (21 LOC)
`moe_wna16.py:373` hard-asserts SiLU-only for the WNA16 fused-MoE runner, which blocks Gemma-4 21B REAP AWQ from serving. Relaxed to `silu | gelu` since `moe_runner/triton.py` already dispatches both.

### 018 — qwen36-vision-config-dict-wrap (R9700 backport, 19 LOC)
`qwen_vl.py` multimodal processor assumes `hf_config.vision_config` is a `PretrainedConfig` (attribute access). llmcompressor-saved CT configs (Qwen3.6 VL, and likely our own Phase 3 Qwen3-VL-32B output) ship `vision_config` as a plain dict, causing `AttributeError: 'dict' object has no attribute 'spatial_merge_size'` at HTTP 500 on first image request. Wrap once at processor init with `SimpleNamespace`. R9700 verified this unblocks `mattbucci/Qwen3.6-35B-A3B-AWQ-thinking-vision`.

### 019 — qwen3_5-moe-vl-config-dataclass-and-model-init (2026-04-24, 60 LOC)
Three-part fix required to load the Qwen3_5MoeForConditionalGeneration path on Python 3.13 + transformers 5.x:
- **Model init dict-wrap (qwen3_vl.py):** symmetric to patch 018 but on the model side — `Qwen3VLForConditionalGeneration.__init__` accesses `config.vision_config.{hidden_size,depth,…}` and `config.text_config`; llmcompressor-saved CT configs ship both as raw dicts / bare PreTrainedConfig. Re-wrap into the proper sub_configs class before super() runs.
- **Explicit `__init__` on subclasses (qwen3_5.py configs):** transformers 5.x on Python 3.13 auto-decorates `PretrainedConfig` subclasses that don't define `__init__` as dataclasses, which **replaces** the inherited init with a generated one that never runs the parent's attribute defaults (`norm_topk_prob=True`, `num_experts=512`, etc.). Reproduced standalone: `Qwen3_5MoeTextConfig(**text_config_dict)` → `hasattr(tc, 'norm_topk_prob') == False`. Fix: add explicit `def __init__(self, **kwargs): super().__init__(**kwargs)` to `Qwen3_5Moe{Vision,Text,}Config`.
- **Drop `model_type` from kwargs when re-wrapping** — it's a class attr, and `PreTrainedConfig.to_dict()` can return `model_type=""` that would overwrite the class default.

Without patch 019, Qwen3_5MoeForConditionalGeneration fails to instantiate (`'dict' object has no attribute 'hidden_size'`) or later raises `'Qwen3_5MoeTextConfig' object has no attribute 'norm_topk_prob'`.

---

## Shipped model history

Narratives for models that are now working. Current-state entries live in the top-level `README.md`.

### Qwen3.6-35B-A3B AWQ-native thinking+vision — shipped 2026-04-24
First successful Qwen3_5MoeForConditionalGeneration load on 3090. Path:
1. Downloaded R9700's `mattbucci/Qwen3.6-35B-A3B-AWQ-thinking-vision` CT upload (20 GB).
2. Attempted direct load → four class-of-bug discoveries: dict-wrap vision_config at model init, dict-wrap text_config, transformers 5.x dataclass replacement of subclass init, model_type kwarg overwriting class attr. All fixed in patch 019.
3. With patch 019: server boots clean but generation emits `_4_4_4_4…` on fp8_e4m3 KV / infinite `"15*17 = 15*17 = …"` loops on bf16 KV.
4. Root cause isolated via safetensors inspection: `shared_expert_gate.weight_packed` has shape `(1, H)` — output dim 1 can't be AWQ-packed (requires divisibility by 8), and the NVIDIA CT loader doesn't fall back to BF16 for unpackable shapes the way R9700's `convert_moe_ct_to_awq.py` does. So every token's shared-expert gate was reading garbage from INT4 weights clamped to ~15 distinct values.
5. Ported R9700's `convert_moe_ct_to_awq.py` verbatim; ran it on the CT checkpoint → 30970 INT4 expert weights + 40 BF16 `shared_expert_gate` fallbacks + 696 passthroughs, 20 GB output.
6. Launched converted native AWQ through patch 019: validator **4/4 PASS** (basic/thinking/vision/video-skipped), **~33 tok/s short-context** (vs R9700's 21.6 on ROCm — +55% from awq_marlin kernel).

Bench curve on 2x 3090, flashinfer attention, bf16 KV: 33.4 short / 21.8 @32K / 5.8 @160K / 2.6 @250K tok/s. Long-context drop is steeper than R9700's flat 20 tok/s @131K ROCm curve — verified by A/B: matching R9700's CHUNKED/DECODE_STEPS/MAMBA_CACHE tuning didn't move the needle, and swapping to `--attention-backend triton` was *worse* on Ampere (3.4 tok/s @131K). Conclusion: R9700's flat long-ctx curve is ROCm-triton-kernel-specific, not portable; Ampere decode on this model is bounded by flashinfer's hybrid-attention path past ~80K.

Detailed bench artifact: `benchmarks/qwen3.6-35b-a3b/awq-native-thinking-vision.json`.

### Qwen3.6-27B Dense AWQ thinking+vision — shipped 2026-04-23 (v3)
Dense VL, 64 layers, 262K native, `Qwen3_5ForConditionalGeneration`. Three calibration attempts:
- **v1 (9.5h):** copied Qwen3-VL script, all `linear_attn` projections INT4 → `!!!!!` garbage everywhere.
- **v2 (7.6h):** added `re:.*linear_attn\..*` to GPTQ ignore (all DeltaNet BF16) → still `!!!!!` garbage.
- **v3 (7h):** ignore ONLY `in_proj_a$` + `in_proj_b$` → 4/4 PASS.

Root cause: SGLang's `Qwen3_5GatedDeltaNet` loader merges `in_proj_qkv + in_proj_z → in_proj_qkvz` (passes outer `quant_config` → expects INT4) and `in_proj_b + in_proj_a → in_proj_ba` (hardcoded `quant_config=None` per `qwen3_5.py:186` → expects BF16). So the loader demands: `in_proj_qkv/z/out_proj` INT4, `in_proj_a/b` BF16, `conv1d` BF16 (it's `nn.Conv1d`, GPTQ doesn't touch it). Cross-validated against R9700's shipped `mattbucci/Qwen3.6-27B-AWQ-thinking-vision` — throughput identical within noise across 1K/8K/32K/131K contexts; quality trade varies with calibration recipe (R9700 used `thinking_vision_video`, we used `thinking_vision`).

Bench on 2x 3090: 30.4 @1K / 30.1 @8K / 29.7 @32K / 21.1 @131K tok/s. DeltaNet Triton kernels cap short-ctx around 30 tok/s (vs Qwen3-VL-32B Dense's 69 tok/s on flashinfer).

### Qwen3-VL-32B Dense AWQ thinking+vision — shipped 2026-04-21
First successful vision-preserving self-calibration. Recipe: `thinking_vision`, 256 samples × 1024 tokens, 13.5h CPU. Vision tower held in BF16 via `ignore=[re:.*visual\..*, re:.*vision_tower.*]`. Validator 4/4 PASS: basic `(reasoning)paris`, thinking 108 tok cleanly terminated, vision `saw=['red','circle','round']` on red-circle probe. Needed patch 018 (R9700 backport) to wrap the dict `vision_config` at processor init.

### Gemma 4 31B Dense AWQ — unblocked 2026-04-23
`QUANT=compressed-tensors EXTRA_ARGS="--attention-backend torch_native --disable-cuda-graph --disable-piecewise-cuda-graph" ./scripts/launch.sh gemma4-31b`. 11.2 GB/GPU, 48K max tokens at 16K context. Validator 3/4: basic+thinking PASS, vision generates (`"the image shows a single cuneiform character"`) — hallucinated, not the plumbing failure the 21B REAP has. Short-ctx bench 28 tok/s, TPOT 35 ms @ 1K. Model config says `compressed-tensors` despite the AWQ directory name.

### Qwen3.5-28B REAP thinking recalibration — cancelled 2026-04-19
v1 died at layer 13/41 on harness restart (lost 7h 45min — this is where the `setsid` detach rule came from). v2 killed at layer 1/41 because R9700 shipped a working thinking-preserving Qwen3.5-27B-AWQ v2 (`mattbucci/Qwen3.5-27B-AWQ-4bit-calibrated`, basic FAIL→PASS) and Qwen3.6-35B-A3B GPTQ passes the thinking validator on both rigs — the regression this was fixing is superseded by switching long-ctx thinking workloads to `qwen36` or `qwen3-ream`. May be re-opened once we have free calibration cycles + the upgraded `thinking_vision_video_audio` recipe R9700 shipped after v2 started.

### Coder-REAP-25B-A3B AWQ — shipped 2026-04-25
Cerebras's REAP prune of Coder-30B (103 routed experts vs 128). First boot hung at CUDA-graph capture; `--disable-piecewise-cuda-graph` cleared it. Validator basic PASS, code probes (Fibonacci, lambda) compile clean. R9700's HF upload `mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ` serves directly — no calibration burn. Preset `coder-reap-25b` in `scripts/launch.sh`. Currently being benched on SWE-bench Lite as the first SWE-bench eval candidate.

### Qwen3-VL-30B MoE AWQ self-calibration — closed 2026-04-25 (loader bug, not calibration)
Three calibration attempts (10.9h v1 + 1.5h v2 smoke + 1.3h v3 smoke) all produced identical multilingual garbage (`各项工作` repetition). v3 used the EXACT pattern R9700 ships successfully for Qwen3MoE / Coder-30B-A3B (`SequentialQwen3VLMoeTextExperts` registered for `Qwen3VLMoeTextExperts`, parent `Qwen3VLMoeTextSparseMoeBlock.forward` untouched). Same failure → calibration is fine; **SGLang's `Qwen3VLMoeForConditionalGeneration` loader is broken on both CT and AWQ output**. The community vLLM AWQ for this class produces the same garbage — broken layer is on load/serve side. To unblock: trace `load_weights` for missing weight or shape mismatch, or wait for upstream SGLang fix. Vendored llmcompressor pattern at `components/llmcompressor/` (commit `30845208`) is REUSABLE for any future Qwen3MoE/Qwen3VLMoe calibration on this stack.

### Calibration ignore-list audit — closed 2026-04-25
After R9700 flagged a `shared_experts` plural typo, swept all our self-calibrated checkpoints. Findings:
- Qwen3-VL-32B-CT (117 entries): vision blocks preserved ✓
- Qwen3.6-27B-CT (207 entries): vision blocks preserved ✓
- gemma-4-26B-A4B-it-AWQ-4bit (312 entries): per-layer `mlp.gate_proj` + `router.proj` + vision tower preserved ✓
- gemma-4-21b-REAP-AWQ-thinking-vision (0 entries): empty `ignore=[]`, everything went INT4. Re-cal v2 with explicit ignore list completed 2026-04-25; output still broken (clippable_linear shim suspected, see Gemma 4 entry in main README).

### R9700 cross-team thread — Qwen3.6-35B-A3B AWQ-native upload + audit fixes (2026-04-24/25)
- R9700 published `mattbucci/Qwen3.6-35B-A3B-AWQ-native-thinking-vision` (19 GB, 10 files) so 3090 users skip the CT→AWQ conversion. R9700 numbers: 21.6 tok/s short / 20.6 @131K flat (ROCm triton moe_wna16). Our 33 short / 2.6 @250K curve on Ampere is +55% short / 8x worse @250K — flashinfer asymmetry, not quant.
- HF upload lesson: plain `hf upload <repo> <dir>` completed the 19 GB push in ~1 minute after `hf upload-large-folder` stalled 11h at `committed: 0/9` (XET worker deadlock). Plain upload for ≤25 GB; large-folder only past 50 GB.
- R9700 fixed: `convert_moe_ct_to_awq.py` now copies `quantization_config.ignore` through (one-line); patched 4 affected HF configs in place.
- R9700 corrected the "native AWQ required for NVIDIA" framing to "required for NVIDIA on SGLang" (vLLM/autoawq/TGI handle CT correctly). Both CT model cards now ship a per-stack recommendation table.
- R9700 fixed `gemma-4-31B-it-AutoRound-AWQ` arch field to `Gemma4ForConditionalGeneration` so the multimodal loader engages.
- R9700 confirmed `mattbucci/gemma-4-26B-AWQ` serves clean text + thinking on RDNA4 → strongly implicates our `clippable_linear` shim as the bug for Gemma 4 generation on 3090 (vision crashes for separate SWA reasons there). Two diagnostic suggestions: (a) `git grep -n soft_cap` in upstream Gemma 4 model code to find the actual clip op, then check whether our shim path bypasses it; (b) A/B with `EXTRA_ARGS="--attention-backend torch_native"` to rule out attention-kernel involvement.
- R9700 advice on the 35B long-ctx regression: patch 011 (FP32 online-softmax accumulation) might port via `--attention-backend triton` + the patch — same bug class hit RDNA4 and Blackwell; flashinfer probably already does FP32 internally so this is an A/B worth running.

---

## Cross-team findings (3090 ⟷ R9700)

The sister RDNA4 project runs the same SGLang v0.5.10 stack. Findings that produced patches or changed how we ship are here; day-to-day sync happens in the two READMEs.

- **BF16 attention precision** affects every new architecture (RDNA4, Blackwell SM12.x). Fix: FP32 accumulation in the online softmax (patch 011).
- **AWQ calibration silently breaks thinking and vision.** Quants calibrated on plain text (Open-Platypus, WikiText2, c4) lose `<think>` stop-token behavior and vision-language alignment. Rule: every new quantized model must validate (a) an image+text roundtrip and (b) a thinking-tagged generation that cleanly terminates, before launch. `scripts/eval/validate_chat_template.py` (static) + R9700's `validate_capabilities.py` (live) are the pre-flight gates.
- **Recommended calibration datasets** (reasoning + vision preserving): `a-m-team/AM-Thinking-v1-Distilled`, `glaiveai/reasoning-v1-20m`, `LLaVA-Instruct-150K`, `AI-MO/NuminaMath-CoT` (+9.81% GPTQ accuracy vs WikiText2 in R9700 measurements). Recipe builder: `scripts/quantize/calibration_datasets.py` (`thinking_text`, `thinking_vision`, `code_vision`, `code_thinking`).
- **Chat template is a silent bug magnet.** Devstral community AWQ's template emits BOS → `<unk>`; Gemma 4 community weights ship with no template at all; Qwen3 family needs `temperature ≥ 0.3` to avoid greedy-decode repetition loops. Always render the template with and without `enable_thinking`, and verify `chat_template is not None`.
- **AutoRound > GPTQ > AWQ for INT4 quality** — Intel AutoRound (arXiv 2309.05516) uses SignSGD for 200 iterations to jointly optimize rounding offsets and clipping ranges. Can export to both GPTQ and AWQ formats. [RedHatAI reports 99.4%+ quality](https://huggingface.co/RedHatAI/gemma-3-27b-it-quantized.w4a16) on CUDA.
- **DeltaNet layers must stay BF16.** INT4 noise accumulates through the recurrent state `S(t) = g*S(t-1) + delta` and destroys quality. Architectural limit.
- **DeltaNet replication mandatory for TP=2.** Qwen3.5-27B: TP RowParallelLinear splits matmul `W_0@x_0 + W_1@x_1` which differs from `W@x` by ~1 ULP in FP16; the recurrent state compounds it across layers. Fix: replicate all DeltaNet + MLP layers (`tp_size=1`), SSM state `tp_world_size=1`. Costs 19 GB/GPU weights replicated, which is why Qwen3.5-27B is context-limited to 32K.
- **Community AWQ fails for DeltaNet** on both rigs. Self-calibrate with GPTQ + CT→AWQ.

---

## MoE quantization lessons

Standard GPTQ/AWQ **fails** for MoE (MoEQuant, ICML 2025):
1. **Inter-expert imbalance.** The router unevenly distributes calibration data; rare experts get zero / garbage calibration. Our Gemma 4 26B baseline: 1/128 experts received calibration data, the rest got inf scales.
2. **DeltaNet/SSM sensitivity.** Recurrent state accumulates INT4 error — must stay BF16.

**Fixes used:**
- Expert-balanced sampling (MoEQuant EBSS, GPTQModel FailSafe).
- Skip recurrent layers from INT4 targets (see `scripts/quantize/quantize_qwen35_28b_moe_reap*.py` `ignore=[...]`).
- **In-memory expert fusion for Qwen3.5 MoE REAP:** BF16 source stores `experts.{id}.{gate,up,down}_proj.weight` per expert, but the HF model class expects fused `experts.gate_up_proj` / `experts.down_proj`. Calibrating without fusing silently produces garbage. Pipeline does the fusion before `oneshot()` runs.

---

## Qwen3.5-27B Technical details

Hybrid DeltaNet (linear attention) + full attention. TP=2 requires replicating all layers to avoid FP16 precision errors accumulating through the DeltaNet recurrence.

**Root cause:** TP RowParallelLinear splits matmul `W_0@x_0 + W_1@x_1` which differs from `W@x` by ~1 ULP in FP16. DeltaNet's `S(t) = g*S(t-1) + delta` compounds this across 48 layers × N tokens.

**Fix:** Replicate all DeltaNet + MLP layers (`tp_size=1`), SSM state `tp_world_size=1`.

VRAM per GPU: ~19 GB model (replicated) + 1.27 GB DeltaNet state + 0.92 GB KV cache (FP8) = ~21 GB. Only 32K context fits on a 3090.

**Pipeline bottleneck analysis (pre-patch 007):**

| Operation | ms/model | %  |
|-----------|:--------:|:--:|
| MLP forward | 19.9 | 45% |
| Recurrent update | 8.3 | 19% |
| QKV projection | 7.9 | 18% |
| Output projection | 2.9 | 7% |
| RMSNorm + gating | 2.1 | 5% |
| Conv1d | 1.3 | 3% |
| **Theoretical** | 44.1 | 22.7 tok/s |
| **Actual** | 74 | 13.5 tok/s |

MoE kernel configs generated for RTX 3090 (Triton 3.5.1): `E=128,N=768` (Coder-30B, Qwen3-VL-30B), `E=128,N=704` (Gemma 4), `E=103,N=768` (Coder-REAP). Auto-loaded by `device_name=NVIDIA_GeForce_RTX_3090`.

---

## Gemma 4 notes (Ampere sm_86)

Blocked by FlashInfer on `head_dim=512`. Partial unblock (text path boots) via patches 015 + 016 + `SGLANG_FORCE_CT_MOE_TRITON=1` + `--attention-backend torch_native`. Historical fix list:

1. **FP16 overflow at layer 2** (hidden_size=5376). Fix: `--dtype bfloat16` + patch 006.
2. **CT→AWQ conversion quality poor** — cosine similarity 0.845 on q_proj. Community CT quants generate garbage; self-calibrate.
3. **Missing chat template** — embed jinja into `tokenizer_config.json`.
4. **`num_experts` is None for Dense** — fix: `getattr(config, "num_experts", 0) or 0`.
5. **MLP down_proj shape mismatch** — patch 015 fixes the CT WNA16 dequant layout.
6. **MoE gelu activation** — patch 016 routes to Triton runner which supports gelu.

**Still open:** generation emits `<pad>` tokens — weight-layout or kernel mismatch in the CUDA Triton MoE path, or AWQ calibration that dropped the gelu activation profile. See top-level README Known Issues.

## FlashInfer head_dim support

| head_dim | FlashInfer (sm_86) | Models |
|:--------:|:------------------:|--------|
| 64-256 | Supported | Qwen, Devstral |
| **512** | **Not supported** | **Gemma 4** |

Possible unblock paths: SDPA fallback (partial — patches 015/016 got the text path booting), [FFPA kernels](https://github.com/DefTruth/ffpa-attn-mma), TRTLLM FMHA, or llama.cpp for Gemma 4 serving (reported 80-110 tok/s).
