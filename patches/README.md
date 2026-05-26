# Patches, Fixes, and Historical Findings — 3090

This file collects the details of **what was fixed and why** — per-patch narratives, root-cause notes, and cross-team learnings. The top-level `README.md` keeps only current state; once an issue is closed with a patch, the narrative lives here.

Patches apply in numeric order against SGLang **v0.5.12** (rebased from v0.5.11 on 2026-05-26 — see the v0.5.12-rebase note below). `scripts/setup.sh` applies every `*.patch` in this directory idempotently. **23 patches** ship currently (`ls patches/*.patch | wc -l`):

- **13 v0.5.11-targeted patches** from the 2026-05-07 v0.5.10→v0.5.11 rebase (commit `1655e46`).
- **002, 028, 029** added post-rebase (cross-team port + Gemma4 mm per-expert + Qwen3.5 CT shared_expert_gate).
- **031, 034, 035, 036** added post-rebase for v0.5.11 Qwen3.5/3.6 + sampler-Inf surface (deltanet AWQ loader, sampler ±Inf detection, MoeForCausalLM EntryClass + HF layer_types fallback).
- **030 (v0.5.10 backport of 028) DELETED 2026-05-09** with the env upgrade — patch 028 now applies natively. Historical narrative kept below.
- **v0.5.11 → v0.5.12 rebase (2026-05-26):** 17 patches applied clean; **005, 011, 021, 028 re-ported** + **037 added**. 005: dropped the obsolete flashinfer-version hunk (0.5.12 pins `flashinfer 0.6.11.post1` and adds `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK`); kept the Ampere fp8 + batch-invariant bf16 fixes. 011: CUDA fp32-softmax casts only (decode all sites + extend p/v accum + q/k dot); HIP-only hunks dropped (R9700's domain). 021: gelu dispatch relocated into 0.5.12's new `clamp_limit` else-branch + `activation` arg added to the `register_custom_op`-wrapped `fused_marlin_moe` signature (it infers `str` schema args). 028: re-cast as a regex+suffix extension to 0.5.12's *native* `per_expert_match` so AWQ `qweight/qzeros/scales` per-expert keys bind to the fused `w13/w2` params — upstream only handled dense/FP8 `weight/weight_scale`, so AWQ experts silently went unloaded → garbage (caught by the correctness gate). 037: drops the new mandatory gRPC Rust ext from `python/pyproject.toml` (needs `protoc`; we serve over HTTP, `grpc._core` has zero imports). All 22 verified applying clean to a fresh v0.5.12 tree; build validated on coder-30b-eval (silu MoE) + gemma4 (gelu MoE) before landing. **038 added (follow-up):** defensive `getattr(config, "norm_topk_prob", True)` in the reused qwen2_moe block — `Qwen3_5MoeTextConfig` omits `norm_topk_prob`; validated on qwen36-ream. **0.5.12 correctness gate: 7/9 coherent** (coder-30b-eval, gemma4, devstral, qwen36-dense, coder-reap-25b, coder-30b-ream, qwen36-ream); qwen36 deferred on a CT-MoE `w2_weight_packed` loader bug, qwen35-moe blocked on a missing checkpoint.

Patches dropped during the v0.5.11 rebase as upstreamed: 001, 002 (original — the new 002 is unrelated cross-team), 006, 008, 009, 014, 015, 016, 019, 020, 022 — historical narratives below kept for context but those `.patch` files no longer ship.

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

### 020 — gemma4-clippable-linear-shim (2026-04-30, 289 LOC, v2 = upstream port)
Gemma 4 26B / 21B-REAP launches need a `ClippableLinear` symbol in `sglang.srt.layers.linear` because the architecture's `gemma4.py` imports it. Initial v1 was a one-line alias-shim (`ClippableLinear = ReplicatedLinear`); that surfaced a subtle 2-tuple/plain-tensor signature mismatch — `ReplicatedLinear.forward` returns `(out, bias)`, while Gemma 4 calls `q, k, v = self.qkv(...)` on the result and silently fed wrong shapes downstream. v2 ports the real upstream `clippable_linear.py` (per-tensor clip buffers via `nn.parameter.Buffer(torch.tensor(±_INF), persistent=False)`, plain-tensor forward returns), restoring correct call shapes. Server boots clean post-patch but Gemma 4 MoE decode still emits `<pad>` garbage — root-cause investigation continues in the top-level Known Issues section.

### 021 — marlin-moe-gelu-activation (2026-04-30, 118 LOC)
Drops `assert runner_config.activation == "silu"` from four Marlin MoE chokepoints (`fused_marlin_moe`, the MoE Marlin runner, `compressed_tensors_wNa16_moe`, `marlin_utils`) and adds activation-dispatch on `runner_config.activation` to call `gelu_and_mul` vs `silu_and_mul`. Required for Gemma 4 MoE (uses `gelu_pytorch_tanh`). Kernel-form check: `sgl_kernel.gelu_and_mul` matches `gelu_pytorch_tanh` to 0.00012 in FP16, so the gelu kernel itself is correct; routing the model through it is what 021 does. Loads cleanly across Qwen3 MoE / Coder / Coder-REAP (silu) and Gemma 4 (gelu) without false-asserting.

### 022 — gemma4-causal-dedup-entry-class (2026-04-30, 24 LOC)
Removes `Gemma4ForConditionalGeneration` from `gemma4_causal.py`'s `EntryClass` list (now `EntryClass = [Gemma4ForCausalLM]`). Pre-patch, that wrapper class was registered by both `gemma4_causal.py` and `gemma4_mm.py`; once 020 v2 made `gemma4_mm.py` actually import-clean, the dual registration started asserting (`AssertionError: Duplicated model implementation for Gemma4ForConditionalGeneration`) and blocked any subsequent model launch. Net behavioral change: `gemma4_mm.py` is now the sole owner of the multimodal class — text-only path stays in `gemma4_causal.py`.

### 023 — gemma4-moe-mlp-quant-detection (2026-05-03 → upgraded 2026-05-09)
**Detects whether the calibration recipe quantized the dense MLP, instead of hardcoding the assumption.** The original 2026-05-03 fix was a hardcoded `quant_config=None` on MoE-block dense MLPs — that worked for the original community Gemma 4 26B AWQ where the recipe explicitly kept dense MLP in BF16. But our own ships (`mattbucci/gemma-4-26B-AWQ` HF mirror, locally-built v3b 21B-REAP) calibrate the dense MLP fully — `quantization_config.ignore` is empty for `mlp.*` so all `mlp.{gate,up,down}_proj` ship as `.qweight`. The hardcoded `quant_config=None` made the model create BF16 `.weight` placeholders that never matched the AWQ keys; loader silently failed (60 not-initialized warnings/checkpoint), MoE forward returned zeros, generation produced `<pad>` for every prompt. The 2026-05-09 upgrade replaces the hardcode with detection: parse `config.quantization_config.ignore`, scan for `\.mlp\.(gate|up|down)_proj` matches, set `_mlp_quant_config = None if matched else quant_config`. Both case (a) BF16 dense MLP and case (b) AWQ dense MLP now work.

Discovery (2026-05-03 case (a)):
- `SGLANG_GEMMA4_TRACE=1` forward-hook localized layer 0's `gate_up_proj` producing **15078 NaN + 3267 inf out of 84480 outputs** from a clean BF16 input (abs_max=43.5).
- Trace artifacts in `benchmarks/quality/gemma4-trace-{00,15}.json` (clean / NaN respectively).
- Safetensors comparison confirmed: original community checkpoint had `mlp.gate_proj.weight` as plain BF16 (case a).

Discovery (2026-05-09 case (b)):
- Both `mattbucci/gemma-4-26B-AWQ` HF mirror and v3b 21B-REAP show `mlp.{gate,up,down}_proj.qweight + scales + qzeros` keys for all 30 MoE layers — recipe quantized the dense MLP.
- Server boot warning: `Some weights are not initialized from checkpoints: ['language_model.layers.0.mlp.down_proj.weight', 'language_model.layers.0.mlp.gate_up_proj.weight', ...]` listing 60 paths.
- Generation: literal `<pad><pad>...` for every prompt regardless of quant routing (`QUANT=moe_wna16 DTYPE=bfloat16` doesn't fix it — the bug is constructor-side).

**Validator post-detection-upgrade on Ampere TP=1 / 4K (v0.5.11):** 4/4 PASS for both v3b (`'a solid red circle with a black outline on a white background.'`) and HF mirror (`'a red circle with a black outline is centered on a white background.'`) — content-aware vision + video on the previously-broken per-expert AWQ path. 0 not-initialized warnings on either build. Calibration recipe alignment: all `quantize_gemma4_*.py` scripts ignore `vision_tower` / `embed_vision` / `multi_modal_projector` only — dense MLP stays in the AWQ-quantized set, so detection routes them through `quant_config` correctly.

### 024 — gemma4-mm-towers-no-quant-config (2026-05-03, 28 LOC)
**Unblocks the Gemma 4 26B MoE multimodal route → basic + thinking VERIFIED, vision validator-passes-but-degraded.** Same shape as patch 023 but for the multimodal towers. The Gemma 4 calibration recipe explicitly leaves the vision tower, audio tower, and embed_vision/embed_audio projections in BF16 (recipe.yaml: `ignore: re:.*vision_tower.*, re:.*embed_vision.*, ...`). The checkpoint stores them as plain `.linear.weight` tensors with no qweight/scales/shapes. Without this patch, `Gemma4ForConditionalGeneration.__init__` constructed `Gemma4VisionEncoder`, `Gemma4MultimodalEmbedder`, and `Gemma4AudioEncoder` with `quant_config=quant_config` (AWQ), causing the loader to silently miss every `vision_tower.encoder.layers.*.mlp.gate_up.gate_up_proj.weight_packed/scale/shape` key. The vision tower stayed at random init → vision check produced unicode/lorem-ipsum hallucinations.

**Fix:** pass `quant_config=None` to all four mm tower components (vision_tower, embed_vision, audio_tower, embed_audio).

**Plus a runtime requirement:** the Gemma 4 26B SigLIP vision tower NaNs in FP16 (overflow in attention softmax — abs_max activations exceed 65504), producing `<pad>` token spam. The launch.sh `gemma4` preset now defaults `DTYPE=bfloat16`. `_ENV_DTYPE` env-override pattern added (mirrors `_ENV_KV_DTYPE`).

**Plus a config-side requirement:** the 26B checkpoint config.json must declare `architectures: ["Gemma4ForConditionalGeneration"]` (multimodal route). With `Gemma4ForCausalLM` it loads as text-only and image_url payloads silently degrade. Some upstream community checkpoints ship with the wrong arch — fix in the model directory.

Validator post-fix on Ampere TP=1 / 2K: **basic + thinking VERIFIED, vision validator-passes-but-degraded** (`gemma4-26b-vision-bf16-fix-May03` and `gemma4-26b-preset-default-May03` JSON entries).

- **Thinking confirmed real (2026-05-03 deeper probe):** ran `/tmp/gemma4_thinking_probe.py` with `skip_special_tokens=False` + `chat_template_kwargs={enable_thinking: True}` against an apple-counting word problem (17 - 5 - 2 + 12 = 22 ÷ 2 = 11). Output: separated `reasoning_content` channel starting with raw `thought\n` marker (the `<\|channel>thought` token rendered with skip-special off), real step-by-step scratch work with intermediate `22` and final `11` both present, then a clean `content` field with the formatted answer. `finish_reason=stop`, 310 reasoning tokens vs 119 content tokens. Not just a validator-keyword-match — the channel is structurally honored.

- **Vision: validator-passes-but-degraded.** The validator's loose keyword grep counts a hit (saw=['red','round']) on the response `'(reasoning)this image features a collection of small, scattered red and black pixels against a white background.'`. Tower loads cleanly (no NaN), but the model is not actually recognizing the red circle — it's describing pixel noise. Quality is below Qwen3-VL/Qwen3.5 baseline. Same response shape on R9700's `mattbucci/gemma-4-26B-AWQ`, so this is calibration/recipe-side, not 3090-loader-side. Suspects to investigate: layer_scalar defaults post-SigLIP-projector, `embed_vision` pre-projection RMSNorm shipping with `with_scale=False`, image-token expansion in the chat template, projector alignment after the recipe's `ignore: re:.*vision_tower.*` left it untouched from BF16 base. Open Known Issue.

**Cross-team portability — 3090-Ampere-specific (R9700 task #65, 2026-05-03 commit `5c3d071`).** R9700 ported, applied, and tested 023+024 against their `mattbucci/gemma-4-26B-AWQ` baseline (already 4/4 PASS pre-patch with content-aware vision response `'red and black pixels...'`). Post-patch on RDNA4: basic 1/4 PASS, then HSAIL 0x1016 in `top_k_top_p_min_p_sampling_from_probs_torch` on the first thinking request — third RDNA4 instance of the same exception class (matches Coder-Next + Gemma4-31B long decode under their open task #18). They reverted via `git apply --reverse` and re-validated 4/4. **Conclusion:** R9700's AWQ loader auto-falls-back to BF16 for empty qweight slots, so the bug 023+024 fixes never manifests on RDNA4. Applying the fix changes the dense-MLP / mm-tower kernel path to plain `nn.Linear` BF16, which trips an unguarded HSAIL in a downstream sampler kernel. Patch files retained on R9700's `patches/` for reference + 3090 traceability but **NOT applied** via their setup.sh. Net: 023+024 are 3090-Ampere-specific. M4 stack untested.

### 026 — gemma4-mm-video-per-frame-batching (2026-05-04, ~17 LOC, real bug fix)
**Closes Gemma 4 video OOM on Ampere AND R9700's bsz==1 assertion in one shot.** `gemma4_mm.py:get_video_feature` previously did:

```python
pv = pv.reshape(-1, pv.shape[-2], pv.shape[-1])  # 4D → 3D, batch=num_frames
pooled, pooler_mask = vt(pv, pp)                 # batched call
for hs, mask in zip(pooled, pooler_mask): ...    # iterate over frames
```

That batched `vt(pv, pp)` call materializes a `[num_frames × num_patches × 2 × position_embedding_size]` one_hot tensor inside `Gemma4VisionPatchEmbedder._position_embeddings` at line 419. For our 12-frame test video at `pooling_kernel_size=3` (so `num_patches=2520` per frame) and `position_embedding_size=10240`, that's `12 × 2520 × 2 × 10240 = 619M` elements — **~1.24 GB peak in bf16**. After LM weights + KV pool consume 22 GB at MEM=0.92, only ~220 MB GPU memory is free. Server OOMs.

R9700 hit a different symptom on the same model: their `attention/vision.py:254` asserts `flatten_batch is True, bsz must be 1` and bails before the OOM line.

Both failures share the same root: the vision tower's `forward(batch=num_frames, ...)` shape isn't supported end-to-end. **Fix processes frames one-at-a-time:**

```python
num_frames = pv.shape[0]
for f in range(num_frames):
    pooled_f, mask_f = vt(pv[f:f+1], pp[f:f+1])  # bsz=1
    real_tokens = pooled_f[0][mask_f[0]]
    all_embeds.append(self.embed_vision(...).squeeze(0))
```

Net behavior: same `all_embeds` output (since downstream code already iterates per-frame), `1/num_frames` peak memory, single-batch calls satisfy both the OOM constraint AND the bsz==1 assertion.

**Validated 2026-05-04** at TP=1 / 4K port 23355 via `validate_capabilities.py`: 4/4 PASS — `[PASS] video saw=['red'] response='(reasoning)the video is a static image of a red dot on a white background.'`. Note: the response shows the same Gemma 4 validator-passes-but-degraded pattern as vision — model says "static image" instead of "moving" — but the modality is now structurally unblocked. Calibration-side quality fix is the task #66 next step.

**Cross-team portability:** Should close R9700's bsz==1 assertion too. Pending their port. Unlike patches 023+024 which trip HSAIL on RDNA4, this patch only changes the call shape into existing kernels, so no new code path on either stack.

### 028 — gemma4-mm-per-expert-awq-loader (2026-05-08, ~28 LOC, dual-format support)

**Closes 21B-REAP rebuild + gemma-4-26B-AWQ HF mirror format-mismatch (both Suggested-next 2026-05-04).** Upstream `gemma4_mm.py` at v0.5.11 only supports HF transformers' fused-experts source format (`experts.gate_up_proj` [E, 2*I, H], one tensor per layer). llmcompressor-saved AWQ checkpoints — `mattbucci/gemma-4-26B-AWQ`, our local `gemma-4-21b-REAP-AWQ-thinking-vision-v2`, etc. — ship the per-expert format with one tensor per expert per projection per attribute (`experts.<i>.gate_proj.qweight` / `qzeros` / `scales` × N experts × 30 layers ≈ 35923 keys total).

R9700's `gemma4_causal.py` at line 920+ already handles BOTH formats via a two-tier mapping (per-expert via `FusedMoE.make_expert_params_mapping` + fused via the original hand-built mapping). Patch 028 is the symmetric port for the multimodal loader.

**Mechanism:**

```python
# Per-expert mapping for AWQ/GPTQ checkpoints saved by llmcompressor
per_expert_params_mapping = (
    FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=num_experts,
    )
    if num_experts > 0 else []
)

# Then in load_weights loop, check per-expert FIRST (before fused),
# routing each (expert_id, shard_id) tuple through FusedMoE's weight_loader
# which handles awq_marlin runtime conversion automatically.
```

After patch:
- llmcompressor per-expert AWQ source (`experts.0.gate_proj.qweight`) → loaded into `experts.w13_qweight` shard `"w1"` expert 0.
- HF transformers fused source (`experts.gate_up_proj` [E, 2I, H]) → unchanged path, still chunks gate/up internally.
- Stacked dense MLP path (`gate_proj` / `up_proj` for non-MoE layers) → unchanged.

**Verification:** `git apply --check` clean against fresh v0.5.11 worktree, `ast.parse()` clean on the post-apply file, structural checks on per_expert + fused + FusedMoE references all pass. **Runtime-verified 2026-05-07 (commit `b3654fc`)** — launched local Gemma 4 21B-REAP at TP=1 / 4K with patch 028 applied; SGLang Server fired up cleanly, no missing-weight warnings on the 30-layer × 128-expert MoE block (where pre-patch the loader silently failed all per-expert keys → uninit MoE → 0/3 PASS). Validator still returned 0/4 PASS but for a separate reason: the 21B-REAP-v2 checkpoint itself is calibration-degenerate (164 all-zero `*.scales`, see Known Issues). Loader path is correct; calibration recal is needed for that specific model. Mapping logic also covered by `scripts/test/test_gemma4_per_expert_mapping.py` (12/12 real R9700 HF mirror keys + 4/4 fused-source negative test).

**Cross-team portability:** R9700 already has this support natively in their gemma4_causal.py (their dual-format loader was the reference). No port needed for them. Patch is purely Ampere-side.

### 029 — qwen35-shared-expert-gate-ct-dequant (2026-05-07, ~52 LOC body, real bug fix)
**Closes the documented Qwen3.6-35B-A3B-AWQ-CT-on-NVIDIA bug; unlocks the calibration-clean CT variant for production.** Pre-patch repro: launch `mattbucci/Qwen3.6-35B-A3B-AWQ-CT` at TP=1 / 2K / quant=compressed-tensors → server boots via `CompressedTensorsWNA16MarlinMoEMethod` + 120 warnings of `Parameter model.layers.{0..39}.mlp.shared_expert_gate.weight_packed not found in params_dict` + 1/4 PASS validator with multilingual word-soup output (random-init `shared_expert_gate` for all 40 layers).

**Root cause:** `qwen2_moe.py:Qwen2MoeSparseMoeBlock.__init__` constructs `shared_expert_gate` as plain `torch.nn.Linear` on the GPU path with no `quant_config` plumbed in. The model's params_dict has only `shared_expert_gate.weight`, never `weight_packed/scale/shape`. CT-format checkpoints ship the quantized triplet (the calibration recipe didn't ignore the gate — and `(1, H)` is too narrow for group-quant to benefit anyway, but llmcompressor quantized it regardless) which has nowhere to land in the model's tensor tree. Native AWQ doesn't hit this because the AWQ export silently skips `shared_expert_gate` and exports BF16 `weight` — which the plain nn.Linear loads cleanly.

**Patch:** intercept the missing keys at the top of `qwen3_5.py:load_weights` (both `Qwen3_5ForConditionalGeneration` and `Qwen3_5MoeForConditionalGeneration` branches — Dense Conditional is no-op since it has no shared_expert_gate keys). After the `language_model` rename + before stacked/expert mapping loops, detect `shared_expert_gate.(weight_packed|weight_scale|weight_shape)` keys and buffer per-layer in `self._seg_buf`. Once all 3 land, dequant via `compressed_tensors.compressors.unpack_from_int32(packed, num_bits=4, shape=target_shape, packed_dim=1)` → int4 in [-8, 7] (signed) → cast to bfloat16 → multiply by `weight_scale.repeat_interleave(group_size, dim=-1)` → write to the layer's `.weight` param. Inferred `group_size = target_shape[1] // weight_scale.shape[1]` (typically 128 for the standard CT WNA16 layout).

**Verification:** TP=1 / 2K / quant=compressed-tensors → **0 missing-param warnings (was 120) + 4/4 PASS in 69.4s** (basic finish=stop, thinking 1007 tok reasoning_seen+answer_ok, vision content-aware (`'(reasoning)... it's a circle.'`), video content-aware (`'a red circle moves to the right.'`)). 102-line patch body (incl. 14-line comment block), applies clean to v0.5.11 + all 16 patches still apply cleanly in sequence per `git apply --check`.

**Production impact:** the calibration-clean `hf-mattbucci/Qwen3.6-35B-A3B-AWQ-CT` variant (0/31010 flagged scales per the 2026-05-07 audit) is now usable on Ampere via `MODEL=$MODELS_DIR/hf-mattbucci/Qwen3.6-35B-A3B-AWQ-CT QUANT=compressed-tensors ./scripts/launch.sh qwen36 --tp 1`. The native AWQ variant currently in the `qwen36` preset has 144 flagged rare-expert scales — switching to CT closes that quality gap, but the preset switch is left for a separate iteration that can also bench at TP=2 / 256K once the second 3090 returns.

**Cross-team portability — scope correction (2026-05-08, R9700 hardware test commit `e5b6175`):** R9700's `convert_moe_ct_to_awq.py` BF16 fallback covers the **AWQ-native serving path** on RDNA4, but R9700 ALSO supports CT-format direct serving via auto-detect in their launch.sh. Earlier "ROCm path unaffected" framing was over-stated — it holds for AWQ-native but R9700's CT path was never independently launched before today. R9700's actual launch test of `Qwen3.6-35B-A3B-AWQ-CT-thinking-vision` at TP=2 surfaced 3 regressions: (1) `moe_runner/triton.py:27 NameError is_hip undefined` — R9700-side patch hygiene from a prior regen auto-resolve; (2) `gemma4_causal.py:1074` `EntryClass` duplicate of `Gemma4ForConditionalGeneration` (collides with `gemma4_mm.py:120`) — R9700-side patch hygiene shape-equivalent to our retired patch 022; (3) MoE `_load_w2 RuntimeError narrow(start=4, length=4, size=4)` at TP=2 — likely an RDNA4-specific TP=2 sharding bug for CT-format MoE, NOT shared with patch 029's `(1, H)` shape (down_proj is per-expert, different code path). Patch 029 is **NVIDIA-side, validated only at TP=1 / 2K so far**; TP=2 remains untested on either stack until 2nd 3090 returns. Recipe-side ignore (`re:.*shared_expert_gate.*` in CT calibration) is still cleaner if R9700 recals — would let downstream consumers skip serving-side patches on either rig.

### 031 — qwen3_5-deltanet-awq-weight-loader (2026-05-09, ~25 LOC, v0.5.11-on-Ampere)
**Restores three v0.5.10 fixes that the v0.5.11 upstream sync silently dropped from `qwen3_5.py`.** Without them every AWQ-quantized Qwen3.5/3.6 DeltaNet build outputs literal `!!!!!` for every prompt — `validate_capabilities.py` 0/4 PASS, `(reasoning)` placeholder + exclamation-mark continuation. Affects `qwen35-tp1` (Qwen3.6-27B-AWQ), `qwen36-tp1` (Qwen3.6-35B-A3B-AWQ), `qwen35-moe` (Qwen3.5-28B-A3B-REAP-AWQ), `qwen3-vl-32b` (Qwen3-VL-32B-AWQ), and any `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration` AWQ ship.

Three gaps in `Qwen3_5GatedDeltaNet`:
1. **Constructor:** `in_proj_ba` was passed `quant_config=quant_config`. AWQ checkpoints ship `in_proj_a` / `in_proj_b` in full precision (only `.weight` keys, no `.qweight/.scales/.qzeros`); forcing AWQ on this layer makes the model expect packed AWQ params the checkpoint doesn't provide. Restored v0.5.10's `quant_config=None` (also gives hipBLASLt scaled_mm alignment for FP8 paths since output dim 48 isn't divisible by TP=2 × 16).
2. **`_bind_packed_weight_loaders`:** attr_name tuple was `(weight, weight_scale_inv, weight_scale, input_scale)`. Restored AWQ trio `(qweight, scales, qzeros)` so the packed-checkpoint loader binds for AWQ-quantized merged columns.
3. **`_get_split_sizes_for_param`:** normal-weight return path returned raw `output_sizes[idx]` without dividing by pack factor. Restored `PackedvLLMParameter` / `PackedColumnParameter` detection that divides by `packed_factor` when split_dim == packed_dim.

**Verification 2026-05-09 against `qwen35-tp1` (Qwen3.6-27B-AWQ) at TP=1 / 4K:** 4/4 PASS in 38.8s (basic 'paris', thinking 1511 tok finish=stop, vision 'a red circle with a black outline is centered on a white background', video 'a red circle moves across a white background'). Pre-patch: 0/4 with `'!!!!!!!!!!'` answer pattern + 96 `linear_attn.in_proj_*.weight not found in params_dict` warnings. Post-patch: 0 not-found warnings.

**Cross-team note for R9700:** their patch 002 (`002-qwen3-deltanet-awq-weight-loader.patch`, commits `eec67c0` + `d1dbc77` adapted) ports the same fix shape but only for `qwen3_next.py` — they explicitly skipped `qwen3_5.py` thinking it was "RDNA4-specific". It isn't — both stacks need this when on v0.5.11. R9700 is currently still on v0.5.10 source so they don't hit the bug yet; cross-team advisory in their README points at this patch.

### 034 — sampler-inf-detection (2026-05-13, ~13 LOC, R9700 backport)
**Extends `--enable-nan-detection` to catch `+/-Inf` logits parallel to the existing NaN branch in `Sampler._preprocess_logits`.** R9700 found this on Gemma 4 26B HSAIL bisects: `softmax(+Inf)` → NaN cascade, then `multinomial` returns an invalid index, then `gather` faults — same crash surface as raw NaN logits but caused by upstream Inf instead. The existing NaN check catches the downstream NaN AFTER softmax has already destroyed the signal about whether the model originally produced Inf or NaN, which made bisection ambiguous.

On Ampere CUDA the surface is different (no HSAIL) but the cascade is the same — Inf logits propagate to NaN through softmax and produce silent garbage tokens; the new branch makes the warning fire AT the offending sample. With `SGLANG_IS_IN_CI=1` the branch escalates to `ValueError("Detected errors during sampling! Inf in the logits.")` so CI runs surface the bug loudly.

Mechanic mirrors the NaN branch: replace +Inf with `+1e4`, -Inf with `-1e4` (signed-aware versions of the NaN→`-1e5` replacement). Source file (`python/sglang/srt/layers/sampler.py`) is identical between R9700 and 3090 at v0.5.11, so the patch is verbatim from R9700 commit ec1cf36.

### 035 — qwen3_5-causal-lm-entry-class (2026-05-19, 7 LOC, v0.5.11 + Qwen3.6 CT)
**Expands `qwen3_5.py`'s `EntryClass` registry to expose the two text-only CausalLM heads alongside the existing ConditionalGeneration ones.** SGLang discovers supported model architectures by walking each model file's module-level `EntryClass` list — anything not in `EntryClass` is treated as "no SGLang implementation" even when the class is defined right there in the file.

`qwen3_5.py` defines four heads (`Qwen3_5ForCausalLM` at line 961, `Qwen3_5MoeForCausalLM` at 1256, `Qwen3_5ForConditionalGeneration` at 1466, `Qwen3_5MoeForConditionalGeneration` at 1661), but `EntryClass` only listed the two `ConditionalGeneration` variants. Result: any Qwen3.5/3.6 checkpoint whose `config.json` declares `architectures: ["Qwen3_5MoeForCausalLM"]` or `["Qwen3_5ForCausalLM"]` fails to load:

    ValueError: Qwen3_5MoeForCausalLM has no SGlang implementation and
    the Transformers implementation is not compatible with SGLang.

This is the failure mode the `qwen36` preset hit during the 2026-05-17 bake-off (cycle exited rc=1 in 12 min). The preset points at `mattbucci/Qwen3.6-35B-A3B-AWQ-CT`, whose `config.json` declares `Qwen3_5MoeForCausalLM` (text-only). The native multimodal mirrors at `Qwen3.6-35B-A3B-AWQ-native-thinking-vision` still load because they declare `Qwen3_5MoeForConditionalGeneration`. Also affects `Qwen3.6-27B-AWQ-CT-balanced` (declares `Qwen3_5ForCausalLM`).

Fix is a straight registration update — `Qwen3_5MoeForCausalLM` inherits from `Qwen3_5ForCausalLM` and both classes already have complete `__init__` + `load_weights` + `forward` implementations. No new code, just expose them to the architecture registry. Cross-team note: R9700 should mirror this on their v0.5.11 source (their `qwen3_5.py` carries the same EntryClass list).

### 036 — qwen3_5-layer-types-hf-fallback (2026-05-19, 9 LOC, sister to 035)
**Defensive read of `layer_types` / `layers_block_type` in `Qwen3_5ForCausalLM.get_layer`.** After patch 035 unblocked the bare causal-LM load path, the next step in `make_layers` hit `AttributeError: 'Qwen3_5MoeTextConfig' object has no attribute 'layers_block_type'`.

There are two classes with the same name `Qwen3_5MoeTextConfig`:
- sglang's at `srt/configs/qwen3_5.py:119` — inherits `layers_block_type` as a `@property` from `Qwen3NextConfig` that returns sglang's internal values (`"attention"` / `"linear_attention"`).
- HF transformers' at `models/qwen3_5_moe/configuration_qwen3_5_moe.py` — has only `layer_types` with HF's values (`"full_attention"` / `"linear_attention"`).

Both declare `model_type = "qwen3_5_moe_text"`. sglang's qwen3_5 config module never calls `AutoConfig.register()` (only qwen3_asr / lfm2* do), so HF's class wins the AutoConfig lookup and the model receives the HF object. The multimodal path is fine because the top-level `Qwen3_5MoeConfig` (model_type `qwen3_5_moe`) declares `sub_configs = {"text_config": Qwen3_5MoeTextConfig}` — that sub_config constructor uses sglang's class explicitly.

Fix: in `get_layer`, prefer `layers_block_type` (sglang's path), fall back to `layer_types` (HF's path), translate HF's `"full_attention"` to sglang's `"attention"` so the lookup in `ALL_DECODER_LAYER_TYPES` matches either way. 9 lines, no new code outside this function.

R9700 cross-team note: their v0.5.11 source has the same shape; they should mirror this whenever they pick up patch 035.

### 030 — gemma4-mm-per-expert-awq-loader-v0510 (DELETED 2026-05-09)
**Deleted on env upgrade to v0.5.11.** This was the v0.5.10 backport of patch 028 — sister patch covering per-expert AWQ multimodal Gemma 4 keys at the v0.5.10 source line offsets (`gemma4_mm.py:802+` instead of v0.5.11's `gemma4_mm.py:714+`). Once the dev rig moved to v0.5.11 source, patch 028 applies natively and 030 became redundant. Historical context: it lived in `patches/` for two days (2026-05-08 → 2026-05-09) with `setup.sh`'s `git apply --check` silently routing to whichever sister patch matched the running source.

The original v0.5.10-source pad-token-only generation bug that 030 was added to debug ended up being unrelated to the loader gap — even after 030 fixed the per-expert loading, three different multimodal Gemma 4 builds still produced `<pad>` tokens. v0.5.11 source resolves both the loader gap (via 028) and the post-load generation issue.

### 025 — gemma4-vision-pooler-padding-fp32 (2026-05-03, ~13 LOC, code-correctness)
**Aligns `Gemma4VisionPooler` with HF reference; does NOT fix the user-visible vision degradation.** Two diffs vs `transformers/models/gemma4/modeling_gemma4.py:573-629`:

1. **Pre-pool `masked_fill` of padding patches.** HF's pooler does `hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)` before the avg-pool. SGLang skipped this step. Padding patches are masked in encoder *attention* (`attention_mask=~padding_positions` at `gemma4_vision.py:585`) but the patch_embedder + MLPs + residual stream still leave non-zero hidden states at padding positions (the position-embedding addition zeros position-side, but `_patch_projection(2*(pad_pixels-0.5))` produces non-zero patch-side). When `_avg_pool_by_positions` clamps `position_ids = -1` → `0` and routes those patches to bucket-0, that bucket gets diluted with padding garbage. With Gemma 4's `pooling_kernel_size=3` (9× compression to `max_soft_tokens=280` from up to 2520 patches) and a small image like the validator's 256×256 red-circle (~256 real patches → 28 buckets), the 2264 padding patches all go to bucket-0, dominating the average there. Fix: zero padding hidden_states before pool.

2. **FP32 accumulation in `_avg_pool_by_positions`.** HF does `output = weights.transpose(1, 2) @ hidden_states.float()` then casts back; SGLang did the matmul in BF16. For k²=9 BF16 sums per bucket at hidden=1152 the precision loss is small but observable. Fix: match HF's FP32 accumulation.

**Validation:** drafted + applied + tested with `mattbucci/gemma-4-26B-AWQ` at `gemma4` preset (port 23348, content prompt = "Describe this image in one short sentence" on a 256×256 red circle, `enable_thinking=False`). Pre-patch baseline: `'a collection of small, scattered red and black pixels against a white background'`. Post-patch: `'A red and white pixelated gradient transitions from the top right corner towards the bottom left.'`. Response shape changed slightly (now structurally directional + drops the "black" pixel claim) but **still not "a red circle"** — the pool diff is real but not load-bearing for this failure mode. Vision degradation root cause is elsewhere (suspect remaining: projector / image-token expansion or LM-side embedding manifold drift under text-heavy calibration). Patch retained because it's strictly correct vs upstream — applying it can only improve, never regress.

**Why ship anyway?** Code-correctness alignment with HF reference is itself useful: future investigators won't re-discover this pool-padding-contamination diff and waste time chasing it as "the fix". The Known Issue suspect list is now narrower as a result. R9700 untested for this patch — they may want to mirror it for the same alignment reason if their `Gemma4VisionPooler` has the same code shape.

---

## Open investigations

### Gemma 4 MoE `<pad>` collapse on 3090 — RESOLVED 2026-05-03 via patch 023
Months-long investigation into the `<pad>` garbage from Gemma 4 26B MoE (post-patches 020/021/022 boot was clean but decode emitted `<pad><pad>...`). Resolved by [patch 023](#023-gemma4-moe-mlp-no-quant-config-2026-05-03-13-loc): the dense MLP weights ship as plain BF16 in MoE-block layers but were being fed into AWQ qweight slots, producing NaN+inf at the very first matmul of layer 0.

**Investigation arc summary (full per-step in commit history):**
1. Triton-precision DISPROVED — `--attention-backend torch_native` produced the same `<pad>` output (commit cf182c5 era).
2. Logit-shaping / sampler DISPROVED — top-N candidates were all vocab-low special tokens (`<pad>`, `<unused1>`, `<unused0>`, `<mask>`), the canonical signature of lm_head seeing zero/uniform hidden states.
3. Tied-embed wiring DISPROVED — both 21B-REAP (gemma4_mm.py) and 26B (gemma4_causal.py) paths produce the same failure; embed_tokens tensor is healthy.
4. Embed-class hypothesis DISPROVED — Gemma 4 31B Dense uses the same `Gemma4TextScaledWordEmbedding` lm_head wiring and PASSES.
5. Real differentiator: MoE. 31B Dense (no MoE) PASS; 26B / 21B-REAP MoE FAIL.
6. **Forward-hook trace via `SGLANG_GEMMA4_TRACE=1`** (commit 38cd3ae): layer 0 routes correctly, MoE output is healthy abs_max=104; **NaN appears at layer 1's entry**.
7. Per-checkpoint probe in `Gemma4DecoderLayer.forward` (`SGLANG_GEMMA4_TRACE=1` extended): NaN first appears at `05_mlp_out` — the **dense MLP path**, not the MoE branch.
8. Inside `Gemma3MLP.forward` probe: `01_gate_up_proj` produces 15078 NaN + 3267 inf out of 84480 outputs from a clean BF16 input (abs_max=43.5).
9. Safetensors comparison: `mlp.gate_proj.weight` ships as **plain BF16** (no `weight_packed/scale/zero`), but the constructor was passing `quant_config` to `MergedColumnParallelLinear` — AWQ loader fed plain BF16 into qweight slot.

**Resolution:** patch 023 — `Gemma4DecoderLayer` now passes `quant_config=None` to `Gemma4MLP` when `enable_moe_block=True` (parallel dense+MoE Gemma 4). Dense-only Gemma 4 (e.g. 31B) still gets `quant_config` since its MLP is fully AWQ-packed. **Patch 024 (same day) unblocked the multimodal route** with the analogous fix on the mm towers (vision_tower / embed_vision / audio_tower / embed_audio), and the launch.sh `gemma4` / `gemma4-31b` presets now default to BF16 (FP16 NaN's the SigLIP vision tower in attention softmax). Combined with a config-arch flip to `Gemma4ForConditionalGeneration` for checkpoints stuck on `Gemma4ForCausalLM`, the triple-fix delivers **basic + thinking VERIFIED** on both 26B MoE and 31B Dense. **Vision passes the validator's keyword grep but the actual responses are degraded** ("scattered red pixels" vs "red circle") — separately tracked as an open Known Issue, not a patch-023/024 regression.

Validation timeline at TP=1 (basic + thinking-only — vision row is keyword-grep, see Known Issue):
- `gemma4-26b-moe-mlp-quant-fix-May03` — patch 023 only, basic + thinking PASS, vision still arch-blocked (Gemma4ForCausalLM route).
- `gemma4-26b-vision-bf16-fix-May03` — patches 023+024 + BF16 + arch flip, 2K ctx, vision tower loads cleanly.
- `gemma4-26b-preset-default-May03` — preset default after baking BF16 in, 2K ctx.
- `gemma4-26b-ctx16k-May03` — same combo at 16K ctx (preset default).
- `gemma4-31b-arch-flip-bf16-May03` — same triple-fix on Dense, 2K ctx.
- `gemma4-31b-ctx16k-May03` — Dense at 16K ctx; KV pool is tight (1947 tokens
  remaining at TP=1 / mem-fraction 0.92), so per-request input cap is ~1.9K.
- `gemma4-thinking-probe-May03` — `/tmp/gemma4_thinking_probe.py` with `skip_special_tokens=False`: structured `<\|channel>thought` markers + correct multi-step arithmetic + clean `finish=stop`. Confirmed thinking is genuinely working, not validator-keyword-match.

Trace artifacts (committed in `benchmarks/quality/gemma4-trace-{00,15}.json`) document the pre-fix NaN-onset pattern.

---

## Shipped model history

Narratives for models that are now working. Current-state entries live in the top-level `README.md`.

### Qwen3.5-28B-A3B-REAP AWQ thinking+vision — shipped 2026-05-02
Recal of `cerebras/Qwen3.5-28B-A3B-REAP` (from local BF16 base). Replaces the broken-thinking Apr-14 version at canonical [`mattbucci/Qwen3.5-28B-A3B-REAP-AWQ`](https://huggingface.co/mattbucci/Qwen3.5-28B-A3B-REAP-AWQ) (commit `2cf434c8`). Pipeline:
1. R9700's `balanced_thinking_vision` recipe (40/60 thinking/non-thinking) ported into `scripts/quantize/calibration_datasets.py`. 256 samples × 2048 tokens. Final mix: 30% am_thinking + 25% llava_instruct (vision) + 35% ultrachat (padded for thestack_code which is HF-gated) + 10% numina_math, 256/256 thinking-tagged.
2. **First recal attempt killed at +6 min** after monitor caught a `liuhaotian/LLaVA-Instruct-150K` `DatasetGenerationError` mid-load — the dataset ships multiple JSON files with diverging schemas, default `load_dataset()` concat fails. The script's pad-from-ultrachat fallback was masking it: vision-source = 0 samples, ultrachat ballooned to 60% — would have produced a vision-broken model after 18+h. Fix: `data_files` field on the `Mix` dataclass + pin `llava_instruct` to `data_files="llava_instruct_150k.json"` (canonical 158K-row file). Commit `489db4f`, ported to R9700 commit `054a10d`.
3. Restarted recal: 18.22h CPU GPTQ at ~32 min/layer steady-state; 92 GiB system swapped 49 GiB once BF16 base + Hessian buffers + intermediate activations exceeded RAM, slowing each step but not stopping progress. ETA was originally 10-13h per CLAUDE.md, revised to ~22h after watching the actual pace; finished at the upper bound.
4. CT save → CT→AWQ conversion via `convert_moe_ct_to_awq.py --group-size 128` (~2 min): 24850 quantized + 696 passthrough tensors, 17.1 GB output.
5. **Two manual fixups required post-conversion** (worth knowing for future recals of `Qwen3_5MoeForConditionalGeneration` BF16 bases):
   - `config.json` had `architectures: ["Qwen3_5MoeForCausalLM"]` from the recal script's text-only override, but the saved weight keys still had `model.language_model.*` prefix because `AutoModelForImageTextToText` loaded the BF16 base as multimodal. Reverted `architectures` to `Qwen3_5MoeForConditionalGeneration` + `model_type` to `qwen3_5_moe` to match the weight layout.
   - `convert_moe_ct_to_awq.py` doesn't carry `preprocessor_config.json` or `video_preprocessor_config.json` forward; copied both from the BF16 base directory.
6. Validator: 3/3 PASS at TP=1 / 8K via `qwen35-moe` preset — basic finish=stop, thinking 3145 tok finish=stop (was BROKEN — emitted "The user is asking..." instead of `<think>...</think>` on the prior REAP-AWQ), vision saw red+circle+round.
7. **Surprise vision-tensor finding:** Cerebras's REAP variant retained 333 visual tensors in the BF16 base (vs `atbender/Qwen3.6-VL-REAP-26B-A3B` BF16 which fully strips its vision tower). All 333 made it through GPTQ→CT→AWQ as passthrough — vision is functional end-to-end. Vision-tower preservation is decided at the BF16-REAP-base layer, before any AWQ calibration. R9700 captured the same finding cross-team in their commit `2484aab`.
8. HF upload via plain `hf upload` stalled at 99% / 16.9 GB for 5h with 0 bytes/s before kill+restart resumed via xet content-store dedup in ~3 min. Confirms R9700's xet-stall pattern (their Coder-Next-REAM saw 12h stalls).

`qwen35-moe` preset's MODEL default repointed to the recal output. Validator's `TEXT_ONLY_MODELS` no longer lists `qwen35-moe` (now multimodal-capable).

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

### Qwen3.5-28B REAP thinking recalibration — cancelled 2026-04-19 (v3 SHIPPED 2026-05-02)
v1 died at layer 13/41 on harness restart (lost 7h 45min — this is where the `setsid` detach rule came from). v2 killed at layer 1/41 because R9700 shipped a working thinking-preserving Qwen3.5-27B-AWQ v2 (`mattbucci/Qwen3.5-27B-AWQ-4bit-calibrated`, basic FAIL→PASS) and Qwen3.6-35B-A3B GPTQ passes the thinking validator on both rigs — the regression this was fixing was superseded by switching long-ctx thinking workloads to `qwen36` or `qwen3-ream`. **Re-opened and shipped 2026-05-02 as v3** with R9700's `balanced_thinking_vision` recipe — see "Qwen3.5-28B-A3B-REAP AWQ thinking+vision" entry above for the v3 pipeline + 3/3 PASS validation.

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

### R9700 cross-team thread — Gemma 4 confirmation + harness port + 35B-v2 config-class fix (2026-04-25/26)

- **R9700 confirmed Gemma 4 26B AWQ serves CLEAN on RDNA4** — ran `validate_capabilities.py` against `mattbucci/gemma-4-26B-AWQ`: basic PASS, thinking 441 tok terminated cleanly, vision correctly described "red round" blocks. Same weights, same chat template, opposite outcome from 3090 → the gap is our `clippable_linear.py` no-op alias. R9700 doesn't ship a shim at all; they run SGLang's stock Gemma 4 model code. (Their video probe hit a separate `flatten_batch is True, bsz must be 1` SGLang assertion — unrelated bsz-handling bug in their video input path; doesn't affect the text/thinking/vision conclusion.)
- **R9700 ported the SWE-bench harness** — mirrored our `evals/swebench/{run_rollouts,score_local}.py` verbatim (their commit `ba3d457`); platform-agnostic since it talks opencode → local SGLang. Coder-REAP-25B is the natural overlap model for the first head-to-head once 3090 publishes a scored result.
- **R9700 correction on Qwen3.6-35B v2 recalibration "loader broken" note** — it was actually a config-class downgrade, not a loader bug. llmcompressor saves multimodal MoE checkpoints with `architectures=Qwen3_5MoeForCausalLM` (text-only) and strips `text_config` + `vision_config` + `{vision,image,video}_token_id`. The SGLang loader's `language_model.` strip is calibrated for the multimodal wrapper class — text-only class registers params under `layers.X...` (no `model.` prefix) so the post-strip lookup misses. Fix is a one-shot config.json patch (no source changes): rewrite `architectures` to `Qwen3_5MoeForConditionalGeneration` + `model_type` to `qwen3_5_moe` + copy missing config fields from a sibling v1. After patch v2 boots cleanly. So if anyone recalibrates a 35B with a corrected `re:.*shared_expert\..*` ignore, ship without waiting on a loader change. R9700 v2 then hits HSAIL 0x1016 on first inference (same exception class as their open #18, Coder-Next + Gemma4-31B long-decode) — likely the BF16 shared_expert path tripping a RDNA4-Triton miscompile. v1 stays their production until #18 lands or v2 is recalibrated with shared_expert quantized.

---

## Cross-team findings (3090 ⟷ R9700)

The sister RDNA4 project runs the same SGLang v0.5.11 stack (both rebased 2026-05-07). Findings that produced patches or changed how we ship are here; day-to-day sync happens in the two READMEs.

- **BF16 attention precision** affects every new architecture (RDNA4, Blackwell SM12.x). Fix: FP32 accumulation in the online softmax (patch 011).
- **AWQ calibration silently breaks thinking and vision.** Quants calibrated on plain text (Open-Platypus, WikiText2, c4) lose `<think>` stop-token behavior and vision-language alignment. Rule: every new quantized model must validate (a) an image+text roundtrip and (b) a thinking-tagged generation that cleanly terminates, before launch. `scripts/eval/validate_chat_template.py` (static) + R9700's `validate_capabilities.py` (live) are the pre-flight gates.
- **Recommended calibration datasets** (reasoning + vision preserving): `a-m-team/AM-Thinking-v1-Distilled`, `glaiveai/reasoning-v1-20m`, `LLaVA-Instruct-150K`, `AI-MO/NuminaMath-CoT` (+9.81% GPTQ accuracy vs WikiText2 in R9700 measurements). Recipe builder: `scripts/quantize/calibration_datasets.py` (`thinking_text`, `thinking_vision`, `code_vision`, `code_thinking`).
- **Chat template is a silent bug magnet.** Devstral community AWQ's template emits BOS → `<unk>`; Gemma 4 community weights ship with no template at all; Qwen3 family needs `temperature ≥ 0.3` to avoid greedy-decode repetition loops. Always render the template with and without `enable_thinking`, and verify `chat_template is not None`.
- **AutoRound > GPTQ > AWQ for INT4 quality** — Intel AutoRound (arXiv 2309.05516) uses SignSGD for 200 iterations to jointly optimize rounding offsets and clipping ranges. Can export to both GPTQ and AWQ formats. [RedHatAI reports 99.4%+ quality](https://huggingface.co/RedHatAI/gemma-3-27b-it-quantized.w4a16) on CUDA.
- **DeltaNet layers must stay BF16.** INT4 noise accumulates through the recurrent state `S(t) = g*S(t-1) + delta` and destroys quality. Architectural limit.
- **DeltaNet replication mandatory for TP=2.** Qwen3.5-27B: TP RowParallelLinear splits matmul `W_0@x_0 + W_1@x_1` which differs from `W@x` by ~1 ULP in FP16; the recurrent state compounds it across layers. Fix: replicate all DeltaNet + MLP layers (`tp_size=1`), SSM state `tp_world_size=1`. Costs 19 GB/GPU weights replicated, which is why Qwen3.5-27B is context-limited to 32K.
- **Community AWQ fails for DeltaNet** on both rigs. Self-calibrate with GPTQ + CT→AWQ.
- **`save_pretrained` OOMs on 32B+ models with default `max_shard_size="5GB"`** (R9700 finding 2026-05-03, commit `e28c43b`). VL-32B died at "Writing model shards: 0%" with exit=137 after 27.5h of GPTQ — 62 GB RAM + 68 GB swap insufficient for the 5GB shard buffer alloc. Two-layer defense: (1) `max_shard_size="2GB"` on final save forces smaller shard buffers; (2) per-layer checkpoint hook on `LifecycleCallbacks.sequential_epoch_end` writes a snapshot every 16 subgraphs (~4 across a 65-subgraph run), keeping only the last 2 to bound disk at ~34 GB. Worst case if final save OOMs: the previous snapshot at `.checkpoints/subgraph_NNN/` survives and ships at lower quality. Apply this pattern to any future 30B+ self-calibration on the 3090; our `quantize_qwen3vl_30b_moe_thinking_vision.py` is currently not active (Qwen3-VL-30B loader closed in README) but inherit the pattern when we re-open or take on a new 30B+ recal.

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
