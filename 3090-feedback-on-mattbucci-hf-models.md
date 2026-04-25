# 3090 Team Feedback on `huggingface.co/mattbucci` Models (2026-04-25)

Tested all 10 models published at https://huggingface.co/mattbucci on 2x RTX 3090 (Ampere, sm_86) with SGLang v0.5.10 + our 19 patches. Downloaded actual HF copies via `hf download` (not relying on local equivalents). Total 158 GB.

---

## Per-model config audit

```
                                       arch                             quant            ignore
Qwen3.5-27B-AWQ-4bit-calibrated        Qwen3_5ForConditionalGeneration  awq              0
Devstral-24B-AWQ-4bit-calibrated       Mistral3ForConditionalGeneration awq              0
Qwen3-Coder-30B-A3B-AWQ                Qwen3MoeForCausalLM              compressed-tensors  1
Qwen3-Coder-REAP-25B-A3B-AWQ           Qwen3MoeForCausalLM              awq              0
Qwen3.6-27B-AWQ                        Qwen3_5ForConditionalGeneration  awq              0
Qwen3.6-27B-AWQ-CT                     Qwen3_5ForConditionalGeneration  compressed-tensors  97
Qwen3.6-35B-A3B-AWQ                    Qwen3_5MoeForConditionalGeneration  awq           0
Qwen3.6-35B-A3B-AWQ-CT                 Qwen3_5MoeForConditionalGeneration  compressed-tensors  101
gemma-4-26B-AWQ                        Gemma4ForConditionalGeneration   awq              0
gemma-4-31B-it-AutoRound-AWQ           Gemma4ForCausalLM                awq              0
```

**Pattern:** every native AWQ upload (post-CT→AWQ conversion) ships `quantization_config.ignore=[]`, while the CT pre-conversion uploads retain their full ignore list (97-101 entries). Verified by safetensors inspection that BF16 fallback is correctly preserved in the WEIGHTS (e.g. `model.language_model.layers.0.router.proj.weight: bf16, (128, 2816)` in gemma-4-26B-AWQ; vision-tower weights ship in BF16 in 35B native AWQ). So the AWQ conversion is doing the right thing at the weight level, just not propagating the `ignore=[...]` field into the saved `quantization_config`. Cosmetic for SGLang load (the awq_marlin loader doesn't look at ignore for runtime), but it makes downstream sanity audits ambiguous — anyone running our `len(ignore) < 50` heuristic on the native uploads would incorrectly flag them as broken. Worth one-liner fix in `convert_moe_ct_to_awq.py` to copy the ignore list through.

---

## Per-model results on 3090

### ✅ Qwen3.6-35B-A3B-AWQ — flagship win
**Validator 4/4 PASS** (basic, thinking 687 tok terminated, vision saw=[red,circle,round]).

```
ctx     30: 33.4 tok/s
ctx  40 K: 21.8 tok/s
ctx 160 K:  5.8 tok/s
ctx 250 K:  2.6 tok/s
```

vs R9700: 21.6 short / 20.6 @131K. **+55% short-ctx win** (Marlin INT4 GEMM > ROCm Triton AWQ at sub-32K) but **3.5x long-ctx loss @131K** (Ampere flashinfer's hybrid-DeltaNet+full-attn path doesn't hold up past ~80K). A/B'd `--attention-backend triton` (worse: 3.4 @131K) and matched all R9700 tunables (CHUNKED/DECODE_STEPS/MAMBA_CACHE — no improvement). Genuine kernel-stack asymmetry.

**Required for NVIDIA Ampere:** the `-CT` upload triggers infinite repetition under `quantization=compressed-tensors` (our SGLang CT loader doesn't do BF16 fallback for `(1, H)` `shared_expert_gate`). The **native AWQ upload Just Works**. Suggest model card promote it to "**required for NVIDIA**" rather than "optional speedup."

Required SGLang fixes for Python 3.13 + transformers ≥5.5 (3090 patch 019 — already cross-posted):
1. dict-wrap `text_config`/`vision_config` at model init
2. explicit `def __init__(self, **kwargs): super().__init__(**kwargs)` on `Qwen3_5MoeVisionConfig/TextConfig/Config` (transformers metaclass auto-decorates as dataclass otherwise, drops parent attribute defaults)
3. drop `model_type` from kwargs when re-wrapping subclass

### ✅ Qwen3.6-27B-AWQ — solid Dense VL
Both `-AWQ` and `-AWQ-CT` work identically. Validator 4/4. ~30 tok/s short, ~21 tok/s @131K. Vision caption quality is cleaner than our self-cal v3 (`thinking_vision_video` recipe vs our `thinking_vision`); reasoning is verbose (your model hits 4096-tok ceiling on simple probes vs our 259 tok). Recipe trade-off — both are valid takes.

### ⚠️ Qwen3.6-35B-A3B-AWQ-CT — broken on NVIDIA, native AWQ fixes it
Loops `"15*17 = 15*17 = ..."` in thinking mode under `quantization=compressed-tensors` on 3090. Fixed by re-running `convert_moe_ct_to_awq.py` and serving as awq_marlin. **Root cause:** `shared_expert_gate.weight_packed` ships shape `(1, H)` — output dim 1 isn't AWQ-packable (needs divisibility by 8). Your conversion script handles this with BF16 fallback. SGLang's NVIDIA CT loader doesn't, so the 1×H gate gets read as garbage INT4 → routing of every token's shared-expert path breaks → cumulative drift through 48 thinking layers.

### ❌ gemma-4-26B-AWQ — broken on 3090, calibration defect
**Validator 1/4** (basic FAIL with `1-1-1-1-1-...` repetition; thinking timeout; vision crashes server). Architecture is `Gemma4ForConditionalGeneration` (multimodal!) — SGLang loads via `gemma4_mm.py` once our patch 020 shim for `clippable_linear` is applied. Server boots, weights load. But generation is broken.

Weight inspection shows router stays BF16 (`router.proj.weight: bf16`) — so the Gemma4 router-with-per-expert-scale design is preserved correctly. The garbage output suggests calibration damaged the experts, possibly (a) router was BF16 but per-expert weights were under-calibrated due to expert imbalance, or (b) the AutoRound-derived clip bounds are missing (our clippable_linear shim is a no-op clip — we alias to plain Linear classes). If Gemma 4 26B was originally calibrated WITH ClippableLinear's actual clip bounds, removing the clip at inference time could destabilize activations through the 60-layer SWA. **Worth investigating on R9700:** does this model serve correctly there? If yes, the ClippableLinear-vs-no-op gap is doing real work and our shim is incomplete.

### ⚠️ gemma-4-31B-it-AutoRound-AWQ — works but vision hallucinates
Validator 3/4 (basic+thinking PASS, vision generates wrong content). Loads as `Gemma4ForCausalLM` (text-only registration despite Gemma 4 31B being a vision-capable Dense model). Throughput 28 tok/s @ 1K. Note `quantization_config.quant_method=awq` but config metadata says compressed-tensors elsewhere — we override `QUANT=compressed-tensors` at launch.

**Suggestion:** bump arch to `Gemma4ForConditionalGeneration` so the multimodal loader path is exercised; that should fix the "generates content but wrong" by routing image tokens through the actual vision tower instead of dropping them. Today the model ALSO accepts image tokens silently (text-only fallback path) and emits a hallucinated caption.

### ✅ Devstral-24B-AWQ-4bit-calibrated — long-context star
56 tok/s @ 217K. No issues. Vision was stripped during calibration (community pattern); we run `--skip-vision` on validator. Best Dense long-ctx model in our lineup. Architecture is `Mistral3ForConditionalGeneration` (multimodal!) but weights ship without vision tower → SGLang serves text-only via the same loader (no warnings, no errors, just no vision). If vision-preserving recal is ever in scope, would be a strict win.

### ✅ Qwen3-Coder-30B-A3B-AWQ — peak short-ctx
193 tok/s @ 16K. No issues. Default coding-agent for our rig. `quantization_config.quant_method=compressed-tensors` despite the directory name (we override `QUANT=compressed-tensors`).

### ⚠️ Qwen3-Coder-REAP-25B-A3B-AWQ — server stuck at CUDA graph capture
Server boots (load weight end at 19s, KV cache allocates), but `Capture piecewise CUDA graph` hangs >10 min — probably an interaction between piecewise CUDA graph and the long 131K context preset on our preset. Did NOT validate — failed pre-validation. Adding `--disable-piecewise-cuda-graph` or shorter context should let it through; not retesting tonight because GPU is needed for other work.

### ✅ Qwen3.5-27B-AWQ-4bit-calibrated — works, ctx-limited
Validator basic+thinking PASS. 32K ctx ceiling on 3090 (DeltaNet replication forces 19 GB/GPU weights, leaving 2.2 GB for KV).

---

## Cross-cutting suggestions

1. **CT→AWQ conversion script: copy `quantization_config.ignore` through.** Currently the native AWQ uploads ship `ignore=[]` even though the WEIGHTS preserve BF16 for routers/vision/etc. Cosmetic for runtime but breaks downstream audits. One-line fix in `convert_moe_ct_to_awq.py`: when constructing the new `quantization_config`, copy the `ignore` field from the source CT config.

2. **CT vs AWQ default for NVIDIA users.** The CT format trips on `(1, H)` `shared_expert_gate` in SGLang's NVIDIA CT loader (your conversion script's BF16-fallback isn't replicated server-side). Promote the native AWQ uploads as "Recommended for NVIDIA Ampere/Hopper" on the model cards; keep CT for ROCm where the fallback isn't needed.

3. **`gemma-4-26B-AWQ` produces garbage on 3090 while loading via the multimodal class.** Need to know if it serves correctly on R9700 — that would tell us whether our clippable_linear shim (no-op alias of Parallel Linear classes) is missing actual clip behavior the calibration relies on. If it's broken on both rigs, the calibration itself needs revisiting. If it's fine on R9700, port your real `ClippableLinear` op (with clip bounds) to your published patch 013 shim so the no-op aliases stop being needed downstream.

4. **`gemma-4-31B-it-AutoRound-AWQ` should register as `Gemma4ForConditionalGeneration`.** Today it's `Gemma4ForCausalLM` so the vision tower never engages — image tokens silently fall through to text-only generation. Quick metadata fix; would unblock vision evaluation.

5. **Audit hook at publish time.** Auto-flag MoE/vision uploads with `len(ignore) < 50` AND `bf16_keys < 50` (both = no preservation at all). Detects the empty-recipe-typo case (ours had a 21B-REAP-AWQ shipped with `ignore=[]` AND no BF16 keys at all — a real bug, not the cosmetic one above).

6. **`hf upload` over `hf upload-large-folder` for ≤25 GB repos.** You already noted this in your README. Confirmed reproducible on our side.

7. **Long-context decode-curve documentation.** The Ampere-vs-ROCm asymmetry on Qwen3.6-35B-A3B-AWQ (33 tok/s short-ctx win + 3.5x long-ctx loss for the same weights) would be useful as a model-card snippet so users know what kernel stack their rig prefers.

---

## Tested and confirmed working on 3090 (recommendable to NVIDIA users)
- Qwen3.6-35B-A3B-AWQ (native) — best multimodal MoE in lineup, 4/4
- Qwen3.6-27B-AWQ (both native and CT) — 4/4
- Qwen3-Coder-30B-A3B-AWQ — 193 tok/s peak short-ctx
- Devstral-24B-AWQ-4bit-calibrated — 56 tok/s @ 217K, long-ctx star
- Qwen3.5-27B-AWQ-4bit-calibrated — basic+thinking work, ctx-limited

## Broken on 3090 / needs work
- gemma-4-26B-AWQ — calibration defect (or our shim incomplete)
- gemma-4-31B-it-AutoRound-AWQ — vision hallucinates (arch should be ConditionalGeneration)
- Qwen3.6-35B-A3B-AWQ-CT — needs native AWQ conversion for NVIDIA (use `-AWQ` instead)
- Qwen3-Coder-REAP-25B-A3B-AWQ — server stuck at CUDA graph capture (needs preset tuning)
