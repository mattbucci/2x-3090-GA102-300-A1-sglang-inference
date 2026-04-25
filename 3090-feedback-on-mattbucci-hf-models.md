# 3090 Team Feedback on `huggingface.co/mattbucci` Models (2026-04-25)

Comprehensive review of all 10 models published at https://huggingface.co/mattbucci, tested on 2x RTX 3090 (Ampere, sm_86) with SGLang v0.5.10 + our 19 patches. R9700 calibration pipeline + 3090 serving stack.

---

## Summary table

| Model | Status on 3090 | Validator (basic/thinking/vision) | Notes |
|---|---|---|---|
| `Qwen3.5-27B-AWQ-4bit-calibrated` | ✅ Working | basic ✓ thinking ✓ (no vision) | 32K ctx ceiling on 3090; prefer `qwen3-ream` for long ctx |
| `Devstral-24B-AWQ-4bit-calibrated` | ✅ Working | basic ✓ thinking N/A vision N/A | 217K @ 56 tok/s — long-ctx star; vision-stripped community AWQ |
| `Qwen3-Coder-30B-A3B-AWQ` | ✅ Working | basic ✓ thinking N/A | 193 tok/s @ 16K — peak throughput on rig |
| `Qwen3-Coder-REAP-25B-A3B-AWQ` | ✅ Working (W4A16 base) | basic ✓ | 131K @ 46 tok/s; haven't tested AWQ-repacked variant directly |
| `Qwen3.6-27B-AWQ` (native) | ✅ Working 4/4 | basic ✓ thinking ✓ vision ✓ | 30 tok/s short → 21 @131K; cross-validated against our self-cal v3 |
| `Qwen3.6-27B-AWQ-CT` | ✅ Working | basic ✓ thinking ✓ vision ✓ | CT pre-conversion of the above; identical perf within noise |
| `Qwen3.6-35B-A3B-AWQ` (native) | ✅ **SHIPPED 4/4** | basic ✓ thinking ✓ vision ✓ | 33 tok/s short → 2.6 @250K. **+55% over R9700 short-ctx** (Marlin vs Triton) but steeper long-ctx drop |
| `Qwen3.6-35B-A3B-AWQ-CT` | ⚠️ Quality issue (CT path on NVIDIA) | basic ✓ thinking loops | CT direct-load triggers infinite repetition under awq_marlin; native AWQ conversion fixes it. See "Issue 1" |
| `gemma-4-26B-AWQ` | ❌ Broken | basic FAIL (`<pad>` output) | MoE expert routing damaged; calibrated as text-only (`Gemma4ForCausalLM`); needs re-calibration |
| `gemma-4-31B-it-AutoRound-AWQ` | ⚠️ Partial | basic ✓ thinking ✓ vision hallucinates | Works via `--attention-backend torch_native`; vision plumbing OK but content wrong |

---

## Detailed findings

### ✅ Qwen3.6-35B-A3B-AWQ (native) — flagship win
HF: https://huggingface.co/mattbucci/Qwen3.6-35B-A3B-AWQ-thinking-vision

Tested via the converted-locally version (we ran R9700's `convert_moe_ct_to_awq.py` ourselves on the CT upload — same weights). Results on 2x RTX 3090:

```
Validator 4/4 PASS
  basic   = "(reasoning)paris"
  thinking = 687 tok cleanly terminated, answer_ok
  vision   = saw=[red, circle, round] on solid-red-circle probe
  video    = skipped (imageio missing in env, not the model's fault)

Bench (flashinfer attention, bf16 KV, TP=2, --enable-multimodal):
  ctx     30: 33.4 tok/s
  ctx  40 K: 21.8 tok/s
  ctx 160 K:  5.8 tok/s
  ctx 250 K:  2.6 tok/s
```

vs R9700 ROCm: 21.6 short / 20.6 @131K (flat curve). On 3090 we win short-ctx by ~55% (Marlin INT4 GEMM is faster than ROCm Triton AWQ for sub-32K) but we **lose** long-ctx by ~3.5x at 131K (5.8 vs 20.6). Ampere flashinfer's hybrid-DeltaNet + full-attn path doesn't hold up past ~80K — verified by A/B'ing against `--attention-backend triton` (got worse: 3.4 @131K) and matching all R9700 tunables (CHUNKED/DECODE_STEPS/MAMBA_CACHE — no improvement). The decode kernel path is genuinely a kernel-stack difference, not configuration.

**Suggestion to R9700:** users on NVIDIA Ampere should expect short-ctx wins but worse long-ctx than your numbers. Worth a one-line note in the model card explaining the kernel-asymmetry. Hopper/Blackwell may close the gap (TMA + larger FlashInfer windows).

---

### ✅ Qwen3.6-27B-AWQ — solid Dense VL, 4/4 cross-validated
Both `Qwen3.6-27B-AWQ` and `Qwen3.6-27B-AWQ-CT` work identically on 3090. We cross-validated R9700's published weights against our own self-cal v3 (`thinking_vision` recipe, ignore=in_proj_a$/in_proj_b$ only):

```
                         ours v3      R9700 native AWQ
  ctx 1K  tok/s           30.4         30.1
  ctx 8K  tok/s           30.1         30.8
  ctx 32K tok/s           29.7         29.8
  ctx 131K tok/s          21.1         21.0
```

Throughput identical within noise. **Quality difference**: R9700's vision caption is cleaner (`"a red circle on a white background"` vs ours `"the flag of japan"`) — likely because R9700 used `thinking_vision_video` recipe while we used `thinking_vision`. Conversely, our model terminates thinking more concisely (259 tok vs R9700's 4096-tok ceiling on the same probe) — recipe trade-off.

**Suggestion to R9700:** the `thinking_vision_video` recipe correctly grounds vision but increases reasoning verbosity. If you want to ship a thinking-balanced variant later, consider mixing in fewer video samples or dialing up math/code (more "stop generating" patterns).

---

### ⚠️ Qwen3.6-35B-A3B-AWQ-CT — CT serve broken on NVIDIA, native AWQ fixes it

The CT-format upload (`mattbucci/Qwen3.6-35B-A3B-AWQ-CT`) loads as `Qwen3_5MoeForConditionalGeneration` and validator passes 4/4 on R9700, but on 3090 with `quantization=compressed-tensors` it produces `"15*17 = 15*17 = 15*17 = ..."` infinite loops in thinking mode. After running R9700's `convert_moe_ct_to_awq.py` locally and serving with `quantization=awq_marlin` it works perfectly.

**Root cause** (confirmed by safetensors inspection): `shared_expert_gate.weight_packed` ships with shape `(1, H)` — output dim 1 isn't AWQ-packable (needs divisibility by 8). Your conversion script handles this with a BF16 fallback. SGLang's NVIDIA CT loader doesn't, so the 1×H gate gets read as garbage INT4 → routing of every token's shared-expert path breaks → cumulative drift through 48 thinking layers.

**Suggestion to R9700:** the model card already mentions running the converter; consider promoting it to "**required for NVIDIA**" rather than "optional speedup." Or upload the converted version under the same name as default and CT-as-`-CT` suffix for ROCm folks who want to pre-convert themselves.

We also hit three SGLang-side bugs while loading the 35B class on Python 3.13 + transformers ≥5.5, fixed in our **patch 019**:
1. `vision_config` / `text_config` come back as raw dicts from llmcompressor's CT save → wrap to proper sub-config classes at model init.
2. transformers' metaclass auto-decorates `PretrainedConfig` subclasses without `__init__` as `@dataclass`, replacing inherited init and dropping ALL parent attribute defaults (`norm_topk_prob=True`, `num_experts`, etc.). Reproduced standalone: `Qwen3_5MoeTextConfig(**dict)` → `hasattr(tc, 'norm_topk_prob') == False`. Fix: add `def __init__(self, **kwargs): super().__init__(**kwargs)` to all `Qwen3_5Moe{Vision,Text,}Config` subclasses.
3. `model_type` from `to_dict()` can be `""` and overwrites the class attr when re-wrapping → drop it before passing kwargs.

This bites anyone running Python 3.13 against any of your published Qwen3.6 / Qwen3.5 native-AWQ checkpoints. Already cross-posted to your README under "evergreen lessons."

---

### ✅ Devstral-24B-AWQ-4bit-calibrated — long-context star
Long-context bench (TP=2, default Devstral preset → 217K with `MEM=0.97 CHUNKED=2048 --disable-cuda-graph --disable-overlap-schedule --disable-radix-cache`):

```
  ctx 1K   : 88.7 tok/s    TPOT 11.3 ms
  ctx 16K  : 82.6 tok/s
  ctx 64K  : 68.5 tok/s
  ctx 131K : 55.8 tok/s
  ctx 200K : 55.9 tok/s    TPOT plateaus — memory-bandwidth bound
```

Hits 56 tok/s flat from 131K through 200K. This is the best Dense-class long-ctx model in our lineup. **No issues**, ships cleanly. **Vision was stripped** during calibration (community pattern). Devstral is image-only natively but neither your nor our calibration preserves it; we run `--skip-vision` on the validator.

**Suggestion to R9700:** if a vision-preserving Devstral re-cal is ever in scope, would be a strict win — `Pixtral-Devstral-Mistral` is image-input-only not interleaved-multimodal so the vision tower would just need to stay BF16 (small footprint). Low priority since most agentic workloads using Devstral are coding/text.

---

### ✅ Qwen3-Coder-30B-A3B-AWQ — peak short-ctx throughput

```
  ctx 16K : 193 tok/s    TPOT 5 ms — best in lineup
```

Default coder-30b preset hits 193 tok/s @ 16K context. No issues. We use this as our coding-agent default.

---

### ✅ Qwen3-Coder-REAP-25B-A3B-AWQ — REAP variant, mid throughput

Tested via the W4A16 base (`Qwen3-Coder-REAP-25B-A3B-W4A16`) since we don't have the AWQ-repacked variant locally:

```
  ctx 131K : 46 tok/s    TPOT 22 ms
```

This is the Cerebras REAP prune (103 experts of original 128, ~20% drop). Throughput OK; quality should be 70%+ MMLU per Cerebras paper. **Quality not independently re-tested on this rig** — would benefit from a head-to-head against the unpruned Coder-30B on a real coding eval (HumanEval-pro / LiveCodeBench).

**Suggestion to R9700:** if you have bench numbers for the AWQ vs W4A16 packing of REAP-25B, would be useful in the model card — we can't tell from the HF page whether the AWQ has the same accuracy as the W4A16 source.

---

### ⚠️ gemma-4-31B-it-AutoRound-AWQ — works but vision hallucinates

```
Validator 3/4
  basic   ✓
  thinking ✓
  vision  generates content but wrong (response: "the image shows a single cuneiform character")
  video   skipped

Launch flags: QUANT=compressed-tensors EXTRA_ARGS="--attention-backend torch_native --disable-cuda-graph --disable-piecewise-cuda-graph"
Throughput: 28 tok/s @ 1K, TPOT 35 ms
```

Vision plumbing works (image tokens reach the model; it generates a caption) — just the caption is wrong content. This is different from the 21B-REAP "i cannot see the image" failure mode (text-only loader fallback). Likely calibration-side: AutoRound on Gemma 4's vision tower may have lost alignment.

Note model config says `compressed-tensors` despite the `AWQ` directory name — we have to override `QUANT=compressed-tensors` at launch.

**Suggestion to R9700:** consider re-tagging the HF repo from `*AutoRound-AWQ` to `*AutoRound-CT` to match the actual checkpoint format; the AWQ name causes downstream tooling to default-launch with awq_marlin which doesn't load this checkpoint correctly. (Or convert to actual AWQ via `convert_ct_to_awq.py` and retag.)

---

### ❌ gemma-4-26B-AWQ — broken on 3090, calibration defect

```
Validator 1/4 (only video skipped passes)
  basic   FAIL: emits literal `<pad>` tokens
  Direct probe: { content: null, reasoning: '<pad><pad><pad>...30x', finish: length }
```

This is the worst result in the lineup. We trace it to TWO independent issues:

1. **Architecture is `Gemma4ForCausalLM` (text-only)** — no vision tower engaged regardless of SGLang's loader.
2. **MoE expert routing is broken** — text-only path emits `<pad>` because the calibration damaged expert selection. Likely the `mlp.gate_proj` per-layer was INT4-quantized when it shouldn't have been (the 26B model card has `mlp.gate_proj` and `router.proj` in ignore list per our audit — but the saved weights still emit pad).

We (3090) have a parallel issue: our locally calibrated `gemma-4-21b-REAP-AWQ-thinking-vision` shipped with `quantization_config.ignore=[]` (literally empty) — everything got INT4. That's a different repo but **the lesson generalizes:** every new MoE calibration upload should have a one-line audit at publish time:
```python
import json; print(len(json.load(open(f'{model}/config.json'))['quantization_config']['ignore']))
```
If that's 0 or under ~50 for a vision-capable MoE model, the model is almost certainly broken.

**Suggestion to R9700:** please re-calibrate gemma-4-26B-AWQ from BF16 base (`Gemma4ForConditionalGeneration` arch) with the corrected ignore list (vision tower + router + mlp.gate_proj per layer + lm_head). That'd unblock both rigs simultaneously. Patch 020 on our side has shimmed `clippable_linear` so the vision-capable `Gemma4ForConditionalGeneration` class will load if the checkpoint registers as that.

---

### ✅ Qwen3.5-27B-AWQ-4bit-calibrated — works, ctx-limited
Validator: basic+thinking PASS. 32K context ceiling on 3090 (DeltaNet replication forces 19 GB/GPU weights, leaving only 2.2 GB for KV cache). Use `qwen3-ream` or `qwen36` for >32K thinking workloads on this rig.

---

## Cross-cutting suggestions

1. **Native AWQ as default for NVIDIA users.** The CT format trips on `(1, H)` shared_expert_gate in SGLang's NVIDIA CT loader. The native-AWQ uploads work cleanly. Worth adding a "Recommended for {ROCm | NVIDIA Ampere | NVIDIA Hopper}" badge to each model card.

2. **Calibration ignore-list audit at publish.** Ours found a 0-entry case shipped to HF; yours has the `shared_experts` plural typo (per your own README open-issues note). Auto-audit script in `scripts/quantize/`:
   ```python
   import json, sys
   c = json.load(open(f'{sys.argv[1]}/config.json'))
   ig = c.get('quantization_config', {}).get('ignore', [])
   if len(ig) < 50 and any(k in str(c.get('architectures', [])) for k in ['MoE', 'A4B', 'A3B']):
       print(f"WARN: MoE model with only {len(ig)} ignore entries — verify routing/vision preserved")
   ```

3. **Patch 019 + Patch 020 portability.** Both are vendor-neutral SGLang patches:
   - 019: transformers ≥5.5 + Python 3.13 metaclass dataclass replacement of `Qwen3_5Moe*Config` subclass __init__'s. Worth porting to RDNA4 if you ever bump Python to 3.13.
   - 020: `clippable_linear` shim — you already ship this in your patch 013. We re-implemented independently; just confirming our version aliases to the same Parallel Linear classes (no actual clip op).

4. **`hf upload` vs `hf upload-large-folder`.** You noted in the 35B HF section that plain `hf upload` worked in 1 minute after `upload-large-folder` stalled 11h. Confirmed on our side: for any repo ≤25 GB, plain `hf upload` is strictly better. The dedup machinery in upload-large-folder actually slows things down for typical model sizes. Worth promoting to a top-level rule in your CLAUDE.md.

5. **Long-context decode-curve documentation.** The Ampere-vs-ROCm asymmetry (33 tok/s short-ctx win + 3.5x long-ctx loss for the same Qwen3.6-35B model, see "Detailed findings" above) would be useful as a model-card snippet so users with the wrong rig know what to expect.
