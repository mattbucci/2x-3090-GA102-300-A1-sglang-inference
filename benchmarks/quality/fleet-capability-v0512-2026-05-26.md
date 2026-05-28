# Fleet capability validation — sglang v0.5.12, 2026-05-26

Each preset launched TP=2 on the v0.5.12 stack (`ENV_NAME=sglang-v0512
SGLANG_DIR=/data/sglang-rebase-v0512`) and run through
`validate_capabilities.py` (now including a **tool_call** probe). 256K context
where the model supports it, else the preset's native max. "n/a" = capability
not applicable to that model (auto-skipped). Receipts in
`benchmarks/quality/capability_check.json` + `qwen36-awq-marlin-rebuild-v0512.json`.

| Preset | ctx | basic | tool_call | thinking | image | video | verdict |
|--------|:---:|:-----:|:---------:|:--------:|:-----:|:-----:|:--------|
| **qwen36** (AWQ-Marlin rebuild) | 256K | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5** — shipped to HF |
| **qwen35-moe** | 256K | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5** |
| **qwen36-dense** | 131K | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5** |
| **gemma4** (26B MoE) | 16K | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5** |
| **qwen3-vl-32b** | 131K | ✅ | ✅ | n/a | ✅ | ✅ | **4/4** (non-thinking VL) |
| **coder-30b** | 256K | ✅ | ✅ | n/a | n/a | n/a | OK (text coder) |
| **coder-30b-ream** | 256K | ✅ | ✅ | n/a | n/a | n/a | OK (text coder) |
| **coder-reap-25b** | 256K | ✅ | ✅ | n/a | n/a | n/a | OK (text coder) |
| **qwen36-ream** (vision-grafted) | 256K | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5** — coding-eval leader (58.7% opencode); vision tower grafted |
| **devstral** (Devstral-2 AWQ rebuild) | 131K | ✅ | ✅ | n/a | ✅ | n/a | **3/3** — in-house Devstral-2-2512 rebuild, tool-calling fixed, shipped to HF |
| **gemma4-31b** (AWQ rebuild) | 256K | ✅ | ✅ | ✅ | ✅ | ✅ | **5/5** — in-house BF16→AWQ rebuild, shipped to HF |
| qwen3-ream | — | — | — | — | — | — | **model not on disk** |

## What got fixed this pass
- **qwen36** rebuilt from BF16 → native AWQ-Marlin (was CT, broke on v0.5.12). 5/5 incl. tool-calling + vision (after BF16 vision-tower merge). Shipped to `mattbucci/Qwen3.6-35B-A3B-AWQ`.
- **qwen35-moe** stale-symlink fixed → 5/5.
- **coder-30b** dead MODEL path repointed to the on-disk CT dir + QUANT corrected (was `awq_marlin` over a CT config).
- **qwen3-vl-32b** missing `hf-mattbucci/Qwen3-VL-32B-AWQ` symlink created → 4/4.
- **validate_capabilities.py** gained a `tool_call` probe + coder-family auto-skip classification.
- **patch 039**: `gemma4_causal` `num_experts` getattr fallback (dense Gemma4 crash on load).

## What got fixed this pass (cont.)
- **gemma4-31b** rebuilt in-house from BF16 → GPTQ W4A16 → native AWQ (17.1h CPU calibration on `balanced_thinking_vision`, vision tower + embed_vision ignored → kept FP16). 5/5 at 256K incl. **content-aware vision + video** — replaces the AutoRound mirror whose text-only calibration left vision hallucinating. Shipped to `mattbucci/gemma-4-31B-AWQ`. Loader fix (patches 039 + 040, the top-level `Gemma4Config` head-dim remap for the dense path) was a prerequisite.

## What got fixed this pass (cont.)
- **devstral** rebuilt in-house: the community AWQ degenerated on the `[AVAILABLE_TOOLS]` context (under-calibrated `[TOOL_CALLS]` pathway). Rebuilt Devstral-Small-2-24B-Instruct-2512 (FP8 base → BF16 dequant → GPTQ W4A16 with the new `code_vision_tools` recipe → AWQ). 3/3 PASS (basic+tool_call+vision). Three serving subtleties were load-bearing: (1) strip the leading `model.` key prefix the VLM `save_pretrained` adds (else SGLang misloads → `<unk>`); (2) ship the official `tokenizer_config.json` (the dequant save dropped `additional_special_tokens`); (3) serve with the checkpoint's embedded canonical Mistral template (with BOS) — must match the calibration template or the tool pathway degenerates. Shipped to `mattbucci/Devstral-Small-2-24B-AWQ`.

## What got fixed this pass (cont.)
- **qwen36-ream** vision fixed by grafting the vision tower: the checkpoint declared `vision_config` but shipped **zero `model.visual.*` weights** (stripped in the REAM/AWQ build) — not "alignment drift". REAM only merges MoE experts, so the Qwen3.6 vision tower from the shipped qwen36 (identical `vision_config`) drops straight in (333 FP16 tensors via `merge_vision_weights.py`, LM untouched). 5/5 PASS — the coding-eval leader now also does content-aware vision + video.

## Remaining gaps (pre-existing, need deeper work)
- **qwen3-ream**: `Qwen3-30B-Instruct-2507-REAM-AWQ` is not on disk; preset references a missing checkpoint. (Text-only, non-thinking; documented not viable for codegen.)
