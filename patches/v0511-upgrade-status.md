# SGLang v0.5.10 → v0.5.11 patch upgrade status

Snapshot 2026-05-07. v0.5.11 fetched as new upstream tag. v0.5.10 → v0.5.11
diff: 2347 files changed, ~309k insertions, ~38k deletions (major release).

Tested all 24 local patches via `git apply --check` against the v0.5.11
worktree. Per-patch verdict:

## ✅ COMPLETE 2026-05-07. 24 → 13 patches, all apply cleanly to v0.5.11.

| Patch | Status | Notes |
|-------|--------|-------|
| 001-upstream-sync | 🗑️ DROPPED | Upstreamed wholesale into v0.5.11 |
| 002-nvidia-model-fixes | 🔄 REPLACED | Old wide-scope 002 superseded by upstream Qwen3_5 wrappers, but a focused fix was still needed → new `002-qwen3-deltanet-awq-weight-loader.patch` ports R9700's `Qwen3GatedDeltaNet._override_weight_loader` extension to cover AWQ attrs (`qweight`, `scales`, `qzeros`) + early-return guard in `_make_packed_weight_loader` for AWQ modules. Without this, AWQ-quantized Qwen3.5/3.6 (qwen35, qwen35-moe, qwen36-tp1) silently miss the packed weight_loader override on their `in_proj_qkvz` / `in_proj_ba` linears at v0.5.11 — produces incorrect TP slicing on fused checkpoint loading. Cross-team port of R9700's regenerated 007 (their commits `eec67c0` + `d1dbc77`). |
| 003-deltanet-triton-dtype-fix | ✅ CLEAN | Applies unchanged |
| 004-gemma4-causal-lm-fix | ✅ REBASED | 3-way auto-rebase (line 185→225) |
| 005-ampere-fp8-triton-fallback | ✅ REBASED | 3-way clean for 2 files; manual resolve for `entrypoints/engine.py` flashinfer pin (`0.6.7.post2` → `0.6.8` allow custom builds) |
| 006-awq-bf16-activation-support | 🗑️ DROPPED | `AWQMarlinConfig.get_supported_act_dtypes` already returns `[half, bfloat16]` upstream (only path 3090 sm_80+ uses) |
| 007-ampere-deltanet-kernel-tuning | ✅ REBASED | 3-way auto-rebase |
| 008-awq-moe-wna16-fallback | 🗑️ DROPPED | `AWQMarlinConfig` MoE→WNA16 fallback now in `awq/awq.py` upstream |
| 009-qwen35-moe-causalLM | 🗑️ DROPPED | `Qwen3_5ForCausalLM` + `Qwen3_5MoeForCausalLM` now in `qwen3_5.py:935` + `:1230` upstream; `_bind_packed_weight_loaders` at `:276` |
| 011-triton-attention-fp32 | ✅ CLEAN | Applies unchanged |
| 012-sliding-window-decode-fix | ✅ CLEAN | Applies unchanged |
| 014-gemma4-reasoning-parser | 🗑️ DROPPED | `Gemma4Detector` class now in `parser/reasoning_parser.py:510` upstream + `"gemma4": Gemma4Detector` in detector dict |
| 015-ct-wna16-dequant-layout-fix | 🗑️ DROPPED | `_dequant_fallback` method removed entirely in v0.5.11 — `apply_weights` now calls `apply_gptq_marlin_linear` directly. If Gemma 4 down_proj Marlin shape incompat resurfaces, will need new path. |
| 016-ct-moe-gelu-triton-route | 🗑️ DROPPED | `SGLANG_FORCE_CT_MOE_TRITON=1` env-var convenience superseded by upstream `moe_runner_backend=triton` config option |
| 017-moe-wna16-gelu-activation | ✅ REBASED | Hand-rebased `silu`-only assert in `MoeWNA16Method.apply` to allow `silu`+`gelu` |
| 018-qwen36-vision-config-dict-wrap | ✅ CLEAN | Applies unchanged |
| 019-qwen3_5-moe-vl-config-dataclass-and-model-init | 🗑️ DROPPED | `Qwen3_5MoeVisionConfig.__init__` (and Text/Config equivalents) now upstream at `configs/qwen3_5.py:115`+ |
| 020-gemma4-clippable-linear-shim | 🗑️ DROPPED | `clippable_linear.py` is now an upstream module |
| 021-marlin-moe-gelu-activation | ✅ REBASED | Hand-rebased gelu support across `fused_marlin_moe.py` (import `gelu_and_mul` from `sglang.jit_kernel.activation`, add `activation` kwarg, dispatch), `compressed_tensors_wNa16_moe.py` (2 silu asserts → silu+gelu, pass kwarg), `moe_runner/marlin.py` (silu assert + pass kwarg), `marlin_utils.py` (relax `supports_activation`) |
| 022-gemma4-causal-dedup-entry-class | 🗑️ DROPPED | v0.5.11 has `EntryClass = Gemma4ForCausalLM` (single value, our patch's intent) |
| 023-gemma4-moe-mlp-no-quant-config | ✅ CLEAN | Applies unchanged |
| 024-gemma4-mm-towers-no-quant-config | ✅ CLEAN | Applies unchanged |
| 025-gemma4-vision-pooler-padding-fp32 | ✅ CLEAN | Applies unchanged |
| 026-gemma4-mm-video-per-frame-batching | ✅ CLEAN | Applies unchanged |

**Counts:** 24 → 17 (7 dropped, 8 unchanged, 2 auto-rebased, 7 manual TODO).

## Summary

- **Clean (8):** 003, 011, 012, 018, 023, 024, 025, 026 — apply unchanged.
- **Upstreamed (2):** 001, 020 — DROP entirely.
- **AWQ refactor (4):** 002, 006, 008, 009 — `awq.py` became `awq/awq.py`. Need to update `+++ b/...` paths in each `.patch` file, then verify the patch still applies cleanly to the new file. If the v0.5.11 `awq/awq.py` already resolves the bug we patched, the patch is effectively upstreamed.
- **Context-shift conflicts (10):** 004, 005, 007, 014, 015, 016, 017, 019, 021, 022 — line numbers shifted in v0.5.11 but the file still exists. Each needs forensic rebase: re-apply the change at the new location + verify the bug it fixes is still present in v0.5.11.

## Rebuild plan

**Phase 1 — drop UPSTREAMED:**
- Remove `patches/001-upstream-sync.patch`.
- Remove `patches/020-gemma4-clippable-linear-shim.patch`.
- Verify our `gemma4_mm.py` (which imports from clippable_linear) still works — the upstream `clippable_linear.py` should match our shim's interface.

**Phase 2 — AWQ refactor (002, 006, 008, 009):**
- Update each patch's `+++ b/` and `--- a/` paths from `awq.py` to `awq/awq.py` (or wherever the change lives in the new layout).
- Re-test `git apply --check`.
- If conflicts remain (the AWQ class methods may have been split across files), do a forensic rebase per-patch.

**Phase 3 — context-shift conflicts (004, 005, 007, 014, 015, 016, 017, 019, 021, 022):**
- For each: read the patch's intent + check whether the bug still exists in v0.5.11 + rebase to new line numbers.
- Patches that fix bugs already addressed upstream become DROP; the rest get re-issued.

**Phase 4 — switch tag + retest:**
- Update `scripts/setup.sh:23 SGLANG_TAG="v0.5.10"` → `v0.5.11`.
- Update line 4 comment "20 local patches" → actual count.
- Re-run `setup.sh` against v0.5.11 with the rebuilt patch set.
- `validate_capabilities.py` + `probe_*.py` sweep across the default-MODELS set to confirm no regressions.

## Estimated effort

- Phase 1: 15 min (delete + verify).
- Phase 2: ~1 h (4 patches, mostly path updates + verification).
- Phase 3: ~3-5 h (10 patches, each needs reading + forensic rebase).
- Phase 4: ~30 min setup + ~30 min validator sweep.

Total: 5-7 hours of focused work. Most of it serial (can't easily parallelize patch rebasing).

## Open questions to surface before starting

1. Should we update `setup.sh` SGLANG_TAG immediately and run on a partial set (just the CLEAN-applying patches), or hold the tag at v0.5.10 until all rebases land?
2. Any v0.5.10 → v0.5.11 release notes to read for breaking changes that could affect our serving (e.g., chat-template parsing, multimodal pipeline, KV-cache layout)?
3. Are any of our patches due to be retired anyway (e.g., 014 reasoning parser if upstream's parser now handles Gemma 4 channel markers)?
