# SGLang v0.5.10 → v0.5.11 patch upgrade status

Snapshot 2026-05-07. v0.5.11 fetched as new upstream tag. v0.5.10 → v0.5.11
diff: 2347 files changed, ~309k insertions, ~38k deletions (major release).

Tested all 24 local patches via `git apply --check` against the v0.5.11
worktree. Per-patch verdict:

| Patch | v0.5.11 status | Action |
|-------|---------------|--------|
| 001-upstream-sync | ⏫ UPSTREAMED | DROP (file/feature now in upstream) |
| 002-nvidia-model-fixes | 🔀 REFACTORED | `quantization/awq.py` → `quantization/awq/awq.py` |
| 003-deltanet-triton-dtype-fix | ✅ CLEAN | Keep as-is |
| 004-gemma4-causal-lm-fix | 🔧 CONFLICT | Rebase (`configs/model_config.py:185` shifted) |
| 005-ampere-fp8-triton-fallback | 🔧 CONFLICT | Rebase (`entrypoints/engine.py:1128` shifted) |
| 006-awq-bf16-activation-support | 🔀 REFACTORED | `quantization/awq.py` → `quantization/awq/awq.py` |
| 007-ampere-deltanet-kernel-tuning | 🔧 CONFLICT | Rebase (`fla/fused_sigmoid_gating_recurrent.py:280` shifted) |
| 008-awq-moe-wna16-fallback | 🔀 REFACTORED | `quantization/awq.py` → `quantization/awq/awq.py` |
| 009-qwen35-moe-causalLM | 🔀 REFACTORED | `quantization/awq.py` → `quantization/awq/awq.py` |
| 011-triton-attention-fp32 | ✅ CLEAN | Keep as-is |
| 012-sliding-window-decode-fix | ✅ CLEAN | Keep as-is |
| 014-gemma4-reasoning-parser | 🔧 CONFLICT | Rebase (`parser/reasoning_parser.py:477` shifted) |
| 015-ct-wna16-dequant-layout-fix | 🔧 CONFLICT | Rebase (`compressed_tensors_wNa16.py:311` shifted) |
| 016-ct-moe-gelu-triton-route | 🔧 CONFLICT | Rebase (`compressed_tensors.py:678` shifted) |
| 017-moe-wna16-gelu-activation | 🔧 CONFLICT | Rebase (`moe_wna16.py:369` shifted) |
| 018-qwen36-vision-config-dict-wrap | ✅ CLEAN | Keep as-is |
| 019-qwen3_5-moe-vl-config-dataclass-and-model-init | 🔧 CONFLICT | Rebase (`configs/qwen3_5.py:145` shifted) |
| 020-gemma4-clippable-linear-shim | ⏫ UPSTREAMED | DROP (`clippable_linear.py` is in v0.5.11 upstream) |
| 021-marlin-moe-gelu-activation | 🔧 CONFLICT | Rebase (`fused_marlin_moe.py:8` shifted) |
| 022-gemma4-causal-dedup-entry-class | 🔧 CONFLICT | Rebase (`gemma4_causal.py:1060` shifted) |
| 023-gemma4-moe-mlp-no-quant-config | ✅ CLEAN | Keep as-is |
| 024-gemma4-mm-towers-no-quant-config | ✅ CLEAN | Keep as-is |
| 025-gemma4-vision-pooler-padding-fp32 | ✅ CLEAN | Keep as-is |
| 026-gemma4-mm-video-per-frame-batching | ✅ CLEAN | Keep as-is |

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
