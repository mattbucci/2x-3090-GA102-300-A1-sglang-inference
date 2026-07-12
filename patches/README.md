# Patches, Fixes, and Historical Findings — 3090

This file collects the details of **what was fixed and why** — per-patch narratives, root-cause notes, and cross-team learnings. The top-level `README.md` keeps only current state; once an issue is closed with a patch, the narrative lives here.

**24 logical patches** apply in numeric order against SGLang **v0.5.15** (the default stack since the 2026-07-12 flip; live tree `/data/sglang-rebase-v0515`, env `sglang-v0515`). **v0.5.14→v0.5.15 rebase (2026-07-12):** first flip with a net patch-count reduction — 21 apply clean, 2 regenerated (**011**: upstream added PAGE_SIZE paging through both extend kernels, all 7 FP32 sites re-ported, and upstream itself adopted `out_dtype=tl.float32` on the extend triangle-branch QK dot — converging on 011 one site at a time; **051**: `hybrid_swa_archs` grew `UnlimitedOCRForCausalLM`, pure context drift), **2 retired as upstreamed** (**054**: v0.5.15 keyword-binds the ministral3 `super().__init__` calls natively, a superset of our fix incl. `start_layer` threading; **055**: transformers **5.12.1** — the new hard pin — ships the entire `gemma4_unified` family natively, so the ~1700-line vendored stack evaporated exactly as its narrative predicted), **+1 new (057** mistral-common-backend-optout — the one functional break the tx 5.8.1→5.12.1 bump introduced: tx 5.12 routes Mistral-family checkpoints shipping `tekken.json` to `MistralCommonBackend`, which never parses special tokens from text, so sglang's render-then-encode chat path fed `[INST]`/`[TOOL_CALLS]` as PLAIN TEXT — fleet probe caught devstral at needle 0.0/all depths, HE 80%→48%, dead tool-calls, while the basic probe still passed; 057 reroutes to `TokenizersBackend` via the `fix_mistral_regex` kwarg-presence escape, token ids verified bit-identical to tx 5.8.1, devstral restored to full parity on re-probe). Full 11-preset probe green (receipts `*-v0515.json`); 3-gate replay green at 24 patches — now scripted, `scripts/test_patch_gates.sh`. Receipt: `v0.5.15-rebase-status.md`. (Historical flips below.)

(v0.5.14 history:) **v0.5.13.post1→v0.5.14 rebase (2026-06-26):** the smallest yet — 21 apply clean, 3 regenerated (017 wNa16_moe refactored to the MoeRunner arch so the 2 silu asserts moved + the old `activation=` kwarg sub-hunk is now upstream → dropped; 051 cohere2moe hybrid-SWA archs-set drift; 053 EVS-video import-block drift), **0 fully upstreamed**, +1 new (**056** gdn-conv-state-dtype-cast — v0.5.14's mamba-track radix path index_puts an fp16 conv source into the bf16 conv cache → qwen36/qwen35-moe DeltaNet crash; boundary cast, sibling of 003). Full probe (9 fleet + image/video/thinking/tool capability) green, no regressions (gemma4-31b/12b 256K-needle even *improved* 0.4/0.6→1.0). **nemotron3-omni** also serves 6/6 on v0.5.14 (incl. video/053 + audio) once the preset forces `QUANT=moe_wna16` — the auto-detected awq_marlin path fails identically on v0.5.13 and v0.5.14 (shared-expert 3712→1856 @TP=2 vs Marlin min_thread_k=128), so it was never a version regression, just a preset gap (the moe_wna16 override wasn't baked in). Receipt: `v0.5.14-rebase-status.md`. (054 was added during the v0.5.13 cycle when the devstral fleet-smoke caught an upstream Mistral3 bug; 055 re-vendored the `gemma4_unified` config after tx 5.8.1 proved to still lack it — the gemma4-12b boot regression — see below.) `scripts/setup.sh` applies every `*.patch` in this directory idempotently (`git apply --check` → apply, else skip). **v0.5.12→v0.5.13.post1 rebase (2026-06-16):** 6 patches dropped as upstreamed (012/028/034/042/043/050 — files deleted); 017/023/039/051 regenerated against the new tree (017 = gelu now only on the Marlin/CT path, the moe_wna16 assert merged into 052; 023 gemma4 towers+dense-MLP; 039 half-2 head-dim remap, half-1 num_experts upstreamed; 051 half-2 hybrid-SWA, config-vendor half dropped). 026 is the **unchanged upstream patch** but it shares `gemma4_mm.py` with 023 (023 < 026): 023 was generated against **pristine+026** so its tower hunk (line ~191) is isolated from 026's video hunk (line ~466), and the clean 023 only inserts +7 lines above 026's region so 026 still applies via git-apply's offset tolerance (verified by gate (a)). 052 absorbs the non-gated **and** gelu `moe_wna16` assert to avoid a shared-line ordering trap with 017. **054** (Mistral3 keyword super-init) was added post-flip when the devstral smoke caught an upstream `ministral3.py` positional-super bug. The whole set was re-verified by the 3-gate pristine replay: applied to pristine v0.5.13.post1 it produces a tree **byte-identical** to `/data/sglang-rebase-v0513`, and re-running the loop on the patched tree applies nothing. Fleet re-validated on v0.5.13: nemotron3-omni 6/6, gemma4-26b/gemma4-31b/qwen36 5/5 (+ R9700 full-fleet resweep). Old v0.5.11/v0.5.12 stacks retained untouched for one-revert rollback. (Prior consolidation 2026-06-10: five multi-file series merged into single logical patches — mapping below.)

## Patch map

Grouped by work area. *Upstream* = status vs `sgl-project/sglang` main `70c71ba18` (2026-06-10): **drop** = main has it, delete at next rebase; **PR** = main still has the bug, upstream-PR candidate; **site** = Ampere/this-rig-specific, carry forever.

| # | Patch | Area | Upstream (main @ 2026-06-10) |
|---|-------|------|------------------------------|
| 002 | qwen3-deltanet-awq-weight-loader | Quantized loading | **PR** — main still force-overrides packed loaders for AWQ |
| 003 | deltanet-triton-dtype-fix | Kernel correctness | **PR** — no conv_state cast in main |
| 004 | gemma4-causal-lm-fix | Gemma 4 | **PR** — no `is_causal_lm` multimodal gate in main |
| 005 | ampere-fp8-triton-fallback | Ampere | **site** (sm_86 fp8e4nv) |
| 007 | ampere-deltanet-kernel-tuning | Ampere | **site** (BV=64 is 3090 tuning; main keeps BV≤32) |
| 011 | triton-attention-fp32 | Kernel correctness | **PR** — main still downcasts at 5 extend + 3 decode sites (audit 2026-07-12), but v0.5.15 adopted `out_dtype=tl.float32` on the extend triangle-branch QK dot — upstream is converging; PR the rest in their idiom. Main also MOVED the kernels to `python/sglang/kernels/ops/attention/` → next rebase re-ports this patch regardless. Bites RDNA4 + SM12.x |
| 012 | sliding-window-decode-fix | Kernel correctness | **drop** — main reworked `window_kv_offsets` plumbing |
| 017 | moe-gelu-activation *(= old 017+021)* | MoE runners | **PR** — main still asserts SiLU-only (moe_wna16 + marlin) |
| 018 | qwen36-vision-config-dict-wrap | Qwen3.5/3.6 | **PR** — no dict-wrap in main's qwen_vl processor |
| 023 | gemma4-quant-config-detection *(= old 023+024)* | Gemma 4 | still needed, **Ampere-only** (trips HSAIL on RDNA4 — do not port) |
| 025 | gemma4-vision-pooler-padding-fp32 | Gemma 4 | partial — fp32 pool landed in main, pre-pool `masked_fill` did not |
| 026 | gemma4-mm-video-per-frame-batching | Gemma 4 | **PR** — main still batches frames (OOM + ROCm bsz==1 assert) |
| 028 | gemma4-mm-per-expert-awq-loader | Gemma 4 | **drop** — main's gemma4_mm now has `make_expert_params_mapping` |
| 029 | qwen35-shared-expert-gate-ct-dequant | Quantized loading | still needed (or fix recipe-side: ignore `shared_expert_gate`) |
| 030 | fused-moe-w2-presharded-detect | Quantized loading | **KEEP** (verified at the v0.5.15 flip): upstream's `use_presharded_weights` plumbing exists at exactly our narrow sites but only `quark_int4fp8_moe` ever sets it — compressed-tensors never does, so the shape-based detect stays load-bearing for CT MoE @TP=2. PR option: wire the flag through the CT MoE method (mirrors quark), or upstream our format-agnostic shape-guard |
| 031 | qwen3_5-deltanet-awq-weight-loader | Quantized loading | **PR** — main's qwen3_5.py still has all three v0.5.11 gaps |
| 034 | sampler-inf-detection | Robustness | **PR** — no ±Inf branch in main (R9700-originated) |
| 035 | qwen3_5-causal-lm-enablement *(= old 035+036+038)* | Qwen3.5/3.6 | **PR** — main's EntryClass still omits both CausalLM heads |
| 037 | skip-grpc-rust-ext | Build | **site** (no protoc here; HTTP-only serving) |
| 039 | gemma4-dense-bringup *(= old 039+040)* | Gemma 4 | half: `num_experts` getattr **landed in main**; top-level head-dim remap did not |
| 041 | devstral-toolcall-omission-recovery | Robustness | **PR** — main's MistralDetector still leaks marker-less calls (R9700-originated). **⚠ R9700 extension available (2026-06-17): patch 056 upgrades the hold-back from exact-match to PREFIX-match.** SGLang streams the tool-name token(s) before `[ARGS]`, so a MULTI-token name (`todowrite`→`todo`+`write`, `webfetch`→`web`+`fetch`) arrives in pieces — the first piece isn't an exact tool name, so 041/040 flushes it as content → the name splits across the flush boundary → recovery still fails. Holding a trailing run that is a *prefix* of any known tool keeps the full name buffered until `[ARGS]` arrives, where exact-match recovery then succeeds. **Backend-independent** (streaming/detector logic, not attention) → applies on your FlashInfer Devstral too if your harnesses use multi-token tool names (opencode/claw do). Prose-safe (a prefix word is held ≤1 increment then flushed on divergence, never dropped); unit-test 8/8. Source: R9700 `patches/056-devstral-multitoken-toolname-omission-recovery.patch`. |
| 042 | cohere2-moe-loader-graft | New arch | **drop** — grafted *from* main (+ 2026-06-11 runtime fix: `RoutingMethodType.Sigmoid*` getattr shim — v0.5.12 enum predates the members; metadata-only on sm_86, real routing is the custom_routing_function) |
| 043 | gemma4-unified-bringup *(= old 043–048)* | Gemma 4 | mostly **drop**: model graft + `model_config` arch-lists + `config.py` remap are in main. **⚠ Correction (2026-06-17): the config/processor vendor shims did NOT evaporate** — the v0.5.13 env is tx 5.8.1 (< 5.10), which still lacks `gemma4_unified`, so `AutoConfig` raised at boot (gemma4-12b regression). The config-register subset returned as **patch 055**; only the model graft + arch-lists were genuinely upstreamed. |
| 049 | tp-load-timeout-cold-cache | Robustness | **site** (R9700 graft — their 048; upstream would want an env-var, low-priority PR) |
| 050 | gemma4-unified-video-token-count | Gemma 4 | vendor-shim fix — **PR to HF transformers** (vendored verbatim 2026-06-07, likely upstream-broken there too) |
| 051 | cohere2moe-256k-enablement | New arch | half **PR** (hybrid-SWA pool classification for `Cohere2MoeForCausalLM` — main lacks it too: without the split the pool is sized full-attention for all 49 layers, ~4× the KV bytes/token); half env-bound (vendored `Cohere2MoeConfig` + registry — evaporates once the env's transformers knows `cohere2_moe`) |
| 052 | moe-wna16-nongated-moe | Quantized loading | **PR** — `moe_wna16` builds/loads a gated `2×intermediate` fused gate_up + asserts `silu`; no non-gated (squared-ReLU) support in main. Generalize via `moe_runner_config.is_gated`: `1×intermediate` gate_up + `up_proj` at offset 0 when non-gated. Unblocks Nemotron-Omni int4 experts (`up_proj`-only, `moe_intermediate_size=1856`). **v0.5.13-targeted; backward-compatible (gated path unchanged). ⚠ shares the `apply()` assert region with the pending 017 GELU regenerate — merge to `not is_gated or activation in (silu,gelu)`.** |
| 053 | evs-video-combined-path-routing | Multimodal | **PR** — `_get_chunked_prefill_embedding` routed single-offset EVS video to the per-image path, which `.reshape()`s the returned `EVSEmbeddingResult` → scheduler abort. Require `not any(EVSDataItem)` for the per-image path so EVS takes the combined (unwrap + redistribute) path. Generic `mm_utils`, arch-agnostic. **Cross-team port of R9700 CANDIDATE 057** (both stacks hit it; their FP8, our int4). v0.5.13-targeted. |
| ~~054~~ | ministral3-v0513-keyword-super-init | New arch | **RETIRED at the v0.5.15 flip — upstreamed**: v0.5.15 keyword-binds all three `super().__init__` calls natively + threads the new `start_layer` (superset of our fix; independently confirmed by R9700 at their same-day rebase). History: **PR (upstream `ministral3.py`)** — v0.5.13 inserted `start_layer` into `LlamaAttention`/`LlamaDecoderLayer.__init__` (after `layer_id`), but upstream `ministral3.py`'s `Ministral3Attention`/`Ministral3DecoderLayer` still call `super().__init__()` **positionally** → the args shift one slot and `prefix` (a str) lands in `quant_config` → `'str' object has no attribute 'get_quant_method'` at load. Breaks devstral (Mistral3-AWQ) on v0.5.13. Fix: keyword-bind the super() calls. Upstream bug (file unmodified by us); **cross-team = R9700 patch 064**. Caught by the post-flip devstral fleet-smoke. |
| ~~055~~ | gemma4-unified-config-register *(= config subset of dropped 043)* | Gemma 4 | **RETIRED at the v0.5.15 flip — the predicted evaporation happened**: transformers 5.12.1 (v0.5.15's pin) ships `gemma4_unified` natively (config + processors + feature extractor + the generic `ProcessorMixin.__call__` placeholder expansion); nothing in sglang imported the vendored modules, gemma4-12b validated 5/5 on the native classes (R9700 confirmed from RDNA4 independently). History: **env-bound** — v0.5.13 upstreamed the gemma4_unified **model** + processor adapter + `model_config` arch-lists + `config.py` model_type remap (so 043 was dropped at the flip), but it does **NOT** vendor the **config class**, and the env's tx 5.8.1 registers `gemma4`/`_text`/`_vision`/`_audio`/`_assistant` but not `gemma4_unified*` → `AutoConfig.from_pretrained` raises at boot. `gemma4-12b` (model_type `gemma4_unified`) was the one fleet member NOT re-validated at the flip, so this slipped through until 2026-06-17. Re-vendors `Gemma4UnifiedConfig` (+ audio/text/vision sub-configs) and the tx processor stack (image/video/feature-extraction/processing), and registers the config in the sglang `_CONFIG_REGISTRY` (→ `AutoConfig`) + the processors via `AutoProcessor`/`AutoImageProcessor`/`AutoVideoProcessor`/`AutoFeatureExtractor`. 6 vendored files + 2 edits (`configs/__init__.py`, `hf_transformers/common.py`); model / processor-adapter / `model_config.py` / `config.py` all untouched (upstream). Evaporates once the env's transformers ships `gemma4_unified` (tx ≥ 5.10). gemma4-12b boots + serves + reasons to 256K on v0.5.13 again. |
| 056 | gdn-conv-state-dtype-cast | Kernel correctness | **PR** — v0.5.14 GatedDeltaNet grew a `has_mamba_track_mask` path (the `extra_buffer` mamba-radix cache) that `index_put`s the fp16 conv source into the **bf16** conv-state cache (`gdn_backend.py:438`) → `RuntimeError: Index put requires the source and destination dtypes match, got BFloat16 (dest) … Half (source)`. Crashes qwen36 / qwen35-moe (any Qwen3.5/3.6 DeltaNet preset, which launch `--dtype float16`) on the first decode forward. Fix: `.to(conv_states.dtype)` cast at the write (no-op when matched) — the torch-level sibling of 003's kernel-side conv_state cast. v0.5.14-net-new (the mamba-track write path is new); applies to v0.5.13.post1 too but isn't needed there. **Re-audited 2026-07-12: still live on main — top upstream-PR candidate** (any fp16 DeltaNet user with mamba-track crashes on first decode). |
| 057 | mistral-common-backend-optout | Robustness | **PR-worthy (sglang, possibly tx)** — transformers ≥5.12 routes Mistral-family checkpoints shipping `tekken.json` to `MistralCommonBackend` (before honoring `backend=`), which by design never parses special tokens out of text; sglang renders the chat jinja to a string then encodes → `[INST]`/`[/INST]`/`[TOOL_CALLS]` become plain text. Fleet-probe signature: devstral needle **0.0 at every depth incl. 1K**, HE 80%→48%, tool_call dead, generation hallucinating `[inst]` turns — `basic` still passes (answer precedes the ramble), so boot smokes don't catch it. Fix in `get_tokenizer`: when AutoTokenizer returns `MistralCommonBackend`, reload with `fix_mistral_regex=False` (kwarg *presence* is transformers' escape hatch to `TokenizersBackend`; ids verified bit-identical to tx 5.8.1: `[INST]`→3, `[TOOL_CALLS]`→9) and skip `_resolve_tokenizers_backend` re-resolution for the rerouted result (it would re-take the MistralCommon route and crash on strict kwargs). AutoProcessor/Pixtral inner tokenizer already safe. Cross-team: warned R9700 same-day (their FP8 devstral bake-off cells on v0.5.15 were suspect). |

Rebase ledger (current, post-v0.5.15): **retired so far** 012, 028, 042, most of 043, 050, 054, 055; **~13 upstream-PR candidates** in priority order 056 (crash, still on main), 011 (upstream converging + kernel-tree move pending), 017, 049 (as env-var), 030 (as CT flag-wiring or shape-guard), 057, then 002, 003, 004, 018, 026, 031, 035, 041, 051's hybrid-SWA half + 025's masked_fill; **5 site-specific** (005, 007, 037, 049-until-PR'd, 023-Ampere-only). Bonus from the same main audit: `Qwen3VLMoeForConditionalGeneration` now exists upstream with a registered EntryClass — re-check the VL-30B loader (top-level README MoE backlog №3) against a graft before sinking calibration time.

### Merged-unit mapping (2026-06-10 consolidation)

| Now | Absorbs | Why one unit |
|-----|---------|--------------|
| 017-moe-gelu-activation | 017-moe-wna16-gelu-activation, 021-marlin-moe-gelu-activation | same feature (Gemma 4 gelu MoE) across two runner stacks |
| 023-gemma4-quant-config-detection | 023-gemma4-moe-mlp-no-quant-config, 024-gemma4-mm-towers-no-quant-config | one principle (respect recipe `ignore`), always shipped/tested as a pair |
| 035-qwen3_5-causal-lm-enablement | 035-…-entry-class, 036-…-layer-types-hf-fallback, 038-…-norm-topk-prob-fallback | one enablement chain; members not independently functional |
| 039-gemma4-dense-bringup | 039-…-num-experts-fallback, 040-…-head-dim-remap | one bring-up (gemma4-31b), authored as a pair |
| 043-gemma4-unified-bringup | 043…048 (loader graft, config register, processor+remap, processor stack, hybrid-swa TP=2, processor `__call__`) | one bring-up (gemma-4-12B); members not independently valid (044 imports a module 046 ships) |

Retired numbers (dropped in upstream rebases; narratives in **Historical** below or `git log`): 001, 006, 008, 009, 010, 013 (lives in the M4 repo — DeltaNet cache wiring), 014, 015, 016, 019, 020, 022, 027, 032, 033, plus the original 030 (v0.5.10 sister of 028).

## Patch-hygiene rules (learned 2026-06-10, the hard way)

Two latent defects surfaced while consolidating — both invisible on the live tree because `setup.sh`'s idempotent loop skips everything there:

1. **Old 045 could never replay.** It was diffed against *pristine* v0.5.12, but old 040 had already rewritten the same `hf_transformers/config.py` block — so on any fresh clone 045 hit "patch does not apply" and was silently *skipped*: fresh installs got a 12B with no `gemma4_unified` config remap. **Rule: generate a new patch as a diff against the predecessor-patched tree, never against pristine.**
2. **Old 026 re-applied onto an already-patched tree.** Hand-written hunk with no unique anchor; `git apply` matched the *image* path's identical `vt(pv, pp)` block and would have patched `get_image_feature` on a `setup.sh` rerun. **Rule: hand-written patches need context that pins the intended site (≥U10 from the real tree); regenerate from real trees when possible.**

**Gate for any new patch (all three, in order):** (a) full glob-order sequence applies on the pristine target tag; (b) the resulting tree is byte-identical to the live serving tree; (c) on the *patched* tree, `git apply --check` **fails** for every patch (rerun safety — a pass means your patch can mis-target a second site). **Scripted since the v0.5.15 flip: `scripts/test_patch_gates.sh`** (`SGLANG_TAG=… SGLANG_DIR=… [PATCH_DIR=…]`) — run it before committing any patch change.

**Graft-validation addendum (learned 2026-06-11 via 042):** an AST constructor-kwarg drift check does NOT catch **missing enum members** — `RoutingMethodType.Sigmoid` parsed fine and died only at first boot. For any graft from a newer tree, additionally grep the graft for `SomeEnum.Member` attribute references and verify each member exists in the v0.5.12 target module.

---

## 2026-06-10 optimization sprint (history)

Full lab notebook with per-experiment receipts: [`../benchmarks/sprint-2026-06-kv-decode/LOG.md`](../benchmarks/sprint-2026-06-kv-decode/LOG.md). One-line verdicts: **Track A** — SGLang's `--swa-full-tokens-ratio 0.8` default was the Gemma KV wall; ratio 0.0625 took the 12B 102K→565K and the 26B 118K→652K full-pool tokens, tool-use 1.0/1.0 verified to 258,085 true tokens on both (patch 050 fixed the 12B video crash found en route → 5/5 omni). **Track B** — CUDA graphs were the whole Gemma decode gap ("triton SWA can't capture" falsified): 26B 34→83 tok/s @1K / 31→41 @256K, 12B 41→107, 31B 33→58, all 5/5; B2′ (26B dense down_proj dequant fallback) microbenched at 1.9% of TPOT and parked; B3 (qwen3-ream 64K→128K "cliff") disproven — smooth slope, 2-point sampling artifact. **Track C** — fleet flat-TPOT audit found the three graph-off Gemmas. **Track D** — kernel-gate audit: pass-1 log fingerprint (7 presets clean, 1 sized+parked hit), pass-2 profiled-decode recipe (`scripts/bench/profile_decode_step.sh`) demonstrated clean on qwen36-dense. **Sprint-2 (same day, concluded):** **A3** — the 31B had been wrongly excluded from Track A ("full-attention-bound" was an unverified footnote); same lever, 24,248→**109,120** full tokens at ratio **0.1** (tool-use 1.0 to 99K true, 5/5, decode unchanged; took 0.1 over 0.0625/134K — swa margin 10,912 vs proven 8,393 floor beats the last 25K of cap under multi-turn radix pressure). **B4** — NGRAM speculative decoding (zero draft VRAM): on hybrids it needs the DFlash recipe (extra_buffer + SPEC_V2) and then hits a 003-class fp16/bf16 Triton assert in the conv1d spec-verify `KERNEL_WIDTH` branches (**parked DFlash-conv1d patch candidate**, no file yet — the 052/053 slots went to the Nemotron-Omni serving fixes; renumber if realized); on qwen3-ream the draft-depth curve ran 1.01× (d12) → **1.24× (d6)** aggregate — parked at the pre-registered 1.3× bar, with copy-heavy prompts at **1.3–2.1×** on record as a one-flag opt-in, and a measurement caveat: repetitive filler corpora produce artifact NGRAM decode numbers (1042 tok/s ≠ real). **B5** — the "qwen36-dense is replicated, shard it for 2×" plan died by measurement: it loads 8.8 GB/GPU (sharded all along; stale VRAM table fixed); the real residual is 3.2 ms/token of NCCL allreduce (22% of TPOT), and **B5′** `--enable-flashinfer-allreduce-fusion` was a **null result** against it. Ops scar: bare systemd-run needs the exec bit (203/EXEC on the fleet script). Chart-regen fleet pass confirmed all sprint numbers on the standard instrument (26B 83→41, 12B 109→34, 31B 58→31@64K).

---

## Per-patch notes

### Quantized-weight loading (AWQ / CT int4)

#### 002 — qwen3-deltanet-awq-weight-loader (48 LOC)
Qwen3-Next `Qwen3GatedDeltaNet` unconditionally overrode the weight loaders of `in_proj_qkvz` / `in_proj_ba` with a packed-`weight` loader. AWQ checkpoints ship `qweight/scales/qzeros` instead of `weight`, so the override broke AWQ loads. `_make_packed_weight_loader` now returns `None` when the module has no `weight` param and the override is skipped. (R9700 carries the same fix as *their* 002 — numbering coincidence, see cross-team numbering note.)

#### 029 — qwen35-shared-expert-gate-ct-dequant (2026-05-07, ~52 LOC body)
**Closes the Qwen3.6-35B-A3B-AWQ-CT-on-NVIDIA bug; unlocks the calibration-clean CT variant.** `qwen2_moe.py` constructs `shared_expert_gate` as a plain `nn.Linear` (no quant_config), but CT-format checkpoints ship the quantized triplet `weight_packed/scale/shape` for it (llmcompressor quantized the `(1, H)` gate even though group-quant can't help there; native AWQ exports skip it). Pre-patch: 120 `not found in params_dict` warnings + multilingual word-soup from a random-init gate. Patch buffers the per-layer triplet at the top of `qwen3_5.py:load_weights`, dequants via `compressed_tensors.unpack_from_int32` → bf16 × `repeat_interleave(group_size)`-expanded scales, and writes the result into the layer's `.weight`. Verified TP=1/2K: 0 warnings, 4/4 PASS. Cleaner long-term fix is recipe-side (`re:.*shared_expert_gate.*` in the CT ignore list). R9700's CT direct-serve path hit 3 separate regressions when they tested (their patch hygiene + an RDNA4 TP=2 `_load_w2` bug = our patch 030's sibling symptom).

#### 030 — fused-moe-w2-presharded-detect (2026-05-09, 74 LOC)
**Closes the CT-format-MoE-at-TP≥2 crash** (`_load_w2`: `RuntimeError: start (4) + length (4) exceeds dimension size (4)`). The CT loader pre-shards w2 to per-rank size before invoking FusedMoE's weight_loader; the loader still called `narrow(shard_dim, shard_size*tp_rank, shard_size)`, walking past the end on rank>0. Native AWQ stores full-global w2 so narrow is correct there — which is why qwen36 TP=2 worked on native AWQ while the CT mirror crashed. Detection: if the loaded tensor's shard-dim size already equals the per-rank size, skip the narrow. TP=1 unaffected (narrow is a no-op at rank 0). Upstream main now has a `use_presharded_weights` flag through `narrow_padded_param_and_loaded_weight` — verify and drop at next rebase.

#### 031 — qwen3_5-deltanet-awq-weight-loader (2026-05-09, ~25 LOC)
**Restores three v0.5.10 fixes the v0.5.11 upstream sync silently dropped from `qwen3_5.py`.** Without them every AWQ Qwen3.5/3.6 DeltaNet build outputs literal `!!!!!` (0/4 validator): (1) `in_proj_ba` must get `quant_config=None` — AWQ ships it full-precision; (2) `_bind_packed_weight_loaders` must bind the AWQ trio `(qweight, scales, qzeros)`, not the FP8 tuple; (3) `_get_split_sizes_for_param` must divide merged-column split sizes by the pack factor for `Packed*Parameter`. Verified on Qwen3.6-27B-AWQ TP=1/4K: 0/4→4/4, 96 loader warnings → 0. Main's `qwen3_5.py` still has all three gaps (it post-dates the v0.5.11 sync) — PR candidate.

### Qwen3.5 / 3.6 family enablement

#### 018 — qwen36-vision-config-dict-wrap (R9700 backport, 19 LOC)
`qwen_vl.py` multimodal processor assumes `hf_config.vision_config` is a `PretrainedConfig`. llmcompressor-saved CT configs ship it as a plain dict → `AttributeError: 'dict' object has no attribute 'spatial_merge_size'` (HTTP 500 on first image request). Wrap once at processor init with `SimpleNamespace`. R9700-verified on `Qwen3.6-35B-A3B-AWQ-thinking-vision`.

#### 035 — qwen3_5-causal-lm-enablement (merged 2026-06-10; members 2026-05-19 + v0.5.12 rebase)
One enablement chain for checkpoints declaring `Qwen3_5(Moe)ForCausalLM` — our text-only CT mirrors (`Qwen3.6-35B-A3B-AWQ-CT`, `Qwen3.6-27B-AWQ-CT-balanced`); the multimodal `ForConditionalGeneration` mirrors never hit any of this:
- **EntryClass registration** *(old 035)*: `qwen3_5.py` implements four heads but only registered the two ConditionalGeneration ones; unregistered = "has no SGLang implementation". This was the qwen36 preset's 12-minute rc=1 bake-off exit on 2026-05-17.
- **`layer_types` HF fallback** *(old 036)*: two classes named `Qwen3_5MoeTextConfig` exist (sglang's with `layers_block_type`, HF's with `layer_types`); HF's wins the AutoConfig lookup for bare CausalLM configs, so `get_layer` must read either and translate `full_attention`→`attention`.
- **`norm_topk_prob` fallback** *(old 038)*: the CausalLM path reuses `qwen2_moe.Qwen2MoeSparseMoeBlock`; Qwen3.5/3.6 text configs omit `norm_topk_prob` → `getattr(config, "norm_topk_prob", True)` (Qwen MoE family renormalizes; observed wsum=1).

Steps 2+3 are dead code without 1; 1 crashes without 2+3 — hence one patch. R9700 should mirror whenever they serve bare-CausalLM checkpoints on ≥v0.5.11.

### Gemma 4 bring-up (26B MoE multimodal / 31B dense / 12B unified)

#### 004 — gemma4-causal-lm-fix (19 LOC)
CausalLM architectures are text-only even if the config class exposes `vision_config`/`audio_config` as class defaults (Gemma4ForCausalLM uses Gemma4Config). Gate `is_multimodal` on `is_causal_lm_only`.

#### 023 — gemma4-quant-config-detection (merged 2026-06-10; members 2026-05-03, detection upgrade 2026-05-09)
**One principle: respect what the calibration recipe actually quantized — never hardcode.** Two halves:
- **Dense MLP in MoE-block layers** *(old 023)*: parse `config.quantization_config.ignore` for `\.mlp\.(gate|up|down)_proj`; pass `quant_config=None` to the dense `Gemma4MLP` only when the recipe kept it BF16. Community 26B AWQ keeps dense MLP BF16; our own ships quantize it — the original hardcoded `None` made our ships create BF16 placeholders that never bound the AWQ keys → 60 not-initialized warnings, MoE forward of zeros, `<pad>` for every prompt. Forensics: `SGLANG_GEMMA4_TRACE=1` hook localized layer-0 `gate_up_proj` emitting 15078 NaN + 3267 Inf from a clean input (trace JSONs in `benchmarks/quality/`).
- **Multimodal towers** *(old 024)*: vision tower, embed_vision, audio tower, embed_audio always ship BF16 (recipe `ignore`) — pass `quant_config=None` to all four or the loader silently misses every tower key and vision degenerates to lorem-ipsum hallucination. Runtime requirements discovered alongside: 26B SigLIP NaNs in FP16 (launch preset defaults BF16) and the checkpoint must declare `architectures: ["Gemma4ForConditionalGeneration"]` (CausalLM silently degrades image payloads).

**Ampere-only.** R9700 hardware-tested the pair 2026-05-03: their loader BF16-falls-back on empty qweight slots (bug never manifests) and the changed kernel path trips an unguarded HSAIL in the sampler — they keep the files for reference but do NOT apply. Validator post-fix: 4/4 PASS both our v3b 21B-REAP and the HF-mirror 26B, 0 not-initialized warnings.

#### 025 — gemma4-vision-pooler-padding-fp32 (2026-05-03, ~13 LOC)
Aligns `Gemma4VisionPooler` with the HF reference: (1) pre-pool `masked_fill` of padding patches — without it bucket-0 of the avg-pool is diluted by up to 2264 padding patches on a small image; (2) FP32 accumulation in `_avg_pool_by_positions`. Shipped as code-correctness alignment; it changed the degraded-vision response shape but was **not** the root cause of the 26B vision-quality issue (that was calibration-side). Upstream main has since adopted the FP32 pool but still lacks the `masked_fill` — the remaining half is a PR candidate.

#### 026 — gemma4-mm-video-per-frame-batching (2026-05-04, ~17 LOC; regenerated 2026-06-10)
**Closes Gemma 4 video OOM on Ampere AND R9700's bsz==1 vision-attention assertion in one shot.** The batched `vt(pv, pp)` call materializes a `[num_frames × num_patches × 2 × position_embedding_size]` one_hot inside `Gemma4VisionPatchEmbedder._position_embeddings` — ~1.24 GB bf16 for a 12-frame video — after LM weights + KV pool have consumed the card. Processing frames one-at-a-time keeps the peak at 1/num_frames and satisfies the RDNA4 `bsz==1` assertion; output is identical (downstream already iterated per-frame). Validated 4/4 PASS. **2026-06-10:** regenerated with ≥U10 context — the original hand-written hunk had no unique anchor and `git apply` would re-target the *image* path's identical block on a `setup.sh` rerun (see Patch-hygiene rules).

#### 028 — gemma4-mm-per-expert-awq-loader (2026-05-08, ~28 LOC)
`gemma4_mm.py` only loaded HF's fused-experts source format (`experts.gate_up_proj` `[E, 2I, H]`); llmcompressor AWQ ships per-expert keys (`experts.<i>.gate_proj.qweight` × N experts × 30 layers ≈ 36K keys). Port of R9700's dual-format two-tier mapping (per-expert via `FusedMoE.make_expert_params_mapping` checked first, fused fallback unchanged). Runtime-verified on 21B-REAP (loader clean; that checkpoint's remaining 0/4 was its own calibration-degenerate scales). Unit-tested in `scripts/test/test_gemma4_per_expert_mapping.py`. **Upstream main now ships the same mechanism in gemma4_mm — drop at next rebase.**

#### 039 — gemma4-dense-bringup (merged 2026-06-10; members 2026-05-26)
One bring-up: dense Gemma4 (`gemma4-31b`) loads via `gemma4_causal.py` + the **top-level** `Gemma4Config`:
- **`num_experts` fallback** *(old 039)*: dense configs omit the attr; `load_weights` read it unconditionally → AttributeError. `getattr(config, "num_experts", 0) or 0` — the expert loop never matches dense checkpoints anyway. *(This exact fix has since landed in upstream main.)*
- **Top-level head-dim remap** *(old 040)*: SGLang's gemma4 config remap (base attrs = SWA layers, `global_*` = full-attention; SGLang expects the opposite) only handled `config.text_config` — the multimodal path. The dense path consumes the top-level config → full-attention layers built at head_dim=256 instead of 512, crashing q/k_norm + k/v loads. Refactored into `_remap_gemma4_head_dims(tc)` applied to both objects; `swa_head_dim` guard keeps it idempotent. *(Still absent in main.)*

Verified 2026-05-26: 31B boots TP=2, basic/tool/thinking/vision PASS; 26B unchanged 5/5.

#### 043 — gemma4-unified-bringup (merged 2026-06-10; members 2026-06-07)
**Everything needed to serve `google/gemma-4-12B` (`Gemma4UnifiedForConditionalGeneration`) on v0.5.12 + transformers 5.6.** Shipped as one patch because the six former members were never independently valid — old 044's `common.py` hunk imports the register module old 046 ships; old 046's processor needs old 048's `__call__`; and old 045 was diffed against pristine v0.5.12 so it could not replay over old 040 on a fresh clone (the consolidation's trigger — see Patch-hygiene rules). Components:
- **Model graft** *(old 043)*: `models/gemma4_unified.py` verbatim from sglang main (thin wrapper reusing our `Gemma4TextModel` + `Gemma4ForConditionalGeneration`), with main's `pp_filter_load_weight` helper inlined faithfully (no-op under PP=1).
- **Config vendor + registration** *(old 044)*: the arch is a transformers-5.10.dev `model_type`; our pinned tx 5.6 can't parse it. Vendor `Gemma4UnifiedConfig` (+ text/vision/audio sub-configs) verbatim from transformers main into `configs/gemma4_unified.py`, register via `_CONFIG_REGISTRY`. Avoids a fleet-wide transformers bump that would risk the qwen3_5 config behavior (patch 035).
- **Processor remap** *(old 045)*: registered SGLang processor subclass; gemma4 head-dim swa-remap extended to `model_type gemma4_unified`; `eoa_token_id` ← `eoa_token_index` alias (12B names it `_index`).
- **Processor stack vendor** *(old 046)*: all four `Gemma4Unified{Processor,ImageProcessor,AudioFeatureExtractor,VideoProcessor}` classes vendored from transformers main + AutoProcessor/AutoImageProcessor/AutoFeatureExtractor/AutoVideoProcessor registration — tx 5.6's AutoProcessor otherwise falls back to a bare GemmaTokenizer. All modalities preserved.
- **Hybrid-SWA KV routing** *(old 047)*: add the unified arch to `is_hybrid_swa_model` / `get_hybrid_layer_ids` / `is_hybrid_swa_compress` — the 12B has mixed per-layer-type KV (sliding 256-dim×8kv, global 512-dim×1kv); the uniform-pool fallback crashed `store_cache` on first prefill (`expected 2 but got 1`). *(Same registration has since landed in main's model_config.)*
- **Processor `__call__`** *(old 048)*: tx 5.10's base `ProcessorMixin.__call__` expands `<|image|>` into N soft tokens; tx 5.6's does not → `split_with_sizes expects 256 got [1]` scheduler crash on first image. Inlined an expansion `__call__` modeled on the 26B's processor.

Bring-up status (2026-06-07): ships as a full omni model — text + reasoning + tool-call + vision verified; int4 AWQ (RTN-from-QAT) at TP=2 awq_marlin, MMLU 80 / HumanEval 95 / needle 100 / 256K tool-use 100%→95K / ~42 tok/s. Open item: KV caps at 102K full / 81K swa — true 256K needs swa-pool/mem-fraction tuning (decode/KV track). On rebase: model graft + arch-lists come from main; the tx-5.6 vendor shims evaporate once the env moves to tx ≥ 5.10.

#### 050 — gemma4-unified-video-token-count (2026-06-10, 1 LOC, sprint A1-V)
First video request to the 12B killed the scheduler: `split_with_sizes expects sum 768 ... got [64]`. The vendored video processor (patch 046, verbatim from transformers main) declared `num_soft_tokens_per_video = merged_patches.shape[1]` — the **per-frame** merged-patch count — while `_embed_patches` emits every frame's non-padding patches (12-frame validator video × 64/frame = 768). `__call__` therefore expanded 64 placeholders against 768 embeddings. Fix: declare the all-frames total (`shape[0] × shape[1]`); per-frame `-1` padding is still dropped model-side. **Live-verified: video crash → PASS ("a red circle moving vertically"), 12B = 5/5 full omni.** Found because the A1 ratio-sweep probe battery exercised video where the bring-up had not. Transformers main likely carries the same defect — upstream report goes there.

#### 053 — evs-video-combined-path-routing (2026-06-16, v0.5.13, Nemotron-Omni video, cross-team port)
Nemotron-Omni video uses EVS (Efficient Video Sampling, `video_pruning_rate=0.7`): the embedding func returns an `EVSEmbeddingResult` (pruned video embeddings + per-frame token counts) that only the **combined** chunked-prefill path unwraps (`.embedding` + `redistribute_pruned_frames_placeholders`). But `_get_chunked_prefill_embedding` chose the per-image path on `all(len(item.offsets)==1)` alone — a single-offset EVS video matched → went to `_get_chunked_embedding_by_item` → `.reshape()` on the result object → `AttributeError: 'EVSEmbeddingResult' object has no attribute 'reshape'` → scheduler abort (the whole server dies on the first video request). The code's own comment already said "Fall back to combined path for non-split items **or EVS**" — the EVS half was never implemented. Fix: `is_per_image` also requires `not any(isinstance(item, EVSDataItem) ...)`. **Cross-team port of R9700's CANDIDATE 057** (their FP8 Nemotron hit the identical bug 2026-06-15; generic `mm_utils`, arch/quant-agnostic). **Live-verified: video crash → PASS ("a red circle moving right"), Nemotron-Omni = 6/6, server stays alive, 0 reshape errors.** Upstream-PR-worthy.

#### 054 — ministral3-v0513-keyword-super-init (2026-06-16 … RETIRED 2026-07-12: upstreamed in v0.5.15, which keyword-binds + threads `start_layer` natively)
The v0.5.13 default-flip's devstral fleet-smoke crashed at load: `AttributeError: 'str' object has no attribute 'get_quant_method'` in `LlamaAttention.__init__`. Root cause: v0.5.13 inserted a `start_layer` positional param into `LlamaAttention.__init__` (after `num_kv_heads`/`layer_id`) **and** `LlamaDecoderLayer.__init__` (after `layer_id`), but upstream `models/ministral3.py` (unmodified — not a 3090 graft) still calls `super().__init__()` **positionally** in `Ministral3Attention` (line 42) and `Ministral3DecoderLayer` (line 99). The inserted param shifts every later positional arg by one, so the trailing `prefix` string lands in the `quant_config` slot → the model loader calls `.get_quant_method` on a str. Devstral loads via `LlavaForConditionalGeneration → ministral3`, so it hit both. Fix: keyword-bind the super() args (`quant_config=…, prefix=…`, etc.) in all three Ministral3 super calls (Model's was correct since `LlamaModel` didn't change, bound anyway for robustness). **Live-verified: devstral 3/3 (basic+tool+vision) on v0.5.13.** Upstream `ministral3.py` bug — PR; **cross-team R9700 patch 064** (same fix, found on their resweep). Lesson reinforced: a default-stack flip MUST smoke at least one model per *distinct arch family* — Gemma/Qwen/Mamba all passed but the Mistral3 family had its own upstream breakage.

### MoE runner activation coverage

#### 017 — moe-gelu-activation (merged 2026-06-10; members 2026-04-30)
Gemma 4 MoE uses `gelu_pytorch_tanh`; every fused-MoE chokepoint asserted SiLU-only. Relax to `silu|gelu` + dispatch on `runner_config.activation` across **both** runner stacks: the MoeWNA16 Triton runner *(old 017)* and the Marlin stack — `fused_marlin_moe`, the MoE Marlin runner, `compressed_tensors_wNa16_moe`, `marlin_utils` *(old 021)*. Kernel-form check: `sgl_kernel.gelu_and_mul` matches `gelu_pytorch_tanh` to 0.00012 in FP16 — the kernels were always capable, only the asserts blocked routing. Loads clean across silu (Qwen3 MoE/Coder/REAP) and gelu (Gemma 4). Main still asserts SiLU-only in both stacks — PR candidate.

#### 052 — moe-wna16-nongated-moe (2026-06-16, v0.5.13, the Nemotron-Omni int4 unblock)
NemotronH's MoE is **non-gated** (squared-ReLU): experts have a single `up_proj` (no gate), `moe_intermediate_size=1856`. `moe_wna16` (the int4 weight-only MoE path) hardcodes the gated layout — `create_weights` builds the fused `w13` gate_up at `2*intermediate_size`, the loader writes `up_proj` at the `shard_size//2` (= gate) offset, and `apply()` asserts `activation == "silu"`. On a non-gated checkpoint this overflows: `start(1856)+length(1856) exceeds dimension size(1856)`. Fix: read `moe_runner_config.is_gated` (FusedMoE already carries it), set `gate_up_mult = 2 if is_gated else 1`, size the three `w13` tensors (`qweight`/`scales`/`qzeros`) by `gate_up_mult * intermediate`, load `up_proj` at offset 0 in the non-gated `w13_qzeros`/weight branch, and relax the assert to `not is_gated or activation == "silu"`. SGLang #21149 added non-gated only to the Marlin/CUTLASS/Triton *runners*, never `moe_wna16`; `awq_marlin` can't take Nemotron's experts anyway (`1856 ∤ 128` = `min_thread_k`). **Live-verified: Nemotron-Omni AWQ loads on `moe_wna16` TP=2 and serves 6/6 caps.** Backward-compatible — gated models default `is_gated=True`, unchanged. ⚠ Touches the same `apply()` assert region as the pending 017 GELU regenerate (merge to `not is_gated or activation in (silu,gelu)`). PR candidate (general non-gated AWQ-MoE support).

### Kernel correctness & precision

#### 003 — deltanet-triton-dtype-fix (51 LOC)
DeltaNet `causal_conv1d` loads `conv_state` without casting to the activation dtype — conv_state may be bf16 while activations are fp16 under AWQ. Cast at load to match the else-branch. Still absent in main.

#### 011 — triton-attention-fp32 (R9700 backport, 129 LOC)
Triton attention kernels accumulate `e_max`/`e_sum`/`re_scale` in BF16 and call `tl.dot()` without `out_dtype=tl.float32` → 15% mean error vs FP32 after 128 KV tokens, compounding catastrophically over 60 layers. RDNA4 and Blackwell SM12.x hit this; Ampere/Hopper mostly tolerate it. FP32 casts throughout decode + extend kernels. We serve FlashInfer (FP32 internally) for most models, so this bites only models forced onto Triton attention (Gemma 4 head_dim=512 path; 64-layer dense candidates). Main still accumulates bf16 — PR candidate with cross-vendor impact.

#### 012 — sliding-window-decode-fix (R9700 backport, 36 LOC)
`window_kv_offsets` was captured then discarded at `triton_backend.py:278` — SWA decode computed on full-pool indices instead of the window slice. **Main has reworked the whole plumbing (offsets flow through CUDA-graph capture + the index tuple) — drop at next rebase.**

### Ampere (sm_86) enablement

#### 005 — ampere-fp8-triton-fallback (46 LOC)
FP8 KV cache on sm_86: Triton emits `fp8e4nv`, unsupported on Ampere → PyTorch fallback path. FlashInfer handles FP8 KV for `head_dim ≤ 256`.

#### 007 — ampere-deltanet-kernel-tuning (85 LOC)
DeltaNet fused-recurrent BV=64 tuning for sm_86 — default `BV≤32, num_warps=1` under-utilizes the 3090; sweep found **1.57×** (0.018→0.011 ms/layer). Site-specific tuning; main keeps the generic config.

### Serving robustness / agentic

#### 034 — sampler-inf-detection (R9700 backport, 2026-05-13, ~13 LOC)
Extends `--enable-nan-detection` to catch ±Inf logits parallel to the NaN branch in `Sampler._preprocess_logits` — `softmax(+Inf)` → NaN cascade → invalid `multinomial` index → `gather` fault, with the NaN check only firing after softmax destroyed the Inf-vs-NaN signal that bisection needs. Replaces +Inf→`+1e4`, −Inf→`−1e4`; escalates to ValueError under `SGLANG_IS_IN_CI=1`. Verbatim from R9700 commit `ec1cf36`.

#### 049 — tp-load-timeout-cold-cache (R9700 graft of their 048, 2026-06-10, 1 LOC)
Big-model TP=2 loads on a cold page cache exceed upstream's 480s `UNBALANCED_MODEL_LOADING_TIMEOUT_S` rank-skew window — the slower rank gets killed mid-load. Bites 30–62 GB checkpoints when NVMe is contended (bake-off docker I/O); directly de-risks the queued Nemotron-3-Nano-Omni 62 GB BF16 TP=2 smoke. 480 → 1800s; genuinely wedged ranks still time out. Numbered 049 (not a reuse of retired 044–048) to keep historical references unambiguous.

#### 041 — devstral-toolcall-omission-recovery (R9700 graft, 217 LOC)
Devstral-Small-2 / Mistral-3 intermittently emit the compact tool call **without** the leading `[TOOL_CALLS]` special token (`tool_name[ARGS]{...}`), especially at temperature — the whole call leaks as assistant text, the agent sees no tool call, and the episode ends with an empty diff. Recovery only fires when the identifier before `[ARGS]` is a *known tool* and the JSON payload parses complete — no false positives on prose. Directly material to the SWE-bench bake-off (empty-diff class). Main still leaks — PR candidate.

#### 057 — mistral-common-backend-optout (2026-07-12, v0.5.15 flip validation catch, ~20 LOC)
transformers 5.12.1 (v0.5.15's pin) silently routes Mistral-family checkpoints shipping `tekken.json` to the new `MistralCommonBackend`, which never parses special tokens out of text — by design (anti-injection). SGLang's chat path renders the jinja template to a string and encodes it, so `[INST]`/`[/INST]`/`[TOOL_CALLS]` entered devstral prompts as plain text: `<s>[INST]` → `['<','s','>[','IN','ST',']']` instead of `[1, 3]`. First sweep pass: needle 0.0 at every depth (incl. 1K), HE 80%→48%, tool_call dead, turn-boundary blow-through — with the basic probe still green, so nothing short of a quality probe catches it. Fix in `get_tokenizer`: reload with `fix_mistral_regex=False` when the first load returns `MistralCommonBackend` (the kwarg's *presence* is transformers' escape hatch back to `TokenizersBackend`), and skip `_resolve_tokenizers_backend` for the rerouted result. Token ids verified bit-identical to tx 5.8.1; devstral re-probe restored full parity (needle 0.80 = v0.5.14 exactly, tool 3/3, MMLU +2). Lesson attached to the 054 one: a stack flip must fleet-probe **quality**, not just boot+basic — the two Mistral-family breaks in two consecutive rebases were both invisible to smoke tests.

### New-arch grafts

#### 042 — cohere2-moe-loader-graft (2026-06-07, 625 LOC, new file; +2 local fixes 2026-06-11 — no longer verbatim)
`models/cohere2_moe.py` grafted from sglang main: `Cohere2MoeForCausalLM` (128-expert fine-grained MoE thinking+agentic coder — **CohereLabs/North-Mini-Code-1.0**, released 2026-06-05 from the BLS-Mini-Code preview name). Two correctness fixes layered on the graft (both upstream bugs — PR candidates):
1. **Enum-member drift (`RoutingMethodType.Sigmoid`/`SigmoidRenorm`)** — added upstream after v0.5.12; first real boot died at layer construction. Fixed with a `getattr(..., Unspecified)` shim (the member is runner-selection metadata for the flashinfer-trtllm path, never used on sm_86; real sigmoid routing is the graft's own `custom_routing_function`). The AST constructor-drift check missed it — **lesson: also grep `Enum.Member` references against the target tree.**
2. **Dense-prefix-layer NaN (R9700 cross-team catch 2026-06-11).** The graft read `getattr(config, "first_k_dense_replace", 0)` at TWO sites (MoE-block selection + attention `force_rope`), but `Cohere2MoeConfig.__post_init__` *pops* that key (it derives `mlp_layer_types` and never stores the int) → getattr returns 0 → (a) every layer builds the sparse MoE block, so the dense prefix layer L0 loads dense BF16 weights against uninitialized experts → **NaN logits on first forward** (HSAIL 0x1016 on gfx1201; warmup NaN-detect on Ampere); (b) L0 silently loses `force_rope` → NoPE instead of RoPE. Both now read the canonical `mlp_layer_types[layer_id] == "dense"` (with a `first_k_dense_replace` fallback). R9700 found and fixed (a) on their FP8 WMMA path (which reaches the MoE forward where our sm_86 e4m3 wall blocked us earlier); we additionally fixed (b), the `force_rope` site they left. GPU-free validated: dense=[0]/sparse=48, force_rope[L0]=True for North's config. **This makes 042 no longer byte-verbatim with upstream — it carries 2 local correctness deltas until upstream fixes them.**

#### 051 — cohere2moe-256k-enablement (2026-06-11, 3 files + 1 vendored config)
Two halves, one bring-up (North-Mini-Code on v0.5.12):
1. **Vendored `Cohere2MoeConfig`** (`configs/cohere2_moe.py`, from transformers main, relative→absolute imports only) + `_CONFIG_REGISTRY`/`AutoConfig` registration — tx 5.6.0 predates `cohere2_moe`, so the checkpoint config didn't parse at all. Evaporates once the env's transformers knows the model type (gemma4_unified precedent, patch 043).
2. **Hybrid-SWA pool classification** (`model_config.py`: `hybrid_swa_archs` + a `layer_types`-derived branch in `get_hybrid_layer_ids`) — North is 1:3 full:sliding (window 4096, 13 global / 36 sliding of 49). Without the split SGLang sizes the KV pool full-attention for all 49 layers (~98 KB/token at fp16) and 256K can't fit beside the weights on 24 GB cards; with it the full pool costs ~26 KB/token and 256K is trivial. **Upstream main lacks this classification too — PR candidate** (the GptOss branch is the same pattern).
GPU-free validation green: config parses through the registry (49L/128e/sigmoid/500K), hybrid split 36/13 (full ids = every 4th layer), model imports. GPU boot validation reached the MoE forward — and hit the **sm_86 fp8e4nv wall** (below).

**North-fp8-on-Ampere verdict (2026-06-11): blocked.** The official fp8 checkpoint (CT `float-quantized`, W8A8-dynamic per-channel) loads and constructs, but the triton fused-MoE kernel can't compile `fp8e4nv` on sm_86 (`supported fp8 dtypes are ('fp8e4b15', 'fp8e5')` — the exact patch-005/A5 wall), and **no marlin W8A16-fp8 MoE scheme exists in v0.5.12 or upstream main** (the CT fp8-MoE scheme routes AITER/triton only; `compressed_tensors_w8a16_fp8.py` covers linears, not MoE). The Ampere ship is the **AWQ-int4 build from upstream BF16** (calibration backlog №1); the fp8 checkpoint is the R9700 ship (AITER + native fp8 WMMA + 32 GB). A from-scratch W8A16-fp8 marlin MoE scheme is possible but is real kernel-integration work — parked unless the AWQ path disappoints.

### Build pragmatics

#### 037 — skip-grpc-rust-ext (18 LOC)
v0.5.12 made the Rust gRPC ext (`sglang.srt.grpc._core`) a mandatory build step; it needs protoc, which this rig lacks, and we serve HTTP-only (zero imports on that path). Comment out the `setuptools-rust` ext-module table. Restore for gRPC serving.

---

## Historical (retired numbers)

Narratives for patches that no longer ship — kept because the *findings* still inform current work.

### 001 — upstream-sync (~3,000 LOC)
Cherry-picks from SGLang main needed at v0.5.10/11: Gemma 4, Qwen3.5/Qwen3-Next, Triton attention updates, pool_configurator. Superseded by the v0.5.12 rebase.

### 006 — awq-bf16-activation-support (15 LOC)
BF16 activations with AWQ dequant — needed for Gemma 4 (hidden_size=5376 → FP16 overflow at layer 2). Marlin accumulates FP32 so BF16 activations are safe. Folded upstream by v0.5.12.

### 008 — awq-moe-wna16-fallback (64 LOC)
`awq_marlin_moe_repack` doubles peak memory during repack (~7 GB/GPU on 128-expert MoE). `SGLANG_FORCE_MOE_WNA16=1` bypassed Marlin repack → MoeWNA16 Triton kernels. CT-format only.

### 009 — qwen35-moe-causalLM
Qwen3.5 MoE text-only CausalLM wrapper (logits processor + mrope) for the REAP-28B path. Superseded by the real CausalLM heads (current patch 035).

### 014 — gemma4-reasoning-parser (40 LOC)
`Gemma4Detector` cherry-picked from upstream PR #21952 (`--reasoning-parser gemma4`). In v0.5.12.

### 015 — ct-wna16-dequant-layout-fix
The CT WNA16 dequant fallback assumed `[in//pack, out]` layout; real CT layout is `[out, in//pack]` — silent garbage on TP-sharded RowParallel layers. Rewrote to keep native `[out, in]` orientation. Folded by v0.5.12.

### 016 — ct-moe-gelu-triton-route (47 LOC)
`SGLANG_FORCE_CT_MOE_TRITON=1` routed CT MoE to the Triton scheme (gelu-capable) on CUDA. Superseded by current 017's direct activation dispatch.

### 019 — qwen3_5-moe-vl-config-dataclass-and-model-init (60 LOC)
Three-part fix for `Qwen3_5MoeForConditionalGeneration` on Python 3.13 + transformers 5.x: model-side dict-wrap of `vision_config`/`text_config`; explicit `__init__` on config subclasses (tx 5.x auto-dataclass-decoration replaced inherited inits — `norm_topk_prob` etc. never set); drop `model_type` from kwargs when re-wrapping. Folded by the v0.5.12 rebase; the *findings* still gate any transformers upgrade (see current 035/043).

### 020 — gemma4-clippable-linear-shim (289 LOC, v2 = upstream port)
`ClippableLinear` for gemma4.py imports; v1 alias-shim hit a 2-tuple/plain-tensor signature mismatch, v2 ported the real upstream class. In v0.5.12.

### 022 — gemma4-causal-dedup-entry-class (24 LOC)
Removed the dual registration of `Gemma4ForConditionalGeneration` (gemma4_causal.py + gemma4_mm.py) that asserted once 020-v2 made gemma4_mm import-clean. In v0.5.12. R9700 independently hit the same shape later.

### 030 (original) — gemma4-mm-per-expert-awq-loader-v0510 (deleted 2026-05-09)
v0.5.10 sister of current 028 at old line offsets; redundant once the rig moved to v0.5.11+. The number was later reused for fused-moe-w2-presharded-detect.

### 010 / 013 / 027 / 032 / 033
Dropped during upstream rebases; 013 (DeltaNet cache wiring) lives on in the M4 repo. Narratives in `git log` if ever needed.

---

## Cross-team findings (3090 ⟷ R9700)

The sister RDNA4 project runs the same SGLang-rebase strategy. Findings that produced patches or changed how we ship are here; day-to-day sync happens in the two READMEs.

- **BF16 attention precision** affects every new architecture (RDNA4, Blackwell SM12.x). Fix: FP32 accumulation in the online softmax (patch 011).
- **AWQ calibration silently breaks thinking and vision.** Quants calibrated on plain text (Open-Platypus, WikiText2, c4) lose `<think>` stop-token behavior and vision-language alignment. Rule: every new quantized model must validate (a) an image+text roundtrip and (b) a thinking-tagged generation that cleanly terminates, before launch. `scripts/eval/validate_chat_template.py` (static) + R9700's `validate_capabilities.py` (live) are the pre-flight gates.
- **Recommended calibration datasets** (reasoning + vision preserving): `a-m-team/AM-Thinking-v1-Distilled`, `glaiveai/reasoning-v1-20m`, `LLaVA-Instruct-150K`, `AI-MO/NuminaMath-CoT` (+9.81% GPTQ accuracy vs WikiText2 in R9700 measurements). Recipe builder: `scripts/quantize/calibration_datasets.py` (`thinking_text`, `thinking_vision`, `code_vision`, `code_thinking`).
- **Chat template is a silent bug magnet.** Devstral community AWQ's template emits BOS → `<unk>`; Gemma 4 community weights ship with no template at all; Qwen3 family needs `temperature ≥ 0.3` to avoid greedy-decode repetition loops. Always render the template with and without `enable_thinking`, and verify `chat_template is not None`.
- **AutoRound > GPTQ > AWQ for INT4 quality** — Intel AutoRound (arXiv 2309.05516) uses SignSGD for 200 iterations to jointly optimize rounding offsets and clipping ranges. Can export to both GPTQ and AWQ formats. [RedHatAI reports 99.4%+ quality](https://huggingface.co/RedHatAI/gemma-3-27b-it-quantized.w4a16) on CUDA.
- **DeltaNet layers must stay BF16.** INT4 noise accumulates through the recurrent state `S(t) = g*S(t-1) + delta` and destroys quality. Architectural limit.
- **DeltaNet replication mandatory for TP=2.** Qwen3.5-27B: TP RowParallelLinear splits matmul `W_0@x_0 + W_1@x_1` which differs from `W@x` by ~1 ULP in FP16; the recurrent state compounds it across layers. Fix: replicate all DeltaNet + MLP layers (`tp_size=1`), SSM state `tp_world_size=1`. Costs 19 GB/GPU weights replicated, which is why Qwen3.5-27B is context-limited to 32K.
- **Community AWQ fails for DeltaNet** on both rigs. Self-calibrate with GPTQ + CT→AWQ.
- **`save_pretrained` OOMs on 32B+ models with default `max_shard_size="5GB"`** (R9700 finding 2026-05-03, commit `e28c43b`). VL-32B died at "Writing model shards: 0%" with exit=137 after 27.5h of GPTQ — 62 GB RAM + 68 GB swap insufficient for the 5GB shard buffer alloc. Two-layer defense: (1) `max_shard_size="2GB"` on final save; (2) per-layer checkpoint hook on `LifecycleCallbacks.sequential_epoch_end` writes a snapshot every 16 subgraphs, keeping the last 2 (~34 GB disk bound). Worst case the previous snapshot ships at lower quality. Inherit for any future 30B+ recal on this box.

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

Reusable rules for Gemma 4 on this rig:

- **Use BF16, not FP16.** FP16 overflows at layer 2 (hidden_size=5376). The SigLIP vision tower NaNs in FP16 attention softmax above 65504 — BF16 is mandatory for vision.
- **head_dim=512 is FlashInfer-unsupported on sm_86.** Route through `--attention-backend triton` or `--attention-backend torch_native`; both work, triton is faster.
- **FP8 KV on the triton-forced path uses `fp8_e5m2`, NOT `fp8_e4m3`.** sm_86 triton can't emit `fp8e4nv` (= e4m3) — the compile error itself lists `fp8e5` (e5m2) as the supported FP8. The triton attention kernels carry no fp8-type hardcode (the type is inferred from the KV buffer), so `--kv-cache-dtype fp8_e5m2` compiles and halves KV where e4m3 fails. e5m2's 2-bit mantissa (vs e4m3's 3) held tool-use retrieval **1.0 to 258K true tokens** on gemma-4-31B (sprint A5), decode unchanged. This took the 31B 130K→260K KV pool = ~256K on 24 GB. (head_dim ≤ 256 models keep e4m3 via FlashInfer — higher quality, and they're not triton-bound.)
- **Self-calibrate, don't ship community CT quants.** Cosine similarity vs BF16 base was 0.845 on q_proj for community CT — produces garbage.
- **Embed chat template into `tokenizer_config.json`.** Community Gemma 4 weights ship with no template.
- **MoE+dense parallel block:** `Gemma4DecoderLayer` MUST pass `quant_config=None` to the dense `Gemma4MLP` only when the recipe kept it BF16 (patch 023 detects this). Vision tower + image/audio projectors also stay BF16 via `modules_to_not_convert`.

## FlashInfer head_dim support

| head_dim | FlashInfer (sm_86) | Models |
|:--------:|:------------------:|--------|
| 64-256 | Supported | Qwen, Devstral |
| **512** | **Not supported** | **Gemma 4** |

Possible unblock paths: SDPA fallback (the text path boots), [FFPA kernels](https://github.com/DefTruth/ffpa-attn-mma), TRTLLM FMHA, or llama.cpp for Gemma 4 serving (reported 80-110 tok/s).
