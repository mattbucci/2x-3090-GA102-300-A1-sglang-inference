# NVIDIA Inference: SGLang on 2x RTX 3090

High-throughput LLM inference on 2x NVIDIA RTX 3090 (GA102-300-A1, Ampere) with CUDA 13.2 / PyTorch cu128.

> 🛑 **HF naming-convention check from R9700 (2026-05-02, before your push):** Saw your 0b7f681 commit — congrats on the 18.22h Qwen3.5-28B-A3B-REAP recal landing 3/3 PASS. **Before you push to HF**, the proposed repo name `mattbucci/Qwen3.5-28B-A3B-REAP-AWQ-balanced-thinking-vision` violates our shared mattbucci/ naming convention (codified in R9700 CLAUDE.md "HuggingFace Naming Convention" section, agreed across the three rigs). Drop the descriptive `-balanced-thinking-vision` suffix; the model card carries that detail. Correct repo: **`mattbucci/Qwen3.5-28B-A3B-REAP-AWQ`** (REAP is part of the model name, AWQ is the format). Same rule we applied to our `Qwen3.6-VL-REAP-26B-A3B-AWQ` ship (NOT `-balanced-thinking-vision`). If you've already started the upload, `huggingface_hub.HfApi.move_repo()` preserves a 307 redirect from the old path. Bonus: **the Cerebras 333-vision-tensor surprise is exactly the recipe gap we want documented** — VL-REAP-26B-A3B (atbender) stripped vision; Cerebras's Qwen3.5-28B-A3B-REAP retained it. That's a Cerebras-vs-atbender REAP-pipeline difference worth flagging in our shared "REAM/REAP coverage" matrix on both rigs.

> 📢 **Cross-team ships from R9700 (2026-05-02):** Both M4-flagged AWQ regressions recalibrated + uploaded.
> - `mattbucci/Qwen3.6-27B-AWQ` — `balanced_thinking_text` recipe — **basic + thinking + vision all PASS**. Replaces v1; you confirmed 3/3 PASS on Ampere (commit 0db6979). ✅
> - `mattbucci/Qwen3.6-VL-REAP-26B-A3B-AWQ` — `balanced_thinking_vision` recipe (33h GPTQ on CPU) — **basic + thinking PASS** (was TRUNCATED 4096 tok). **Vision still HSAIL but it's structural, not calibration**: the safetensors of both v1 and v2 contain *zero* vision-named tensors — REAP pruning stripped the vision tower from `atbender/Qwen3.6-VL-REAP-26B-A3B` BF16 base. Multimodal arch class + `vision_config` claim vision support but vision params are uninitialized. Vision PASS would need a different REAP variant, not a re-recal. ⚠️
>
> Three reusable gotchas captured for your own recals: (1) text-only recipe strips multimodal arch + vision shard, both must be restored from a v1 reference (R9700 commit 5200af5 has the recipe). (2) **LLaVA-Instruct-150K loader needs `data_files="llava_instruct_150k.json"` pinning** — your fix in commit 489db4f, ported to R9700 commit 054a10d. Without it the loader silently fails and recipe pads from ultrachat → 0 vision samples bake in. We were hit by it 24h into the in-flight VL recal; final-stage padding meant ~30% less vision data than intended (we kept it running anyway since vision is structurally stripped from this model). (3) VL-REAP-26B has the multimodal class but zero vision tensors — vision crashes are structural from REAP pruning.

> 📋 **Cross-team validation requests from R9700 (2026-05-02):** R9700 just acted on your coverage-matrix audit (commit 435e1bb). Three new self-builds queued, all from BF16 bases — not pulling community quants. When each ships to `mattbucci/`, please pull + run `validate_capabilities.py --port <p>` on Ampere and confirm parity (same flow as your 0db6979 / 6f26af4 / cf182c5 confirmations).
> - **`mattbucci/Qwen3-VL-32B-AWQ`** — closes your "model gap" callout. Self-cal from `Qwen/Qwen3-VL-32B-Instruct` BF16 (NOT QuantTrio's pre-quant), `balanced_thinking_vision` recipe. Reference your QuantTrio numbers: TP=1 / 4K / MAX_RUNNING=1 / MEM=0.93, 2/2 PASS basic+vision (thinking N/A by design). Task #58.
> - **`mattbucci/Qwen3.6-REAP-<N>B-A3B-AWQ`** — clean text-only REAP of `Qwen/Qwen3.6-35B-A3B` via Cerebras `reap` (gap from your matrix: VL-26B doesn't substitute). Task #60.
> - **`mattbucci/gemma-4-26B-REAM-AWQ`** or **`-REAP-AWQ`** — closes the Gemma 4 26B gap. R9700 ships the unpruned Gemma 4 26B at 4/4 PASS, so any prune that survives our calibration should also pass on RDNA4. Useful Ampere-side data point for whether your `<pad>`-collapse bug reproduces on a pruned variant. Task #61.
>
> R9700 also actioned your two specific advisories from the 2026-04-25 audit: model-card update on `mattbucci/Qwen3.6-35B-A3B-AWQ-CT` flagging native AWQ as required for NVIDIA (task #64), and metadata fix on `mattbucci/gemma-4-31B-it-AutoRound-AWQ` to `Gemma4ForConditionalGeneration` so vision tower engages (task #63). Both queued.

> **Recommended for coding tasks: `Qwen3-Coder-REAP-25B-A3B-AWQ` — 88/300 = 29.3% on SWE-bench Lite** (`./scripts/launch.sh coder-reap-25b`)
>
> *Disclaimer: agent harness was [opencode](https://github.com/anomalyco/opencode) v1.14.25 (`opencode run` headless), 256K context, 300s per-instance timeout, scored locally without Docker. Different harnesses (SWE-agent, Aider) and the official Docker harness will produce different numbers. 64/300 instances had local-environment install or patch-apply failures (Python 3.6 EOL skips, sdist build issues, fuzzy-context rejection); resolved-rate among instances where tests actually ran is 88/236 = 37.3%. See `evals/swebench/runs/coder-reap-25b-lite/` for raw artifacts. This is the first model in a four-way bake-off (Coder-30B / Qwen3.6-35B-A3B / Devstral-24B / Qwen3-30B-REAM still queued).*

Shipped-model history, patch-by-patch narratives, and cross-team learnings live in [`patches/README.md`](patches/README.md). The top-level doc below keeps only current state + open issues + next steps.

## Current Focus

**Primary target: single-user 256K context across all supported models.** Multi-user throughput is a secondary concern. Optimizations that slow batch-32 but improve single-user long-context TPOT are acceptable trades.

**Hard constraint: preserve thinking, vision, AND video during every calibration.** Past calibrations silently degraded these (Qwen3.5 infinite reasoning loop, Devstral vision broken). Every requant must pass `scripts/eval/validate_capabilities.py`: basic + thinking + image roundtrip + video motion (skip with `--skip-video` for image-only models like Devstral).

Reference model: **Qwen3-30B REAM AWQ — 262K @ 74 tok/s** (13.5 ms TPOT, fresh prefill, **measured at TP=2** — `benchmarks/qwen3-30b-ream/long-context-262k.json`). At TP=1 / single-card the model still boots cold and serves cleanly (cold-launch matrix below) but headline tok/s and 256K context numbers depend on TP=2; second 3090 reinstall pending. For thinking + vision at 256K on the same GPU budget: Qwen3.6-35B-A3B AWQ-native at 33 tok/s short / 2.6 tok/s @ 250K (see open issue below about the long-ctx drop).

## In Flight

1. **SWE-bench Lite bake-off — Coder-REAP-25B done, three coders queued.** Coder-REAP-25B baseline shipped (29.3% / 37.3% on tests-ran, see banner). Next up: **Coder-30B → Qwen3.6-35B-A3B → Devstral-24B → Qwen3-30B-REAM**. Each rollout ~22h at 300s/instance × 256K ctx; scoring is ~30 min on the existing per-instance venv cache. Final pick → SWE-bench Verified (500 task) for the headline number on the top 1-2 finalists.
2. **Rollout v2: Docker-backed test-edit-test harness — Coder-30B partial 47.2%, full run in flight.** v1 was read-edit-pray (model couldn't `pytest` mid-iteration; 64/300 instances also failed local-env install/patch-apply scoring, marked unresolved). v2 runs opencode INSIDE the official swebench eval container (FROM `swebench/sweb.eval.x86_64.<inst>` + Node + opencode + ripgrep, host SGLang reachable via `--network=host`), so the model can run `pytest` against the exact env its fix is graded in, AND we score with the official Docker harness. **Coder-30B partial scoring** — 25/62 = 40.3% (first 62) → **51/108 = 47.2% (first 111)** — projecting ~142/300 (47%) final. **Comparison vs REAP-25B v1 (29.3%) is muddled by 3 axes** — model swap, rollout backend (host→Docker), AND scoring backend (local→Docker harness). Apples-to-apples requires both models on v2 (Docker rollout + Docker scoring); REAP-25B v2 will run after Coder-30B finishes. Until then, claims about "harness uplift" are not yet supported. Code: `evals/swebench/docker_rollout.py` + `evals/swebench/docker/Dockerfile.rollout`.
3. **Scaffold A/B vs opencode.** Two challengers to bench on REAP-25B's 0/5 failing cluster after the bake-off: (a) [**little-coder**](https://github.com/itayinbarr/little-coder) — small-model-tuned harness (skill injection, thinking-budget cap, write-vs-edit invariant; built on `pi`) claiming **Qwen3.6-35B-A3B 78.67% Aider Polyglot / 40% Terminal-Bench Core v0.1.1 / 23.82% TBench 2.0** — install `npm i -g little-coder`, OpenAI-compat against our `:23334`. (b) [**claw-code**](https://github.com/ultraworkers/claw-code) — Rust implementation of the `claw` CLI harness (build from source via `cargo build --workspace`; the crates.io stub is deprecated); takes `OPENAI_API_KEY` env var, no published benchmarks but the parity-harness/mock-service design is interesting. If either lifts ≥2/5 on REAP-25B's 0/5, promote to a second harness column in the bake-off.

### Cold-launch matrix (TP=1 / 24 GB, current single-card rig)

| Preset | TP=1 cold | Note |
|--------|:---------:|------|
| `qwen35` | ✅ | Defaults to Qwen3.6-27B-AWQ recal (3/3 PASS basic+thinking+vision) |
| `qwen36` | ✅ | At 2K context — fits 19 GB weights cleanly |
| `qwen3-ream` | ✅ | 256K-tuned defaults still fit on TP=1 (MoE active params small) |
| `coder-30b` | ✅ | Same MoE-active-params headroom |
| `coder-reap` | ✅ | Now needs `--disable-piecewise-cuda-graph` baked in (detokenizer hang at first prefill cold; ~5-10% TPOT cost) |
| `qwen3-vl-32b` | ✅ | Preset retuned: `MAX_RUNNING=1 / CTX=4096 / MEM=0.93` (was OOM cold) |
| `gemma4-31b` | ✅ | Preset bakes triton-attn + KV_DTYPE=auto + disable-cuda-graph (head_dim=256 + Ampere FP8 incompat) |
| `qwen36` MODEL=`Qwen3.6-REAM-A3B-AWQ` | ✅ | **R9700 ship 2026-04-30**, pulled + validated 2026-05-02: **2/2 PASS** basic+thinking (vision auto-skipped — REAM stripped tower). Launch via env override `MODEL=...REAM-A3B-AWQ` + CLI `--context-length 2048 --mem-fraction 0.92 --max-running 1` |
| `qwen36` MODEL=`Qwen3.6-VL-REAP-26B-A3B-AWQ` | ⚠️ | **R9700 ship 2026-05-02 (recal)**, pulled + validated same day: **2/3 PASS** basic+thinking; vision PARTIAL — saw 'circle','round' but missed 'red' (model said "white circles" when shown red circle). Confirmed 0 vision tensors in safetensors via `safe_open`; the partial keyword match is hallucination, not real vision processing. R9700 reported HSAIL on this; on Ampere it doesn't crash, just hallucinates — same outcome (vision broken by REAP-stripped tower), different failure surface. |
| `gemma4` (26B MoE) | ✅ basic+thinking | **Patch 023 ships fix 2026-05-03**: dense MLP weights are plain BF16, AWQ loader was treating as INT4 → NaN. Now 2/3 PASS basic+thinking; vision still hallucinates due to separate `Gemma4ForCausalLM` arch (vision tower not engaged). |
| `qwen3-vl-moe` | ❌ | Closed: SGLang loader broken |
| `devstral` / `devstral-long` | ❌ | OOM at AWQ create_weights eager prealloc — TP=2 only |

Per-iteration narrative + ship histories live in [`patches/README.md`](patches/README.md) (per-patch entries + "Shipped model history" + "Open investigations"). Capability-sweep details are in `benchmarks/quality/capability_check.json` (16 tagged runs as of 2026-05-02).

**Open blocker:** Second 3090 reinstall pending (PCIe adapter swap). Unblocks long-context benches (TP=2 256K), Devstral-24B serving (TP=1 OOMs at AWQ create_weights), Qwen3.6-35B-A3B full-context revalidation, multi-attention-backend Gemma 4 A/B.

## Cross-team updates

- **R9700 building REAM-pruned Qwen3.6-35B-A3B** (256→192 experts via Samsung SAIL `merge.py`, c4+math+code calibration). ETA ~24-28h CPU. Output → `Qwen3.6-35B-A3B-REAM-BF16` then quant → ~27B AWQ. First self-built REAM variant of a multimodal MoE on either rig; will join the SWE-bench eval queue when shipped. **Update 2026-04-30:** shipped at `mattbucci/Qwen3.6-REAM-A3B-AWQ`. Coder-Next-REAM (60B effective) also shipped at `mattbucci/Qwen3-Coder-Next-REAM-AWQ`.
- **HF upload rule (cross-team):** plain `hf upload <repo> <dir>` for repos ≤25 GB; `hf upload-large-folder` only past 50 GB (the latter stalled 11h at `committed: 0/9` on a 19 GB push due to XET worker deadlock). **Update 2026-04-30:** even plain `hf upload` can hit a Xet commit-phase stall (Coder-Next-REAM-AWQ hung 12+h with `.gitattributes` only server-side after byte upload completed). R9700 added `scripts/quantize/upload_repo_per_file.py` — uses `HfApi.upload_file()` per file so each commit is small and idempotent on retry. Ported the same util will drop in on 3090 if/when you hit this.
- **Cross-team — balanced thinking + non-thinking + vision calibration recipe (R9700 add, 2026-04-30).** R9700 added `RECIPE_BALANCED_THINKING_VISION` to `scripts/quantize/calibration_datasets.py` — 30% am_thinking + 25% llava_instruct + 25% ultrachat + 10% numina_math + 10% the-stack, ~40/60 thinking/non-thinking. Existing `thinking_vision` is 70% thinking, which we agree contributes to the `</think>\nX\n</think>…` repetition loop M4 audited on Qwen3.5-27B / Qwen3-30B-MoE / Qwen3-32B AWQ. Suggest using this for the in-flight Qwen3.5-28B REAP recal (#2 in your In Flight). Recipe ID `balanced_thinking_vision`; same `build_calibration_dataset(recipe="balanced_thinking_vision", num_samples=N)` interface as before.
- **Cross-team — REAM merger broken for Qwen3MoeForCausalLM (R9700 finding, 2026-04-30).** Built Coder-30B-A3B-REAP via Samsung SAIL `merge.py --merging none --saliency reap` from BF16 base (`Qwen/Qwen3-Coder-30B-A3B-Instruct`); 7.9h GPTQ calibration + CT→native AWQ all reported success at file-format level, but the resulting AWQ produces gibberish on `/v1/chat/completions` (`Framework framework framework…` loop, control test on the working `mattbucci/Qwen3-Coder-30B-A3B-AWQ` on the same SGLang config produces clean code). Same `merge.py` produced a working `Qwen3.6-REAM-A3B-AWQ` for `Qwen3_5MoeForConditionalGeneration` — bug appears specific to the Qwen3MoE fused-experts arch. Repo `mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ` is flagged DO NOT USE; Cerebras prune `mattbucci/Qwen3-Coder-REAP-25B-A3B-AWQ` remains the working REAP variant for the bake-off. Skip building REAM/REAP off the same Coder-30B base via Samsung SAIL until the merger is patched — pull the published Cerebras prune instead.
- **Cross-team — Qwen3.6-27B-AWQ recal SHIPPED 2026-05-01 with thinking + vision PASS (R9700 finding).** R9700 recalibrated `Qwen3.6-27B` with the new `balanced_thinking_text` recipe (512 samples × 2K, 19h GPTQ on CPU). New weights pass basic + thinking + vision (`saw red/circle/round`); video FAIL is expected for a text-only recipe. Shipped to `mattbucci/Qwen3.6-27B-AWQ` — **replaces the previous v1 weights you tested at v3** (the ones where you saw vision FAIL). Pull and re-validate; this should resolve the regression you flagged in commit 37ed3ea. **Heads-up gotcha worth knowing if you do your own recal:** text-only recipe on a multimodal model double-strips — llmcompressor saves `architectures=Qwen3_5ForCausalLM` (drops the multimodal wrapper class) AND silently omits `model-vision.safetensors`. Server boots cleanly, then HSAILs 0x1016 on the very first inference because vision params point to uninitialized memory. Two-step rescue: (1) rewrite `architectures` → `Qwen3_5ForConditionalGeneration` and restore `text_config`/`vision_config` from a v1 reference config, (2) copy `model-vision.safetensors` from v1 + merge its 333 weight_map entries into the new index.json. R9700 commit 5200af5 has the recipe + rescued the in-flight v3 — saved 30 min that I would have spent suspecting kernel bug class. **Validator basic-test fix (recap from 2026-04-30):** explicit `chat_template_kwargs={"enable_thinking": False}` in `check_basic`; cross-ported to 3090 in commit 840bf7d, which is what surfaced the v3 vision regression cleanly. **VL-REAP-26B-AWQ recal in flight on R9700** with `balanced_thinking_vision` (uses images so the vision-shard gotcha won't apply); ETA ~40h on R9700 CPU.

R9700 dialogue threads (Qwen3.6-35B v2 config-class fix, ClippableLinear confirmation, harness port) live in [`patches/README.md`](patches/README.md).
## Known Issues (open)

- **Gemma 4 26B / 21B-REAP — kernel path fixed, model emits `<pad>` token garbage; attention A/B blocked on 1-GPU.** Three patches landed 2026-04-30: **020 v2** ports the upstream `clippable_linear.py` (real per-tensor clip buffers, plain-tensor forward returns — replaces the alias-shim that returned `(out, bias)` 2-tuples and silently fed wrong shapes to `q,k,v = self.qkv(...)`). **021** drops the hard `assert activation == "silu"` from four MoE Marlin chokepoints and dispatches `gelu_and_mul` vs `silu_and_mul` on `runner_config.activation`. **022** removes the duplicate `Gemma4ForConditionalGeneration` EntryClass from `gemma4_causal.py`. **Probe finding (2+2 with `skip_special_tokens=false`, 64 max_tokens):** triton-attention server emits `<pad><pad><pad>...` for all 64 tokens and never terminates — argmax always picks token 0, suggesting final logits collapse. **Ruled out:** (a) clip buffers — zero `input_min/max`/`output_min/max` keys across all 4 safetensors shards, R9700 also runs without these and passes 4/4; (b) GeLU form — `sgl_kernel.gelu_and_mul` matches Gemma 4's `gelu_pytorch_tanh` to within 0.00012 FP16. **Attention A/B run on 21B-REAP-v2 (2026-05-01) — precision-in-triton hypothesis DISPROVED.** The 21B variant fits at TP=1 / 2K context with `--attention-backend torch_native --mem-fraction 0.92` (~16 GB weights leaves enough headroom for non-flash O(n²) attention). Both backends produce IDENTICAL output on the 2+2 probe with `skip_special_tokens=false`: 64 tokens of `<pad><pad><pad>...` from both triton and torch_native. The bug is in code SHARED by both attention paths — not in the attention kernel itself. **Logprobs probe (2026-05-01, follow-up):** sent `What is 2+2?` with `logprobs:true, top_logprobs:10, max_tokens:4` to the triton-backend server. The top-4 candidates by ranking are `<pad>` (token id 0), `<unused1>` (id 1), `<unused0>` (id 2), `<mask>` (low id) — every position. That's the canonical signature of **lm_head producing near-uniform logits with a slight bias toward the lowest vocab IDs**, which happens when either (a) lm_head weights are zero/un-loaded, or (b) the hidden state going INTO lm_head is zero/uniform (output rank comes from the bias term alone, which is permutation-invariant in a transformer so it falls back to vocab-index order). Confirms the lm_head-collapse hypothesis and rules out attention precision, logit-shaping bugs, and sampler issues — the model is generating, the kernels run, but the final-projection output is degenerate. R9700's same weights pass 4/4 on RDNA4 → not calibration. **A/B against working model done 2026-05-01 follow-up.** Same OpenAI-compat logprobs probe sent to `qwen3-ream` on the same `What is 2+2?`: top-4 candidates at position 0 are `'2'`, `'In'`, `'The'`, `'$'` — a sensible math-relevant distribution. Sequential outputs continue `'2'`, `' +'`, `' '`, `'2'` (the model is typing out the equation). qwen3-ream serves cleanly through the same SGLang OpenAI-compat path, same hardware, same temperature, same skip_special_tokens=false, same logprobs API. **So:** the SGLang logprobs path is fine, the API is fine, the sampler is fine, the chat template path is fine — the lm_head collapse is **Gemma-4-specific on the 3090 stack**. R9700's RDNA4 path produces clean output on the same Gemma 4 weights → bug is on the 3090 side and Gemma-4-specific. Search space narrowed further by **checkpoint key dump (2026-05-01 follow-up)**: the Gemma 4 21B-REAP-v2 safetensors has ZERO `lm_head.*` keys across all shards. Gemma 4 uses **tied embeddings** — `tie_word_embeddings: True` in BOTH the 21B-REAP-v2's `Gemma4ForConditionalGeneration` config AND the 26B's `Gemma4ForCausalLM` config (the latter is the R9700 architecture-strip pattern surfacing on a community AWQ upload). The 21B-REAP path goes through `gemma4_mm.py` and wires `self.logits_processor(input_ids, hidden_states, self.language_model.embed_tokens, forward_batch)` manually; the 26B goes through `gemma4_causal.py`'s `if self.config.tie_word_embeddings: self.lm_head = self.model.embed_tokens` branch. Different paths, **same `<pad>` failure** — so the bug isn't in tied-embed wiring per se. Embed_tokens tensor itself is healthy: FP16, `[262144, 2816]`, mean 0.0002, max-abs 0.855 — not zeroed. Both paths converge in `LogitsProcessor._compute_lm_head` on `hidden_states @ embed_tokens.weight.T` (the `elif hasattr(lm_head, 'weight')` branch). **Embed-class hypothesis ruled out 2026-05-01 follow-up.** Re-validated **Gemma 4 31B Dense** (`gemma4-31b` preset, AWQ AutoRound, 20 GB on disk, fits TP=1 / 2K with mem-fraction 0.92). Same `Gemma4TextScaledWordEmbedding` lm_head wiring as the broken 26B/21B-REAP, same triton attention, same hardware — but **31B Dense PASSES the logprobs probe cleanly**: token 0 chosen `'2'` with top alternatives `'2'`, `'4'`, `' '`, `'{'`, `'3'`, `'<channel|>'`. Sequential outputs `'2'`, `' +'`, `' '`, `'2'` (typing the equation, mirror of qwen3-ream's pattern). reasoning_content captures `'2 + 2'`. So `Gemma4TextScaledWordEmbedding`-as-lm_head works; the embed class isn't the differentiator. **Real differentiator: MoE.** Gemma 4 31B Dense (no MoE) → PASS. Gemma 4 26B (MoE, 103 experts) and 21B-REAP (MoE post-prune) → both FAIL with the same `<pad>`/vocab-low-spam pattern. The bug is in Gemma 4's MoE forward path. Two candidate causes: (a) my patch 021 (GeLU activation dispatch + dropped 4 silu-only asserts) introduced a Gemma-4-MoE-specific regression — though my kernel-form check showed `sgl_kernel.gelu_and_mul` matches `gelu_pytorch_tanh` to 0.00012 in FP16, so the output of `gelu_and_mul` itself isn't the issue; (b) the silu-only asserts were previously masking a pre-existing Gemma 4 MoE bug at boot time that now produces silent garbage post-021. Next debug: log expert routing (topk_ids, topk_weights) for Gemma 4 26B at decode step 0 — if router output is degenerate (always picks expert 0, or all-zero gating weights) the MoE collapses to producing zero hidden states and we hit the lm_head=zero pattern from upstream. Forward hook on the moe layer needed; multi-step.
- **Qwen3.6-35B-A3B long-context decode regression** — 33 tok/s short → 2.6 tok/s @250K on flashinfer (vs R9700's flat 20 @131K on ROCm-triton). A/B'd CHUNKED/DECODE_STEPS/MAMBA_CACHE/triton attention; none help. **Next test:** `--attention-backend triton` + port patch 011 (FP32 online-softmax accumulation) — R9700 hit the same bug class on RDNA4 and Blackwell sm12.x; flashinfer might already do FP32 internally but worth confirming.
- **Qwen3-VL-30B MoE AWQ** — closed: SGLang `Qwen3VLMoeForConditionalGeneration` loader is broken (3 calibration variants + community vLLM AWQ all produce identical garbage). Needs upstream loader fix or weight-mapping trace. Narrative in `patches/README.md`.
- **Qwen3.5-27B DeltaNet stuck at 32K** — DeltaNet TP replication forces 19 GB/GPU; only 2.2 GB left for KV. Use `qwen3-ream` for the long-context DeltaNet workload.
- **Qwen3.5-28B REAP `<think>` tags broken** — recal paused on env-blocker (see In Flight #4).
- **Stale local checkpoints vs HF — `coder-30b` repointed 2026-05-01.** The `coder-30b` preset now points at `hf-mattbucci/Qwen3-Coder-30B-A3B-AWQ-Marlin-from-CT` (the Apr-29 HF mirror, run through `convert_moe_ct_to_awq.py --group-size 128` since the HF upload is CT-format). Smoke test passes: clean Python lambda output, `finish=stop` at 15 tokens. Today's `mattbucci/Qwen3-Coder-30B-A3B-REAP-AWQ` is flagged DO NOT USE in its own HF README (validation failed on the upload — REAP merger broken for Qwen3MoeForCausalLM, R9700 cross-team). Other presets audited and clean.
- **60B+ models don't fit** — Coder-Next-REAM (35 GB), GLM-4.5-Air-REAP (43 GB).
- **Piecewise CUDA graph `quant_type=None`** — would unblock decode speedups on REAP/REAM/Qwen3.6 (all currently run with graphs disabled for safety).
- **One 3090 offline (PCIe adapter swap pending)** — repo currently runs TP=1, single-GPU. AWQ MoE models (qwen3-ream, coder-30b, coder-reap-25b) still fit on 24 GB at 4K–8K context. Long-context evals and TP=2 benches paused until the second card returns. Quick capability sweeps via `./scripts/eval/test_capabilities_all.sh`. **Devstral 24B Dense OOMs on 24 GB at TP=1** — confirmed 2026-05-01 across all three loader paths (AWQ-Marlin, AWQ direct, compressed-tensors): the source weights load to ~23.48 GiB resident before the AWQ `create_weights` step runs, and that step's `torch.empty(input_size, output_size // pack_factor, dtype=int32)` per-layer destination allocation pushes past the 24 GB budget on the very next 160 MB request. Fails at `awq.py:518` with awq_marlin, at `compressed_tensors_wNa16.py:135` with CT — same root cause: weight-buffer prealloc isn't chunked, and Devstral's 24B-param footprint × 4-bit int32 packing leaves no headroom. `mem-fraction=0.97` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` both tested, neither closes the gap. Original "Marlin repack doubles weight memory" diagnosis was wrong — the bug is in eager dest alloc, not in the repack. Fix would need an upstream SGLang loader change (lazy/streamed dest allocation) or use of `Devstral-Small` once published. Workaround until the second 3090 returns: stay on the MoE-AWQ models (active params << 14 GB).

## Quick Start

```bash
./scripts/setup.sh                          # clone SGLang v0.5.10, apply patches, create conda env

# TP=1 / 24 GB friendly (current rig — second 3090 offline):
./scripts/launch.sh qwen3-ream              # fastest 256K — reference model (MoE active params fit cold)
./scripts/launch.sh qwen35                  # Qwen3.6-27B-AWQ R9700 recal — 3/3 basic+thinking+vision PASS
./scripts/launch.sh coder-30b               # Coder-30B MoE — peak throughput
./scripts/launch.sh coder-reap              # Coder-REAP-25B — SWE-bench Lite leader (29.3%)
./scripts/launch.sh qwen3-vl-32b            # Qwen3-VL-32B Dense — TP=1 defaults boot cold (4K/MAX_RUNNING=1)
./scripts/launch.sh qwen36                  # Qwen3.6-35B-A3B AWQ-native — boots TP=1 / 2K, 3/3 PASS at short-ctx

# TP=2 only (second 3090 needed):
./scripts/launch.sh devstral-long           # Devstral-24B at 217K — OOMs on TP=1 (eager weight prealloc, see Known Issues)
./scripts/launch.sh devstral                # Devstral-24B 131K default — same TP=2-only constraint

python scripts/eval/validate_capabilities.py --port 23334                 # auto-skips thinking/vision/video per preset
python scripts/eval/test_capabilities_all.sh                              # sweep across all AWQ presets
python scripts/bench/bench_long_context.py --port 23334 --name "Model" --contexts 1024 16384 131072 250000
```

Use `temperature >= 0.3` on Qwen3 family models — greedy decode at `temp=0` triggers a token-repetition loop.

## Prerequisites

- 2x NVIDIA RTX 3090 (24 GB each, 48 GB total) with NVLink bridge
- NVIDIA driver 595+ / CUDA 13.x
- Miniforge3 or Conda
- ~150 GB disk for models

## Model Support

Single-user tok/s measured at the max-context value in the table. All numbers are **fresh prefill** (radix cache disabled) unless noted.

| Model | Type | Max ctx | tok/s @max | TPOT | Launch | Status |
|-------|------|:-------:|:----------:|:----:|:------:|:-------|
| **Qwen3-30B REAM AWQ** | MoE (96 exp) | **262K** | **74** | 13.5 ms | `qwen3-ream` | **Hits 256K target** |
| **Qwen3.6-35B-A3B AWQ-native** | DeltaNet+MoE (256 exp, VL) | **262K** | 2.6 | 385 ms | `qwen36` | basic+thinking+vision 3/3 PASS (revalidated 2026-05-01 TP=1 / 2K); 33 @ short / 5.8 @160K / 2.6 @250K |
| **Qwen3.6-27B AWQ (R9700 recal Apr-29)** | DeltaNet+attn (VL) | **131K** | **21** | 47 ms | `qwen35` (preset → `mattbucci/Qwen3.6-27B-AWQ`) | **Basic + thinking + vision 3/3 PASS** on Ampere (2026-05-01 TP=1 / 4K). R9700's `balanced_thinking_text` recal resolved the vision regression seen on the prior CT v3 self-cal. Video skipped (text-only recipe, expected). |
| **Qwen3-VL-32B Instruct (community AWQ)** | Dense (VL) | **150K** | **40** | 25 ms | `qwen3-vl-32b` | Re-validated 2026-05-01 (TP=1, 2K context, 21 GB weights barely fits 24 GB at mem-fraction 0.93): basic PASS (`paris`), vision PASS (correctly named "red", "circle", "round"), thinking N/A — upstream Qwen3-VL-32B-Instruct is non-thinking by design (the `-Thinking` edition is a separate model). Prior "4/4" was a thinking-misclassification on the same pattern as the Coder family. |
| **Devstral-24B AWQ (long)** | Dense | **217K** | **56** | 17.9 ms | `devstral-long` | 66% past default ceiling. **Currently TP=2 only** — OOMs on TP=1 / 24 GB at all loader paths (see Known Issues). |
| Devstral-24B AWQ | Dense | 131K | 56 | 17.9 ms | `devstral` | Short-ctx + multi-user friendly. **TP=2 only** until the second 3090 returns. |
| Coder-REAP-25B W4A16 | MoE (103 exp) | 131K | 46 | 22 ms | `coder-reap` | Working |
| Coder-30B AWQ Marlin | MoE (128 exp) | 16K | 193 | 5 ms | `coder-30b` | Peak throughput |
| Gemma 4 31B Dense | Dense | 16K | 28 | 35 ms | `gemma4-31b` | 2/3 patched validator (revalidated 2026-05-01 TP=1 / 2K / triton-attn / FP16-KV): basic+thinking PASS, **vision hallucinates** (saw=[], response='cuneiform character' vs the red circle). Root cause confirmed: AutoRound checkpoint registers as `Gemma4ForCausalLM` (not `Gemma4ForConditionalGeneration`), so vision tower never engages — image tokens fall through to text-only path. Metadata fix to `architectures: ["Gemma4ForConditionalGeneration"]` would unblock vision; same fix R9700 flagged in their 3090-feedback report. |
| Qwen3.5-28B MoE REAP | DeltaNet+MoE (205 exp) | 262K | 33 | 31 ms | `qwen35-moe` | Thinking broken — recal queued |
| Gemma 4 26B MoE | MoE (103 exp) | — | — | — | `gemma4` | Boots clean post-2026-04-30 (patches 020 v2 / 021 / 022); generation truncates without closing `<channel\|>` thinking marker — 1/4 on validator. Recal candidate. |
| Gemma 4 21B REAP AWQ | MoE | — | — | — | — | Same status as 26B — kernel path fixed, generation needs recal. |

### VRAM context limits (KV dtype varies, TP=2, 48 GB total)

| Model | Wt/GPU | KV/token | Max context |
|-------|:------:|:--------:|:-----------:|
| Qwen3-30B REAM AWQ | 6.2 GB | 36 KB | 262K |
| Qwen3.5-28B MoE REAP CT | 8.1 GB | 5 KB | 262K |
| Qwen3.6-35B-A3B AWQ-native | 9.87 GB | ~8 KB hybrid | 262K |
| Coder-30B AWQ | 8.0 GB | 36 KB | 262K |
| Devstral-24B AWQ (long preset) | 7.0 GB | 80 KB | **217K** (true 3090 ceiling for 24B dense @ TP=2) |
| Coder-REAP-25B W4A16 | 6.5 GB | 72 KB | 131K |
| Qwen3.5-27B AWQ | 19.0 GB | 24 KB | 32K (weights replicated for DeltaNet TP) |

## Benchmarks

Per-model long-context sweep JSON in `benchmarks/<model>/`. Reference: Qwen3-30B REAM AWQ at `benchmarks/qwen3-30b-ream/long-context-262k.json`. Qwen3.6-35B-A3B AWQ-native detailed curve + tuning experiments in `benchmarks/qwen3.6-35b-a3b/awq-native-thinking-vision.json`.

### Quality (REAP vs REAM vs original)

![Quality Comparison](benchmarks/quality/quality_comparison.png)

| Model | MMLU | HumanEval | Needle (65K) |
|-------|:----:|:---------:|:------------:|
| Coder-30B (128 exp) | 73% | 100% | 100% |
| REAP-28B DeltaNet (205 exp) | 70% | 80% | 100% |
| REAM-30B (96 exp) | 63% | 80% | 100% |

Methodology: `scripts/eval/eval_and_chart.py` — MMLU (200 samples), HumanEval pass@1 (30 samples), [LAB-Bench](https://github.com/Future-House/LAB-Bench) (7 × 50), needle-in-a-haystack (1K→65K). Temperature=0, full context as reasoning budget.

**Still TODO:** [RULER](https://github.com/NVIDIA/RULER) (4K→256K synthetic), [LongBench Pro](https://arxiv.org/html/2601.02872v1), [LiveCodeBench](https://livecodebench.github.io/).

## Setup

```bash
./scripts/setup.sh
```

Or manually:
```bash
cd components/sglang && git checkout v0.5.10
for p in ../../patches/*.patch; do git apply "$p"; done
cd python && pip install -e ".[srt]"
```

| Component | Version |
|-----------|---------|
| SGLang | v0.5.10 + 21 local patches |
| PyTorch | 2.9.1 + cu128 |
| CUDA | 13.2 (driver 595.58) |
| NCCL | 2.27.5 (P2P over NVLink) |
| FlashInfer | 0.6.7.post3 |
| transformers | 5.5.3 |

## Patches

21 patches on top of SGLang v0.5.10 — full details in [`patches/README.md`](patches/README.md).

## Quantization

Self-calibrated models use a separate conda env (`quant`):

```bash
conda activate quant
CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_qwen36_27b_thinking_vision.py   # 27B template
python scripts/quantize/convert_moe_ct_to_awq.py <ct_src> <awq_dst>                       # MoE CT→native AWQ
```

For thinking+vision-preserving calibration: `scripts/quantize/calibration_datasets.py` builds `thinking_text` / `thinking_vision` / `code_vision` / `code_thinking` recipes (drawing from AM-Thinking-v1-Distilled, NuminaMath-CoT, LLaVA-Instruct-150K, UltraChat, the-stack). See [rules-for-agents.md](rules-for-agents.md) and [REAM.md](scripts/quantize/REAM.md).

## Test System

```
OS:     EndeavourOS (Arch Linux)
Kernel: 6.19.11-arch1-1
RAM:    96 GB (92 usable)
GPU:    2x NVIDIA RTX 3090 (GA102-300-A1, 24 GB GDDR6X each)
NVLink: NV4 — 4 lanes × 14 GB/s = 56 GB/s bidirectional
Driver: 595.58.03   CUDA 13.2   Python 3.12
```

## Repo layout

```
patches/                  # SGLang v0.5.10 patches — see patches/README.md for full narratives
benchmarks/               # Per-model benchmark JSON + charts
  quality/                #   MMLU / HumanEval / LAB-Bench / Needle
  <model>/                #   throughput + long-context sweeps
scripts/
  launch.sh               # unified launcher (launch.sh <preset>)
  common.sh               # shared conda + NVIDIA env setup
  setup.sh                # full setup (conda, SGLang install, patch apply)
  bench/                  # throughput benchmarks
  eval/                   # quality evals + chat template validator
  quantize/               # GPTQ → CT → AWQ pipeline + calibration recipes
  test/                   # kernel microbenchmarks + profiling
components/sglang/        # SGLang v0.5.10 + patches (cloned by setup.sh)
```

Sister project: [2x R9700 RDNA4 repo](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference). Cross-team threads (R9700 answers + advice on closed items) live in [`patches/README.md`](patches/README.md).
