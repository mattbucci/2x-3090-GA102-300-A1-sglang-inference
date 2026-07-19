# 3090-F: Trial-port R9700 069 decode-topk sparse-KV decode to triton-forced Gemma presets (graphs-gated)

| | |
|---|---|
| **Type** | experiment |
| **Status** | ready |
| **Execution host** | 3090-box |
| **Wall clock** | 2-3 days elapsed (pre-check same-day; port ~1 day; A/B + gates ~1 day), interleavable with bake-off queue gaps |
| **GPU time** | ~10-14h GPU occupancy on the 3090 eval box: eager pre-check ~2h; port smoke ~1h; topk sweeps + recall/caps probes ~6-10h (26B arm adds ~3h if pursued). Schedulable around bake-off cells. |
| **Depends on** | None blocking: donor patch + receipts readable in the local R9700 checkout; both Gemma AWQ checkpoints and post-depth-fix baselines already on the 3090 box; no new downloads, no calibration. |
| **Provides to** | R9700 team: portability verdict + the 069 server_args hunk rebased to v0.5.15 Annotated style (their live tree has no decode_topk yet; feeds the 067-070 CANDIDATE promotion decision and re-prices their v3.3 fixed-shape/cuda-graph follow-up, which the Gemma graphs-interaction data directly motivates); 3090 fleet-audit queue: closes bullet 6 of README 'Fleet-audit action queue (2026-07-18)'; Fleet: first sparse-KV-decode datapoint on an SWA-hybrid architecture (donor validation was pure full-attention only) |

## Objective

Port R9700's CANDIDATE patch 069 (--decode-topk-pages Quest-bbox sparse-KV decode, 1.77x @~245K on their stack) to the 3090's triton-forced Gemma presets, where post-depth-fix 256K decode sits at 12.9 tok/s (31B) / 24.0 tok/s (26B) and the occupancy levers are receipted CLOSED at 72%-of-roofline. Sparse top-K cuts KV bytes READ ~8-16x on exactly the layers that carry ~all of Gemma's depth-attributable decode cost (the full-attention layers), which is orthogonal to kernel efficiency — but v1 auto-disables cuda graphs, and graphs are a receipted 2.4x@1K lever on these presets, so the experiment is gated on a cheap eager-baseline pre-check before any porting effort. Closes fleet-audit queue item 6 and hands R9700 a v0.5.15 rebase of their own candidate.


## Hypothesis

On triton-forced Gemma presets, R9700's --decode-topk-pages sparse-KV decode beats the graphs-on baseline at true >=250K depth by >=1.3x on gemma4-31b DESPITE v1's cuda-graph auto-disable, because (a) the depth-attributable decode cost (58.9 of 77.4ms TPOT) is concentrated in the 10 full-attention layers whose 262K-token KV reads topk cuts >=8x (SWA layers read <=1024 tokens and contribute ~nothing), while (b) the graphs-off penalty is only ~13ms/step on the dense 31B. Corollary prediction to falsify cheaply: the 26B MoE nets only ~1.2x because its eager penalty (~17ms) is comparable to its whole depth term (29.4ms).


## Background & receipts

- Donor: /home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/patches/069-decode-topk-sparse.patch.CANDIDATE (296 lines, 2 files: triton_backend.py fused Triton bbox scorer `_bbox_crit_kernel` + v1 eager / v2 cached-reps page selection feeding the unchanged decode kernel; server_args.py adds --decode-topk-pages / --decode-topk-page-size / --decode-topk-scorer, and auto-sets disable_cuda_graph=True). Opt-in, default -1/off; gate already skips SWA layers: `layer.sliding_window_size is None or <= -1`.
- R9700 receipts (2x-R9700.../benchmarks/FINDINGS.md 'Query-selected sparse decode'): 12.85 -> 14.83-15.90 tok/s @~128K (page 32/64, 2048-token budget, needles exact); 8.29 -> 14.71 tok/s @~245K fused bbox = 1.77x, needle near-exact (<=1-char); 6-instance agentic A/B 2/6 = 2/6 both arms, applied diffs 5/6 -> 6/6. Gates: scripts/eval/decode_topk_agentic_ab.sh (flags used: --decode-topk-pages 128 --decode-topk-page-size 32), scripts/bench/decode_topk_needle_ab.sh. Donor's validated target was Qwen3-VL 32B (pure full-attention) — SWA-hybrid Gemma is untested for 069 anywhere.
- R9700 evergreen lessons (memory project-decode-topk-sparse-kv + FINDINGS): sparse budget must SCALE with context (1.6% ok @128K, 0.8% under-selects @256K and fails); throughput is budget-insensitive 2048 vs 16384 at depth (GPU-work-bound); the deep +-1-char near-miss is budget-insensitive; single-needle recall is a weaker bar than the agentic gate.
- 3090 post-depth-fix Gemma baselines (fixed instrument, actual_input_tokens verified; benchmarks/bench-depth-bug-2026-07-14.md): benchmarks/gemma4-31b/results.json = 54.1 tok/s @1K (TPOT 18.5ms) -> 12.9 @261,916 true (77.42ms) => depth-attributable ~58.9ms/step; benchmarks/gemma4-26b-awq/results.json = 81.4 @1K (12.29ms) -> 24.0 @261,916 (41.72ms) => depth-attributable ~29.4ms/step.
- Gemma 4 geometry (verified in /home/letsrtfm/AI/models/gemma-4-26B-A4B-it-BF16/config.json and /home/letsrtfm/AI/models/gemma-4-31B-AWQ/config.json): 31B = 60 layers, 10 full_attention / 50 sliding (window 1024), global KV 4 heads x 512 dim; 26B = 30 layers, 5 full / 25 sliding, global KV 2 x 512. SWA layers read <=1024 tokens regardless of depth, so ~100% of the depth-attributable decode cost sits in the full-attention layers topk applies to — the SWA-hybrid 'dilution' is negligible; the REAL dilution is the graphs-off penalty.
- Graphs penalty receipts (benchmarks/sprint-2026-06-kv-decode/LOG.md, B1/B1b): gemma4-26b triton graphs-on 82.9 vs graphs-off 34.1 tok/s @1K => eager overhead ~+17.2ms/step; gemma4-31b 58 vs 33 @1K => ~+13.1ms/step. Predicted topk-on (graphs off, >=8x read cut) @262K: 31B ~18.5+13.1+58.9/8+~2 ~= 41ms -> ~24 tok/s ~= 1.9x over 12.9 (GO); 26B ~12.3+17.2+29.4/8+~1 ~= 34ms -> ~29 tok/s ~= 1.2x over 24.0 (MARGINAL — the port caveat in the queue item is real for the 26B, not the 31B). Note B1 deep points predate the depth fix; @1K anchors are trustworthy, eager-at-depth must be re-measured (step 2).
- Port surface verified against pristine v0.5.15 (`git show v0.5.15:...` in /data/sgl-v0515, tag f63458b5be): the forward_decode hunk's context (`kv_indptr = self.forward_metadata.kv_indptr` ... `if layer.k_scale is not None and layer.v_scale is not None:`) matches pristine exactly; the __init__ hunk and BOTH server_args hunks do NOT apply — 069 was cut on the v0.5.13-era stack (contexts reference R9700 patches 067/068 and the old dataclass+add_argument server_args style). v0.5.15 server_args is Annotated `A[T, Arg(...)]` style, `disable_cuda_graph` is `Arg(no_cli=True)`, and the supported switches are `disable_decode_cuda_graph` / `--cuda-graph-backend-decode disabled` (+ `_handle_cuda_graph_config()` in `__post_init__`). R9700's live v0.5.15 tree contains NO decode_topk symbols (grep verified) — nobody has rebased 069 to v0.5.15 yet; this port is the first and its server_args rework is reusable by R9700.
- Hybrid-pool compatibility verified (git show v0.5.15:python/sglang/srt/mem_cache/swa_memory_pool.py): SWAKVPool.get_key_buffer(layer_id) routes full-attention layers to full_kv_pool — 069's `token_to_kv_pool.get_key_buffer(layer.layer_id)` works unmodified on Gemma's hybrid pool. gemma4-31b preset uses KV_DTYPE=fp8_e5m2; v2's `_pagereps` already casts non-fp16/bf16/fp32 KV to bf16 for amin/amax, and e5m2-auto has k_scale=1.0 so raw values = real values (ranking unaffected in principle; recall gate still required).
- Clean port surface on 3090: no patch in /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/patches/ touches triton_backend.py or server_args.py (grep verified); series ends at 057 => next free number 058. Serving tree: /data/sglang-rebase-v0515 (scripts/common.sh SGLANG_DIR), env sglang-v0515. Gemma presets expose the `${_ENV_GEMMA_GRAPH:-...}` env hook (launch.sh lines 290/316/348) — the eager pre-check needs zero code.
- Instruments verified on 3090: scripts/bench/bench_long_context.py (pins --random-range-ratio 1, records actual_input_tokens, flags depth_shortfall, supports --contexts subset); scripts/eval/probe_256k_tooluse.py (valid_toolcall + correct_action at server-verified true depths, ladder to ~258K true); scripts/eval/probe_256k_quality.py; scripts/eval/validate_capabilities.py. Fleet queue item being executed: README.md 'Fleet-audit action queue (2026-07-18)' bullet 6.


## Method

1. Read the donor + receipts end-to-end: ../2x-R9700-RDNA4-GFX1201-sglang-inference/patches/069-decode-topk-sparse.patch.CANDIDATE, benchmarks/FINDINGS.md ('Query-selected sparse decode'), scripts/bench/decode_topk_needle_ab.sh, scripts/eval/decode_topk_agentic_ab.sh (arm flags: --decode-topk-pages 128 --decode-topk-page-size 32 @64K).
2. Eager pre-check, zero code (go/no-go for each preset): `_ENV_GEMMA_GRAPH="--cuda-graph-backend-decode disabled --disable-piecewise-cuda-graph" ./scripts/launch.sh gemma4-31b --port 23334`, then `python scripts/bench/bench_long_context.py --port 23334 --name gemma4-31b-eager --contexts 2048 262144 --output benchmarks/gemma-topk-port/eager-control-31b.json`. Compute eager penalty = TPOT_eager@2K - 18.5ms and depth term = TPOT_eager@262K - TPOT_eager@2K; apply the kill rule below. Repeat for gemma4 (26B) only if its numbers clear the rule.
3. Port 069 to a git worktree of /data/sglang-rebase-v0515 (env sglang-v0515): (a) the fused `_bbox_crit_kernel`/`fused_bbox_score` block and both `_build_topk_kv_indices*` helpers + the forward_decode gate hunk apply against pristine-anchored context (verified: the `kv_indptr = self.forward_metadata.kv_indptr` ... `if layer.k_scale is not None` context matches v0.5.15 exactly); (b) hand-anchor the __init__ attribute block near `self.req_to_token_pool = model_runner.req_to_token_pool` (~line 145), dropping the R9700 067/068 force-*-window context lines — 069's added code has no dependency on them; (c) REWRITE the server_args side in v0.5.15 Annotated style: three `A[int/str, Arg(...)]` fields (decode_topk_pages=-1, decode_topk_page_size=32, decode_topk_scorer='bbox') + in `__post_init__` BEFORE `self._handle_cuda_graph_config()`, if decode_topk_pages > 0: log a warning and set `self.disable_decode_cuda_graph = True` (v0.5.15's supported decode-phase switch; `disable_cuda_graph` is no_cli internal).
4. Smoke per the version-rebase gate lesson (apply+py_compile is NOT sufficient): eager-import boot-chain, then boot gemma4-31b at CTX=32768 with `EXTRA_ARGS="--decode-topk-pages 64 --decode-topk-page-size 64"`, confirm the auto-disable warning in the server log, coherent multi-sentence output, and that SWA layers are untouched (gate is `layer.sliding_window_size is None or <= -1`; SWAKVPool.get_key_buffer routes full layers to full_kv_pool — verified v0.5.15).
5. Deep A/B on gemma4-31b, one mechanism changed: baseline = existing benchmarks/gemma4-31b/results.json (graphs-on); topk arm = preset + `--decode-topk-pages 256 --decode-topk-page-size 64` (budget 16,384 tokens ~= 6.2% of 262K — above R9700's validated 1.6% fraction, and throughput is budget-insensitive 2048->16384 at depth; page 64 was their fastest). Sweep `--contexts 2048 32768 131072 262144`, detached via setsid; cross-check decode tok/s against server-log gen-throughput. Note fp8_e5m2-KV caveat: bbox reps computed on raw e5m2 (k_scale=1.0) — covered by step 6 gates.
6. Recall + capability gates on the topk arm (same server): `python scripts/eval/probe_256k_tooluse.py --port 23334 --tag gemma4-31b-topk --lengths 16384,65536,131072,196608,256000` (correct_action must be 1.0 at every rung — the preset's 1.0@258K-true receipt is the bar), `probe_256k_quality.py` needle (near-exact tolerated per R9700 precedent), `validate_capabilities.py` 5/5 (thinking+vision+video preserved).
7. If 31B clears >=1.3x: locate the crossover depth (the context where topk-eager overtakes graphs-on — expected 64-131K) for the README guidance; then run the 26B arm ONLY if its step-2 pre-check passed (predicted ~1.2x marginal), and optionally gemma4-21b-reap (same A4B geometry as 26B).
8. Decision + write-up: WIN -> promote as patches/058-decode-topk-sparse.patch via the 3-gate test, wire an opt-in env hook (e.g. `${_ENV_GEMMA_TOPK:-}` appended to the gemma presets' EXTRA_ARGS, default empty — graphs-on default UNCHANGED since topk loses at short/medium context), update README perf notes + check off queue bullet 6, and push the rebased v0.5.15 server_args hunk + 'portable: yes/no + numbers' to R9700's README (feeds their 069 promotion decision and re-prioritizes their v3.3 cuda-graph-compatible fast path, which this MoE/graphs interaction now motivates). NULL -> verdict.md receipt + one line in README Decode ideas.


## Baseline & instrument

gemma4-31b graphs-on preset depth curve, already on file post-depth-fix: benchmarks/gemma4-31b/results.json (2026-07-14) = 54.1 tok/s @1K -> 12.9 tok/s @261,916 server-verified true tokens (TPOT 18.5 -> 77.42ms); gemma4-26b-awq/results.json = 81.4 -> 24.0. Before any code: a NEW eager (graphs-off) control at {2048, 262144} on the same preset via `_ENV_GEMMA_GRAPH="--cuda-graph-backend-decode disabled --disable-piecewise-cuda-graph" ./scripts/launch.sh gemma4-31b` + `python scripts/bench/bench_long_context.py --contexts 2048 262144 --name gemma4-31b-eager --output benchmarks/gemma-topk-port/eager-control-31b.json` — this measures the true graphs-off penalty at depth on the fixed instrument.


## Success criteria

- Primary: topk arm decode >= 1.3x the graphs-on baseline on gemma4-31b at >=250K server-verified true depth (baseline 12.9 tok/s / TPOT 77.42ms from benchmarks/gemma4-31b/results.json; instrument bench_long_context.py with actual_input_tokens recorded and no depth_shortfall flag; cross-checked against server-log gen-throughput). Predicted ~1.9x.
- Recall: probe_256k_tooluse.py correct_action = 1.0 at every rung through ~256K true depth with topk on; needle probe no worse than the donor's near-exact (+-1-char) class.
- Capabilities: validate_capabilities.py 5/5 (thinking + tool + vision + video) unchanged with topk enabled.
- Crossover depth measured and documented; gemma preset defaults unchanged (opt-in env hook only).
- On WIN: patches/058-decode-topk-sparse.patch passes the 3-gate patch test; receipts under benchmarks/gemma-topk-port/; R9700 handed the v0.5.15-rebased server_args hunk + verdict.
- A clean NULL (pre-check kill or <1.15x measured) with receipts equally closes the queue item — negative results are findings.


## Kill criteria

- Pre-check kill (before any porting): if measured eager penalty (TPOT_eager - TPOT_graphs @2K) >= 0.5x the preset's depth-attributable TPOT @262K, predicted net is <~1.2x — record the null in verdict.md and stop for that preset. Expected: 31B passes (13 vs 58.9ms), 26B fails (17 vs 29.4ms) — do not start the port for the 26B alone.
- Perf kill: topk arm < 1.15x vs graphs-on baseline at >=250K server-verified depth (bench_long_context.py, no depth_shortfall) after ONE budget-scaling retry (>= 6% of context per R9700's budget-scales lesson) — null with receipts.
- Recall kill: probe_256k_tooluse.py correct_action < 1.0 at any rung, or needle worse than R9700's +-1-char near-exact class, not fixed by budget scaling (the deep near-miss is budget-INSENSITIVE per R9700 — more budget will not rescue it); or validate_capabilities.py < 5/5.
- Port time-box: no booting smoke after ~1.5 days of rebase debugging — park with worktree diff + notes in verdict.md (the v0.5.15 server_args rework is the known-unknown; the attention-side hunks are verified to anchor).
- External: bake-off queue contention or a kernel-BUG reboot mid-sweep — resume from receipts, do not re-baseline a half-measured arm.


## Deliverables

- benchmarks/gemma-topk-port/eager-control-31b.json (+ eager-control-26b.json if pursued) — graphs-off control sweeps, fixed instrument
- Worktree port of 069 rebased to v0.5.15; on WIN promoted as patches/058-decode-topk-sparse.patch (3-gate tested, applied by setup.sh glob) + patches/README.md entry
- benchmarks/gemma-topk-port/topk-31b-results.json (sweep at 2K/32K/131K/262K, actual_input_tokens recorded) + tooluse-topk-31b.json (probe_256k_tooluse.py ladder) + caps receipt (validate_capabilities.py output)
- benchmarks/gemma-topk-port/verdict.md — win/null one-pager: measured eager penalty, per-depth A/B table, crossover depth, budget used, recall results (negative result is a finding, keep it)
- launch.sh: opt-in env hook (e.g. ${_ENV_GEMMA_TOPK:-} appended to gemma preset EXTRA_ARGS), defaults unchanged; README: queue bullet 6 checked off + perf-table/Tooling note
- Cross-repo: rebased v0.5.15 server_args hunk + portability verdict pushed to R9700's README (feeds their 069 CANDIDATE->promoted decision and re-prices their v3.3 fixed-shape/cuda-graph follow-up)


## Constraints

- READ the live repo state first: patches 083-089-era note applies to R9700 only, but 3090's working tree may have moved past this spec — re-verify patch numbering (058 next-free) and the queue item before starting.
- Rule 1 / Rule 2 (CLAUDE.md): no concurrent calibration+serving/eval, no concurrent rollout+score; this experiment occupies the eval box GPUs — coordinate with the resuming SWE-bench bake-off queue (evals/swebench/run_all_cycles.sh), do not interleave.
- Kernel-BUG reboots every ~9-17h under docker I/O: detach every >30min sweep via the setsid pattern, write receipts to benchmarks/ incrementally so a hard reset loses nothing.
- Fleet measurement invariants: decode tok/s from server-log gen-throughput or bench_long_context.py's server-verified actual_input_tokens ONLY (never client TPOT, never random without --random-range-ratio 1); one mechanism at a time; A/B at short+medium+deep context.
- Preserve thinking+image+video (+audio where applicable): validate_capabilities.py must stay 5/5 with topk enabled before any WIN claim.
- Preset defaults must NOT change: topk auto-disables decode cuda-graphs, which regresses short/medium context (agentic median ~41K) — wire only an env-hook opt-in (mirror the _ENV_GEMMA_GRAPH pattern), keep graphs-on defaults.
- Patch promotion only via the 3-gate test in CLAUDE.md (pristine glob-order apply; byte-identical to live tree; git apply --check fails on already-patched tree), diff generated against the predecessor-patched tree.
- READ-ONLY courtesy to R9700: push the rebased-to-v0.5.15 server_args hunk and the portability verdict to their README/patch inbox; do not edit their tree.


## Risks

- Eager penalty may be larger at depth than the @1K estimate (overhead may not be purely additive with attention time) — mitigated by measuring the eager control at 262K in step 2 rather than extrapolating; the June B1 deep points are pre-depth-fix and must not be reused.
- fp8_e5m2 KV on the 31B: bbox scoring on raw e5m2 keys (coarse 2-bit mantissa) is unvalidated — R9700's real-key scorer validation was on bf16 keys; ranking could degrade. Fallback: KV_DTYPE=auto arm (halves KV capacity, still ~260K pool per preset comments) or run the 26B (KV auto) as the recall canary.
- Fused kernel BLOCK = next_power_of_2(Hkv*D) = 1024 on the 31B (2 heads/rank x 512) is untested on sm_86 (donor ran gfx1201) — the patch's eager fallback path covers correctness; perf-only risk.
- v2 rep-cache is bs==1-only (bs>1 falls back to v1 O(ctx)/step) and keyed on slot0 request identity — safe under MAX_RUNNING=1 presets, but any concurrency test must expect the slow path.
- Gemma-specific attention features (attn softcapping, 26B attention_k_eq_v shared-KV layers with the k=None decode path) interact with selection only via kv_indices, which the decode kernel already consumes index-agnostically — but the 26B's KV-shared-layer route is exactly the kind of merge-remnant the boot-chain smoke exists to catch.
- Even on a WIN this ships as an opt-in deep-context lane, not a preset default: below the ~64-131K crossover graphs-on wins, and agentic median context is ~41K — the win is real for the 256K single-user mission, narrow for mixed workloads.


---
*Vetted 2026-07-18: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
