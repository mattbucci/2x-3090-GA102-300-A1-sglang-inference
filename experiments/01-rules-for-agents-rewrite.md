# 3090-C: Rewrite stale rules-for-agents.md + fix CLAUDE.md dead sister path

| | |
|---|---|
| **Type** | task |
| **Status** | ready |
| **Execution host** | 3090-box |
| **Wall clock** | 2-4 h (docs + scripted consistency gate + adversarial review pass; no GPU) |
| **GPU time** | none — verification is textual against launch.sh, README, and benchmarks/ receipts; do not launch servers. Step-8's flag-existence check may use `python -m sglang.launch_server --help` in the sglang-v0515 env (no GPU) or fall back to grepping SGLang's argparse source — neither needs a GPU. |
| **Depends on** | None — all source material is committed repo state; independent of the other 3090 queue items (the baselines.json arming item is explicitly out of scope here). |
| **Provides to** | 3090 team's standing optimization loop — every fresh agent session reads rules-for-agents.md/CLAUDE.md first; accurate flags/caps/instruments prevent forfeiting the graphs win and mis-measured benchmarks.; Fleet-audit queue closure (README.md:11 item). |

## Objective

rules-for-agents.md is the first doc a fresh agent reads and it currently teaches three receipted-wrong behaviors: forfeit the 4.15x cuda-graphs win, cap context at 4-32K when the matrix standard is 262144, and base work on cyankiwi community quants against the prune-ourselves rule. Rewriting it from live launch.sh + README state (and fixing CLAUDE.md's dead R9700 path that breaks the sync-first loop step) removes a standing source of wrong first moves on the rig that owns all evals.


## Background & receipts

- rules-for-agents.md:24 mandates `--disable-cuda-graph --disable-custom-all-reduce` as "required, CUDA graph capture OOMs with TP=2" — but this is receipted-wrong: CUDA graphs are ON for every launch.sh preset except one. Mechanism (verified against scripts/launch.sh): the default is `CUDA_GRAPH=""` (launch.sh:66) which appends no flag and inherits SGLang's graphs-on default; ~10 presets additionally pin `--cuda-graph-max-bs 1` for single-user bs=1 capture (launch.sh:109,139,158,193,227,245,257,534,599,644), and the 4 gemma presets pin it via EXTRA_ARGS (launch.sh:290,316,348,386). qwen36-vl-reap (launch.sh:444) is the SOLE preset that disables graphs, with `--disable-cuda-graph --disable-piecewise-cuda-graph`. The rewrite must NOT state "--cuda-graph-max-bs 1 in every preset" or "23 of 24 via --cuda-graph-max-bs 1" — that count fails the step-8 gate; frame it as "graphs ON everywhere except the one qwen36-vl-reap disable."
- Graphs-win receipt: launch.sh qwen36 preset comment (scripts/launch.sh:589-599, dated 2026-06-07) — enabling graphs took single-user decode 31 -> 129 tok/s @262K (4.15x, TPOT 32.1 -> 7.8 ms), 5/5 capabilities under graph replay; earlier devstral receipt commit 45c4810 (+25%, cited in benchmarks/allreduce-accel-null-2026-06-15.md).
- `--disable-custom-all-reduce` is NOT stale and is a SEPARATE flag from the graph disable: applied globally at scripts/launch.sh:808 (`[[ -z ENABLE_CUSTOM_AR ]] && CMD+=(--disable-custom-all-reduce)`) because custom AR breaks graph capture on sm_86 TP=2, re-confirmed 2026-06-15 (benchmarks/allreduce-accel-null-2026-06-15.md). The per-preset graph disable at launch.sh:444 is `--disable-cuda-graph --disable-piecewise-cuda-graph` (piecewise pairing), NOT custom-AR. The rewrite must split the two: graphs ON default, custom-AR OFF global, and anchor the sole graph-disable to the piecewise pairing at :444.
- rules-for-agents.md:32-39 VRAM table caps context at 4K-32K and lists three dead models (Coder-Next-REAM-60B, GLM-4.5-Air-REAP-82B, Coder-Next-80B — none is a preset); live state is 17 of 24 presets at CTX=262144, matrix standard TP=2/262144/MAX_RUNNING=1 (CLAUDE.md:73), per-model server-verified depth caps in README "Performance — single-user decode at 256K".
- rules-for-agents.md:25 recommends `--quantization compressed-tensors` "for cyankiwi community checkpoints" — README Direction section: "What we don't ship: random community quants. Every mattbucci/*-AWQ is calibrated end-to-end from the upstream BF16 base... Pre-quantized 3rd-party AWQ uploads are reference points only." CT flag remains valid only for own CT-format checkpoints (qwen36-dense-ct preset exists, launch.sh:482).
- rules-for-agents.md:61-68 instructs running calibrations on this box (CPU-only llmcompressor) — division of labor since 2026-05-19 (CLAUDE.md:47,88): calibrations run on the separate same-repo calibration device; the eval box only `git pull --rebase`; Rule 1 forbids concurrent calibration + serving/eval.
- rules-for-agents.md:127-135 Benchmarking section says "Always measure TPOT with sglang.bench_serving" + concurrency sweep 1-32 — violates fleet measurement invariant (decode tok/s from server-log gen-throughput or server-verified actual_input_tokens at true depth, never client TPOT; single-user M=1 is the primary target per CLAUDE.md:39-40). Instrument caveats receipted in benchmarks/bench-depth-bug-2026-07-14.md; client TPOT under-measures ~2x (fleet memory).
- bench_regression.sh exists (scripts/bench/bench_regression.sh) but benchmarks/baselines.json is literally `{}` — the rewrite must not claim a working regression tripwire (arming it is a separate fleet-audit queue item).
- Repo-wide grep `grep -rn rdna4-inference-triton36 --include="*.md" --include="*.sh" .` returns TWO live hits, not one: CLAUDE.md:45 ("Sync first" loop step, the dead R9700 path) AND README.md:11 (the audit-queue bullet that quotes the path). Both are cleared by this same commit — CLAUDE.md:45 via the path fix, README.md:11 via the completed-bullet removal — which is why the success criterion legitimately reaches 0. CLAUDE.md:85 already has the correct `~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference` form.
- Still-valid content to PRESERVE unchanged in the rewrite (verified present in rules-for-agents.md and consistent with CLAUDE.md Calibration Rules + Operational Lessons): quant-vs-serving conda env split, DeltaNet/SSM no-INT4 with narrow `in_proj_a$/b$` ignore, MoE >=512-sample calibration, chat-template verification block, AWQ format/Marlin alignment notes. The check_awq_scales.py scale-audit gate is NOT currently in rules-for-agents.md (grep count 0) — it must be ADDED (sourced from CLAUDE.md), not "preserved."
- rules-for-agents.md last touched at commit eec2d9f (v0.5.11 era; stack is now v0.5.15 + 24 patches, flipped 2026-07-12 per README header). Authoritative queue bullet: README.md:11.
- launch.sh:844 carries its own stale comment ("DeltaNet+MoE hybrids whose presets default to --disable-cuda-graph", plural) while only one preset (qwen36-vl-reap, :444) actually disables graphs. launch.sh is out of scope (do not modify), but the rewrite and the step-9 adversarial pass must NOT treat that comment as authority for "multiple presets disable graphs" — verify against the actual CUDA_GRAPH assignments, not the comment.
- Vetter status: both vetters APPROVE with fixes; strategy is sound and not-yet-done (rules-for-agents.md:24/25 stale mandates present, README.md:11 bullet open). All fixes are factual-precision corrections folded into this revision; status stays ready.


## Method

1. 1. Sync + freshness check: `git pull --rebase`; confirm no session is actively editing scripts/launch.sh presets (`git log --since=2.days -- scripts/launch.sh` + check running jobs). Re-read the live sources the rewrite derives from: rules-for-agents.md, CLAUDE.md, README.md (Direction + Performance + audit queue), scripts/launch.sh (all 24 presets, the CUDA_GRAPH default at line 66 and per-preset assignments, and the global CMD block at lines ~795-810), scripts/common.sh.
2. 2. Record the baseline snapshot greps (see baseline) for the commit message — including the TRUE rdna4-inference-triton36 count of 2 hits (CLAUDE.md:45 + README.md:11), not 1.
3. 3. One-line fix in CLAUDE.md:45: `~/AI/rdna4-inference-triton36` -> `~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference` (match the correct form already at CLAUDE.md:85). Note the second grep hit (README.md:11) is cleared by step 10's bullet removal, not here. Verify after both edits: repo-wide grep for `rdna4-inference-triton36` returns 0.
4. 4. Rewrite rules-for-agents.md Server Launch/flags section with the CORRECT graph mechanism: CUDA graphs are ON for every preset except qwen36-vl-reap. Frame it exactly as: 'graphs run ON by default — the launch.sh default is `CUDA_GRAPH=""` (launch.sh:66, no flag appended, SGLang's own graphs-on default); several presets additionally pin `--cuda-graph-max-bs 1` for single-user bs=1 capture; qwen36-vl-reap (launch.sh:444) is the ONE preset that disables graphs, via `--disable-cuda-graph --disable-piecewise-cuda-graph`.' Do NOT write '--cuda-graph-max-bs 1 in every preset' or '23 of 24'. Cite the 4.15x receipt (launch.sh:589-599: 31->129 tok/s @262K, TPOT 32.1->7.8 ms) and instruct never to override graphs off without an A/B receipt. Document `--disable-custom-all-reduce` as a SEPARATE launch.sh global default (launch.sh:808 — breaks graph capture on sm_86 TP=2; retest only via ENABLE_CUSTOM_AR=1; receipt benchmarks/allreduce-accel-null-2026-06-15.md). Point to `scripts/launch.sh` presets as the single source of truth for per-model flags — do NOT restate every flag (duplication is what made this doc rot). Ignore the stale plural :844 comment.
5. 5. Replace the VRAM/context table: state the matrix standard (TP=2, CTX=262144, MAX_RUNNING=1, per CLAUDE.md Current Hardware State); per-model true-depth caps and decode tok/s come from the README Performance section (server-verified depths — copy the caps, cite the section). Use README's exact framing for examples, e.g. devstral 202K KV pool (42 tok/s @196K) and qwen3-vl-32b 131K model-card cap (35 tok/s @127K — do NOT state the cap as 127K; 127K is the measured decode depth, 131K is the card cap per README:259). Delete dead rows (Coder-Next-REAM-60B, GLM-4.5-Air-REAP-82B, Coder-Next-80B); keep the true rule "80B+ does not fit in 48GB".
6. 6. Rewrite the Quantization Pipeline section: (a) division of labor — calibrations run on the separate same-repo calibration device, THIS box only pulls (Rule 1: no concurrent calibration + serving/eval); keep the pipeline description and the quant-vs-serving conda env split as reference for the calibration device; (b) replace the cyankiwi line with the prune-ourselves policy from README Direction (own BF16->GPTQ->CT->AWQ builds only; community quants are reference points, never ships/bases); keep `--quantization compressed-tensors` documented only for own CT-format checkpoints (example: qwen36-dense-ct preset, launch.sh:482); (c) PRESERVE unchanged the content already present in the doc: DeltaNet/SSM no-INT4 (narrow in_proj_a$/b$ ignore), MoE >=512-sample rules, chat-template verification, AWQ/Marlin format notes; and ADD the check_awq_scales.py scale-audit gate reference (sourced from CLAUDE.md — it is currently absent from rules-for-agents.md, grep count 0, so this is an addition not a preservation).
7. 7. Rewrite the Benchmarking section to the fleet invariants: primary target is single-user (M=1) decode at TRUE depth; decode tok/s from server-log gen-throughput or server-verified actual_input_tokens — never client TPOT, never bench_serving random without --random-range-ratio 1; instruments are scripts/bench/bench_long_context.py and scripts/eval/run_v0512_fleet_eval.sh; instrument caveats: benchmarks/bench-depth-bug-2026-07-14.md; receipts land under benchmarks/. Note bench_regression.sh exists but benchmarks/baselines.json is `{}` — no tripwire claim until the separate queue item arms it.
8. 8. Scripted consistency gate: extract every `--flag` token from the new rules-for-agents.md and require each to appear in scripts/launch.sh or `python -m sglang.launch_server --help` (sglang-v0515 env; may fall back to grepping SGLang's argparse source so this is not a hard serving-env dependency); `test -e` every file path mentioned; every number (tok/s, context caps, sample counts) must trace to a launch.sh comment, README table, or a file under benchmarks/. In particular the graph-mechanism wording must contain NO false preset count and the qwen3-vl cap must read 131K (card) / 127K (measured). Any claim that fails tracing gets CUT, not kept.
9. 9. Review gate (adversarial pass): a fresh-context agent reads ONLY the new rules-for-agents.md + scripts/launch.sh + README.md + CLAUDE.md and lists every contradiction or unverifiable claim (must independently re-derive the graph mechanism from CUDA_GRAPH assignments, not from the stale :844 comment). Zero findings required; if findings, fix and re-run the pass once.
10. 10. Commit both files + delete the completed bullet from README "Fleet-audit action queue" (README.md:11) in the same self-contained commit — this removal also clears the second rdna4-inference-triton36 grep hit (README discipline: remove completed tasks, don't mark DONE); include the truthful baseline greps and review-gate result in the commit message; push. Re-verify success criteria post-commit.


## Baseline & instrument

Pre-state snapshot (paste into the commit message, truthful counts): `grep -n \"disable-cuda-graph\" rules-for-agents.md` (1 hit, line 24 \"required\" mandate), `grep -in cyankiwi rules-for-agents.md` (1 hit, line 25), `grep -rn rdna4-inference-triton36 --include=\"*.md\" --include=\"*.sh\" .` (TWO hits: CLAUDE.md:45 + README.md:11 — both cleared by this commit), `grep -cn check_awq_scales rules-for-agents.md` (0 — confirms it is an addition), stale VRAM-table caps at rules-for-agents.md:32-39.


## Success criteria

- `grep -rn rdna4-inference-triton36 --include="*.md" --include="*.sh" .` over the repo returns 0 hits (was 2: CLAUDE.md:45 fixed + README.md:11 bullet removed).
- `grep -n "disable-cuda-graph" rules-for-agents.md` returns hits only inside the documented qwen36-vl-reap exception (paired with --disable-piecewise-cuda-graph, anchored to launch.sh:444); no "required"/blanket-disable language remains, and no false all-preset graph count is stated.
- `grep -in cyankiwi rules-for-agents.md` returns 0; the prune-ourselves policy statement is present and matches README Direction.
- No context cap stated in the doc is below its preset's live CTX or README-receipted server-verified depth; the three dead model rows are gone; the qwen3-vl-32b entry reads 131K model-card cap (35 tok/s @127K), not a 127K cap.
- check_awq_scales.py gate is present in the rewritten doc (added from CLAUDE.md); the DeltaNet/MoE-512/chat-template/AWQ-Marlin preserve content survives unchanged.
- Step-8 consistency gate passes 100% (every flag in launch.sh/--help/argparse source, every path exists, every number traced to a receipt); step-9 adversarial pass records zero contradictions.
- Benchmarking section names server-log gen-throughput / server-verified depth + bench_long_context.py / run_v0512_fleet_eval.sh; no client-TPOT or concurrency-sweep-as-primary instruction survives; no armed-tripwire claim while baselines.json is {}.
- README audit-queue bullet removed in the same commit; commit pushed.


## Kill criteria

- If step 1 shows launch.sh presets under active modification by a running bake-off/optimization session, pause the rewrite until that lands — a rewrite against a moving flag set re-creates the staleness it fixes.
- If the adversarial pass still finds contradictions after one fix cycle, do not commit; escalate the specific contradictions to the README queue as a finding.
- If any launch-flag or perf claim cannot be traced to launch.sh, README, or a benchmarks/ receipt, cut the claim; never commit unverified text to a rules doc.


## Deliverables

- Rewritten /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/rules-for-agents.md (derived from launch.sh + README, stale mandates removed, valid safety content preserved, check_awq_scales.py gate added).
- One-line path fix in /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/CLAUDE.md (line 45).
- README.md audit-queue bullet (line 11) deleted in the same commit — this also clears the second rdna4-inference-triton36 grep hit.
- Commit message containing the truthful baseline greps (rdna4 grep = 2 hits), the consistency-gate result, and the review-gate result (the receipts for this doc task).


## Constraints

- No GPU/serving work: verification is textual; do not launch servers or touch nvidia-smi state (Rules 1/2 unaffected but stay aware of running jobs).
- Do not modify scripts/launch.sh — it is the source of truth this task documents, not the target. Do not treat its stale :844 comment as authority (verify against the actual CUDA_GRAPH assignments).
- README discipline: no /tmp paths, setsid mechanics, or session banners in the public docs; remove the completed queue bullet rather than marking it done.
- Doc single-source-of-truth: rules-for-agents.md must point at launch.sh presets for per-model flags instead of duplicating them; do not duplicate CLAUDE.md loop/process content.
- Preserve the still-valid calibration safety content listed in step 6c (DeltaNet/chat-template/MoE-sample/AWQ-Marlin) and ADD the check_awq_scales.py gate; over-deletion of these rules is a regression, not a cleanup.
- One small self-contained commit at a clean boundary; push after the review gate.


## Risks

- Preset churn during the rewrite (active bake-off resuming per README) makes copied caps stale on landing — mitigated by step-1 freshness check and by citing README/launch.sh as authority instead of over-copying numbers.
- Over-deletion risk: the doc mixes wrong ops guidance with correct hard-won calibration safety rules; the keep-list in step 6c and the adversarial pass guard this.
- The qwen36-vl-reap exception could be re-generalized into a blanket graph-disable rule by a careless rewrite (and the stale launch.sh:844 plural comment invites exactly that error) — the doc must frame it as the ONE preset-level exception with its launch.sh:444 `--disable-cuda-graph --disable-piecewise-cuda-graph` anchor, and must not cite :844 as authority.
- Propagating a false graph-preset count (e.g. '23/24 via --cuda-graph-max-bs 1') or the wrong qwen3-vl cap (127K vs the 131K card) into a precision doc — mitigated by the corrected step-4/step-5 wording and the step-8 gate that traces every number.
- CLAUDE.md has other aging references (e.g. run_v0512_fleet_eval.sh name predates v0.5.15 but the file exists and is the live harness) — renaming scripts is out of scope; touching only line 45 avoids scope creep.


---
*Vetted 2026-07-18: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
