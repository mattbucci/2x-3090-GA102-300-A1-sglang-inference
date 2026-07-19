# 3090-A: Adopt R9700 patch 056 prefix-match toolcall hold-back into 3090 patch 041 (as new patch 058)

| | |
|---|---|
| **Type** | task |
| **Status** | ready |
| **Execution host** | 3090-box |
| **Wall clock** | ~0.5-1 day total: steps 1-8 in ~2-4 h hands-on; the observational opencode cell adds ~2-6 h detached rollout+score. |
| **GPU time** | 3090-box only: ~15 min (server boot + 2×10-trial probes + capability probes) + one devstral×opencode full-300 rollout+score occupancy (~2-6 h). Steps 1-7 are GPU-free. |
| **Depends on** | None blocking: the donor patch is committed in the R9700 repo (readable checkout at ~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference, or GitHub) — note the 3090 CLAUDE.md sister path `~/AI/rdna4-inference-triton36` is DEAD; use the full repo-name path.; Scheduling only: the devstral×opencode cell re-run (step 9) must slot into the bake-off queue per Rule 1/2 — coordinate with the resuming little-coder/claw cells (README Active work #1). |
| **Provides to** | 3090 SWE-bench bake-off: devstral cells (opencode now; future claw-code/little-coder devstral cells inherit the fix) — removes one silent empty-diff mechanism from the mistral-parser lane.; 3090 upstream-PR ledger: the 041 PR-candidate package should fold in 058 (R9700's existing upstream package lacks the prefix extension — flag to R9700).; R9700: adoption confirmation + the committed unit test file (their 8/8 was never committed; the 3090 test is portable back to their stack verbatim). |

## Objective

Port R9700 patch 056 (prefix-match streaming hold-back for tool names) onto the 3090's patch 041 so multi-token tool names (todowrite, webfetch — the names opencode/claw use) stop being flushed piecewise as content, which currently defeats [TOOL_CALLS]-omission recovery and silently produces empty diffs on devstral agentic SWE-bench cells. Closes the oldest flagged cross-repo adoption in the 3090 patch map (row 041, flagged 2026-06-17) and hardens the mistral parser lane for the resuming bake-off. Backend-independent detector logic — no kernels, no checkpoint changes.


## Hypothesis

n/a (task — the mechanism was already validated live on R9700; the falsifiable sub-claim, checked at step 3, is that the pre-058 3090 tree still leaks the first streamed piece of a multi-token tool name as content)


## Background & receipts

- Donor verified: `/home/letsrtfm/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/patches/056-devstral-multitoken-toolname-omission-recovery.patch` changes ONLY `_trailing_known_tool_name_len` in `mistral_detector.py` from exact-match to trailing-PREFIX match (+16/-3 logic lines); introduced in their commit 9628f9b (2026-06-15), listed Candidate in their `patches/README.md` line 154.
- Applicability verified by simulation: 3090's `patches/041-devstral-toolcall-omission-recovery.patch` is byte-identical to R9700's `patches/040-devstral-toolcall-omission-recovery.patch` modulo `index` lines (diff run 2026-07-18), and R9700's 056 applies with strict `git apply` (zero fuzz) onto a 041-patched `mistral_detector.py`, which then passes `py_compile` (scratchpad sim: /tmp/claude-1000/-home-letsrtfm-AI/31b4fdbe-1d15-4b21-95c9-77b2ec9abd6b/scratchpad/portcheck/sim).
- No interleaving conflicts: grep shows 041 is the ONLY 3090 patch touching `python/sglang/srt/function_call/mistral_detector.py`; on R9700 only their 040+056 touch it. 3090 numbers 056 (gdn-conv-state-dtype-cast) and 057 (mistral-common-backend-optout) are taken, so the port must be numbered 058.
- The flag is real and open: 3090 `patches/README.md` line 30 (row 041) has carried the ⚠ R9700-extension-available note since 2026-06-17; README fleet-audit queue line 9 is the same ask. Mechanism: SGLang streams tool-name tokens before `[ARGS]`, so multi-token names (`todowrite`->`todo`+`write`, `webfetch`->`web`+`fetch`) have their first piece flushed as content by 041's exact-match hold-back, splitting the name across the flush boundary and defeating omission recovery.
- CORRECTION to the audit bullet's 'unit-tested 8/8': the R9700 unit test was ad-hoc and NOT committed — their commit 9628f9b describes the 8 cases in prose only; the only committed instrument is the live 2-turn probe `scripts/eval/devstral2_toolprobe.py` + receipt (their commit 500715e, TURN2 todowrite structured, zero leak, on FP8 TP2). The 3090 port must therefore WRITE a committed unit test; it cannot copy one.
- Do not diff against R9700's committed `components/sglang` tree: its `mistral_detector.py` contains 040 but NOT 056's prefix loop (grep `startswith(trailing)` = absent) — their live serving tree is `/data/sgl-v0515` on their box. Use the patch FILE as the donor.
- Confounded baseline on record: `benchmarks/quality/bakeoff-devstral-opencode.json` = 40/300 resolved (13.3%), 173/300 empty_patch, run_dir `evals/swebench/runs/devstral-opencode-v2`, timestamp 2026-06-16 — predates the v0.5.15 flip AND patch 057 (MistralCommonBackend opt-out) which restored devstral needle/HE/tool parity, so a cell delta cannot be attributed to 058 alone. devstral has no claw-code / little-coder cells (only `bakeoff-devstral-opencode.json` exists in `benchmarks/quality/`).
- 3090 instruments confirmed present: `scripts/test_patch_gates.sh` (scripted 3-gate: pristine glob-order apply / byte-identity vs live tree / re-apply rejection; requires SGLANG_DIR, default `/data/sglang-rebase-v0515` from `scripts/common.sh`), `evals/swebench/run_model_cycle.sh` (SCAFFOLDS env override, Rule-2 sequencing, full-300 default), `scripts/eval/validate_capabilities.py`, `scripts/eval/probe_256k_tooluse.py`. Serving env `sglang-v0515`, default PORT 23334 (`common.sh` line 39).
- devstral preset (launch.sh lines 79-124): `--tool-call-parser mistral --sampling-defaults model`, CTX 262144, canonical template `scripts/devstral2_chat_template.jinja` — the parser and template this port exercises.
- Patch-count surfaces that must be bumped 24->25 on landing: `scripts/setup.sh` line 5 comment and README.md line 429 ('24 logical patches').


## Method

1. Preflight (GPU-free, on the 3090 box): confirm parity of the donor base — `diff <(grep -v '^index' ~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/patches/040-devstral-toolcall-omission-recovery.patch) <(grep -v '^index' patches/041-devstral-toolcall-omission-recovery.patch)` must print nothing (verified 2026-07-18 from the R9700 checkout).
2. Pre-change gate sanity: `SGLANG_DIR=/data/sglang-rebase-v0515 scripts/test_patch_gates.sh` must be green at 24/24 before touching anything (isolates any new failure to this work).
3. Write `scripts/eval/test_mistral_detector_prefix_holdback.py`: instantiate `sglang.srt.function_call.mistral_detector.MistralDetector`, tools = read/task/todowrite/webfetch; drive `parse_streaming_increment(chunk, tools)` with 1-4-char chunks and concatenate `normal_text`/`calls`. 8 cases mirroring R9700 commit 9628f9b: (1) `todowrite[ARGS]{...}` split as `todo`|`write`|`[ARGS]`|json -> 1 structured call, name never in normal_text; (2) same for `webfetch`; (3) single-token `task[ARGS]{...}` -> recovered (041 behavior preserved); (4) canonical `[TOOL_CALLS]` form -> parsed; (5) prose ending in a tool-name-prefix word that then diverges (`todo list...`) -> flushed verbatim, 0 calls; (6) prose containing a full tool name with no `[ARGS]` -> flushed, 0 calls; (7) stray `[ARGS]{...}` after an UNKNOWN identifier -> 0 calls; (8) non-streaming `detect_and_parse` omission recovery intact. Run in the `sglang-v0515` env (activate via `scripts/common.sh` + `activate_conda`): cases 1-2 MUST FAIL now — capture the output as the mechanism baseline.
4. Apply the donor to the live tree: `cd /data/sglang-rebase-v0515 && git apply ~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference/patches/056-devstral-multitoken-toolname-omission-recovery.patch` (verified to apply strict-clean on a 041-patched `mistral_detector.py`; `py_compile` passes).
5. Regenerate as a numbered 3090 patch per the 3-gate discipline (committed-baseline method): `git worktree add /tmp/wt-058 v0.5.15`; apply `patches/*.patch` (all 24) in glob order; `git add -A && git commit` (baseline); apply the R9700 donor; `git diff > patches/058-devstral-multitoken-toolname-prefix-holdback.patch`. The diff is generated against the predecessor-patched tree, never pristine.
6. Run the full gate: `SGLANG_DIR=/data/sglang-rebase-v0515 scripts/test_patch_gates.sh` -> gate (a) 25/25 apply on pristine v0.5.15, gate (b) byte-identical to live tree, gate (c) 25/25 rejected on re-apply. Capture output.
7. Re-run the unit test -> 8/8 PASS. Commit patch + test + doc updates (patches/README.md 058 row, strip row-041's ⚠ flag, README fleet-audit checkbox, patch counts 24->25 in setup.sh comment and README line 429).
8. Live e2e A/B (minutes of GPU): port R9700's `scripts/eval/devstral2_toolprobe.py` (change default port to 23334); `./scripts/launch.sh devstral`; run 10 turn-2 (todowrite) trials post-058: require >=9/10 `finish_reason=tool_calls` with structured `todowrite` call and 0/10 leak signature (`[TOOL_CALLS]`/`[ARGS]` in content). Then regression probes on the same server: `scripts/eval/validate_capabilities.py --port 23334` (3/3 basic+tool_call+vision) and `scripts/eval/probe_256k_tooluse.py` (devstral must stay 1.0/1.0, 132K-firm per README tool-use table). Receipt to `benchmarks/quality/devstral-toolprobe-058-<date>.txt`.
9. Observational cell re-run (schedule per Rule 1/2, detach via setsid): `SCAFFOLDS="opencode" ./evals/swebench/run_model_cycle.sh devstral` — full 300, regenerates `benchmarks/quality/bakeoff-devstral-opencode.json`. Before starting, if the old run dir `evals/swebench/runs/devstral-opencode-v2` still exists on the eval box, grep its transcripts for the leak signature (`todowrite[ARGS`, `webfetch[ARGS`, bare `[ARGS]{` in assistant content) and record the incidence count — that quantifies how much of the 173-empty-patch class this defect explains, for free, before burning GPU.
10. Report adoption to R9700 (sister-README ask): their upstream-PR package `patches/upstream-prs/main/mistral-toolcall-omission.patch` does NOT include the prefix extension (grep verified) — flag that the eventual 041-class upstream PR should carry it.


## Baseline & instrument

Two-part: (1) mechanism baseline — run the new offline unit test against the CURRENT live tree (v0.5.15 + 24 patches) in the `sglang-v0515` env; the multi-token streaming cases MUST fail by leaking the first name piece as `normal_text` (deterministic receipt that the defect exists pre-058); (2) cell baseline on record — `benchmarks/quality/bakeoff-devstral-opencode.json`: 40/300 resolved, 173/300 empty_patch (2026-06-16, pre-v0.5.15/pre-057 stack — confounded, comparison is observational only).


## Success criteria

- 3-gate green via scripts/test_patch_gates.sh: (a) 25/25 patches apply on pristine v0.5.15, (b) patched worktree byte-identical to /data/sglang-rebase-v0515, (c) 25/25 rejected on re-apply.
- Unit test scripts/eval/test_mistral_detector_prefix_holdback.py: multi-token cases FAIL pre-058 (captured) and 8/8 PASS post-058, GPU-free in the sglang-v0515 env.
- Live probe (port of R9700 devstral2_toolprobe.py, port 23334): >=9/10 turn-2 todowrite trials return finish_reason=tool_calls with a structured call; 0/10 leak signatures in content.
- No serving regression: validate_capabilities.py devstral 3/3 (basic+tool_call+vision); probe_256k_tooluse.py devstral unchanged at 1.0/1.0 (132K-firm).
- Observational: regenerated full-300 devstral×opencode cell committed with empty_patch count reported against the 173/300 June cell (stack-flip confound stated in the README/patch-map note).
- Docs: fleet-audit item checked off, patches/README.md row 058 added + row-041 flag removed, patch counts bumped 24->25.


## Kill criteria

- Step-3 kill: if the pre-058 unit test's multi-token cases PASS on the current v0.5.15+24-patch tree, the defect is already fixed (upstream or 057 side-effect) — set the fleet-audit item done-as-null, record in patches/README.md row 041, do NOT port.
- Regression kill: any single-token or canonical-[TOOL_CALLS] unit case failing post-058, or any 3-gate failure that survives one regeneration cycle — revert the live tree (`git checkout -- python/sglang/srt/function_call/mistral_detector.py` + re-apply 041) and record.
- Efficacy kill: post-058 live probe still shows leak signatures in >=2/10 turn-2 trials — the port is insufficient on this stack; timebox investigation to 2 h, then revert and record the negative finding with the probe receipt (negative results are findings).
- Do NOT kill on the bake-off cell staying weak: 173 empty-patches are multi-cause (repetition loops, timeouts, model capability); the cell is observational, not the port's pass/fail gate.


## Deliverables

- patches/058-devstral-multitoken-toolname-prefix-holdback.patch (in /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/patches/)
- scripts/eval/test_mistral_detector_prefix_holdback.py — committed offline unit test (8 cases), with pre-058 FAIL output and post-058 8/8 PASS output captured in the commit message or benchmarks/quality/ note
- scripts/eval/devstral2_toolprobe.py — port of R9700's probe (default port 23334) + A/B receipt benchmarks/quality/devstral-toolprobe-058-2026-07.txt (pre/post leak counts over 10 turn-2 trials)
- Regenerated devstral×opencode full-300 cell: benchmarks/quality/bakeoff-devstral-opencode.json + run dir evals/swebench/runs/devstral-opencode-v2 (re-run on current stack)
- Doc updates in the same commit: patches/README.md new 058 row + strip the ⚠ extension flag from row 041; README.md fleet-audit checkbox line 9 removed/checked; patch counts 24->25 in setup.sh line-5 comment and README line 429
- 3-gate receipt: scripts/test_patch_gates.sh output (25/25 a, byte-identical b, 25/25 rejected c) quoted in the commit


## Constraints

- READ the 3090 ops rules: Rule 1 (no concurrent calibration + serving/eval) and Rule 2 (no concurrent rollout + score; run_model_cycle.sh already sequences — don't bypass).
- One mechanism at a time: no other serving/preset/template changes ride along with 058 in any A/B; the unit test + live probe are the attribution instruments, not the bake-off cell.
- Patch hygiene (CLAUDE.md 3-gate): generate 058 against the predecessor-patched tree (never pristine), unique anchoring context, then `scripts/test_patch_gates.sh` green before commit.
- Detach the bake-off cycle via the setsid pattern (>30 min job); docker rollout I/O can trigger the ~9-17h kernel-BUG reboot — predictions on disk survive; resume with `--skip-existing`; if `swebench-bakeoff.service` auto-starts the default queue on reboot, `sudo systemctl stop swebench-bakeoff` before relaunching the devstral-only run.
- Validate behavior, not exit status: read actual probe/unit-test output; a leak signature (`[ARGS]` in message content) is a fail even if the script exits 0.
- This is detector/streaming logic — no checkpoint is touched, so no modality re-validation of weights is needed, but run the standard capability probes to prove no serving regression.


## Risks

- Kernel-BUG reboot during the docker rollout (~9-17h exposure window) — mitigated by setsid + --skip-existing resume; predictions persist.
- Over-attribution: the June cell's 173 empty patches are multi-cause; the stack has since gained 057 (tokenizer-backend fix) which alone moves devstral quality — any cell improvement is 057+058+stack combined. The unit test and probe carry attribution; state this in the README update.
- Live probe stochasticity: [TOOL_CALLS] omission is intermittent and temperature-dependent — a passing pre-058 probe would NOT prove absence of the defect (don't run it as the baseline; the unit test is the deterministic instrument).
- Prefix-hold delays flushing prose words that prefix a tool name by one streaming increment (never dropped) — cosmetic latency accepted on R9700 2026-06-15; watch for it in probe content, not a regression.
- Diff-context drift: if any future 3090 patch starts touching mistral_detector.py before this lands, regenerate 058 against the new predecessor chain (currently only 041 touches it).


---
*Vetted 2026-07-18: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
