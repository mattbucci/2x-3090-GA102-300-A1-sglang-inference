# 3090-H: Shared ~/AI/models manifest + defuse the BF16-name→AWQ-weights symlink + user-gated purge brief

| | |
|---|---|
| **Type** | task |
| **Status** | ready |
| **Execution host** | multi-rig |
| **Wall clock** | 3-5h authoring+receipts; purge execution gated on Matt, minutes once approved |
| **GPU time** | none — pure filesystem/script work; the calibration scripts already hide the GPU (CUDA_VISIBLE_DEVICES=\"\") so no allocation ever happens, and the guards must abort BEFORE the 49G weight/model load |
| **Depends on** | External: Matt's sign-off required before any deletion (purge half only; manifest/defuse/guards proceed without it); Host access: manifest runs and symlink fixes on the 3090 eval + calib boxes must be executed by the 3090 team on-box (audited-host state is receipted here from the R9700 box); the R9700-host trap symlink defuse is the R9700 team's on-box filesystem action |
| **Provides to** | R9700 team: guard-port notice for their identical run_all_calibrations.sh:60 / quantize_gemma4_26b_thinking_vision.py copies, an instruction to defuse the trap symlink on their own host filesystem, the test_glm_moe_isolation.py:133 path fix if Tier A lands, and the finding that coder-next's default MODEL Qwen3-Coder-Next-AWQ is absent at ~/AI/models on their host; All three rigs: MANIFEST.json as the authoritative what-is-this-dir record for every ~/AI/models consumer (launch presets, quantize pipelines, disk_hygiene); 3090 calibration backlog (README Active work #3-4): trustworthy BF16 bases for every future gemma4-26b recalibration; Disk headroom on the audited host (~85G Tier A, ≈96G with small dirs) for the parked bigger-RAM work (e.g. GLM own-build resume) |

## Objective

All four fleet entry points default MODELS_DIR=$HOME/AI/models (3090 scripts/common.sh:38, R9700 scripts/common.sh:36, M4 scripts/common.sh:13, 3090 scripts/quantize/run_all_calibrations.sh:40), yet the dir has no manifest, and the name gemma-4-26B-A4B-it-BF16 is a symlink that resolves to AWQ-int4 weights — the exact path run_all_calibrations.sh feeds as the gemma4-26b calibration BF16 base, i.e. a queued silent double-quantization of the 16h-loss class. Produce a per-host manifest generator with trap detection, retarget the symlink to the real BF16, add loud guards to the quantize entry points, and hand Matt an itemized purge brief (~96G immediately-actionable + ~82G verify-then-propose); the audited host's filesystem is at 99% (22G free), so approved deletions also unblock disk headroom.


## Background & receipts

- Trap verified live: ls -la ~/AI/models shows gemma-4-26B-A4B-it-BF16 -> gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed -> /data/cache/huggingface/hub/models--mattbucci--gemma-4-26B-AWQ/snapshots/06b09ed1...; its config.json has quantization_config quant_method=awq bits=4. The real base sits case-colliding at ~/AI/models/gemma-4-26b-a4b-it-BF16 (lowercase, 49G, 2 shards, tokenizer.json present, NO quantization_config).
- Symlink predates the real BF16: link mtime May 26 01:57, lowercase dir May 27 07:09 — the alias was created before the true base existed and never retargeted.
- Double-quantization vector receipted: 3090 scripts/quantize/run_all_calibrations.sh:60 passes $MODELS_DIR/gemma-4-26B-A4B-it-BF16 as the gemma4-26b BF16 base (field 7, BF16_BASE, of the JOBS row); scripts/quantize/quantize_gemma4_26b_thinking_vision.py:58 defaults BF16_MODEL to the same path; scripts/quantize/run_full_pipeline.sh:54 sets BF16_BASE to it; scripts/quantize/merge_vision_weights.py:16 documents it as --base. R9700 carries identical copies (their run_all_calibrations.sh:60, quantize_gemma4_26b_thinking_vision.py) plus launch.sh:303 GEMMA_TOK tokenizer probe and debug scripts scripts/debug/repro_gemma4_hsail.py:44 (TOKENIZER_PATH = MODELS / "gemma-4-26B-A4B-it-BF16") and scripts/debug/awq_kernel_layer0_test.py:22 (BF16 = Path(os.environ.get("BF16_MODEL", ...gemma-4-26B-A4B-it-BF16))). All 8+ consumers expect BF16 semantics at that name; none needs AWQ there — and the retargeted lowercase base carries tokenizer.json so the launch.sh:303 tokenizer probe stays satisfied — so retargeting the link fixes every consumer in both repos with zero code edits.
- Existing tooling gap: 3090 scripts/maint/disk_hygiene.sh targets MODELS_DIR=/data/models (3090-box-local), not ~/AI/models; its gc_bases loop uses [ -d "$path" ] which symlinks-to-dirs pass, and candidate_repo (disk_hygiene.sh:51) maps gemma-4-26B-A4B-it-BF16 -> google/gemma-4-26B-A4B-it, so the trap link would be reported as a RE-DOWNLOADABLE base. (Note: du of the symlink argument reports the link size (~0), not the 49G target, unless -L is passed — the flag is the re-downloadable classification, not an inflated size figure.) No manifest exists: find ~/AI/models -maxdepth 1 -type f returns nothing.
- Purge Tier A (verified superseded, ~85G by du): GLM-4.5-Air-REAP-AWQ 43G — superseded by GLM-4.5-Air-REAP-AWQ-native which is the served preset (R9700 launch.sh:267); sole remaining ref is R9700 scripts/test_glm_moe_isolation.py:133 debug hardcode. Qwen3.5-27B-AWQ-CT-gdn 25G and Qwen3.5-27B-AWQ-CT-toolcall 17G — CT-stage intermediates whose final AWQ exports exist alongside (Qwen3.5-27B-AWQ-gdn 26G, -toolcall 18G, both referenced by R9700 scripts/eval/int4_agentic_sweep/); R9700 scripts/download_all_awq.sh:2 explicitly excludes '-CT' from ships.
- Small spec-decode experiment dirs ~11G (immediately-actionable alongside Tier A): Qwen3.5-27B-DFlash 3.3G, gemma-4-31B-it-DFlash 2.9G, Qwen3.6-35B-A3B-DFlash 0.9G, Qwen3.6-27B-EAGLE3-specdrift 1.2G, Gemma-4-31B-Eagle3 1.3G, gemma-4-31B-it-assistant 0.9G, Aurora-Spec-Coder-Next 1G; DFlash is receipted net-negative on DeltaNet-MoE. Tier A 85G + small dirs 11G ≈ 96G of immediately-actionable reclaim.
- Purge Tier B (verify-then-propose, AutoRound pair ~82G): Qwen3-Coder-Next-int4-AutoRound 41G + -routerbf16 41G — no launch preset in any repo references either; download_all_awq.sh:2 calls AutoRound 'legacy'; R9700 coder-next preset (launch.sh:225) serves $MODELS_DIR/Qwen3-Coder-Next-AWQ which does NOT exist on the audited host (ls: No such file or directory — separate finding for R9700).
- NOT purge candidates despite looking stale: EAGLE3-Coder-30B-A3B (361M) is the live SPEC_DRAFT at R9700 launch.sh:800; gemma-4-12B-AWQ (7.3G) is served by R9700 launch.sh:373; all mattbucci snapshot symlinks resolve into /data/cache/huggingface/hub (verified resolvable on the audited host).
- Disk pressure receipt: df -h ~/AI/models on the audited (R9700) host = /dev/nvme1n1p2 1.8T, 99% used, 22G free; du -sh ~/AI/models = 1.2T. 3090-host layout of ~/AI/models unverified from here (only their repo defaults are receipted) — step 1 turns that into a per-host receipt.
- Host ownership note: the trap is verified LIVE on this R9700 (audited) host, but per constraints repo edits land only in the 3090 repo and R9700 scripts/filesystem are the R9700 team's. Defusing the actual ~/AI/models symlink ON the R9700 host is a filesystem action owned by the R9700 team (carried in the guard-port notice), distinct from the 3090-box defuse.
- Fleet-audit queue bullet being executed: 3090 README.md:16.


## Method

1. Recon each host (3090 eval box, 3090 calib box; audited-host values already in background): run `findmnt -T ~/AI/models; readlink -f ~/AI/models; df -B1 --output=size,avail ~/AI/models; ls -la ~/AI/models | head -50` and record whether ~/AI/models exists, is local or a link into /data/models, and whether the May-26 gemma-4-26B-A4B-it-BF16 symlink is replicated there (`readlink -f ~/AI/models/gemma-4-26B-A4B-it-BF16`).
2. Author scripts/maint/models_manifest.py in the 3090 repo (stdlib only, no GPU): for every top-level entry emit {name, entry_type dir|symlink, readlink -f target, du -sb bytes (use -L / resolve through the symlink so the reported size is the real target, not ~0), mtime, shard_count from model.safetensors.index.json (or single-shard), quant = config.json quantization_config quant_method+bits or 'none/BF16', torch_dtype, consumers = grep -rl of the entry name across all repo checkouts found under ~/AI (best-effort, record which checkouts were scanned)}. Built-in flag rules: TRAP = name contains 'BF16' (case-insensitive) AND quantization_config present; COLLISION = names equal under casefold(); DANGLING = symlink whose target does not exist. CLI: `python scripts/maint/models_manifest.py --models-dir ~/AI/models --out ~/AI/models/MANIFEST.json --md ~/AI/models/MANIFEST.md`.
3. Run the generator on each host; commit the JSON as benchmarks/models-manifest-<hostname>-<date>.json in the 3090 repo. First run MUST flag gemma-4-26B-A4B-it-BF16 as TRAP and the upper/lowercase pair as COLLISION on the audited host — that is the generator's self-test.
4. Rule-1 precondition (calib box only): before editing or running anything against the shared quantize scripts on the calibration box, confirm no calibration is in flight — e.g. `pgrep -f run_all_calibrations` and `pgrep -f quantize_.*_thinking_vision` return nothing. If a calibration is running, defer the script edits (steps 5-6) until it completes; manifest/recon (steps 1-3) are read-only and may proceed.
5. Defuse: on every host where the trap resolves to AWQ, retarget instead of delete (all verified consumers want BF16 semantics at this name). PRECONDITION (case-insensitive-fs guard): before retargeting, assert the fs is case-sensitive — `realpath ~/AI/models/gemma-4-26B-A4B-it-BF16` (current link) must differ from `realpath ~/AI/models/gemma-4-26b-a4b-it-BF16` (target); if they are already equal the link name and target are the same fs entry (APFS/case-insensitive) and retarget would create a self-referential loop — abort and record host divergence. Then: `ln -sfn gemma-4-26b-a4b-it-BF16 ~/AI/models/gemma-4-26B-A4B-it-BF16`. Verify: `readlink -f` now ends in .../gemma-4-26b-a4b-it-BF16, `python3 -c "import json; c=json.load(open('/home/letsrtfm/AI/models/gemma-4-26B-A4B-it-BF16/config.json')); assert 'quantization_config' not in c"` passes, and the index lists 2 shards. Re-run the manifest: zero TRAP flags. (On the R9700 host this filesystem retarget is the R9700 team's action, carried in the guard-port notice.)
6. Guard the quantize entry points in the 3090 repo so the trap class cannot recur silently. In scripts/quantize/quantize_gemma4_26b_thinking_vision.py (after resolving BF16_MODEL, ~line 58) abort loudly if <BF16_MODEL>/config.json contains quantization_config. In scripts/quantize/run_all_calibrations.sh, apply per-row to field 7 (BF16_BASE) with this exact scoping so valid rows are not false-aborted: if BF16_BASE is empty -> skip (qwen36-moe and coder-30b legitimately have empty field 7); if BF16_BASE is not an existing local directory -> skip (devstral's mistralai/Devstral-Small-2507 is a legitimate HF repo id, not a local path); only if <base>/config.json exists AND contains quantization_config -> abort loudly. Negative test as receipt: `BF16_MODEL=~/AI/models/gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed python scripts/quantize/quantize_gemma4_26b_thinking_vision.py` must exit non-zero BEFORE any model/weight load (the script already sets CUDA_VISIBLE_DEVICES="" at line 45 before importing torch at line 49, so the guard's job is to stop the 49G weight load + calibration compute, not GPU allocation); capture output.
7. Patch scripts/maint/disk_hygiene.sh gc_bases to skip symlinks (add `[ -L "$path" ] && { echo "  SKIP (symlink) $d"; continue; }` ahead of the -d test) so a BF16-named link is never reported as a re-downloadable base.
8. Write benchmarks/models-purge-brief-<date>.md for Matt: an immediately-actionable table = Tier A (GLM-4.5-Air-REAP-AWQ 43G, Qwen3.5-27B-AWQ-CT-gdn 25G, Qwen3.5-27B-AWQ-CT-toolcall 17G = ~85G) plus the ~11G small spec-decode experiment dirs (≈96G total) — each row: size, supersessor + serving receipt, remaining consumers from the manifest consumer-scan, exact rm command; and a verify-then-propose table = Tier B AutoRound pair (Qwen3-Coder-Next-int4-AutoRound 41G + -routerbf16 41G ≈82G) with the open provenance questions. Reuse disk_hygiene.sh's hf_exists check to mark which candidates are the sole copy of a shipped mattbucci artifact (those need explicit call-out). Include the R9700 side-finding that coder-next's default MODEL path Qwen3-Coder-Next-AWQ is absent on the audited host.
9. Only after Matt's written sign-off: execute approved deletions, re-run the manifest, commit the post-purge benchmarks/models-manifest-<hostname>-<date>.json plus df before/after, and tick the README.md:16 queue checkbox. Notify the R9700 team (via their README per reference-sister-teams) to add the same guard to their copies of run_all_calibrations.sh / quantize_gemma4_26b_thinking_vision.py, to defuse the trap symlink on their own host filesystem, and to fix test_glm_moe_isolation.py:133 if Tier A is approved.


## Baseline & instrument

First MANIFEST.json run per host (du -sb per entry through resolved targets + df -B1 ~/AI/models) is the baseline artifact; audited-host pre-state already receipted in background: 1.2T dir on a 99%-full 1.8T fs, 22G free, trap symlink present.


## Success criteria

- scripts/maint/models_manifest.py committed; on the audited host it covers 100% of top-level entries and its first run flags exactly the known TRAP (gemma-4-26B-A4B-it-BF16) and COLLISION (upper/lowercase gemma pair) — receipt: benchmarks/models-manifest-<hostname>-<date>.json.
- Post-defuse: readlink -f ~/AI/models/gemma-4-26B-A4B-it-BF16 resolves into gemma-4-26b-a4b-it-BF16, the config.json assert (no quantization_config) passes, and a manifest re-run reports zero TRAP flags on every host that had one.
- Guard negative-test receipt shows quantize_gemma4_26b_thinking_vision.py exits non-zero on an AWQ-quantized base BEFORE the 49G weight/model load, the run_all_calibrations.sh guard aborts the gemma4-26b row on an AWQ base while NO-OPing (skipping) on the empty-field-7 rows (qwen36-moe, coder-30b) and the HF-id base (devstral), and both still run normally (past the guard) on the true BF16 base.
- disk_hygiene.sh gc-bases dry-run no longer lists any symlink as a re-downloadable base.
- benchmarks/models-purge-brief-<date>.md delivered with per-dir evidence and exact commands; zero bytes deleted without Matt's sign-off; if approved, post-purge manifest + df receipt shows the freed space (Tier A alone ≈85G, with small dirs ≈96G).


## Kill criteria

- Symlink defuse: if the manifest consumer-scan or host recon surfaces ANY consumer that requires AWQ content at the uppercase-BF16 name (none found in both repos today), stop, do not retarget, and escalate to Matt with the receipt.
- Case-insensitive filesystem: if the retarget precondition finds realpath(link)==realpath(target) (link and lowercase base are the same fs entry, e.g. APFS), do NOT retarget — record host divergence and scope the defuse out for that host.
- Purge rows: any candidate that turns out to be referenced by a live launch preset, or to be the sole copy of a shipped mattbucci artifact absent from HF (hf_exists 404), is struck from the brief — recorded as a KEEP row with reason, not deleted.
- Host divergence: if a 3090 host has no ~/AI/models or a materially different layout, do not force the convention — record the per-host reality in the manifest header as a finding and scope the symlink/purge steps to hosts where the entries actually exist.
- If Matt declines the purge, land manifest + defuse + guards anyway and record the brief as a standing decision doc — the disk-space half becomes a null with receipts, not a retry loop.


## Deliverables

- /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/scripts/maint/models_manifest.py (new generator, stdlib-only)
- ~/AI/models/MANIFEST.json + ~/AI/models/MANIFEST.md per host, with committed receipts at /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/benchmarks/models-manifest-<hostname>-<date>.json (pre- and post-change)
- Retargeted symlink ~/AI/models/gemma-4-26B-A4B-it-BF16 -> gemma-4-26b-a4b-it-BF16 on every affected Linux host (case-sensitivity precondition asserted), with readlink+config-assert verification in the receipt
- Guard edits: scripts/quantize/quantize_gemma4_26b_thinking_vision.py (abort on quantized BF16_MODEL before weight load), scripts/quantize/run_all_calibrations.sh (per-row field-7 guard: skip empty, skip non-local/HF-id, abort only on local quantized base), scripts/maint/disk_hygiene.sh (gc_bases symlink skip) + negative-test output receipt
- /home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference/benchmarks/models-purge-brief-<date>.md — immediately-actionable Tier A ~85G + small dirs ~11G (≈96G) / verify-then-propose Tier B AutoRound ~82G itemized decision brief for Matt
- README.md fleet-audit queue line 16 checked off with pointer to the receipts


## Constraints

- Deletions are user-gated: the brief proposes, Matt disposes — mirrors disk_hygiene.sh's stated SAFETY RULE (in-house quants never auto-deleted).
- No serving, benches, or GPU work involved; the calibration scripts already hide the GPU (CUDA_VISIBLE_DEVICES=""), so guards must trip before the 49G weight/model load. Do not edit the quantize scripts while a calibration is in flight on the calib box — check pgrep first (Rule 1 discipline).
- Repo edits land only in the 3090 repo; R9700's script copies AND its host-side symlink defuse are their team's to change (share via README per fleet convention).
- Retarget, don't delete, the trap symlink — 8+ verified consumers across two repos resolve BF16 semantics through that name; deletion converts a silent trap into scattered breakage.
- Negative results are findings: a declined purge or a divergent 3090-host layout gets a receipt, not silence.


## Risks

- 3090-host ~/AI/models layout is unverified from the authoring box (their disk_hygiene.sh uses /data/models; hf-mattbucci/ subdir exists there but not on the audited host) — step 1 recon de-risks before any change; consumer grep is best-effort where sister checkouts are absent.
- M4 is APFS (typically case-insensitive): the upper/lowercase gemma pair cannot coexist if ever synced to the Mac — the COLLISION flag, the step-5 realpath precondition, and the retarget (making both names one content on Linux) contain this, but any rsync of the old layout to M4 would silently merge them.
- mattbucci snapshot symlinks point into /data/cache/huggingface/hub — purge commands must never follow links (rm the link target only when the target itself is the approved candidate); manifest records resolved targets precisely to prevent this.
- Consumer-scan false negatives: paths built by string concatenation or env override won't grep-match a dir name; brief mitigates by also requiring the hf_exists sole-copy check and Matt's eyeball before rm.
- Tier B AutoRound pair (82G) has weak provenance (one FINDINGS.md line about an incoherent Coder-Next AutoRound conversion, R9700 benchmarks/FINDINGS.md:232) — kept out of the immediately-actionable bucket deliberately; deleting the wrong one could destroy the only working router-bf16 variant.
- run_all_calibrations.sh guard must be scoped exactly (skip empty field-7, skip non-local/HF-id bases): a naive 'every row' <base>/config.json check would false-abort qwen36-moe, coder-30b (empty base) and devstral (mistralai/Devstral-Small-2507 HF id), regressing the shared production calibration script — the guard fires only on a local dir whose config.json contains quantization_config.


---
*Vetted 2026-07-18: drafted against live repo state, adversarially checked (feasibility + design), revised. Part of the fleet-audit experiment queue.*
