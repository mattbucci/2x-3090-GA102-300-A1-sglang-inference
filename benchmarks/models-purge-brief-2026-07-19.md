# Models-dir manifest + BF16→AWQ trap defuse — 3090 eval box (letsrtfm-amd)

Fleet-audit lane 3090-H, executed 2026-07-19 on the 3090 **eval** box. Manifest,
calibration guards, and disk_hygiene fix are **landed**; the purge half is a
**null-with-receipt** on this host (no disk pressure, no candidates present).
Deletions of any model data remain user-gated (propose, Matt disposes).

## Host reality (differs materially from the R9700 audited host)

| | 3090 eval box (`letsrtfm-amd`) | R9700 audited host |
|---|---|---|
| `~/AI/models` | symlink → `/data/models` (nvme1n1p1, ext4, whole-disk) | resolves onto nvme1n1p2 |
| Models fs | 1.8T, **69% used, 554G free** | 1.8T, **99% used, 22G free** |
| gemma-4-26B-A4B-it-BF16 **trap** | **ABSENT** (both upper/lowercase names absent) | **PRESENT** (BF16 name → AWQ weights) |
| Manifest flags | 2 DANGLING, **0 TRAP, 0 COLLISION** | expected 1 TRAP + 1 COLLISION |

Receipt: `benchmarks/models-manifest-letsrtfm-amd-2026-07-19.json` (46 entries,
1312.4G resolved). Generator: `scripts/maint/models_manifest.py` (stdlib-only).
Two-nvme layout documented in `rules-for-agents.md` "Host filesystem layout".

**The disk-pressure/purge urgency is R9700-specific.** None of the R9700
Tier A/B candidates (GLM-4.5-Air-REAP-AWQ, Qwen3.5-27B-AWQ-CT-{gdn,toolcall},
Qwen3-Coder-Next-int4-AutoRound pair, the DFlash spec-decode dirs) exist on this
box — all verified absent. Inverse side-finding: `Qwen3-Coder-Next-AWQ`
(46G, awq-4bit) **is present here**, whereas the audit reported it absent on R9700.

## This host — actionable items

1. **Symlink defuse: N/A.** The BF16→AWQ trap name is absent; nothing to retarget
   (kill-criterion "host divergence" recorded, not forced).
2. **2 DANGLING alias links** (manifest-flagged, **zero data**, safe to remove — a
   broken symlink stores nothing; not the live serving path):
   - `Qwen3.6-35B-A3B-AWQ-thinking-vision` → `…/Qwen3.6-35B-A3B-AWQ-CT-thinking-vision` (target GC'd). *Not* the qwen36 serving path — the preset serves `hf-mattbucci/Qwen3.6-35B-A3B-AWQ → Qwen3.6-35B-A3B-AWQ-marlin-20260526` (real dir, healthy).
   - `Qwen3.6-35B-A3B-GPTQ-Int4` → GC'd HF-cache snapshot (`palmfuture/…`), old community reference, never a ship.
   - Cleanup (optional, no sign-off needed — deletes no data): `rm /data/models/Qwen3.6-35B-A3B-AWQ-thinking-vision /data/models/Qwen3.6-35B-A3B-GPTQ-Int4`
3. **No purge proposed.** 554G free is ample for the calibration backlog. The one
   pressure point is `/` at 84% (274G) from `/var/lib/docker` rollout images —
   handled by `scripts/maint/disk_hygiene.sh gc-docker`, unrelated to models-dir.

## Calibration guards landed (protect every host, incl. the calib box)

Both validated on-box 2026-07-19:

- `scripts/quantize/quantize_gemma4_26b_thinking_vision.py` — resolves `BF16_MODEL`
  before the torch import and **aborts (exit 1) if the base carries
  `quantization_config`**, before the 49G weight load. Negative test:
  `BF16_MODEL=…/gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed python3 …thinking_vision.py`
  → `exit=1` + FATAL, no torch loaded. Control (true BF16) passes the guard.
- `scripts/quantize/run_all_calibrations.sh` — per-row field-7 guard, scoped so it
  does not false-abort valid rows. Verified decisions:
  `qwen36-moe`/`coder-30b` (empty field-7) → SKIP; `devstral`
  (`mistralai/Devstral-Small-2507` HF id) → SKIP; a local AWQ base → ABORT; a true
  local BF16 base → PROCEED.
- `scripts/maint/disk_hygiene.sh` — `gc_bases` now skips symlinks, so a BF16-named
  alias is never classified re-downloadable.

## Cross-team → R9700 (their host, their filesystem actions)

Carried to their README (guard-port notice):
1. Add the same double-quant guard to their identical
   `run_all_calibrations.sh` / `quantize_gemma4_26b_thinking_vision.py` copies.
2. **Defuse the trap symlink on their host filesystem**: retarget
   `~/AI/models/gemma-4-26B-A4B-it-BF16` → the real lowercase
   `gemma-4-26b-a4b-it-BF16` (case-sensitive-fs precondition asserted). Retarget,
   don't delete — 8+ consumers across both repos resolve BF16 semantics there.
3. Fix `scripts/test_glm_moe_isolation.py:133` if their GLM Tier A purge lands.
4. This must complete **before any next gemma4-26b recalibration on either rig** so
   the recal reads a real BF16 base, not double-quantized AWQ.

## Standing decision

No deletion requested on this box; disk is healthy. Manifest + guards + hygiene
fix are the durable deliverables. Re-run `scripts/maint/models_manifest.py` after
any large model add/remove to keep the per-host record current.
