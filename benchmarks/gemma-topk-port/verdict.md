# 3090-F verdict — decode-topk sparse-KV on gemma4-31b: WIN (2.03× @262K, exact recall)

Lane 3090-F (experiments/09), executed 2026-07-19 on v0.5.15 + 25 patches.
Donor: R9700 `069-decode-topk-sparse.patch.CANDIDATE` (their 1.77× @245K was on
pure-full-attention Qwen3-VL-32B, gfx1201) — this is the first SWA-hybrid and
first CUDA/sm_86 datapoint for 069, rebased to v0.5.15 as **patch 059**.

## Pre-check (go/no-go gate)

Eager control (`_ENV_GEMMA_GRAPH` disable, zero code), fixed instrument:
TPOT 33.5ms @2,048 / 79.0ms @261,916 actual. Eager penalty = 33.5−18.5 =
**15.0 ms/step**; depth-attributable = 58.9 ms → 15.0 < 0.5×58.9 → **GO**.
Sub-finding: at 262K the graphs-on and eager TPOT nearly converge (77.4 vs
79.0ms) — cuda graphs are worth ~everything at short depth (18.5 vs 33.5) and
~nothing at full depth on this preset.
26B arm: killed by the pre-check rule as predicted (eager penalty 17.2 ≥
0.5×29.4) — not ported, per spec.

## Deep A/B (gemma4-31b, budget 256 pages × 64 = 16,384 tokens ≈ 6.2% of 262K)

| depth (actual) | graphs-on baseline | topk arm (graphs off) | ratio |
|---:|---:|---:|---:|
| 2,048   | 53.3 tok/s (18.8ms) | 29.3 (34.1ms) | 0.55× |
| 32,768  | 38.7 (25.8ms) | 27.2 (36.8ms) | 0.70× |
| 131,072 | 20.8 (48.1ms) | 26.6 (37.5ms) | **1.28×** |
| 261,916 | 12.9 (77.4ms) | **26.2 (38.1ms)** | **2.03×** |

The topk depth curve is essentially FLAT (34.1 → 38.1ms over 128×
context growth): selection overhead ≈ 4.6 ms/step at 262K replaces the 45.5
ms/step eager depth term. **Crossover ≈ 80-90K** (baseline crosses ~37ms
between 65K and 131K). Gate ≥1.3× @250K: **PASS at 2.03×** (predicted ~1.9×).

## Recall + capability gates (same server, topk on)

- `probe_256k_tooluse` ladder: **1.0 valid / 1.0 correct at every rung**,
  exact id through **255,957 actual tokens** — equals the preset's no-topk
  receipt; exceeds the donor's ±1-char near-exact class.
  (fp8_e5m2-KV bbox scoring concern: resolved by measurement — exact recall.)
- `validate_capabilities`: **5/5** (basic, tool, thinking, vision, video).
- `probe_256k_quality` multi-depth reasoning ladder: **100% at every rung**
  (multikey + variable-tracking + aggregate, 1,159 → 255,800 actual tokens).
- One transient: the first quality-probe attempt wedged the server's
  detokenizer ~40 min into the session (watchdog kill, heartbeat lost at
  20:26:44); the identical probe on a fresh boot ran clean end-to-end — logged
  as a one-off watchdog event, not a 059 defect (no repro).
- Agentic A/B (R9700 gate adopted 2026-07-19, `scripts/eval/topk_agentic_ab_gemma.sh`
  + depth-binned `context_reliability_curve.py`): same 6 Lite instances, Docker
  harness, only variable = the flag. Results appended below when complete.

## Ship shape

Patch `patches/059-decode-topk-sparse.patch` (opt-in, default off; decode
cuda-graph auto-disabled only when enabled). Presets UNCHANGED — graphs-on
default wins below the ~80-90K crossover and agentic median is ~41K. Opt-in
hook: `_ENV_GEMMA_TOPK="--decode-topk-pages 256 --decode-topk-page-size 64"`
on any gemma preset for deep-context single-user sessions.
