# Gemma 4 — `--triton-attention-num-kv-splits` bracket (2026-10-22)

**Lever:** the triton decode kernel's split-KV cap (CUDA default 8), the same flag that
took `flint` from 36 to 90 tok/s at 262K the same morning
([bracket](flint/kv-splits-bracket-2026-10-22.md)). The adaptive formula
(`get_num_kv_splits_triton`, `triton_backend.py`) wants `82 SMs × log2(seq/64) /
token_grid` splits and is clamped to this flag. All four Gemma 4 presets run the triton
backend (the only Ampere backend with `head_dim=256` + `SWAKVPool`), and at M=1 / TP=2
their token grids are tiny — 26B / 12B / 21B-REAP: 8 q-heads / 4 KV heads per rank →
grid 4 (wants ~246 splits at 262K, gets 8 = 32 CTAs on 82 SMs); 31B: 16 q / 8 KV per
rank → grid 8 (wants ~123, gets 8 = 64 CTAs). Every full-attention layer walks the 262K
KV in 8 serial chunks; the sliding-window layers size their own splits from the 1K
window and are unaffected. Prefill is untouched.

This retires the README's old explanation for the Gemma depth collapse ("group-32 AWQ +
TP-hostile shapes fall off Marlin") — a weight-kernel fallback cannot produce a
depth-dependent 3–4× slowdown, and the flag recovers it with the weights untouched.

**Method:** bare-metal `scripts/launch.sh <preset>` (TP=2 / 262144, cuda graphs on,
each preset's shipped KV dtype / SWA ratio) with `OVERRIDE_ARGS="--triton-attention-num-kv-splits N"`,
M=1, fresh prefill, 100 output tokens — `bench_regression.sh` (1K / 32K / 261,916
actual). Baseline column = the 2026-07-14 depth sweeps in `benchmarks/<preset>/results.json`
(`gemma4` also has the saved tripwire level 82.2 / 64.2 / 24.2, which the 64 run PASSed
against).

| preset | splits | 1K | 32K | 261,916 actual | TPOT @262K | deep gain |
|---|---:|---:|---:|---:|---:|---:|
| `gemma4` (26B MoE) | 8 (default) | 81.4 | 63.5 | 24.0 | 41.4 ms | — |
| | **64** | **81.8** | **77.6** | **58.3** | **17.2 ms** | **+143 %** |
| | 128 | 81.4 | 76.5 | 57.7 | 17.3 ms | +140 % |
| `gemma4-31b` (dense) | 8 (default) | 54.1 | 38.7 | 12.9 | — | — |
| | **64** | **55.0** | **49.3** | **29.6** | **33.8 ms** | **+129 %** |
| | 128 | 54.8 | 49.4 | 29.5 | 33.9 ms | +129 % |
| `gemma4-12b` (omni) | 8 (default) | 100.0 | 63.7 | 17.5 | — | — |
| | **64** | **103.8** | **93.7** | **57.7** | **17.3 ms** | **+230 %** |
| | 128 | 102.9 | 91.3 | 56.9 | 17.6 ms | +225 % |
| `gemma4-21b-reap` (MoE) | 8 (default) | 81.4 | 63.4 | 24.0 | — | — |
| | **64** | **81.5** | **77.3** | **58.3** | **17.2 ms** | **+143 %** |
| | 128 | 81.2 | 76.4 | 57.5 | 17.4 ms | +140 % |

tok/s single-user decode. 64 wins or ties 128 on every preset (128 pays ~1 % more
partial-result reduce work, exactly the flint shape), so **64 ships on all four**. The
depth curves are now nearly flat for the three grid-4 models (26B / 21B-REAP 82 → 58, 12B
104 → 58 over 1K → 262K) and halve instead of quartering on the 31B (55 → 30). The
decode-topk lever (patch 059, opt-in) had bought `gemma4-31b` 26.2 at depth by skipping
KV pages; 64 splits beats it with exact attention, so topk is no longer the deep-context
answer for Gemma — it stays available as a stacking experiment.

**Shipped:** every Gemma preset's `EXTRA_ARGS` →
`--triton-attention-num-kv-splits ${GEMMA_KV_SPLITS:-64}`; `gemma4` tripwire baseline
re-saved on the as-shipped level (81.8 / 77.6 / 58.3). Depth sweeps on the as-shipped
presets refresh `benchmarks/<preset>/results.json` and the fleet charts.

Receipts: `benchmarks/regression/<preset>-2026-10-22-kv{64,128}.json`; serve logs
in the session's `/var/tmp` chain. Prior levels: `benchmarks/gemma4-26b-awq/results.json`,
`gemma4-31b/results.json`, `gemma4-12b/results.json`, `gemma4-21b-reap/results.json`
(all 2026-07-14, default 8).
