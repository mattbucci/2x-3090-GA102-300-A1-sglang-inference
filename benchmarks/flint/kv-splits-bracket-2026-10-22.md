# Flint — `--triton-attention-num-kv-splits` bracket (2026-10-22)

**Lever:** the triton decode kernel's split-KV cap. CUDA default 8 (`server_args.py`;
HIP gets 16, R9700's patch 086 raises gfx1201 to 64 for the same reason). The adaptive
formula (`get_num_kv_splits_triton`, `triton_backend.py`) wants
`82 SMs × log2(seq/64) / token_grid` splits and is clamped to this flag. For Flint at
M=1 / TP=2 the token grid is 2 (16 q-heads and 2 KV heads per rank → `block_h` 8), so at
262K it asks for ~490 splits and gets 8: **16 CTAs on 82 SMs**, and each of the 14
full-attention NoPE layers walks 262K tokens in 8 serial chunks. Sliding-window layers
compute their own splits from the 513-token window and are unaffected.

**Method:** bare-metal `scripts/launch.sh flint` (TP=2 / 262144, AWQ-Marlin INT4, bf16
KV, cuda graphs on) with `OVERRIDE_ARGS="--triton-attention-num-kv-splits N"`, M=1,
fresh prefill, 100 output tokens — `bench_regression.sh` (1K / 32K / 261,916 actual) +
`bench_long_context.py --contexts 65536 131072`. Baseline = the saved tripwire level
(default 8; the docker-parity smoke re-measured it the same morning: 166.1 / 115.1 / 36.1).

| splits | 1K | 32K | 64K | 131K | 261,916 actual | TPOT @262K |
|---:|---:|---:|---:|---:|---:|---:|
| 8 (default) | 165.6 | 114.8 | 87.5 | 59.2 | 35.9 | 27.8 ms |
| 32 | 166.1 | 146.8 | 131.8 | 109.1 | 80.3 | 12.5 ms |
| **64** | **163.9** | **147.5** | **135.3** | **116.0** | **89.5** | **11.2 ms** |
| 128 | 161.3 | 144.5 | 132.3 | 113.1 | 88.1 | 11.4 ms |

tok/s single-user decode. 64 is the knee: +28 % @32K, +55 % @64K, +96 % @131K,
**+149 % @261,916**, 1K inside 1 %; 128 loses ~1–2 % everywhere (more partial-result
reduce work), 32 leaves 10 % on the table at depth (64 CTAs still under-fill the 82 SMs).
Prefill/TTFT unchanged (the flag is decode-only).

**Shipped:** `launch.sh flint` → `--triton-attention-num-kv-splits ${FLINT_KV_SPLITS:-64}`;
tripwire baseline re-saved on the as-shipped preset (`benchmarks/baselines.json` → `flint`).

Receipts: `benchmarks/regression/flint-2026-10-22-kv{32,64,128}.json` (tripwire points),
`flint-2026-10-22-kv{32,64,128}-131k.json` (64K / 131K), `flint-2026-10-22-docker-smoke.json`
(default 8, same day). The same lever applies to every preset on the triton backend —
the four Gemma 4 presets (`gemma4`, `gemma4-31b`, `gemma4-12b`, `gemma4-21b-reap`) are
being bracketed next; their token grids are 4–8, so the default under-fills them too.
