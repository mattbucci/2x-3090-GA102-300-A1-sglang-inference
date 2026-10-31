# Flint — `fp8_e5m2` KV vs bf16 KV at depth (2026-10-31)

**Why:** Flint decodes 150 tok/s @32K but 93 @255K; the slope is its 14 full-attention NoPE layers walking the whole
KV every step (the triton split-KV fix already took the deep point 36 → 90). The preset ships bf16 KV because triton
rejects E4M3 on sm_86 — but `fp8_e5m2` compiles there (the Gemma 4 31B preset has served it since the A5 sprint), and it
halves the bytes those 14 layers read per step. The SWA layers' 513-token window is unaffected either way.

**Instrument:** `scripts/bench/bench_long_context.py` (fleet invariants: M=1, 100 output tokens, server-verified
`actual_input_tokens`), both arms in the same boot profile (300 W cap, CPU boost off, gpu-unlock on, bare-metal
`launch.sh flint`, TP=2 / 262144, `--triton-attention-num-kv-splits 64`), bf16 arm first. The bf16 arm reproduces the
2026-10-25 fleet refresh to 0.1 tok/s at every point, so the instrument noise is ≪ 1 %.

| actual tokens | bf16 KV tok/s | e5m2 KV tok/s | Δ decode | TTFT bf16 / e5m2 |
|---:|---:|---:|---:|---:|
| 1,024 | 166.1 | 159.5 | **−4.0 %** | 68 / 73 ms |
| 32,768 | 149.9 | 148.4 | −1.0 % | 139 / 129 ms |
| 131,072 | 119.5 | 125.8 | **+5.3 %** | 392 / 368 ms |
| 261,916 | 93.2 | 104.4 | **+12.0 %** | 766 / 697 ms |

KV pool at `--context-length 262144`, MEM 0.85: **588,610 → 1,177,221** tokens (full-attention pool; SWA pool scales the same).

**Read:** a depth trade. The fp8 path costs a fixed ~4 % at short context (dequant on the read side of every attention
step, both layer types) and buys back bandwidth as the full-attention KV grows; crossover is ≈ 64K. Against the
project's primary target (single-user decode at 256K) it is a **+12 % win**; against the SWE-bench Lite prompt
distribution (median 41K, p90 82K, max 230K) it is roughly neutral.

Receipts: [`flint-2026-10-31-kv-auto.json`](flint-2026-10-31-kv-auto.json), [`flint-2026-10-31-kv-e5m2.json`](flint-2026-10-31-kv-e5m2.json).

## Quality gate — e5m2 at parity with bf16 (same day, same profile, same instruments)

Needle = natural Open-Platypus haystack (`needle_band_probe.py --haystack benchmarks/quality/haystack-platypus.txt`,
`--flush`, 2048-token budget). Cells are read **strictly**: a hit is a `stop`-finished turn with the password in the
final content. (The fleet rule counts the needle anywhere in reasoning + content; the first e5m2 131K @0.9 greedy draw
quoted the haystack until the budget and was scored "found" with empty content — the probe now records `answered`
alongside `found`.)

| cell | bf16 KV | e5m2 KV | receipts |
|---|:--:|:--:|---|
| 131K @0.1 / @0.5 greedy | found / found (2026-10-22) | **hit / hit** | `flint-kv-e5m2-natural-131k.json` |
| 131K @0.9 greedy ×3 | **3/3** (55–550 tok) | **3/3** (43–287 tok; first draw of the day: budget, unanswered — knife-edge) | `flint-kv-{bf16,e5m2}-natural-131k-d09-greedy-x3.json` |
| 131K @0.9 model sampling ×5 | **5/5** | **5/5** | `…-131k-d09-default-x5.json` |
| 252K @0.1 greedy ×2 (e5m2 ×3) | miss / miss | miss / miss / miss | `…-250k-greedy-x2.json`, `flint-kv-e5m2-natural-250k.json` |
| 252K @0.5 greedy ×2 (e5m2 ×3) | miss / miss | miss / miss / miss | same |
| 252K @0.9 greedy ×2 (e5m2 ×3) | **hit / hit** | **hit / hit / hit** | same |
| 252K @0.5 model sampling ×3 | 0/3 (`No password.`, `s3cr3t`, `C`) | 0/3 (`B`, `C`, `C`) | `…-250k-default-x3.json` |
| 256K tool-use probe (stock filler, depth 0.5) | 0.4 / 0.4, max 64,910 | 0.4 / 0.4, max 64,910 — identical row by row | `tooluse256k-flint.json`, `tooluse256k-flint-kv-e5m2.json` |
| HumanEval 40 (raw completions, greedy) | 28/40 (70.0 %) | 29/40 (72.5 %) | `benchmarks/quality/flint.json`, `flint-kv-e5m2.json` |
| MMLU (chat, 200-token cap — weak on a thinking model, same instrument both arms) | 39/57 | 38/57 | same |

**Two findings that are not about the KV dtype:** (1) Flint has a **retrieval cliff between 243.5K and 252K actual
tokens** — on 2026-10-22 the bf16 ship found 243.5K @0.5 and @0.9 greedy; today at 252,401 actual both arms miss
@0.1 and @0.5 on every draw (confident wrong answers: `C`, `B`, `hunter2`, "I don't know") and hit @0.9 on every draw.
The served window is 262,144; the last ~8 % of it is not reliable for retrieval of a mid-context fact. (2) The stock
repeated-sentence filler of `probe_256k_tooluse.py` is the same repetition attractor the needle receipt documented —
Flint length-stops from 131K on it, identically on both arms; a natural-haystack variant of the tool probe is the
instrument Flint needs.

**Cross-stack confirmation (R9700, same day, FP8 per-channel build on ROCm, bf16 KV — their `07432a6`, `benchmarks/quality/flint/needle-natural-fp8-2026-10-31.md`):** 243,481 actual @0.1 **miss** / @0.5 found / @0.9 found; 248,383 @0.1 miss / @0.5 **miss** (sampled 0/3, the model reasons that no password is present and invents one) / @0.9 found; every greedy miss reproduces byte-for-byte from a cold cache, and their sink kernels match a torch reference. Three quant formats (INT4 g64 bf16 KV, INT4 g64 e5m2 KV, FP8) and two stacks agree: the cliff is the checkpoint's — depth 0.1 goes first at ~243K, depth 0.5 by ~248K, depth 0.9 holds to the window. The 243.5K / 248K INT4 pass at all three depths on both arms is the table below.

## The deep band, both arms (greedy, flushed, ×2 per cell; strict read)

| actual tokens | arm | @0.1 | @0.5 | @0.9 | receipt |
|---:|---|:--:|:--:|:--:|---|
| 240,840 | e5m2 | miss / miss | miss / miss | hit / hit | `flint-natural-243k-band-greedy-x2-2026-10-31.json` |
| 240,840 | bf16 | miss / miss | miss / miss | hit / hit | `flint-kv-bf16-natural-243k-band-greedy-x2-2026-10-31.json` |
| 246,531 | e5m2 | miss / miss | **hit / hit** | hit / hit | `flint-natural-248k-band-greedy-x2-2026-10-31.json` |
| 246,531 | bf16 | **hit** / miss | miss / miss | hit / hit | `flint-kv-bf16-natural-248k-band-greedy-x2-2026-10-31.json` |
| 252,401 | e5m2 (×3) | miss ×3 | miss ×3 | hit ×3 | above |
| 252,401 | bf16 | miss / miss | miss / miss | hit / hit | above |

Depth 0.9: 14/14 on each arm. Depths 0.1 + 0.5: e5m2 2/14, bf16 1/12, and the hits do not line up by arm (e5m2 takes
246.5K @0.5 twice, bf16 takes 246.5K @0.1 once at the identical prompt — a greedy knife-edge, i.e. TP-2 nondeterminism, not
the dtype). So the correct statement is not a cliff at one length but a **marginal zone from ≈240K at depths ≤ 0.5** whose
cells flip with the surrounding text (R9700 saw the same text dependence in a 32K band), with depth 0.9 reliable to the
window; the 2026-10-22 "243.5K @0.5 found" was one such cell. **Onset bracket (e5m2 arm, greedy, flushed, ×2, depths 0.1 / 0.5 — `flint-natural-onset-160k-200k-220k-greedy-x2-2026-10-31.json`):** 155,462 actual 4/4, 195,012 actual 4/4, 214,245 actual **0/4** — and at 214K both depths and both draws return the *same* wrong content (a haystack sentence about license plates), i.e. the model continues the Open-Platypus text adjacent to the question instead of answering; a text-triggered continuation, the shape R9700 documented in a 32K band, rather than the deep band's confident wrong passwords. So the reliable range ends between 195K and 214K actual, and the last ~20 % of the 262,144 window is where retrieval becomes text-dependent. Nothing in the band separates e5m2 from bf16 KV.

**Decision:** `fp8_e5m2` is the `flint` preset default from 2026-10-31 (`KV_DTYPE=auto` restores bf16 KV). Ship steps:
fleet re-sweep → `benchmarks/flint/results.json` (bf16 sweep kept as `results-bf16kv-2026-10-25.json`), tripwire
`BASELINE=save`, serving-image rebuild (the image bakes `launch.sh`), parity smoke.
