# 13 · 3090-M · Measure the agentic follow-up in chat shape; check the hybrid-cache checkpoint that serves it

From the M4 (cross-team note, M4 `780635d`). Kind: task. Box: 3090-box. Effort: ~2 h attended.

## Why

The "turn tax" we all measured with a raw `/v1/completions` append never crosses the chat template's
generation tail. An agentic client replays the history through `/v1/chat/completions`, and the Qwen3
template renders a history turn's `<think>\n` as one `\n\n` token (Qwen3.8 content-only history) or no think
block at all (Qwen3.6), so the follow-up diverges 1–2 tokens before the primed prompt end. The
end-of-generation hybrid state never serves an agentic turn; only checkpoints inside the prompt do. The
R9700 found this under `no_buffer` and moved to `extra_buffer`; the M4 found that its runner tracked the
64-token-grid state only in the first prefill chunk and fixed the continuation chunks (M4 patch 026):
first follow-up on Qwen3.8-27B 5–14 s → ~1 s at 8.7K–52K, on Qwen3.6-35B-A3B 0.8–8.2 s → 0.2–0.5 s.

## What to run

1. `scripts/bench/measure_extend_cost.py --multi-turn --prime-max-tokens 64` from the M4 repo (plain HTTP
   client; needs `--enable-cache-report` on the server) against the qwen38 and qwen36 presets, depths
   8192/24576/49152, thinking on. Read `recomputed` on the first follow-up (run0): it is the distance from
   the deepest hybrid checkpoint below the divergence point to the new prompt end.
2. `scripts/eval/probe_multiturn_hit_determinism.py` (same repo): two hits from different checkpoints vs a
   cold run; report the first divergent token. Hits from a grid state inside a chunk are expected to differ
   from cold like two chunk sizes do; chunk-end hits should be identical.
3. If run0 recomputes a whole chunk on `extra_buffer`, trace every write to the tracked-state marker
   before adding buffer: the M4's first fix was dead code because the end-of-prefill checkpoint consumes the
   marker before request finish.
4. Check the hybrid-cache pool budget (`--max-mamba-cache-size`, upstream default 4 × max-running): a
   second tree branch between turns (a tool-call side request) evicts the primed turn's checkpoints and the
   follow-up recomputes the whole prompt (M4: 80 s at 8.7K, 545 s at 52K with 4 slots).

## Deliverable

A row per model with run0 recompute tokens and TTFT at the three depths, the determinism verdict, and the
pool size the production presets run with. Receipt pattern:
`benchmarks/session-endurance/turn-tax-chat-<tag>-<date>.json` as on the M4.
