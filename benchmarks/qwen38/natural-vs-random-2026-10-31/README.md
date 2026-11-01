# qwen38: natural-text vs random-prompt sweep, plain and DSpark arms — 2026-10-31

`bench_long_context.py --prompt-source natural` (Open-Platypus haystack streamed through `/v1/chat/completions`, `ignore_eos`, 100 output tokens, `/flush_cache` before every point, prompt length rescaled from the server's own token count) against the fleet instrument (`sglang.bench_serving --dataset-name random`), both on the same boot per arm. 1K / 32K / 131K / 262K requested; actual input tokens within 1 % of the label on every point. 300 W cap, gpu-unlock on, custom allreduce + INT8 `lm_head` as shipped.

| arm | prompts | 1K | 32K | 131K | 262K | TTFT @262K |
|---|---|---:|---:|---:|---:|---:|
| plain | natural | 86.6 | 80.8 | 69.3 | 58.8 | 182.9 s (fresh prefill) |
| plain | random | 86.3 | 80.6 | 69.3 | 58.6 | 0.45 s (prefix-cache hit) |
| DSpark `SPEC_DECODE=1` | natural | **168.5** | **108.6** | **84.0** | **65.0** | 184.8 s |
| DSpark `SPEC_DECODE=1` | random | see `spec-random.json` — the artifact (degenerate continuation, near-perfect draft) | | | | |

**Plain arm:** the two instruments agree within 0.5 % on decode; only the random instrument's TTFT is wrong (its 262K prompt is served from the warm-up's prefix cache — the `results.json` TTFTs across the fleet carry this, decode does not).

**DSpark arm on natural text:** 1.95× / 1.34× / 1.21× / 1.11× over the same-boot plain arm at 1K / 32K / 131K / 262K (the 2026-09-27 receipts read 1.6× / 1.35× / 1.3× / 1.21× against a slower plain arm; the levers landed since make the plain arm faster while the draft's verify cost does not shrink with them, so the ratio at depth narrows — absolute spec decode is up at every depth: 124.7 → 168.5 short, 58.5 → 65.0 deep). The short-context ratio is task-dependent — a passage summary is a predictable continuation; the 09-27 1.6× was on a code prompt.

The tripwire now carries this as the `qwen38-spec` arm (`bench_regression.sh arm qwen38-spec`: SPEC_DECODE=1 launch of `qwen38`, natural prompts), baseline saved the same day. Files: `plain-natural.json`, `plain-random.json`, `spec-natural.json`, `spec-random.json`.

**Tripwire baseline (`qwen38-spec`, same evening, 3 points):** 169.4 / 108.5 / **74.8** tok/s at 1K / 32K / 262K. The deep point differs from the sweep's 65.0 by 15 % with an identical config — on a spec arm the deep decode rate follows the acceptance of that particular continuation (the prompt slice end differs by a few hundred characters between the two runs as the chars/token estimate converges differently), so a spec arm's deep point carries ±15 % run-to-run and a tripwire flag there needs a second run before it is read as a regression; the plain arm's points repeat within 0.5 %.
