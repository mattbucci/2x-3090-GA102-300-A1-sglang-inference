# Chat-template boundary vs the prefix cache — agentic-shaped turn on this stack (2026-11-01)

Trigger: R9700's R97-R (their `b8c9276`): under the HIP default `no_buffer` a Mamba/GDN state exists only at 8K chunk ends, and the Qwen3 template re-renders a history turn's `<think>\n` as one `\n\n` token, so every follow-up turn diverged one token before the primed prompt end and re-prefilled up to 8K tokens; `--mamba-radix-cache-strategy extra_buffer` took their qwen38 1-token turn at 176K from 5.7 s to 0.82 s.

**Template divergence reproduces here** (CPU, `tokenizer` + `apply_chat_template`, visible-only history): Qwen3.8 shares the cached prefix to **1 token before** the primed prompt end (`\n` → `\n\n` merge), Gemma 4 to **4 tokens before** (the thought-channel opener is not re-rendered), Flint to the prompt end exactly. Replaying `reasoning_content` moves Qwen3.8 to the prompt end.

**Our servers already run `extra_buffer`** (server_args of every hybrid preset: `mamba_radix_cache_strategy='extra_buffer'`, `mamba_track_interval=256`, overlap scheduling on — the CUDA default in this tree), so the question was only the cost of that divergence. Ported probe: `scripts/bench/measure_extend_cost.py --multi-turn --prime-maxtok 200` (R9700's; prime a prefix, let the model answer 200 tokens, replay the visible answer as history, add a new user message, time the first token; `--enable-cache-report` makes the server return `cached_tokens`). Medians of 2 runs; first requests after boot carry a one-off (JIT / plan) that the k = 1 median at 8K sometimes shows.

| preset (backend) | prompt tokens | cached by the server | turn with 1 new token | 64 new tokens | same-message append (1 / 64) |
|---|---:|---:|---:|---:|---:|
| qwen38 (flashinfer, GDN) | 117,071 | 117,056 | 296 ms | 265 ms | — |
|  | 175,876 | 175,744 | 425 ms | 395 ms | — |
| qwen38, `--mamba-track-interval 64` | 117,067 / 175,876 | 117,056 / 175,744 | 307 / 403 ms | 284 / 402 ms | — |
| qwen36 (flashinfer, GDN MoE) | 116,953 | 116,928 | 277 ms | 265 ms | 248 / 252 ms |
|  | 175,762 | 175,744 | 325 ms | 342 ms | 343 / 346 ms |
| gemma4-31b (triton, route on) | 117,489 | 117,483 | 280 ms | 456 ms | 275 / 454 ms |
|  | 176,564 | 176,558 | 392 ms | 667 ms | 386 / 668 ms |

## Reading

- **No cliff on this stack.** The server reuses the cache to within 15–132 tokens on the GDN presets (the reuse boundary sits on 64-token multiples — the FLA chunk — on both track intervals, so `--mamba-track-interval` is not a lever here) and to within 6 on Gemma 4. The agentic-shaped turn costs the same as a same-message append at every depth.
- **What the turn costs instead** is what the rest of today's work is about: server-side re-tokenization of the whole conversation (patch 072, ≈ 70–100 ms of the qwen38 numbers at these depths, ≈ 110–170 ms of Gemma's), the launch-bound flashinfer extend (qwen38 / qwen36, ~180 ms; prefill-graph A/B queued) and, on triton, the serial extend for turns above the 32-token route cap (Gemma 4's 64-token rows: 456 / 667 ms vs 280 / 392 ms for a routed 1-token turn).
- **Flint (triton, route on), fixed probe:** cached 116,935 of 116,937 and 175,744 of 175,746 across the turn boundary; 1-token turn **238 / 320 ms** at 117K / 176K (same-message 232 / 332), 64-token turn 286 / 427 ms (above the route cap) — the turn is the tokenizer (1.7 µs/token) plus the routed extend. qwen38 same-message append at depth: 253 / 357 ms (1 token) vs 296 / 425 ms multi-turn — within two-run noise with ~80 more recomputed tokens.

Receipts: `*.json` in this directory; R9700 relay in their `CROSS-TEAM.md`.

## Patch 072 (incremental tokenization) on the real chat path — SHIPPED (2026-11-01)

Same probe, same session, `SGLANG_INCREMENTAL_TOKENIZE=0` then `1` (verify off), medians of 3 runs; agentic shape (200-token reply replayed as history, new user message of k tokens):

| preset | prefix | 1-token turn off → on | 64-token turn off → on |
|---|---:|---:|---:|
| flint | 32K | 120 → **76 ms** | 115 → 71 ms |
|  | 131K | 239 → **82 ms** | 281 → 135 ms |
|  | 197K | 331 → **90 ms** | 442 → 191 ms |
| qwen38 | 32K | 123 → **87 ms** | 132 → 95 ms |
|  | 131K | 254 → **120 ms** | 283 → 136 ms |
|  | 197K | 364 → **142 ms** | 370 → 145 ms |

- py-spy on the tokenizer-manager process during 1-token turns at 208K (Flint, off): **94 % of samples in `tokenizer.encode`** via `_render_and_encode_chat_template → parallel_prompt_encode` (its parallel branch is not taken on this stack; `_plan_for` falls back to the plain encode), jinja render < 1 %. qwen38 (off, 229K): 77 % encode + **8.5 % `tokenizer.decode`** — the chat path decodes the whole prompt back to text for its template cache on multimodal-flagged models; next item.
- The first served A/B read no gain because the hook sat on a helper that chat requests never take; the profile found the real path. A single growing message with no special token inside reuses nothing (by design: boundaries are added tokens) and now costs a plain encode (the first version's offset mapping made it 20 % slower).
- Remaining per-turn floor: Flint 76–90 ms flat in depth (route + launch + HTTP), qwen38 87–142 ms (the launch-bound flashinfer extend plus the decode-back; prefill-graph A/B queued).
