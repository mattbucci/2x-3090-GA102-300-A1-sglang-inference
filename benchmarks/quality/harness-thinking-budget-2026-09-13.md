# Harness thinking + output budget audit (2026-09-13)

**Question.** Does every bake-off scaffold let the model think at its maximum and give it enough
output room? (User directive: "we also need to make sure all harnesses enable max thinking".)
The 2026-09-11 context-budget fix (`9c31fff`) had already shown that scaffold defaults, not the
server, decide what the model sees — so the same question was asked about reasoning effort and
`max_tokens`.

**Method.** A fake OpenAI endpoint (`scaffold_request_audit.py`, now Phase 0 of every cycle) runs
each scaffold's real rollout command against a pinned rollout image and captures the first
request body verbatim; the historical cells were read from their rollout logs.

## What the scaffolds were sending (qwen38 cycle, before the fix)

| lane | effort on the wire | output cap | consequence |
|---|---|---|---|
| opencode, opencode+DCP | none (template default `xhigh`) | **8192** (`opencode.json` `limit.output`) | thinking truncated mid-turn: 38/293 opencode and 51/282 DCP sessions hit `reason: length`; **35 of 51 (69 %)** and **43 of 69 (62 %)** empty patches were length-truncated, vs 3/242 (1 %) of the non-empty sessions |
| little-coder (pi 0.68), little-coder+RTK (pi 0.83) | **`reasoning_effort: medium`** (pi's default `thinkingLevel`) | 16384 (`maxTokens` cloned from the models.json template) | the model reasoned at `medium` for every little-coder cell ever run; 252/300 of the 256K re-roll had already landed at medium when stopped |
| prime (prime-agent 0.8.1) | `xhigh` (explicit) | 16384 in the models entry, **32000** on the wire (`Math.min(maxTokens, 32000)` in pi-ai `simple-options.js`) | cap only |
| dcode (`/v1/responses`) | none (template default) | none | — |
| Gemma 4 presets (any scaffold) | — | — | `gemma4` chat templates render thinking **off** unless `enable_thinking: true` is passed; no scaffold passes it |

Why the empties matter: a thinker at `xhigh` spends most of an 8K budget inside `<think>`; the
turn ends with no tool call, opencode records the step as finished, and the session drifts to an
empty diff. A non-thinking coder loses far less to the same cap, so the defect was **not
uniform across families** — the superseded cross-family reads (`qwen36` vs the coder trio) are
withdrawn, not just re-scaled.

## Fix (harness input, recorded per cell)

- **Effort:** every scaffold sends **no** `reasoning_effort`. The served chat template's default
  is the model's maximum (Qwen3.8 `xhigh`; Qwen3.5/3.6 thinking on). Sending an explicit value
  was rejected: pi clamps `xhigh`/`max` → `high` (`clampReasoning`) and the Qwen3.8 template
  raises on `high`; little-coder's models entry therefore sets
  `compat.supportsReasoningEffort: false` so pi omits the field. Gemma 4 presets add
  `--default-chat-template-kwargs '{"enable_thinking": true}'` in `launch.sh` (per-request
  kwargs still win via `setdefault`).
- **Output budget:** uniform `OUTPUT_BUDGET = 32768` written into every scaffold per run —
  `opencode.json` `limit.output` + `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX`, little-coder /
  prime `maxTokens`, and the same number as pi's `compaction.reserveTokens`
  (`~/.pi/agent/settings.json`, `/opt/rtk-home/.pi/agent/settings.json`,
  `/root/.prime/agent/settings.json`) so compaction triggers before the server's
  prompt + max_tokens > 262144 → HTTP 400. pi 0.68 and prime clamp the wire value to 32000, so
  the audit floor is `OUTPUT_CAP_FLOOR = 32000`.
- **Tripwire:** `evals/swebench/scaffold_request_audit.py` runs before the server launches in
  `run_model_cycle.sh` (Phase 0). It fails the cycle if any scaffold's first request has the
  wrong model id, a `reasoning_effort` outside `--allow-effort` (default `xhigh`), any
  `enable_thinking: false` / `thinking.type: disabled`, or a cap below 32000. Receipt per cycle:
  `/tmp/run-model-cycle-logs/<preset>/scaffold-audit.json`; the 2026-09-13 qwen38 run passed 6/6
  in 16 s (opencode 32768, little-coder 32000, RTK 32768, prime 32000, dcode uncapped, no effort
  field anywhere).
- **Provenance:** `meta.json` records `output_budget` + `thinking`; each cell JSON carries
  `scaffold_output_budget` next to `scaffold_context_window`.

Scaffold identity that is deliberately *not* normalised: little-coder sends `temperature: 0.3`
(its benchmark profile), opencode sends `top_p: 1`; the rest of the sampling comes from the
preset's `--sampling-defaults model`.

## Quarantine

Run dirs renamed (gitignored): `qwen38-{opencode,opencode-dcp}-v2-out8k`,
`{qwen36,qwen36-ream,qwen35-moe,coder-30b-eval,coder-reap-25b,coder-30b-ream}-opencode-v2-out8k`,
`qwen38-little-coder-v2-effmed16k`. Receipts renamed to
`benchmarks/quality/bakeoff-<preset>-opencode-out8k.json` with a `superseded` note and
`scaffold_output_budget: 8192` (the chart skips `superseded` cells). DCP engagement receipts of the
8K lane moved to `benchmarks/quality/dcp-engagement-out8k-2026-09-13/`.

The 256K re-roll queue restarted 2026-09-13 14:45 with every `qwen38` lane from scratch, then
opencode + little-coder for `qwen36-dense`, `qwen36`, `qwen36-ream`, `qwen35-moe`, little-coder then
opencode for the three coders, and `devstral` opencode. Cells that ran under an earlier harness
are never re-labelled — they stay as receipts until the re-rolled cell lands.
