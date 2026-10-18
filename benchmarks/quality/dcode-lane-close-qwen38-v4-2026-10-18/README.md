# qwen38 dcode v4 lane close — 2026-10-18 — DEFECT-CONFOUNDED (quarantined, re-roll owed)

**Lane:** `qwen38-dcode-v4` (deepagents-code 0.1.65 → SGLang v0.5.20 `/v1/responses`, Qwen3.8-27B-AWQ under
DSpark spec at 256K, docker-served, network=none). Closed 2026-10-18 12:15 PDT, 300/300 rc 0, 0 wall hits
(2 instances at dcode's own 1950 s inner cap), **87 patches / 213 empties (71 %)**.

**Verdict:** the empty rate is a **server-side harness defect in SGLang's Responses-API history path**, not
model behaviour. Fixed by **patch 066** (`patches/066-responses-merge-assistant-turn-across-phase.patch`),
gated by `scripts/eval/test_responses_merge_assistant_turn_phase.py` (pristine 2/5 → patched 5/5) and by step 4
of `scripts/eval/probe_responses_api.py` (Phase 1a, wire-level, gates the serving image). The cell is
quarantined (`-respsplit`) and re-rolls as a new versioned cell once the serving image carries 066. Upstream
`main` still has the bug (checked 2026-10-18) — PR candidate alongside 063/065.

## The defect

`OpenAIServingResponses._merge_consecutive_assistant_messages` folds consecutive assistant dicts into one chat
turn only when `merged[-1].get("phase") == msg.get("phase")`. `phase` ("commentary" / "final_answer") only
survives on `message` items; the `function_call` and `reasoning` items of the same turn are normalised without
one, so `"commentary" != None` and the text and its own tool call become **two assistant turns**.
langchain-openai 1.7.1 (what dcode speaks) forwards `phase` on every echoed assistant text block
(`_construct_responses_api_input`: `new_item["phase"] = phase`), so every prior step rendered as:

```
<|im_start|>assistant\n<think>\n\n</think>\n\nLet me check the reader:<|im_end|>
<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>…</tool_call><|im_end|>
```

(`render-pristine-vs-patched.txt`: a 2-step history renders as **4** assistant turns pristine, **2** with 066 or
without `phase`.) The model learns in-context that an announcement sentence ends a turn; deepagents ends the
run on any turn without a tool call → empty patch. Scaffolds on `/v1/chat/completions` (opencode, pi,
prime) never touch this path.

## Evidence (all from the per-instance `sessions/<iid>/.deepagents/.state/sessions.db` snapshots)

| final main-thread turn | empties (213) | patched (87) |
|---|---:|---:|
| text only, no tool call | 191 (190 match the announce pattern "Let me / Now I'll … :") | 87 (normal "done" summary) |
| blank / reasoning only (imitates the empty `"\n\n"` split turn) | 20 | 0 |
| ends with a tool call | 2 | 0 |

* **Per-turn hazard:** 211 of 6,186 main-thread assistant turns end a run with no tool call = **3.4 %**;
  little-coder on the same model/instances: 8 announce-only turns among 17,455 and **0** text-only empties
  (0.05 %). Mid-run text-only turns in dcode: **0** — the hazard is entirely "last turn".
* **Nothing was dropped server-side** (`token-balance.json`): for all 6,186 turns
  `usage.output_tokens − visible(text + rendered tool calls) − usage.output_token_details.reasoning` is within
  1 token on every final no-call turn (6,171/6,186 overall within 40; the rest is the rough tool-call
  re-rendering). Empty finals carried a median **2,185** reasoning tokens then stopped after the sentence;
  patched finals 171. Every final `status: completed`, 0 invalid tool calls.
* 178 of the 213 empties made **no edit at all**; median 25 tool calls per empty run vs 24 per patched run,
  median rollout 516 s vs 374 s — the runs are full investigations that quit at the first announcement.
* Paired vs opencode-v4 (259 paired): **173 patch→empty, 0 empty→patch**; vs little-coder-v4 (271 paired):
  193 patch→empty, 1 empty→patch. dcode "speedup" 1.42–1.49× is the early quit, not throughput.

## Lane-close checks

* Leak gate `audit_leakage.py --require-isolation`: **0 exposed**, attempted 143/300 (48 %, all failed
  under network=none), isolation 300/300 on all four proofs (`leak-audit.json`). Two classifier gaps found
  and fixed on this lane before the gate read true: dcode's `fetch_url` tool (151 calls, `url` arg only)
  was invisible to the bash-only fallback, and `calls_from_dcode_db` never paired `ToolMessage` outputs, so
  every fetch read as outcome-unknown; dcode's `execute` trailers (`[Command failed with exit code 6]`,
  `<no output>`) joined the scaffold-line filter. Closed lanes unchanged (opencode / lc / prime 0 exposed).
* Walls: 0 rc-124; `audit_wall_causes.py` has no dcode reader (inner timeout, not a kill).
* Sawtooth / lane_overhead: not available — the server container was recycled for the Phase-4 re-roll before
  the close and `lane_overhead.py` has no dcode reader.

## Fix + gates

* `patches/066-responses-merge-assistant-turn-across-phase.patch` — a missing phase on either side is
  compatible with anything (the merged turn adopts the phase that exists); two *message* items with
  different phases still stay apart. 3-gate clean on v0.5.20 (32/32 apply, byte-identical live tree,
  32/32 refuse on the patched tree). Live tree carries 065 + 066; the docker serving image is rebuilt
  at the qwen38 cycle boundary (Rule 2 — never during a lane / re-roll).
* `scripts/eval/test_responses_merge_assistant_turn_phase.py` — 5 cases incl. an end-to-end
  `_construct_input_messages` over a 3-step phase-tagged history (3 turns expected; pristine gives 6).
* `scripts/eval/probe_responses_api.py` step 4 `history_merge` — replays the probe's own step-2 output
  with and without `phase` and requires equal `usage.input_tokens`; FAIL names patch 066. Runs at Phase 1a
  before any dcode lane, so an image without 066 cannot start a dcode cell again.

## Re-roll plan (cycle boundary)

PAUSE the queue → rebuild `sglang-cuda-3090:local` (065 + 066) → `smoke_docker_serving.sh` parity + live
`probe_responses_api.py` → rename `qwen38-dcode-v4` → `qwen38-dcode-v4-respsplit` (quarantine suffix never
matches the aggregator) → roll `qwen38` dcode as `-v5` → unpause. R9700 runs the same upstream code: their
dcode cells carry the same confound (relayed).
