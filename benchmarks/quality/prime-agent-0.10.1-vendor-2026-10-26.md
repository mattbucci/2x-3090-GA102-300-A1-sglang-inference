# prime-agent 0.10.1 vendored footprint — build + offline warm check (2026-10-26)

Upstream stable moved 0.10.0 → **0.10.1** on 2026-10-21 (128 commits; release body empty, commit
list via `compare/v0.10.0...v0.10.1`). Pin bump staged on branch `harness/prime-agent-0.10.1`
for the flint cycle boundary (per-instance image builds read `Dockerfile.rollout` live). Nothing
rolls prime in the pruned queue; this is the artifact + proof so a prime cell can be asked for.

## Upstream changes that touch the harness
- `3e08932` model-facing code tool renamed `ipython` → `python_repl` (#3400). `audit_leakage.PY_CODE_TOOLS`
  already carries both names; the vendor warm-up stub now picks the name by version (`PY_TOOL`).
- `b3496ee` / `9eec445` daemonless (`-p`) sessions: subagents and wake-ups unavailable, inline RLM
  subagents. A headless session therefore records **7** Python skills (`agent_message attach_image
  compact edit goal refine websearch`); `agent_observe` and `rlm_heartbeat` stay in the payload but
  are not resolved. The vendor gate now derives from the payload (recorded ⊆ payload, core 7 present)
  instead of the hardcoded 9 (which failed the first build).
- `6465da8` compact at tool-turn thresholds, `a498cbd` configurable automatic compaction percentage,
  `f87fa01` retry a length-capped compaction summary once, `0ee10e1` mandatory thinking for auxiliary
  summaries. Scaffold-behaviour changes → recorded per cell (`scaffold_pin`), never re-labelled.
- Not found in the commit list: a retry after a `length` finish on the main turn (0.8.1 / 0.10.0 never
  retried; verify on the first 0.10.1 cell's empties).

## Artifact
- upstream `prime-agent-0.10.1-linux-x64.tar.gz` sha256 `21ec2191…d41945c` (release SHA256SUMS; mirrored)
- vendor tgz `prime-agent-0.10.1-vendor.tgz` 168,499,214 B sha256 `8a67140f…e06a78b`
  (`vendor_prime_agent.sh --build`, ubuntu:22.04, uv 0.13.5, kernel CPython 3.11.17; mirrored at
  `mattbucci/swebench-scaffold-vendor`)
- build gates: `--version` = 0.10.1, second `--prime-agent-bootstrap` pass a no-op, runtime imports,
  stub session → skills recorded (7/9, see above), skill imports.

## Offline warm check (ubuntu:22.04, `--network none`, tgz extracted at `/`, uv wrapped to log calls)
```
version ok
first request: model=stub stream=true stream_options.include_usage store=false
               max_completion_tokens=32768 tools=[python_repl] roles=[system,user,user]
uv invocations during the session: 0
session: tool call round-trip + final DONE
```
Same wire shape as 0.10.0 (no `reasoning_effort`, no `temperature`, task = user message #2 after the
`[harness-digest]` turn, `max_completion_tokens` = OUTPUT_BUDGET unclamped). Phase-0
`scaffold_request_audit.py` remains the gate when the pin merges.
