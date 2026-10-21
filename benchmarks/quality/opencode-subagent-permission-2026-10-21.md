# opencode `task` subagent permission hang — fix + GPU-free gate (2026-10-21)

**Defect.** opencode 1.14.25 `run --dangerously-skip-permissions` answers permission asks for the
*main* session only (`cli/cmd/run.ts`: `if (permission.sessionID !== sessionID) continue`). Every
agent's default ruleset (`agent/agent.ts`) keeps three `ask` sources even with the flag —
`external_directory` (any `bash`/`read`/`edit` path outside the project dir; `bash.ts` scans `cd`,
`cat`, `cp`, `mv`, `rm`, `mkdir`, `touch`, `chmod`, `chown` arguments and the `workdir`), `doom_loop`,
and `read` of `*.env` files. A `task` subagent whose bash does `cd /tmp && …` therefore asks, nobody
answers, and the tool part stays `running` until the rollout wall: rc 124 with the server idle
(qwen38 opencode v4: `django-16820` `cat /etc/resolv.conf`, `matplotlib-23562` `cd /tmp && python -c …`
— 2 of 33 rc-124 walls, both `task` subagents; `audit_wall_causes.py` class `permission_hang`).

**Fix.** A global `permission` block — user config is merged *after* every agent's defaults
(`Permission.merge(defaults, …, user)`) and the evaluator takes the *last* matching rule
(`permission/evaluate.ts` `findLast`), so it reaches build, general, explore and custom agents alike:

```json
{
  "external_directory": "allow",
  "doom_loop": "allow",
  "read": {
    "*.env": "allow",
    "*.env.*": "allow"
  }
}
```

Written two ways: in the static rollout `evals/swebench/docker/opencode.json` (the DCP lane's
`/opt/dcp-home` copy inherits it at image build) and per run by `docker_rollout._opencode_budget_snippet`
(`docker_rollout.OPENCODE_PERMISSION`, logged as `permission: opencode.json …` in the instance stderr
— the Phase-0 request audit records the line). `--dangerously-skip-permissions` stays (main-session
`question`/`plan_*` denies are session rules, unaffected).

**Gate: `evals/swebench/probe_opencode_subagent_permission.py`.** GPU-free, runs the *rolled*
opencode invocation (`build_scaffold_invocation`, first run only) in the real rollout image against a
stdlib HTTP server that plays the model: main request → tool call `task` (general subagent, prompt
carries `'cd /tmp && ls'`) → subagent request → tool call `bash` `cd /tmp && ls` → (allowed?) →
subagent "DONE" → main "DONE". `--mode control` strips the block before `opencode run` and must hang;
`--mode both` passes only if control hangs and fix completes.

| mode | rc | wall | phases seen | opencode log (`service=permission`) |
|---|---|---|---|---|
| control (block removed) | 124 | 92.2 s (timeout 90) | no-tools, main-task, subagent-bash | `external_directory pattern=/tmp/* action=ask` → `asking` (no responder) |
| fix | 0 | 3.0 s | no-tools, main-task, subagent-bash, subagent-done, main-done | `external_directory pattern=/tmp/* action=allow`; subagent bash result `node-compile-cache` |

Verdict **PASS** (image `swebench-rollout/sympy__sympy-24909:latest`, served name `qwen38`). Receipt:
`opencode-subagent-permission-2026-10-21.json`.

**Landing.** Branch `opencode-subagent-permission`, merged at the qwen38→qwen36-dense boundary under
the PAUSE hold (scaffold-config changes are cycle-boundary only). Cells rolled before the merge keep
the wall class — the qwen38 v4 opencode / DCP pair ran the same opencode in both arms, so their
paired read is unbiased, but their wall counts carry the 2 hangs. Re-run the gate at every opencode
pin bump (`--mode both`); a future opencode that answers subagent asks under `run` makes the control
run complete, which the probe reports as FAIL — that is the signal to retire the block, not a defect.
