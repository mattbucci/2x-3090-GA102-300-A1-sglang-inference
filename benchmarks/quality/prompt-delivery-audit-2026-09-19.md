# Task-prompt delivery audit — argv → stdin (2026-09-19)

**Trigger:** R9700 relay `86f25bb` (same harness lineage): their fifth v3 start died on
`django__django-11422` (rc 143 at 1713 s, no diff) because the task prompt — issue text
included — was every scaffold's positional argument, so the agent's own
`pkill -f "manage.py runserver"` (a phrase from the issue) SIGTERMed its scaffold; and
opencode `run` re-quotes a positional message that contains spaces.

**Reproduced here the same day, GPU-free**, with `scaffold_request_audit.py` (real
`build_scaffold_invocation()` commands in the rollout image against a capture endpoint that
now records the first user message verbatim and sweeps `ps -eo args` while each scaffold runs).
Probe prompt: `Reply with the single word OK and stop. Probe text: run \`python -c "print(1)"\` — it has "quotes" and spaces.`

| scaffold | before (argv `"$PROMPT"`) | after (stdin `< /sandbox/prompt.md`) | argv sweep (after) |
|---|---|---|---|
| opencode 1.14.25 | **RE-QUOTED** — `"Reply … \"print(1)\" … \"quotes\" and spaces."` + `\n` | verbatim (+ trailing `\n`) | clean |
| opencode-dcp | **RE-QUOTED** (same) + `<dcp-message-id>` tag | verbatim (+ `\n`, + dcp tag) | clean |
| little-coder 1.1.0 (pi 0.68) | verbatim | verbatim | clean |
| little-coder-rtk 1.19.0 (pi 0.83) | verbatim | verbatim | clean |
| prime-agent | verbatim | verbatim | clean |
| dcode (`--stdin`) | verbatim | verbatim | clean |

Receipts: [`…-before.json`](prompt-delivery-audit-2026-09-19-before.json),
[`…-after.json`](prompt-delivery-audit-2026-09-19-after.json).

## What it means for the cells

- **Every opencode / opencode-dcp cell ever rolled here received the task wrapped in `"…"` with
  every inner `"` backslash-escaped** — our own template says `python -c "..."`, and 176/300
  Lite issue texts contain a `"`. The pi lanes, prime and dcode were verbatim, so the opencode
  lanes were the only ones reading a distorted task. Prompt-fidelity confound on top of the
  network-exposure confound; the v3 restart (from scratch, network=none) supersedes them all.
- **The self-kill class was silent here.** Our inner script continues past a killed scaffold
  (`|| true` → cleanup pass → diff capture), so a `pkill -f` hit produced a truncated session
  with rc 0 and whatever diff existed — indistinguishable from "model stopped early" without
  the session store. With the prompt off argv the phrase can no longer match.

## Fix (harness, this commit)

- `docker_rollout.py`: the prompt is written to `<run>/logs/<iid>.prompt.md` (exact bytes
  delivered) and bind-mounted read-only at `/sandbox/prompt.md`; every scaffold reads it on
  stdin (`opencode run … < file`, `little-coder … < file`, `prime-agent -p < file`,
  `dcode --stdin … < file`); the fixed cleanup prompt goes through a bash here-string. No
  `--env PROMPT`. `meta.json` carries `prompt_delivery: stdin-file`; argv-era cells have no key.
- `scaffold_request_audit.py` (Phase 0 of every cycle) gained two gate columns: **prompt**
  (`verbatim` / `verbatim*` trimmed / `RE-QUOTED` / `WRAPPED` / `ALTERED` / `MISSING`) and
  **argv** (`clean` / `EXPOSED` from the in-container ps sweep); policy checks now run on
  every generation request, not only the first (opencode's first call is its title request).
  The pre-fix delivery fails this gate (`RE-QUOTED` on both opencode lanes).

## Main-loop mount check (canned endpoint, 1 instance, 2026-09-19 21:0x)

Ran the real `docker_rollout.py` main loop (opencode lane, astropy__astropy-12907, `--server-url` at a
canned OpenAI-compatible capture server on a side port, `--network-mode none` default) and diffed the
first user message on the wire against `logs/<iid>.prompt.md` read as bytes: **1969 vs 1968 chars,
sha-equal after opencode's single prepended `\n`** (the `verbatim*` class); the 38 CRLF line endings in
the dataset's problem statement survive intact (a text-mode read of the receipt file collapses them —
compare bytes). Log header reads `# command (prompt on stdin: logs/<iid>.prompt.md -> /sandbox/prompt.md,
never argv)`, `meta.json` carries `prompt_delivery: stdin-file`, both `isolation:` lines present.
