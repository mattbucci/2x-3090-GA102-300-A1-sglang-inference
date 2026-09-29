# opencode 1.14.25 headless permission hang — a harness wall class (2026-09-28)

**Found live** on `django__django-16820` (qwen38 opencode v4 lane, DSpark arm): GPUs at 0 % for 20 min with the rollout container up, no tool subprocess, no TCP connection on either side of the net bridge, the server's last request finished at 00:24:36 UTC. opencode's own log (`~/.local/share/opencode/log/`) ends with:

```
service=bash-tool arg=/etc/resolv.conf resolved=/etc/resolv.conf resolved path
service=permission permission=external_directory pattern=/etc/* ruleset=[{"permission":"*","action":"allow","pattern":"*"},{"permission":"doom_loop","action":"ask","pattern":"*"},{"permission":"external_directory","pattern":"*","action":"ask"}, …]
service=permission permission=external_directory pattern=/etc/* action={"permission":"external_directory","pattern":"*","action":"ask"} evaluated
service=permission id=per_… permission=external_directory patterns=["/etc/*"] asking
service=bus type=permission.asked publishing
```

`--dangerously-skip-permissions` prepends `*: allow`; the built-in defaults (`external_directory: ask`, `doom_loop: ask`, `read *.env: ask`) come later in the ruleset and the last match wins. `opencode run` has no permission responder, so the tool part stays `running` forever and the instance idles to the 1800 s host timeout (rc 124, empty patch). The trigger here was a `task` subagent the model spawned to "find the upstream fix" after `webfetch` failed under the no-network sandbox; it moved on to `cat /etc/resolv.conf; getent hosts github.com …`. The two earlier v4 cases are the main agent writing test output to `/tmp/*.txt` and reading it back.

## Classification of the rolling v4 opencode walls (n = 116 rolled, 9 rc-124 walls)

`evals/swebench/audit_wall_causes.py <run> --walls-only` reads the session snapshot's sqlite (`sessions/<iid>/.local/share/opencode/opencode.db` + `-wal`) and classes the last part across all sessions of the instance (incl. subagents):

| class | n | instances |
|---|---|---|
| `generating` (open reasoning part — model mid-think at the wall) | 6 | django-11564, 13448, 14730, 15213, 15252, 16408 |
| `permission_hang` (open `bash` part, path outside `/testbed`) | 3 | django-11742 (`/tmp/base_failures.txt` …), django-15789 (`/tmp/base_failures.txt`), django-16820 (`task` subagent, `/etc/resolv.conf`) |

`django-16820` walled at rc 124 while this was written (1974 s, empty patch, snapshot captured). Two further rows per arm read `rollout_seconds ≥ 1795` with rc 0 (`django-14999`, `django-16229` in v4; `django-15996`, `django-16229` in v3): they finished — `rollout_seconds` spans the ~175–300 s image build. `paired_lane_compare.py` now counts walls as rc 124 only (the 31-pair receipt is unchanged: all its walls were rc 124; the 107-pair read moves from 10 vs 14 to 8 vs 12).

The v3 no-spec opencode cell (53 rc-124 walls; the README's earlier "65" counted 12 finished rows whose `rollout_seconds` passed 1795 s inside the image build — corrected to 53 / 17.7 %) predates capture-at-wall, so its walls cannot be classed; both arms ran the same opencode 1.14.25 image, so the paired spec-vs-no-spec read is unbiased by this class, but every opencode-family absolute wall count (v3 and v4, opencode and opencode-dcp) carries it.

## Fix (cycle boundary — never mid-cell)

Add to the rollout `opencode.json`:

```json
"permission": { "external_directory": "allow", "doom_loop": "allow" }
```

and gate it in the Phase-0 scaffold audit by reading opencode's `ruleset=` log line on the probe instance (the user-config rules must sort after the defaults; if they do not, the alternative is a `permission.asked` responder via the plugin API or opencode's `OPENCODE_PERMISSION` env). `doom_loop: ask` is the same hang shape for a model that repeats one tool call three times. Also re-check whether the `read *.env` ask can fire on SWE-bench repos (any `.env` file in a testbed). After the fix, rerun `audit_wall_causes.py` on the first walls of the next opencode cell: `permission_hang` must be 0.

Receipt: `opencode-permission-hang-2026-09-28-v4-walls.json` (the classified rows at n = 116).
