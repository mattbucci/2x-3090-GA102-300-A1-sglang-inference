# SWE-bench Lite answer leakage — qwen38 opencode lanes (2026-09-19)

**Finding.** Every bake-off cell rolled before 2026-09-19 ran its container with
`--network=host`. The scaffolds' web tools (`webfetch`) and bash (`curl`, `gh`)
reached the internet, and the model used them to read the upstream fix. R9700
found it first on their lanes (their `af460d0`, 45–61 % exposed); this is the
measurement on ours, with the reader promoted to `evals/swebench/audit_leakage.py`.

| cell (300 instances, 299 transcripts) | web UPSTREAM | own PR fetched (`pull/<n>` any form) | own PR `.diff` | web OTHER | git READ / LIST attempts | **exposed** |
|---|---|---|---|---|---|---|
| `qwen38-opencode-v2-netopen` | 169 | 80 (27 %) | 62 (21 %) | 14 | 14 / 56 | **168 (56 %)** |
| `qwen38-opencode-dcp-v2-netopen` | 160 | 72 (24 %) | 61 (20 %) | 6 | 13 / 39 | **159 (53 %)** |

*exposed* = a web UPSTREAM or web SEARCH tool call that returned content.
Hosts hit (opencode cell): `api.github.com` 291 calls, `github.com` 215,
`raw.githubusercontent.com` 140, `code.djangoproject.com` 66,
`patch-diff.githubusercontent.com` 7. Typical evidence:

```
bash:     curl https://github.com/astropy/astropy/pull/7746.diff
webfetch: https://github.com/django/django/pull/10924/files
webfetch: https://github.com/django/django/pull/11133.diff
webfetch: https://raw.githubusercontent.com/astropy/astropy/main/astropy/io/ascii/rst.py
webfetch: https://api.github.com/repos/django/django/commits?path=django/db/models/sql/compiler.py&since=2019-05-02&until=2019-08-30
```

**Copying is visible in the patches, not just the transcripts.** Added-line
overlap between `model_patch` and the SWE-bench gold patch:

| group | n with a patch | overlap ≥ 80 % | median overlap |
|---|---|---|---|
| opencode, exposed | 159 | 122 (77 %) | **1.00** |
| opencode, isolated | 109 | 50 (46 %) | 0.67 |
| opencode+DCP, exposed | 150 | 117 (78 %) | **1.00** |
| opencode+DCP, isolated | 125 | 54 (43 %) | 0.50 |

The exposed group's typical patch *is* the upstream diff.

**Git channel: closed on our side.** The official `sweb.eval` images hold no ref
past the base commit and no unreachable object (django, sympy: every ref ≤ the
"SWE-bench" HEAD commit; release tags reachable from HEAD leak only unrelated
backports). The 14 `git READ` attempts (`git show origin/main:…`, `git log
--all -p`) found nothing to read — reported, not counted. docker_rollout.py
still strips tags / non-HEAD refs / remotes before the scaffold starts
(belt-and-braces; `isolation: refs=1 tags=0` in every v3 log).

**Consequence.** Every historical cell in the README table and the four qwen38
lanes finished under this harness are exposure-confounded, not SWE-bench
results. The 256K re-roll queue restarts from scratch as `-v3` cells under
network isolation (`docker_rollout.py --network-mode none`, the default:
`--network none` + loopback bridge to the server over a bind-mounted unix
socket, `net_bridge.py`; per-instance session-store snapshot under
`<run>/sessions/<iid>/`). `audit_leakage.py --require-isolation` runs at every
v3 lane close and gates the cell on 0 exposed + isolation proof on 300/300.

**Exposure study (scored 2026-09-19, `exposure_study.py`).** The two cells were
scored purely to measure the effect size — never as bake-off cells:

| cell | resolved (all 300) | exposed: resolved / n | isolated: resolved / n | gold-overlap median exposed / isolated |
|---|---|---|---|---|
| qwen38-opencode-v2-netopen | 250 (83.3 %) | **159 / 168 (94.6 %)** | 91 / 131 (69.5 %) | 1.00 / 0.67 |
| qwen38-opencode-dcp-v2-netopen | 253 (84.3 %) | **145 / 159 (91.2 %)** | 107 / 140 (76.4 %) | 1.00 / 0.50 |

Exposure is worth +25 / +15 points on these cells, and the exposed group's
patches are the gold patch. The "isolated" column is NOT a clean number: it is
the self-selected subset of instances where the agent happened not to make an
*observed* upstream/search call while the network was open (channels the
transcript does not record stay possible), so it is at best an upper-bound hint
for what the v3 cell will measure under `--network none`. Neither column goes
into the bake-off tables. Per-cell JSON: `exposure-study.json` (here) and
`<run>/exposure-study.json`.

Receipts: `swebench-leak-audit-qwen38-netopen-2026-09-19/*.leak-audit.json`
(per-instance source, classification, evidence, gold overlap, isolation proof).
Reproduce: `python evals/swebench/audit_leakage.py --run evals/swebench/runs/qwen38-opencode-v2-netopen`.
The little-coder lane (17 instances) has no transcript — pi's session store
lived only inside the container in v2; the v3 snapshot fixes that.
