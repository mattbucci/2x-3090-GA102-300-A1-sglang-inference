# opencode+DCP lane boot tax — root cause + fix (2026-10-21)

**Symptom (qwen38 v4, `lane_overhead.py --run`).** opencode-dcp `pre` (image build + boot → first
model event) median **264 s** vs plain opencode **193 s**; DCP `post` (last event → exit) pinned at
**120 s** on 265/276 instances with a cleanup session present on only **11/276** (control: 0 s,
276/276). The 71 s was filed as "offline npm, 1.14.25-specific"; the 120 s cleanup pass was
assumed to be killed by its own `timeout 120` for the same reason. Both were true; neither was
understood.

**Root cause (opencode 1.14.25 source).** `Config` forks `Npm.install(<config dir>, add:
@opencode-ai/plugin@<opencode version>)` for every config directory at startup (`config.ts`,
`Effect.forkDetach`), and **plugin loading awaits those fibers** (`plugin/index.ts`: `if
(plugins.length) yield* config.waitForDependencies()`). The plain lane has no plugin, so the fiber
runs detached and is never waited on; the DCP lane is the only one that blocks on it. The image's
`opencode plugin … --global` step exits before the same fiber finishes, so
`/opt/dcp-home/.config/opencode` ships without `node_modules` and `Npm.install` reifies at every
start. Under the rollout's `--network none`, DNS fails instantly (`EAI_AGAIN`) but npm retries
each registry fetch three times (`fetch-retries 2`, 10 s then 60 s) before falling back to its
stale cache — **70 s per packument, sequentially down the dependency tree**:

```
20:47:11 GET https://registry.npmjs.org/@opencode-ai%2fplugin 70100ms (cache stale)
20:48:21 GET https://registry.npmjs.org/@opentui%2fcore        70089ms (cache stale)
20:49:32 GET https://registry.npmjs.org/web-tree-sitter        70083ms (cache stale)
20:50:42 GET https://registry.npmjs.org/@types%2femscripten    70082ms (cache stale)
20:51:52 GET https://registry.npmjs.org/@opentui%2fsolid       70072ms (cache stale)
20:53:02 GET https://registry.npmjs.org/solid-js               70087ms (cache stale)
20:54:12 GET zod / effect / @opencode-ai%2fsdk                 70095–70251ms (cache stale)   … rc 124 at 520 s
```

Why production only paid one round: the per-instance image is built **minutes** before the run,
so the build's packuments are still inside their 5-minute `max-age` and resolve from cache; the
single 70 s is a tarball the build never fetched (`uuid-13.0.2.tgz`). The cleanup pass runs
~10 min later against a now-stale cache → every packument costs 70 s → killed at 120 s before its
first request. A GPU-free probe against a 2 h-old image reproduces the stale case: **no model
request in 600 s**.

**Fix (branch `opencode-subagent-permission`, lands with the permission block at the
qwen38→qwen36-dense boundary).**

1. `Dockerfile.rollout`: `npm install --save-exact --ignore-scripts @opencode-ai/plugin@${OPENCODE_VERSION}`
   into `/root/.config/opencode` with network, and copy `package.json` + `package-lock.json` +
   `node_modules` into `/opt/dcp-home/.config/opencode` before the plugin step. `Npm.install`
   then short-circuits (`node_modules` exists, the lock declares the package) — **no npm at run
   time** in either lane.
2. `docker_rollout.py`: `NPM_OFFLINE_ENV = npm_config_offline=true` on both opencode scaffolds.
   npm's offline mode fails a cache miss immediately (`cache mode is 'only-if-cached'`) instead of
   retrying for 70 s; the plugin still loads. Belt for any future opencode dependency the
   pre-install does not cover.
3. Gate: `probe_opencode_subagent_permission.py --scaffold opencode-dcp` now reports
   `boot_seconds`, `external_plugin_loaded` (PASS requires `service=plugin … loading plugin`) and
   `dependency_install_failed`.

| run (image) | rc | boot → first model request | DCP plugin | dep-install warn |
|---|---|---|---|---|
| baseline, 2 h-old image, stale cache | 124 @ 200 s / 600 s | never | — | — |
| `npm_config_offline=true` only (same image) | 0 | **2.9 s** | loaded | yes (0.6 s, `uuid` tarball miss) |
| pre-installed config dir, simulated (`--pre-run`) | 0 | 3.1 s | loaded | no |
| v2 image: pre-install + env (the branch) | 0 | **2.3 s** | loaded | no |
| v2 image: pre-install only (`unset npm_config_offline`) | 0 | 2.1 s | loaded | no |
| v2 image: plain opencode, control / fix | 124 / 0 | 2.0 s / 2.0 s | n/a | no |

Receipt: `opencode-dcp-boot-tax-2026-10-21.json`. The v2 image is `Dockerfile.rollout` from the
branch on `swebench/sweb.eval.x86_64.sympy_1776_sympy-24909` (pre-install step 3.1 s at build).

**Reading the cells.** The qwen38 v4 DCP cell (and every earlier DCP cell) ran with a dead cleanup
pass — the DCP arm kept its reproducer / debug files in the captured diff far more often than the
control arm. That is a harness asymmetry inside the A/B, not a plugin effect; it is recorded here
and the paired read of those cells should discount diff-hygiene differences. Cells rolled after
the merge carry both fixes (`scaffold_pin` unchanged — the scaffold versions are the same; the
harness version is the cycle's). R9700 never saw it for a different reason than the version:
their sandbox bind-mounts the opencode config dirs **read-only** (`run_rollouts.py`
`DOCKER_MOUNTS`), and `Npm.install` returns before doing anything when the directory is not
writable (`afs.access(dir, {writable: true})` → `if (!canWrite) return`). We cannot take that
route as-is — the per-run budget/permission snippet rewrites `opencode.json` in that directory —
so the directory is pre-populated instead. The "1.14.25-specific" note in README item 4 was wrong
about the mechanism and is retired.
