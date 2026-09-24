# SWE-bench Lite harness — the v3 isolation contract and the 2026-09-20 relaunch boundary

Receipt for the restart of the 256K re-roll queue as `-v3` cells. Every historical cell
(and the four qwen38 lanes finished 2026-09-11 → 09-19) is exposure-confounded and is not a
SWE-bench number; the queue restarted from scratch on 2026-09-20 under the contract below,
served from the v0.5.20 OCI image. `aggregate_bakeoff.py` keeps the highest run version per
cell, so the tables only ever show v3 once a v3 cell exists.

## Why (2026-09-19)

R9700's session-store audit (their `af460d0`) found their agents reaching the upstream fix on
45–61 % of instances. Ours, [`audit_leakage.py`](../../evals/swebench/audit_leakage.py) over the
qwen38 opencode lanes: **56 % / 53 % of instances exposed** (168/299 opencode, 159/299
opencode+DCP). The rollout container ran `--network=host`, so `webfetch` / bash `curl` / `gh`
reached GitHub — 27 % of instances fetched their own PR (`…/django/pull/11133.diff`), and the
exposed group's patches overlap the gold patch at a median 100 % of added lines (isolated
group: 67 %). Scored effect size on the two finished opencode lanes: exposed instances resolve
at **94.6 % / 91.2 %** vs 69.5 % / 76.4 % for the rest (`qwen38-opencode-v2-netopen`,
`…-dcp-v2-netopen`) — an effect size, not a cell; the un-exposed subset is self-selected under
an open network. Full receipt:
[`swebench-leak-audit-qwen38-netopen-2026-09-19.md`](swebench-leak-audit-qwen38-netopen-2026-09-19.md).
The git channel was already closed (the official `sweb.eval` images carry no ref past the base
commit and no unreachable objects — verified on django and sympy).

## What every v3 cell carries

| Property | Mechanism | Proof (per instance / per lane) |
|---|---|---|
| **No network** | `docker_rollout.py` default `--network-mode none`; the container's `127.0.0.1:23334` is a loopback bridge over a bind-mounted unix socket the host forwards to the server ([`net_bridge.py`](../../evals/swebench/net_bridge.py)) | `isolation: network=none bridge=…` line; `BRIDGE CHECK FAILED` → exit 97 (infra, re-rolled) |
| **No refs past HEAD** | tags / non-HEAD refs / remotes deleted before the scaffold starts | `isolation: refs=1 tags=0` |
| **No benchmark cues in mounts** | every per-instance bind is staged under `/var/tmp/rs-*` (`/proc/self/mountinfo` prints bind *sources*; the old `…/runs/<preset>-<scaffold>-v3/sessions/<iid>` path handed the sandbox benchmark + instance + model + scaffold in one read — found by R9700's benchmark-recall audit); `assert_no_harness_cues()` refuses a command whose mounts / env / script match `swe.?bench` or the instance id | `isolation: mounts=…` line, `audit_leakage.py --require-isolation` (`mounts clean n/n`) |
| **No benchmark cue in git history** | the official image's HEAD is authored `SWE-bench <setup@swebench.com>`, titled `SWE-bench` (R9700 tied 17/19 wall hits to sessions naming the benchmark 30+ times); `/testbed` is re-initialised to one `eval@local` commit holding the identical tracked set | `isolation: git=reinit commits=1 dirty=0 author=eval@local` (`dirty≠0` = tracked-set drift) |
| **Prompt on stdin, never argv** | task written to `<run>/logs/<iid>.prompt.md`, bind-mounted ro at `/sandbox/prompt.md`, read on stdin (opencode `run` re-quotes a positional prompt; argv is visible in `ps`) | Phase-0 `scaffold_request_audit.py` columns `prompt` (verbatim on the wire) + `argv` (task absent from `ps`); meta `prompt_delivery: stdin-file` ([receipt](prompt-delivery-audit-2026-09-19.md)) |
| **256K window, template-max thinking, 32768 output budget** | `docker_rollout.py` reads `max_model_len` from `/v1/models` and writes it into every scaffold per run; no `reasoning_effort` on the wire; `OUTPUT_BUDGET = 32768` | `context budget:` line per instance; Phase-0 audit fails the cycle on a bad first request; meta `context_window` ([receipt](harness-thinking-budget-2026-09-13.md)) |
| **Pinned server** | `SERVE_MODE=docker`: the cycle's SGLang runs from `sglang-cuda-3090:local` (strict stale-image check vs the repo's `launch.sh`), per-cycle minted API key carried by every scaffold, models read-only | meta `serve_mode: docker`, `serve_image_id`, `api_auth: true`; `serve-backend.json` per cell |
| **Session stores kept** | each scaffold's session store snapshotted to `<run>/sessions/<iid>/` before the diff; on a wall hit the harness copies it live (`docker exec`) before the kill — added 2026-09-21 at the opencode lane's instance 52, so that cell's 1800 s wall hits carry an empty snapshot (`no_snapshot` in `recall-audit.json`); every later lane covers them | feeds `audit_leakage.py` and `audit_benchmark_recall.py` (Phase 7b, informational) |

Lane close gate: `audit_leakage.py --require-isolation` = 0 exposed + all four isolation proofs on
300/300. Any cell without `isolation:` lines is exposure-confounded — quarantined (`-netopen`),
never compared. Residual cues shared with R9700's sandbox: `/testbed`, the `testbed` conda env,
the issue text, the prompt's "Do not modify tests" line.

## Boundary receipts (2026-09-19 → 09-20)

1. **v0.5.20 flip + GPU campaign** — 21/21 presets within the n=30 noise band, 8/8 tripwire
   `[ok]`, the one quality flag (`gemma4-12b` needle 131K @ 90 %) closed as checkpoint softness
   ([status](../../patches/v0.5.20-rebase-status.md)).
2. **Docker-serving parity smoke** ([`smoke_docker_serving.sh`](../../evals/swebench/smoke_docker_serving.sh);
   image `0d926588ba00`, sglang 0.5.20 / torch 2.13.0+cu130 / tx 5.12.1):

   | preset | scaffold audit (auth) | boot | anon → 401 | bench vs bare baseline (1K / 32K / 261,916) | `/v1/responses` | 2-instance isolated rollout | leak audit |
   |---|---|---|---|---|---|---|---|
   | `qwen38` | PASS | 53 s | PASS | 70.6→70.2 / 67.2→66.1 / 48.4→48.3 tok/s (−0.6 / −1.6 / −0.2 %) | PASS | 2/2 real diffs, meta docker/none/262144/stdin-file/neutral/reinit | 0 exposed, isolation 2/2 on all four proofs |
   | `qwen36` | PASS | 253 s | PASS | 210.5→217.9 / 189.4→192.3 / 122.0→124.5 tok/s (+3.5 / +1.5 / +2.0 %) | PASS | see the janitor defect below; re-run → PASS | PASS on the re-run |

3. **Four harness defects caught by the smoke before any v3 cell rolled** (each would have hit the
   relaunch):
   - `scripts/common.sh` re-points `SCRIPT_DIR` at `scripts/`, so `run_model_cycle.sh` /
     `smoke_kernel.sh` / the smoke sourced `scripts/serve_backend.sh` (missing) — the first
     docker-mode server launch would have died. `SWEBENCH_DIR` captured before the source (`eaa0782`).
   - Official-image git HEAD is a benchmark cue → `/testbed` re-init + gate (`df92eb3`).
   - 15 of 21 preset checkpoints are `hf-mattbucci/<name>` **absolute** symlinks to
     `/data/models/<real>`; inside the container only `/models` existed, so they dangled
     (transformers: `Repo id must be in the form 'repo_name'`, 45 s into boot). `serve_backend.sh`
     now binds the resolved models root at its own host path too and runs a GPU-free checkpoint
     preflight (`DRY_RUN=1 launch.sh` → `test -e/-r` per `/models` argv entry, ~2 s) before every
     `docker run`; `qwen35-moe`'s checkpoint (a symlink into `~/.cache/huggingface`) materialised
     under `/data/models`. 21/21 presets pass (`08f972e`).
   - `rollout_image_janitor.sh` deleted every rollout image whose instance appears in the *newest*
     `runs/*/predictions.jsonl` — a finished 300/300 lane stays newest until the next lane's first
     prediction lands, so the next lane's (here: the qwen36 smoke's) freshly built image vanished
     between `docker build` and `docker run` (rc=125 `Unable to find image … locally`, empty
     prediction). Now deletes only images older than the lane's `logs/<iid>.log`;
     `audit_predictions.py` classes the failure `infra_rollout_image_missing` (`ae0a8bd`).

## In-flight findings (qwen38 opencode v3, 2026-09-22, 158/300)

Running the lane-close gate on the in-flight cell surfaced two audit defects and one
harness defect. The audit fixes landed (`d0ecc49`, `06aa9cb`); the harness fix waits for
the cycle boundary because it changes patch content.

- **`psf__requests-863` is unresolved-by-construction, in every harness version.** The
  official `swebench/sweb.eval.x86_64.psf_1776_requests-863` image ships an untracked,
  un-ignored `build/` (1012 KB, `git status --porcelain` → `?? build/`; `.gitignore` lacks
  it). The re-init keeps the tracked set bit-identical, so the tree is `dirty=1` before the
  scaffold runs, and the `git add -A && git diff --cached` capture then emits 68 `new file`
  hunks under `build/lib/requests/**` (218 KB `cacert.pem`, vendored chardet2 / urllib3)
  around the real 542 B `requests/models.py` change — 873,784 B in this cell. At scoring
  every SWE-bench 4.1.0 apply method fails on it (`git apply` "already exists",
  `git apply --reject`, then `patch --batch --fuzz=5 -p1` prints 27,510 "Assuming -R" lines
  and *reverses* the real hunk; `devstral-opencode-v2/…/psf__requests-863/run_instance.log`).
  Historical sweep: junk-bearing in 26/32 cells (~874 KB each); the 5–6 clean cells are
  runs where the model deleted `build/` itself. No other instance shows image-shipped
  junk (others appear in ≤3/30 cells — model-created). Uniform across models → no
  leaderboard reorder; absolute scores carry a ≤1/300 floor.
  - Gate: `audit_leakage.py` required `dirty=0` and would have false-failed the lane
    close on this instance (157/158). It now requires `dirty == dirty_before`
    (`docker_rollout.py` prints the image's own untracked count on the `isolation: git=`
    line from the next lane on); logs without the field fall back to
    `image_untracked_baseline.json` (`{"psf__requests-863": 1}`). 158/158 after the fix.
  - Harness (cycle boundary, README next-step 3): write the image's pre-existing untracked
    paths to `.git/info/exclude` at re-init, so the model sees a clean `git status` and
    the capture skips them. R9700's sandbox is immune (its base commit `git add -A`s the
    whole tree), so their requests-863 cells score normally — a cross-rig comparison on
    that instance is confounded until we match.
- **Five false `EXPOSED` verdicts.** `outcome()` treated any non-empty tool output as
  fetched content. All five were `curl -s` / `wget -q` failures wrapped in the model's own
  scaffolding: `EXIT: 0`, `=== 4.1 ===` loop banners, `---` + `-w %{http_code}` → `000`,
  opencode's `(no output)` placeholder. `fetched_content()` drops separator / exit-trailer /
  000 / placeholder lines and literals present in the command text. EXPOSED 5 → 0; the
  network proof (`isolation: network=none` 158/158) already said the same thing.
- **`rollout_seconds` is not a wall-hit proxy.** It starts before the per-instance image
  build (1–3 min); django-15996 finished its session at 1785 s, 1983 s elapsed, rc=0, with a
  patch. A wall hit is `rollout_returncode == 124` (the harness SIGKILL). `audit_benchmark_recall.py`
  used `>= 1795 s` (`b34e25b` fixed); `audit_predictions.py` / `ab_lane_receipt.py` already
  keyed on rc. R9700's copy keys on `WALL_S = 1795` — same over-count if their elapsed spans
  the build.
- **Lane-close re-audit of the closed qwen38 opencode v3 cell (2026-09-24, 300/300) — one
  false `EXPOSED`, two classifier blind spots.** The Phase-4.5 gate would have `exit 1`-ed
  the whole cycle unscored, so the fix is audit-side (attribution), not gate semantics.
  (a) `pytest-8365`: an argument-less `pip download --no-deps 2>/dev/null` inside a compound
  command whose *other* segments (`git show --stat`, `ls`, `find … "*pytest*"`) named the
  project — `classify_call` read the whole command as one UPSTREAM fetch and the local
  segments' output as fetched content. Now each `;`/`&&`/`||`/newline segment is classified
  on its own (quote-aware `command_segments()`), a `pip` with no requirement is not a fetch
  (`pip_has_requirement()`), and the outcome stays the conservative shared-output reading —
  the first attempt (deterministic-tool-without-success-signature → not fetched) un-exposed
  5 *real* quiet fetches in the netopen cell (`pip … -q | tail -2; ls /tmp/…`, `gh pr diff`,
  gold overlap 1.0) and was reverted. (b) Inline-python fetches (`python -c` / heredoc with
  `urlopen` / `requests.get` / `httpx` / `http.client` / `socket.create_connection`) were never
  detected — `matplotlib-25442`'s "curl" was a `User-Agent: "curl"` string inside a urllib
  heredoc. Detected now (`PY_EXEC_RE` + `PY_FETCH_RE`); a traceback whose frames pass through
  the network stack counts as a failed fetch even when the model's `| head -5` cut the
  `URLError` line (`django-13964`: `PY_NET_TRACEBACK_RE`). v3: attempted 148 → 159, EXPOSED
  1 → 0, all four proofs 300/300. Same classifier on the stored netopen cell: 168 → **186/299
  (62 %)** exposed — 18 real python/gh/pip fetches the 2026-09-19 study missed
  (`astropy-6938`, `django-11283/13658`, `matplotlib-22835/23299`, `requests-2148`,
  `xarray-3364`, `pytest-6116`, `sklearn-14983`, `sphinx-8435`, eight sympy). The study's
  56 % is a floor; its receipt is left as published. Offline cases:
  `scripts/eval/test_audit_leakage_segments.py` (15).
