# GPU utilization and harness overhead of the v3 re-roll (qwen38 opencode / opencode+DCP lanes)

**Question.** The 256K re-roll is ~5,900 instances from done. How much of the campaign's wall clock is the model working, and where does the rest go?

**Method.** `scripts/gpu_telemetry.sh` (nvidia-smi, both GPUs, 30 s samples, `/var/tmp/gpu-telemetry/`, running since 2026-09-22 20:13) read per lane window; buildkit step timings from the lane's `rollout-<scaffold>.log` (`#N DONE t`); per-instance anatomy from the opencode JSON event timestamps in `runs/<cell>/logs/<iid>.log` against the log mtime and `# elapsed`. Cells: `qwen38-opencode-v3` (300/300) and `qwen38-opencode-dcp-v3` (first 95). Read-only; nothing was added to the box while the lane rolled.

## GPU duty cycle

| Lane window | Hours | busy (util ≥ 50 %) | mean util | SM clock (busy) | power (busy) | throttle reason | temp max / median | fan max |
|---|---:|---:|---:|---:|---:|---|---|---:|
| opencode (09-22 20:13 → 09-24 13:36) | 41.4 | **73.2 %** | 72 % | 1560 / 1515 MHz | 259 W | `0x4` SW power cap (the 260 W profile) | 60 / 56 °C · 73 / 65 °C | 89 % |
| opencode+DCP (09-24 13:38 → 09-26 00:00) | 34.2 | **65.4 %** | 65 % | 1560 / 1515 MHz | 259 W | `0x4` | 62 / 55 °C · 73 / 64 °C | 89 % |

(GPU0 / GPU1; both GPUs track within 0.1 pp because TP=2.) While busy the model runs exactly at the power cap — the serving side is where the roofline work in README item 2 applies; the idle third is harness.

**Idle gaps, DCP lane, GPU0:** 188 gaps, **11.8 h of 34.2 h**. By length: 79 gaps of 5–10 min = 8.5 h (one per instance — image build + container boot + the DCP plugin tax + cleanup pass, below), 30 of 2–5 min = 1.8 h (in-session test runs, cleanup), 77 under 2 min = 1.0 h, 2 over 10 min = 0.6 h.

## Where an instance's wall goes (medians)

| | opencode-v3 (n=300, 53 walls) | opencode-dcp-v3 (n=95, 19 walls) |
|---|---:|---:|
| `rollout_seconds` (build → exit), non-wall rc=0 | 924 s | 1102 s |
| docker image build (inside `rollout_seconds`) | 176 s | 176 s |
| build → first opencode event ("pre") | 198 s | **271 s** |
| ⇒ container boot after the build | ~22 s | **~95 s** |
| session (first → last event), non-wall rc=0 | 735 s | 710 s |
| last event → exit ("post"), non-wall rc=0 | **0 s**; cleanup pass = 2nd session on 235/235 | **120 s** on 71/72; 2nd session on 1/72 |

(`python evals/swebench/lane_overhead.py --run <cell> --rollout-log <lane log> --telemetry <csv> --since <T>` reproduces every number here.)

**Image build — 176 s median (mean 180, max 477, min 166), 14–16 % of `rollout_seconds`.** Per-instance, because every layer is per-parent on a unique `swebench/sweb.eval.x86_64.<iid>` base. Step medians: `#18` dcode conda env + pip 68 s, `#21` export + unpack 60 s, `#17` prime-agent install 12 s, `#10` npm opencode + little-coder 10 s, `#14` DCP plugin 8 s, `#8` apt 7 s. Campaign scale: 5,908 remaining instances × 176 s ≈ **289 h ≈ 12 days** of GPU-idle wall, the largest harness-side lever (README next-step 4 already names it; these are the measured numbers behind it).

## New: the opencode+DCP lane pays ~190 s per instance to an offline npm attempt, and its cleanup pass never runs

From the in-container opencode log of a live DCP instance (`/opt/dcp-home/.local/share/opencode/log/`, read by `docker exec` while it ran):

```
INFO  06:39:20 +0ms     service=plugin name=lm loading internal plugin
WARN  06:40:30 +70502ms service=config dir=/opt/dcp-home/.config/opencode
      error=Cause([Fail(NpmInstallFailedError (cause: FetchError: request to
      https://registry.npmjs.org/uuid/-/uuid-13.0.2.tgz failed ...
INFO  06:40:30 +1ms     service=plugin path=@tarquinen/opencode-dcp@3.1.15 loading plugin
INFO  06:40:30 +17ms    service=server method=POST path=/session request
```

opencode 1.14.25's plugin loader re-resolves the configured `@tarquinen/opencode-dcp@3.1.15` on every start and tries to fetch a transitive dependency (`uuid-13.0.2.tgz`) from the registry. The `--network none` sandbox cannot answer, the attempt takes **70.5 s** to fail, then the pre-installed copy loads and the session proceeds normally (this is the 271 − 198 = 73 s boot difference above). The **cleanup pass** (`timeout 120 opencode run … <<<"$CLEANUP_PROMPT"`) starts a second opencode under the same HOME, whose log stops at `loading internal plugin`: at 106 s it had still not passed the npm step, and `timeout` kills it at 120 s. Every DCP instance therefore (a) starts its session ~70 s later than the control, (b) burns 120 s GPU-idle at the end, and (c) **never gets the self-clean pass** the control lane gets (the pass that deletes model-written `reproduce_*.py` / `debug_*.py` helpers before the diff; the score-time fallback in `filter_predictions.py` catches only the pytest-collected class).

Consequences and handling:

- **A/B caveat for the DCP cell** — the session budget under the 1800 s wall is ~1730 s on DCP vs ~1778 s on control, and sessions that finish in the last ~120 s before the wall would be lost to the dead cleanup pass on DCP only. Of the 19 DCP wall hits so far, none shows the signature (a finished step followed by ≥ 120 s of silence): the two that were silent ≥ 100 s before the kill (102 s, 106 s) had no finished step — mid-think, the recall pattern, same as 11/53 on the control lane. No wall is attributable to it yet; `lane_overhead.py --run` re-checks at lane close. The missing cleanup pass is read at lane close as the count of new top-level helper files in DCP vs control diffs on the same ids.
- **Not fixed mid-lane.** A fix changes the DCP lane's behaviour (boot, budget, cleanup) between instance 95 and 96 — the A/B pair must keep one rule for the whole lane. It lands at the qwen38 cycle boundary with the other harness decisions (README next-step 3/4): vendor the plugin so the loader's install step is a no-op offline (verification: a `--network none` boot reaches `loading plugin` in < 2 s, and a second `opencode run` in the same HOME emits its JSON events inside the 120 s cleanup budget), or drop the loader round-trip by pointing `plugin` at a local path. The little-coder+RTK lane is checked for the same class on its first instance (pre-first-event vs the little-coder control).
- Cost at campaign scale: 190 s × 205 remaining DCP instances ≈ 11 h on this lane; every future DCP lane the same until fixed.

## Standing numbers for the loop

Per hourly wakeup: busy % over the last window, idle-gap count ≥ 5 min (should equal instances completed), median `post` for the current lane (0 s = cleanup pass runs; 120 s = it does not), decode tok/s from the server's `Decode batch` lines (60–65 tok/s at 50–70K tokens of context, the qwen38 preset's expected range). One tool: [`evals/swebench/lane_overhead.py`](../../evals/swebench/lane_overhead.py) (`--telemetry` duty cycle + gap histogram, `--rollout-log` build step medians, `--run` pre / session / post per cell).

## Server-side decode profile during the live lanes

From the `bakeoff-sglang` container's `Decode batch` / `Prefill batch` lines over each lane (single-request throughout — `#running-req: 1` on 100 % of decode lines, so this is the single-user number the project optimises for):

**Decode tok/s is a clean, stable function of context depth** (qwen38 preset, TP=2, 260 W cap), identical across both lanes:

| context | 0–20K | 20–40K | 40–60K | 60–80K | 80–100K | 100–120K | 120–140K | 140–160K | 160–180K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tok/s (median) | 67 | 65 | 63 | 61 | 60 | 58 | 56 | 55 | 54 | 52 |

≈ −1.5 tok/s per 20K tokens; p10–p90 within ±1 tok/s in every bin (no thermal or batching variance — the busy-clock is pinned at the power cap). Matches the [workload profile](../qwen38-agentic-workload-profile-2026-09-10.md) (55 % of the bandwidth roofline); nothing regressed. **Prefill is not the cost:** radix cache hit is 97–98 %, median 373–474 genuinely-new tokens per prefill batch — the re-prefill after each tool turn is nearly free, so ~86 % of server time is decode as the profile says. The serving levers in README item 2 (NGRAM spec-decode, `lm_head` INT8) act on exactly this decode curve.

**DCP changes the server workload, not just the plugin's own bookkeeping** — over the first 95 DCP instances vs the 300 opencode instances, per instance:

| | opencode | opencode+DCP |
|---|---:|---:|
| decode tokens / instance | 21.1K | **50.0K** |
| decode-line context median | 63K | **49K** |
| prefill batches / instance (tool turns) | 24 | **53** |

DCP prunes/compresses conversation context, so the model decodes at a **shorter** median depth (49K vs 63K → a few tok/s faster per step) but runs **more than twice** the turns and generates **2.4× the output tokens** per instance — the longer agentic sessions the engagement receipt already showed (median 1,197 s vs 730 s). Net: DCP's GPU cost per instance is higher, and its wall time is dominated by more-but-shorter decode passes plus the per-instance harness tax above — the plugin is doing work, the question the scored cell answers is whether that extra work resolves more instances.
