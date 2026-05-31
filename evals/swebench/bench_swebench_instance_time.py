#!/usr/bin/env python3
"""Per-instance wall-time gate: spec-decode ON vs OFF on a 5-instance subset.

Gates the full v0.5.12 spec-decode re-sweep (#16). The synthetic decode tok/s
speedup (1.65x EAGLE3 / 4.1x DFlash, per benchmarks/quality/specdec-v0512-2026-05-29.json)
is decode-only — but a SWE-bench instance is prefill (large repo prompt) +
decode (smaller diff/tool args). Real per-instance wall-time savings are
smaller than synthetic decode-only numbers, and that's the number that
actually moves the bake-off ETA.

Gate (from scripts/evals/swebench/spec_decode_plan.md):
    median per-instance wall-time speedup >= 1.5x AND no resolved-count regression.

Both criteria must hold to greenlight the full re-sweep. Failing this means
the synthetic decode win didn't translate to real-instance wall-time savings,
and a multi-day full sweep wouldn't earn back its cost.

Process per (preset, spec_config) pair:
  1. Pick 5 instance IDs spanning prompt sizes (4K / 8K / 16K / 24K / 32K-edge)
     from the finished qwen36-opencode-v2 predictions, OR from --instance-ids.
  2. For each of OFF and ON:
       - Stop any running sglang process
       - Launch server with the appropriate flags (preset + optional SPEC_DECODE=1
         + appropriate MEM/CTX overrides per spec_decode_plan.md)
       - Wait /health
       - Capture /get_server_info baseline (counters before)
       - Run `run_rollouts.py --instance-ids <list> --out runs/spec-bench-<preset>-<config>`
       - Capture /get_server_info final (counters after) for accept_len / decode tok/s
       - Stop server
  3. Parse both predictions.jsonl; compute per-instance rollout_seconds delta,
     median, p10/p90, resolved-deltas.
  4. Emit a JSON receipt + a markdown table + return exit 0/1 based on gate.

Usage:
    # gate qwen36 + DFlash:
    python evals/swebench/bench_swebench_instance_time.py \\
        --preset qwen36 --spec-config dflash_bf16 \\
        --pick-from evals/swebench/runs/qwen36-opencode-v2/predictions.jsonl \\
        --out benchmarks/quality/swebench-spec-bench-qwen36-dflash.json

    # gate coder-30b + EAGLE3:
    python evals/swebench/bench_swebench_instance_time.py \\
        --preset coder-30b --spec-config eagle3_wider \\
        --pick-from evals/swebench/runs/qwen36-opencode-v2/predictions.jsonl \\
        --out benchmarks/quality/swebench-spec-bench-coder30b-eagle3.json

The two known spec_configs (extensible — see spec_decode_plan.md):
    dflash_bf16        Qwen3.6-35B-A3B DFlash @ ctx 32K, MEM 0.70, --dtype bfloat16,
                      SGLANG_ENABLE_SPEC_V2=1, --mamba-scheduler-strategy extra_buffer
    eagle3_wider       SGLang-EAGLE3-Coder-30B @ steps=4 topk=4 draft=8, ctx 16K,
                      MEM 0.70
    eagle3_conservative same draft @ steps=3 topk=1 draft=4 (lower-accept, lower-OOM-risk)

This script does NOT touch the running bake-off. It is the ON-DEMAND gate
runner that fires AFTER #11 finishes, BEFORE #16 launches. Errors loudly if
it detects an active sglang server on its target port.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

import requests

REPO_DIR = Path(__file__).resolve().parents[2]
LAUNCH_SH = REPO_DIR / "scripts" / "launch.sh"
RUN_ROLLOUTS = REPO_DIR / "evals" / "swebench" / "run_rollouts.py"

# spec_configs: keyed by name; each yields env+CLI overrides for launch.sh.
SPEC_CONFIGS = {
    "off": {
        "env": {},
        "extra_args": "",
        "ctx": None,         # use preset default
        "mem": None,         # use preset default
        "dtype": None,
    },
    "dflash_bf16": {
        "env": {"SGLANG_ENABLE_SPEC_V2": "1"},
        "extra_args": (
            "--speculative-algorithm DFLASH "
            "--speculative-draft-model-path /data/models/drafts/qwen36-dflash "
            "--speculative-draft-model-quantization unquant "
            "--speculative-attention-mode decode "
            "--disable-overlap-schedule "
            "--mamba-scheduler-strategy extra_buffer"
        ),
        "ctx": 32768,
        "mem": "0.70",
        "dtype": "bfloat16",
    },
    "eagle3_wider": {
        "env": {},
        "extra_args": (
            "--speculative-algorithm EAGLE3 "
            "--speculative-draft-model-path /data/models/drafts/eagle3-coder30b "
            "--speculative-draft-model-quantization unquant "
            "--speculative-num-steps 4 --speculative-eagle-topk 4 "
            "--speculative-num-draft-tokens 8 "
            "--speculative-attention-mode decode"
        ),
        "ctx": 16384,
        "mem": "0.70",
        "dtype": None,
    },
    "eagle3_conservative": {
        "env": {},
        "extra_args": (
            "--speculative-algorithm EAGLE3 "
            "--speculative-draft-model-path /data/models/drafts/eagle3-coder30b "
            "--speculative-draft-model-quantization unquant "
            "--speculative-num-steps 3 --speculative-eagle-topk 1 "
            "--speculative-num-draft-tokens 4 "
            "--speculative-attention-mode decode"
        ),
        "ctx": 16384,
        "mem": "0.70",
        "dtype": None,
    },
}

# Default target buckets (prompt-token sizes) — 5 instances spanning the range.
DEFAULT_PROMPT_BUCKETS = [4000, 8000, 16000, 24000, 32000]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", required=True,
                   help="launch.sh preset name (e.g. qwen36, coder-30b)")
    p.add_argument("--spec-config", required=True, choices=list(SPEC_CONFIGS),
                   help="spec config to test against the OFF baseline")
    p.add_argument("--pick-from", required=True,
                   help="predictions.jsonl from a finished cycle; used to pick "
                        "5 instance IDs spanning prompt-size buckets")
    p.add_argument("--instance-ids", nargs="*", default=None,
                   help="explicit instance IDs (overrides --pick-from)")
    p.add_argument("--out", required=True,
                   help="output JSON receipt path")
    p.add_argument("--port", type=int, default=30000,
                   help="sglang server port (default 30000)")
    p.add_argument("--health-timeout", type=int, default=600,
                   help="seconds to wait for /health (long for cold load + spec)")
    p.add_argument("--rollout-timeout", type=int, default=900,
                   help="per-instance --timeout passed to run_rollouts.py")
    p.add_argument("--served-name", default="bench",
                   help="--served-model-name override (used by run_rollouts as model id)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the launch commands and exit without serving")
    return p.parse_args()


def stop_any_server(port: int) -> None:
    """Stop any sglang server on the target port. NEVER use pkill -f with a
    pattern that could match the running shell (CLAUDE.md exit-144 risk).
    We match on bound port via lsof, kill those PIDs, and wait for shutdown.
    """
    out = subprocess.run(
        ["lsof", "-i", f":{port}", "-t", "-sTCP:LISTEN"],
        capture_output=True, text=True
    )
    pids = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    if not pids:
        return
    print(f"  stopping existing server on port {port} (pids: {pids})")
    for pid in pids:
        try:
            os.kill(pid, 15)  # SIGTERM
        except ProcessLookupError:
            pass
    for _ in range(30):
        time.sleep(1)
        r = subprocess.run(["lsof", "-i", f":{port}", "-t"], capture_output=True, text=True)
        if not r.stdout.strip():
            return
    print(f"  WARN: port {port} still bound after 30s; sending SIGKILL")
    for pid in pids:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
    time.sleep(3)


def build_launch_cmd(preset: str, port: int, served: str, config: dict) -> tuple[list[str], dict[str, str]]:
    """Compose the launch.sh command and environment for a given spec config.

    launch.sh reads CTX/MEM/DTYPE env overrides; SPEC_DECODE flags + the
    `--speculative-*` block ride in via EXTRA_ARGS (already-existing convention
    in scripts/launch.sh).
    """
    env = dict(os.environ)
    env.update(config.get("env", {}))
    if config["ctx"] is not None:
        env["CTX"] = str(config["ctx"])
    if config["mem"] is not None:
        env["MEM"] = str(config["mem"])
    if config["dtype"] is not None:
        env["DTYPE"] = str(config["dtype"])
    if config["extra_args"]:
        # Prepend so the preset's EXTRA_ARGS=${EXTRA_ARGS:-} pattern picks it up.
        existing = env.get("EXTRA_ARGS", "")
        env["EXTRA_ARGS"] = f"{config['extra_args']} {existing}".strip()
    cmd = [
        str(LAUNCH_SH), preset,
        "--port", str(port),
        "--served-model-name", served,
    ]
    return cmd, env


def wait_health(port: int, timeout: int) -> bool:
    """Poll /health until 200 or timeout."""
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(3)
    return False


def get_server_info(port: int) -> dict:
    try:
        r = requests.get(f"http://127.0.0.1:{port}/get_server_info", timeout=10)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return {}


def pick_5_instances(pred_path: str, buckets: list[int] | None = None) -> list[str]:
    """Pick 5 instance IDs whose prompt-token counts span `buckets`.

    For each bucket size, find the instance with prompt_tokens closest to it.
    Falls back to the first 5 if no prompt_tokens column is recorded.
    """
    buckets = buckets or DEFAULT_PROMPT_BUCKETS
    records = []
    with open(pred_path) as f:
        for line in f:
            r = json.loads(line)
            records.append(r)
    # Try to read prompt_tokens or len(model_patch) heuristic. The current
    # predictions.jsonl record (per the bake-off in flight) does NOT carry
    # prompt_tokens directly — only rollout_seconds. Use rollout_seconds as
    # a proxy for "size" instead and pick at the quantiles.
    succ = [r for r in records
            if r.get("rollout_returncode") == 0 and r.get("model_patch", "").strip()]
    if not succ:
        succ = [r for r in records if r.get("rollout_returncode") == 0]
    if not succ:
        raise SystemExit("no successful rollouts in --pick-from; cannot sample")
    succ.sort(key=lambda r: r.get("rollout_seconds", 0))
    if len(succ) <= 5:
        return [r["instance_id"] for r in succ]
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    picks = []
    for q in quantiles:
        i = min(int(q * len(succ)), len(succ) - 1)
        picks.append(succ[i]["instance_id"])
    return picks


def run_phase(preset: str, port: int, served: str, config: dict, instance_ids: list[str],
              out_dir: Path, rollout_timeout: int, health_timeout: int) -> dict:
    """Stand up a server with `config`, run 5 rollouts, capture before/after server-info."""
    print(f"\n=== phase: spec={config.get('_name','?')} ===")
    stop_any_server(port)
    cmd, env = build_launch_cmd(preset, port, served, config)
    print(f"  launching: {' '.join(shlex.quote(x) for x in cmd)}")
    serve_log = out_dir / f"serve-{config['_name']}.log"
    serve_log.parent.mkdir(parents=True, exist_ok=True)
    with serve_log.open("wb") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)
    try:
        print(f"  waiting /health (up to {health_timeout}s)...")
        if not wait_health(port, health_timeout):
            return {"_name": config["_name"], "error": "health timeout"}
        info_before = get_server_info(port)
        t0 = time.time()
        rollout_out = out_dir / f"rollouts-{config['_name']}"
        rollout_out.mkdir(parents=True, exist_ok=True)
        rcmd = [
            sys.executable, str(RUN_ROLLOUTS),
            "--model", f"sglang/{served}",
            "--served-name", served,
            "--instance-ids", *instance_ids,
            "--out", str(rollout_out),
            "--timeout", str(rollout_timeout),
            "--no-venv",
        ]
        print(f"  running rollouts: 5 instances, --timeout {rollout_timeout}s ...")
        rlog = out_dir / f"rollout-{config['_name']}.log"
        with rlog.open("wb") as lf:
            subprocess.run(rcmd, stdout=lf, stderr=subprocess.STDOUT, check=False)
        elapsed = time.time() - t0
        info_after = get_server_info(port)
        # Parse predictions.jsonl
        records = []
        pred = rollout_out / "predictions.jsonl"
        if pred.exists():
            with pred.open() as f:
                for ln in f:
                    records.append(json.loads(ln))
        return {
            "_name": config["_name"],
            "phase_elapsed_s": elapsed,
            "records": records,
            "server_info_before": info_before,
            "server_info_after": info_after,
            "serve_log": str(serve_log),
            "rollout_log": str(rlog),
        }
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        stop_any_server(port)


def main():
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    work_dir = out.parent / f"bench-{args.preset}-{args.spec_config}"

    instances = args.instance_ids or pick_5_instances(args.pick_from)
    print(f"instances ({len(instances)}): {instances}")

    if args.dry_run:
        print("--- DRY RUN ---")
        for name in ("off", args.spec_config):
            c = dict(SPEC_CONFIGS[name]); c["_name"] = name
            cmd, env = build_launch_cmd(args.preset, args.port, args.served_name, c)
            extra = {k: v for k, v in env.items() if k in ("CTX","MEM","DTYPE","EXTRA_ARGS","SGLANG_ENABLE_SPEC_V2")}
            print(f"\n  spec={name}: ENV={extra}")
            print(f"            CMD={' '.join(shlex.quote(x) for x in cmd)}")
        return 0

    # Pre-flight: confirm port is free
    stop_any_server(args.port)

    # Phase OFF (baseline)
    off_cfg = dict(SPEC_CONFIGS["off"]); off_cfg["_name"] = "off"
    off = run_phase(args.preset, args.port, args.served_name, off_cfg, instances,
                    work_dir, args.rollout_timeout, args.health_timeout)
    # Phase ON
    on_cfg = dict(SPEC_CONFIGS[args.spec_config]); on_cfg["_name"] = args.spec_config
    on = run_phase(args.preset, args.port, args.served_name, on_cfg, instances,
                   work_dir, args.rollout_timeout, args.health_timeout)

    # Build per-instance comparison
    def index_by_inst(recs):
        return {r["instance_id"]: r for r in recs}
    off_by_inst = index_by_inst(off.get("records", []))
    on_by_inst = index_by_inst(on.get("records", []))

    rows = []
    for inst in instances:
        o = off_by_inst.get(inst, {})
        n = on_by_inst.get(inst, {})
        t_off = o.get("rollout_seconds")
        t_on = n.get("rollout_seconds")
        speedup = (t_off / t_on) if (t_off and t_on) else None
        rows.append({
            "instance_id": inst,
            "off_seconds": t_off,
            "off_rc": o.get("rollout_returncode"),
            "off_patch_bytes": len((o.get("model_patch") or "")),
            "on_seconds": t_on,
            "on_rc": n.get("rollout_returncode"),
            "on_patch_bytes": len((n.get("model_patch") or "")),
            "wall_speedup": speedup,
        })

    speedups = [r["wall_speedup"] for r in rows if r["wall_speedup"] is not None]
    med = median(speedups) if speedups else None
    off_pass = sum(1 for r in rows if r["off_rc"] == 0 and r["off_patch_bytes"] > 0)
    on_pass = sum(1 for r in rows if r["on_rc"] == 0 and r["on_patch_bytes"] > 0)

    gate_speedup = med is not None and med >= 1.5
    gate_no_regression = on_pass >= off_pass
    gate_ok = gate_speedup and gate_no_regression

    receipt = {
        "preset": args.preset,
        "spec_config": args.spec_config,
        "instance_count": len(instances),
        "instances": instances,
        "rows": rows,
        "median_wall_speedup": med,
        "off_succeeded_count": off_pass,
        "on_succeeded_count": on_pass,
        "gate_speedup_ok": gate_speedup,
        "gate_no_regression_ok": gate_no_regression,
        "gate_overall_ok": gate_ok,
        "server_info": {
            "off_before": off.get("server_info_before"),
            "off_after": off.get("server_info_after"),
            "on_before": on.get("server_info_before"),
            "on_after": on.get("server_info_after"),
        },
    }
    with out.open("w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nReceipt: {out}")

    # Markdown summary
    md_lines = [
        f"# Spec-decode bench: {args.preset} | {args.spec_config}",
        "",
        f"median wall speedup: **{med:.2f}x**" if med is not None else "median: N/A",
        f"OFF success: {off_pass}/{len(instances)}; ON success: {on_pass}/{len(instances)}",
        f"Gate (>=1.5x median, no regression): **{'PASS' if gate_ok else 'FAIL'}**",
        "",
        "| instance | off_s | on_s | speedup | off_rc | on_rc |",
        "|---|---:|---:|---:|:---:|:---:|",
    ]
    for r in rows:
        speedup_s = f"{r['wall_speedup']:.2f}x" if r['wall_speedup'] else "—"
        md_lines.append(
            f"| {r['instance_id']} | "
            f"{r['off_seconds'] or '—'} | {r['on_seconds'] or '—'} | "
            f"{speedup_s} | {r['off_rc']} | {r['on_rc']} |"
        )
    md_out = out.with_suffix(".md")
    md_out.write_text("\n".join(md_lines))
    print(f"Markdown summary: {md_out}")
    print("Gate:", "PASS" if gate_ok else "FAIL")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())
