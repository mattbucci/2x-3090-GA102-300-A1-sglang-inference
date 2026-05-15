#!/usr/bin/env python3
"""Docker-backed rollout: a coding-agent scaffold runs INSIDE a swebench
instance container. Same env as the official scoring harness (Python
3.6/3.8/3.9/..., conda testbed activated, all system deps installed) —
the model can run pytest mid-iteration against the actual environment
its fix will be graded in.

Per-instance flow:
  1. Pull the upstream eval image  swebench/sweb.eval.x86_64.<normalized>:latest
     (instance_id with __ -> _1776_ per the swebench tag convention).
  2. Build a rollout image  swebench-rollout/<instance_id>:latest  by adding
     Node + opencode + little-coder + ripgrep on top of the eval image
     (Dockerfile.rollout). claw-code lives in Dockerfile.rollout-claw
     because of its Rust build + Anthropic-proxy requirement.
  3. Run a container with --network=host so the scaffold reaches the host
     SGLang server at http://127.0.0.1:23334. Working directory is /testbed
     (already cloned + conda-activated by the upstream eval image).
  4. Exec the scaffold against the problem statement; capture the resulting
     `git diff` as the prediction patch.
  5. Append to predictions.jsonl in the v1 schema + a `rollout_scaffold`
     field so downstream tools can group by scaffold.

Per-(model, scaffold) results are isolated to their own output dir so
predictions, logs, and scores never collide across scaffolds:

    evals/swebench/runs/<preset>-<scaffold>-v2/
        predictions.jsonl                # rollout output
        predictions/<inst>.diff
        logs/<inst>.log
        scores-docker.jsonl              # filled by score_docker.py
        meta.json                        # scaffold + model + run metadata

Usage:
    python evals/swebench/docker_rollout.py \\
        --model sglang/coder-30b-eval \\
        --scaffold opencode \\
        --out evals/swebench/runs/coder-30b-opencode-v2 \\
        --skip-existing
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
DOCKERFILE = THIS_DIR / "docker" / "Dockerfile.rollout"
DOCKER_CTX = THIS_DIR / "docker"


SUPPORTED_SCAFFOLDS = ("opencode", "little-coder", "claw-code")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Scaffold-side model id. opencode/little-coder format: "
                        "<provider>/<served-name>. Examples: sglang/coder-30b-eval, "
                        "openai/coder-30b-eval. Provider is opaque to the rollout — "
                        "just controls the scaffold's request routing.")
    p.add_argument("--scaffold", default="opencode", choices=SUPPORTED_SCAFFOLDS,
                   help="Which agent scaffold to invoke inside the container.")
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite",
                   help="HF dataset id (Lite=300, Verified=500)")
    p.add_argument("--split", default="test")
    p.add_argument("--instances", type=int, default=0,
                   help="Limit to first N instances (0 = all)")
    p.add_argument("--instance-ids", nargs="*", default=None,
                   help="Specific instance IDs to run (overrides --instances)")
    p.add_argument("--out", required=True,
                   help="Output dir for predictions + logs (per-scaffold convention: "
                        "evals/swebench/runs/<preset>-<scaffold>-v2/)")
    p.add_argument("--timeout", type=int, default=1800,
                   help="Per-instance scaffold timeout (seconds). v2 opencode "
                        "distribution: p50=121s, p90=210s, p99=1014s, max=1328s — "
                        "1800s catches all but pathological cases. little-coder + "
                        "claw-code default to the same ceiling pending real data.")
    p.add_argument("--server-url", default="http://127.0.0.1:23334",
                   help="SGLang server base URL (used for preflight)")
    p.add_argument("--served-name", default=None,
                   help="Served model name (defaults to model id after slash)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip instances that already have a prediction")
    p.add_argument("--max-empty-streak", type=int, default=10,
                   help="Abort after this many consecutive empty diffs")
    p.add_argument("--keep-containers", action="store_true",
                   help="Don't `docker rm` after each instance (debug)")
    p.add_argument("--no-pull", action="store_true",
                   help="Skip `docker pull` of the upstream eval image (use only if "
                        "already cached locally; otherwise the build will fail)")
    p.add_argument("--rebuild-image", action="store_true",
                   help="Force `docker build` even when the rollout image already "
                        "exists. Use after Dockerfile.rollout changes (e.g. adding "
                        "a new scaffold) so the layered binaries are present.")
    return p.parse_args()


def build_scaffold_invocation(scaffold: str, model: str, served_name: str) -> tuple[list[str], str]:
    """Return (docker_run_extra_envs, inner_shell_command) for the given
    scaffold. The inner command runs opencode-equivalent against $PROMPT
    in /testbed and emits `=== DIFF ===\\n<git diff>` to stdout for the
    parent process to extract.

    All three scaffolds share the same diff-capture protocol so the parent
    `_extract_diff_from_stdout` works uniformly.
    """
    if scaffold == "opencode":
        # opencode reads ~/.config/opencode/opencode.json which the Dockerfile
        # provisions. The `sglang/<served-name>` model id wires there.
        envs = []
        inner = (
            f"set -e\n"
            f"git config --global user.email eval@local\n"
            f"git config --global user.name eval\n"
            f"git config --global --add safe.directory /testbed\n"
            f"opencode run --dir /testbed --model {model} "
            f"  --format json --dangerously-skip-permissions \"$PROMPT\" || true\n"
            f"echo === DIFF ===\n"
            f"rm -rf /testbed/.claw /testbed/.opencode /testbed/.sandbox-tmp /testbed/.sandbox-home /testbed/.cache\n"
            f"git -C /testbed add -A\n"
            f"git -C /testbed diff --cached\n"
        )
        return envs, inner

    if scaffold == "little-coder":
        # little-coder wraps pi-ai, which reads provider baseUrls from a
        # packaged models.json — not from OPENAI_BASE_URL. Dockerfile.rollout
        # repoints the "llamacpp" provider's baseUrl at our SGLang endpoint,
        # so we route through llamacpp/<served-name>. pi will warn that the
        # model id isn't in its known list and fall back to "Using custom
        # model id" — that warning is benign and the request still reaches
        # SGLang's OpenAI-compat endpoint.
        oc_model = f"llamacpp/{served_name}"
        envs = [
            "--env", "LLAMACPP_API_KEY=noop",
        ]
        inner = (
            f"set -e\n"
            f"git config --global user.email eval@local\n"
            f"git config --global user.name eval\n"
            f"git config --global --add safe.directory /testbed\n"
            f"cd /testbed\n"
            f"little-coder --model {oc_model} \"$PROMPT\" || true\n"
            f"echo === DIFF ===\n"
            f"rm -rf /testbed/.claw /testbed/.opencode /testbed/.sandbox-tmp /testbed/.sandbox-home /testbed/.cache\n"
            f"git -C /testbed add -A\n"
            f"git -C /testbed diff --cached\n"
        )
        return envs, inner

    if scaffold == "claw-code":
        # claw natively supports OpenAI-compat: OPENAI_BASE_URL +
        # OPENAI_API_KEY, model id "openai/<served>". The openai/ prefix
        # wins over the ambient credential sniffer, so DashScope's
        # qwen-/qwen prefix routing won't intercept (claw USAGE.md
        # "Provider matrix" + PR 3001 reasoning_content support). The
        # rollout image must include the pre-built claw binary at
        # /usr/local/bin/claw — Dockerfile.rollout COPYs it.
        oc_model = f"openai/{served_name}"
        envs = [
            "--env", "OPENAI_BASE_URL=http://127.0.0.1:23334/v1",
            "--env", "OPENAI_API_KEY=noop",
        ]
        inner = (
            f"set -e\n"
            f"git config --global user.email eval@local\n"
            f"git config --global user.name eval\n"
            f"git config --global --add safe.directory /testbed\n"
            f"cd /testbed\n"
            f"/usr/local/bin/claw --model {oc_model} prompt \"$PROMPT\" || true\n"
            f"echo === DIFF ===\n"
            f"rm -rf /testbed/.claw /testbed/.opencode /testbed/.sandbox-tmp /testbed/.sandbox-home /testbed/.cache\n"
            f"git -C /testbed add -A\n"
            f"git -C /testbed diff --cached\n"
        )
        return envs, inner

    raise ValueError(f"unknown scaffold: {scaffold}")


# --- helpers ---------------------------------------------------------------

def sh(*args, check=True, capture=False, cwd=None, env=None, timeout=None):
    """subprocess.run wrapper. capture=True returns (rc, stdout, stderr)."""
    kwargs = {
        "cwd": cwd,
        "env": env,
        "timeout": timeout,
    }
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    proc = subprocess.run(list(args), check=False, **kwargs)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, args,
                                            output=proc.stdout if capture else None,
                                            stderr=proc.stderr if capture else None)
    if capture:
        return proc.returncode, proc.stdout, proc.stderr
    return proc.returncode


def _swebench_image_tag(instance_id: str) -> str:
    """Upstream eval image tag for a SWE-bench instance.

    SWE-bench normalizes `<org>__<repo>-<n>` to `<org>_1776_<repo>-<n>` for
    Docker tag compatibility (image tags can't contain `__` consistently).
    """
    normalized = instance_id.replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{normalized}:latest"


def rollout_image_tag(instance_id: str) -> str:
    """Local rollout image tag (eval image + Node + opencode + ripgrep)."""
    return f"swebench-rollout/{instance_id}:latest"


def ensure_rollout_image(instance_id: str, *, no_pull: bool = False,
                         rebuild: bool = False) -> str:
    """Build the per-instance rollout image if not already present.

    Idempotent: docker build is a no-op when the tag exists with all layers
    cached, so re-runs after the first instance build are fast. With
    rebuild=True, forces a fresh build even if the tag already resolves —
    use after Dockerfile.rollout changes (e.g. adding a new scaffold).
    """
    rollout_tag = rollout_image_tag(instance_id)
    base_tag = _swebench_image_tag(instance_id)

    if not rebuild:
        rc, out, _ = sh("docker", "image", "inspect", rollout_tag,
                        check=False, capture=True)
        if rc == 0:
            return rollout_tag

    if not no_pull:
        rc_pull = sh("docker", "pull", base_tag, check=False)
        if rc_pull != 0:
            raise RuntimeError(f"docker pull {base_tag} failed (rc={rc_pull})")

    sh("docker", "build",
       "-t", rollout_tag,
       "--build-arg", f"BASE={base_tag}",
       "-f", str(DOCKERFILE),
       str(DOCKER_CTX),
       check=True)
    return rollout_tag


PROMPT_TEMPLATE = """\
You are working on a GitHub issue in this repository.

The repo is already installed in editable mode in the active conda environment
on your PATH (testbed). You can run `pytest` and `python -c "..."` to verify
imports, exercise edge cases, and re-run tests after each edit. Use this:
write a fix, run the relevant tests, observe failures, refine until green.

Read the problem carefully, locate the relevant code, and write the minimal
patch that fixes the bug. Do not modify tests. Do not add new files unless
strictly required. When you're confident the fix is correct AND the tests
exercise it correctly, stop — your final state will be captured as a `git diff`.

# Problem

{problem_statement}

# Hints (optional, may be empty)

{hints}
"""




def run_in_container(image_tag: str, instance_id: str, prompt: str, model: str,
                     timeout: int, log_path: Path, *, keep: bool = False) -> tuple[int, str, str]:
    """Run opencode inside the rollout container; return (rc, stdout, stderr).

    The container runs in a fresh process group so we can SIGKILL the whole
    docker subtree on timeout. Default removes the container after exit;
    --keep-containers preserves them for debugging.
    """
    container_name = f"swebench-rollout-{instance_id}-{int(time.time())}"
    rm_flag = [] if keep else ["--rm"]

    # `git config --global` ensures the diff-on-exit path doesn't trip on
    # missing user.email/name inside the container.
    inner = f"""set -e
git config --global user.email eval@local
git config --global user.name eval
git config --global --add safe.directory /testbed
opencode run --dir /testbed --model {model} --format json --dangerously-skip-permissions "$PROMPT"
"""

    cmd = [
        "docker", "run",
        *rm_flag,
        "--name", container_name,
        "--network=host",
        "--env", f"PROMPT={prompt}",
        "--env", "HOME=/root",
        "--workdir", "/testbed",
        image_tag,
        "bash", "-lc", inner,
    ]

    t0 = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        rc = proc.returncode
        elapsed = time.time() - t0
        log_path.write_text(
            f"# command (PROMPT in env)\n{' '.join(cmd[:-3])} bash -lc <inner>\n"
            f"# elapsed {elapsed:.1f}s\n# returncode {rc}\n"
            f"# stdout\n{stdout}\n# stderr\n{stderr}\n"
        )
        return rc, stdout, stderr
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        # Belt-and-suspenders: ask docker to kill the container too. The
        # docker CLI process getting SIGKILLed doesn't necessarily reap the
        # daemon-managed container.
        sh("docker", "kill", container_name, check=False, capture=True)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        log_path.write_text(
            f"# TIMEOUT after {timeout}s (process group + docker kill)\n"
            f"# stdout\n{stdout}\n# stderr\n{stderr}\n"
        )
        return 124, stdout or "", stderr or ""


def capture_diff(image_tag: str, instance_id: str) -> str:
    """Run a fresh container against the same image to capture a `git diff` is
    not viable — the rollout container is gone (--rm) and held the worktree.
    Instead we capture inside the same `inner` shell script in run_in_container
    by appending `git add -A && git diff --cached`. This function exists only
    to keep the v1 schema symmetry; the actual diff capture happens in
    run_in_container's stdout (parsed below)."""
    raise NotImplementedError("diff capture is inlined into run_in_container; see _extract_diff_from_stdout")


def _extract_diff_from_stdout(stdout: str) -> str:
    """The inner shell appends `=== DIFF ===` then `git diff --cached` output.
    Strip everything before that marker."""
    marker = "=== DIFF ==="
    idx = stdout.rfind(marker)
    if idx == -1:
        return ""
    return stdout[idx + len(marker):].lstrip("\n")


# --- preflight + dataset ---------------------------------------------------

def preflight_canary(server_url: str, served_name: str) -> tuple[bool, str]:
    """Mimic opencode's wire format (assistant turn with prior tool_calls,
    arguments as JSON string per OpenAI spec) to catch chat-template bugs
    before burning hours on rollouts. Same shape as run_rollouts.py."""
    payload = {
        "model": served_name,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "1", "type": "function",
                             "function": {"name": "glob",
                                          "arguments": '{"pattern": "**/*.py"}'}}]},
            {"role": "tool", "tool_call_id": "1", "content": "a.py\nb.py"},
            {"role": "user", "content": "continue"},
        ],
        "max_tokens": 30,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{server_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read())
            if "choices" in body and body["choices"]:
                content = body["choices"][0]["message"].get("content") or ""
                return True, f"OK ({len(content)}B content)"
            return False, f"unexpected response: {body!r}"
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read())["message"]
        except Exception:
            err = str(e)
        return False, f"{e.code}: {err}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def load_dataset(dataset_id: str, split: str):
    from datasets import load_dataset as _ld
    return _ld(dataset_id, split=split)


# --- main ------------------------------------------------------------------

def main():
    args = parse_args()

    served = args.served_name or args.model.split("/", 1)[-1]
    print(f"Preflight: canary chat completion against {args.server_url} (model={served})...", flush=True)
    ok, info = preflight_canary(args.server_url, served)
    if not ok:
        print(f"  PREFLIGHT FAILED: {info}", flush=True)
        print(f"  refusing to start rollout — fix the server / chat template first", flush=True)
        return 2
    print(f"  preflight {info}", flush=True)

    out = Path(args.out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    # meta.json declares scaffold + model + run dates so future tooling can
    # group predictions.jsonl files by scaffold without inferring from the
    # parent dirname. Idempotent: appends a new run record on resume.
    meta_path = out / "meta.json"
    meta = {"runs": []}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            pass
    meta.setdefault("runs", []).append({
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scaffold": args.scaffold,
        "model": args.model,
        "served_name": args.served_name or args.model.split("/", 1)[-1],
        "dataset": args.dataset,
        "split": args.split,
        "timeout_sec": args.timeout,
    })
    meta["scaffold"] = args.scaffold
    meta["model"] = args.model
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Scaffold: {args.scaffold}   Model: {args.model}   Out: {out}", flush=True)

    print(f"Loading dataset {args.dataset}/{args.split}...", flush=True)
    ds = load_dataset(args.dataset, args.split)
    print(f"  {len(ds)} instances total", flush=True)

    if args.instance_ids:
        ds = [r for r in ds if r["instance_id"] in args.instance_ids]
        print(f"  filtered to {len(ds)} via --instance-ids", flush=True)
    elif args.instances:
        ds = list(ds)[: args.instances]
        print(f"  truncated to first {len(ds)} via --instances", flush=True)

    predictions_path = out / "predictions.jsonl"
    existing = set()
    if args.skip_existing and predictions_path.exists():
        for line in predictions_path.read_text().splitlines():
            try:
                existing.add(json.loads(line)["instance_id"])
            except Exception:
                pass
        print(f"  resume: {len(existing)} predictions already on disk", flush=True)

    empty_streak = 0
    with predictions_path.open("a") as fp:
        for i, row in enumerate(ds):
            iid = row["instance_id"]
            if iid in existing:
                print(f"[{i+1}/{len(ds)}] {iid}  SKIP (exists)", flush=True)
                continue

            print(f"[{i+1}/{len(ds)}] {iid}  repo={row['repo']}  base={row['base_commit'][:8]}", flush=True)
            t0 = time.time()
            try:
                image_tag = ensure_rollout_image(
                    iid, no_pull=args.no_pull, rebuild=args.rebuild_image,
                )
                prompt = PROMPT_TEMPLATE.format(
                    problem_statement=row["problem_statement"],
                    hints=row.get("hints_text", "") or "(none)",
                )
                scaffold_envs, inner_with_diff = build_scaffold_invocation(
                    args.scaffold, args.model, served,
                )
                container_name = f"swebench-rollout-{iid}-{int(time.time())}"
                cmd = [
                    "docker", "run",
                    *([] if args.keep_containers else ["--rm"]),
                    "--name", container_name,
                    "--network=host",
                    "--env", f"PROMPT={prompt}",
                    "--env", "HOME=/root",
                    *scaffold_envs,
                    "--workdir", "/testbed",
                    image_tag,
                    "bash", "-lc", inner_with_diff,
                ]
                log_path = out / "logs" / f"{iid}.log"
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    errors="replace",
                    start_new_session=True,
                )
                try:
                    stdout, stderr = proc.communicate(timeout=args.timeout)
                    rc = proc.returncode
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    sh("docker", "kill", container_name, check=False, capture=True)
                    try:
                        stdout, stderr = proc.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        stdout, stderr = "", ""
                    rc = 124

                elapsed = round(time.time() - t0, 1)
                log_path.write_text(
                    f"# command (PROMPT in env)\n"
                    f"# elapsed {elapsed}s   rc={rc}\n"
                    f"# stdout\n{stdout}\n# stderr\n{stderr}\n"
                )

                diff = _extract_diff_from_stdout(stdout)
                (out / "predictions" / f"{iid}.diff").write_text(diff)
                entry = {
                    "instance_id": iid,
                    "model_name_or_path": args.model,
                    "model_patch": diff,
                    "rollout_returncode": rc,
                    "rollout_seconds": elapsed,
                    "rollout_scaffold": args.scaffold,
                }
                fp.write(json.dumps(entry) + "\n")
                fp.flush()

                non_empty = "yes" if diff.strip() else "EMPTY"
                print(f"  done rc={rc} elapsed={elapsed}s diff={non_empty} ({len(diff)}B)", flush=True)

                if diff.strip():
                    empty_streak = 0
                else:
                    empty_streak += 1
                    if empty_streak >= args.max_empty_streak:
                        print(f"\nABORT: {empty_streak} consecutive empty diffs — re-run preflight before resuming.", flush=True)
                        return 3
            except Exception as e:
                import traceback
                print(f"  SKIP (instance crashed): {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                fp.write(json.dumps({"instance_id": iid, "model_name_or_path": args.model,
                                     "model_patch": "", "rollout_returncode": -1,
                                     "rollout_error": f"{type(e).__name__}: {e}",
                                     "rollout_seconds": round(time.time() - t0, 1)}) + "\n")
                fp.flush()
                continue

    return 0


if __name__ == "__main__":
    sys.exit(main())
