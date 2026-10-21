#!/usr/bin/env python3
"""Gate for opencode's subagent permission hang — GPU-free, runs in the real
rollout image against a scripted model.

opencode 1.14.25 `run --dangerously-skip-permissions` answers permission asks
for the MAIN session only (cli/cmd/run.ts skips events whose sessionID is not
its own). A `task` subagent whose bash/read/edit touches a path outside
/testbed hits the agent default `external_directory: ask`; nobody answers, the
tool part stays `running` until the rollout wall (rc 124, idle server — 2 of
11 walls in the qwen38 opencode v4 cell). docker_rollout.OPENCODE_PERMISSION
is the fix (a global `permission` block, merged after every agent's defaults).

This probe reproduces the mechanism end to end without a model: a stdlib
HTTP server on 127.0.0.1:23334 plays the model —

  main session  request 1 → tool call `task` (general subagent, prompt carries
                            SUBAGENT_MARKER)
  subagent      request 1 → tool call `bash` PROBE_CMD (`cd /tmp && ls`: `cd`
                            is in bash.ts FILES, /tmp is outside /testbed →
                            external_directory ask)
  subagent      request 2 → arrives ONLY if the ask was allowed → "SUBAGENT-DONE"
  main session  request 2 → task result → "MAIN-DONE"

and the scaffold command is the rolled one (docker_rollout.build_scaffold_
invocation, first run only — so the per-run opencode.json snippet that writes
the permission block is exercised, not a hand-made config). PASS = opencode
exits inside --timeout with both DONE phases observed. `--mode control`
strips the block before `opencode run` and must FAIL (rc 124, subagent stuck
after its bash call) — proves the probe discriminates; `--mode both` runs
control then fix and passes only if control hangs and fix completes.

Usage:
  probe_opencode_subagent_permission.py --served-name qwen38 [--scaffold opencode|opencode-dcp]
      [--image swebench-rollout/<iid>:latest] [--mode fix|control|both] [--timeout 120]
      [--receipt out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import docker_rollout as dr  # noqa: E402
from scaffold_request_audit import first_run_only  # noqa: E402

SUBAGENT_MARKER = "PERMPROBE-SUBAGENT"
PROBE_CMD = "cd /tmp && ls"
MAIN_PROMPT = "PERMPROBE-MAIN: this is a harness probe; follow the tool calls."

MODEL_SERVER = r'''
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
SERVED, CTX = sys.argv[1], int(sys.argv[2])
MARKER, CMD = sys.argv[3], sys.argv[4]
LOG = open("/cap/requests.jsonl", "a")
SUB_PROMPT = MARKER + ": run the bash command `" + CMD + "` and report its output verbatim."

def text_of(c):
    if isinstance(c, list):
        return "".join(x.get("text") or "" for x in c if isinstance(x, dict))
    return c if isinstance(c, str) else ""

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.endswith("/models"):
            return self._json(200, {"object": "list", "data": [{"id": SERVED, "object": "model", "max_model_len": CTX}]})
        self._json(200, {})
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try: body = json.loads(self.rfile.read(n))
        except Exception: body = {}
        msgs = body.get("messages") or []
        tools = [t.get("function", {}).get("name") for t in body.get("tools") or [] if isinstance(t, dict)]
        users = [text_of(m.get("content")) for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
        tool_results = [text_of(m.get("content")) for m in msgs if isinstance(m, dict) and m.get("role") == "tool"]
        subagent = any(MARKER in u for u in users)
        phase, reply, call = "", None, None
        if not tools:
            phase, reply = "no-tools", "probe"            # title / summary requests
        elif subagent:
            if tool_results:
                phase, reply = "subagent-done", "SUBAGENT-DONE"
            else:
                phase, call = "subagent-bash", ("bash", {"command": CMD, "description": "permission probe"})
        else:
            if tool_results:
                phase, reply = "main-done", "MAIN-DONE"
            elif "task" in tools:
                phase, call = "main-task", ("task", {"description": "permission probe", "prompt": SUB_PROMPT,
                                                     "subagent_type": "general"})
            else:
                phase, call = "main-bash-no-task-tool", ("bash", {"command": CMD, "description": "permission probe"})
        LOG.write(json.dumps({"t": round(time.time(), 2), "path": self.path, "phase": phase, "tools_n": len(tools),
                              "has_task_tool": "task" in tools, "tool_results": [r[:400] for r in tool_results],
                              "stream": bool(body.get("stream"))}) + "\n"); LOG.flush()
        if not self.path.endswith("/chat/completions"):
            return self._json(200, {})
        usage = {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22}
        if call:
            tc = {"index": 0, "id": "call_" + phase.replace("-", "_"), "type": "function",
                  "function": {"name": call[0], "arguments": json.dumps(call[1])}}
        if body.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            def chunk(delta, fin=None, u=None):
                d = {"id": "probe", "object": "chat.completion.chunk", "created": int(time.time()), "model": body.get("model"),
                     "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}
                if u: d["usage"] = u
                self.wfile.write(("data: " + json.dumps(d) + "\n\n").encode())
            if call:
                chunk({"role": "assistant", "content": None, "tool_calls": [tc]})
                chunk({}, "tool_calls", usage)
            else:
                chunk({"role": "assistant", "content": reply})
                chunk({}, "stop", usage)
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        else:
            msg = {"role": "assistant", "content": None,
                   "tool_calls": [{k: v for k, v in tc.items() if k != "index"}]} if call else \
                  {"role": "assistant", "content": reply}
            self._json(200, {"id": "probe", "object": "chat.completion", "created": int(time.time()), "model": body.get("model"),
                             "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if call else "stop"}],
                             "usage": usage})
HTTPServer(("127.0.0.1", 23334), H).serve_forever()
'''

CONTROL_STRIP = (
    "node -e 'const fs=require(\"fs\");const p=process.env.HOME+\"/.config/opencode/opencode.json\";"
    "const d=JSON.parse(fs.readFileSync(p,\"utf8\"));delete d.permission;"
    "fs.writeFileSync(p,JSON.stringify(d,null,2));"
    "console.error(\"permission: opencode.json CONTROL (block removed)\")'\n"
)


def build_driver(scaffold: str, served: str, ctx: int, timeout: int, control: bool, pre_run: str = "") -> str:
    envs, inner = dr.build_scaffold_invocation(scaffold, f"sglang/{served}", served,
                                              timeout=timeout, context_window=ctx)
    kv = [envs[i + 1] for i in range(0, len(envs), 2) if envs[i] == "--env"]
    run = first_run_only(inner)
    inject = (CONTROL_STRIP if control else "") + (pre_run.rstrip("\n") + "\n" if pre_run else "")
    if inject:
        lines = run.splitlines(keepends=True)
        i = next(k for k, ln in enumerate(lines) if "opencode run" in ln)
        run = "".join(lines[:i]) + inject + "".join(lines[i:])
    parts = ["set +e", "mkdir -p /cap",
             f"python3 /cap/permsrv.py {shlex.quote(served)} {ctx} {shlex.quote(SUBAGENT_MARKER)} {shlex.quote(PROBE_CMD)} &",
             "sleep 1", "export HOME=/root"]
    parts += [f"export {shlex.quote(e)}" for e in kv]
    parts.append(dr.ACTIVATE_TESTBED.rstrip("\n"))
    parts.append(f"mkdir -p {os.path.dirname(dr.PROMPT_FILE)} && printf '%s' {shlex.quote(MAIN_PROMPT)} > {dr.PROMPT_FILE}")
    parts.append('echo "### start $(date +%T)" >&2')
    parts.append(f"timeout {timeout} bash -c {shlex.quote(run)} > /cap/opencode.out 2> /cap/opencode.err")
    parts.append('rc=$?; echo "$rc" > /cap/rc; echo "### rc=$rc $(date +%T)" >&2')
    # opencode's own log is the direct evidence (service=permission evaluate /
    # evaluated / asking lines); session snapshots never carry log/.
    parts.append('for f in "$HOME"/.local/share/opencode/log/*.log; do [ -f "$f" ] && grep -h "service=permission" "$f"; done > /cap/permlog.txt 2>/dev/null')
    # Plugin lines: on the opencode-dcp scaffold the run must show the external
    # plugin loading (service=plugin "loading plugin" path=@tarquinen/...), or a
    # fast boot just means opencode skipped it. "background dependency install
    # failed" is opencode's Config forking `npm install @opencode-ai/plugin`
    # into the config dir at every start (plugin loading awaits it; offline
    # that is 70 s per stale packument - the DCP lane's boot tax).
    parts.append('for f in "$HOME"/.local/share/opencode/log/*.log; do [ -f "$f" ] && grep -h "service=plugin\\|dependency install\\|service=npm" "$f"; done > /cap/pluginlog.txt 2>/dev/null')
    parts.append('grep -h "permission: opencode.json\\|context budget:" /cap/opencode.err > /cap/snippet.txt 2>/dev/null')
    return "\n".join(parts) + "\n"


def redact(s: str) -> str:
    k = dr.api_key()
    return s.replace(k, "<api-key>") if k and k != dr.API_KEY_PLACEHOLDER else s


def run_once(image: str, scaffold: str, served: str, ctx: int, timeout: int, control: bool, work: Path,
             pre_run: str = "") -> dict:
    work.mkdir(parents=True, exist_ok=True)
    (work / "permsrv.py").write_text(MODEL_SERVER)
    (work / "run.sh").write_text(build_driver(scaffold, served, ctx, timeout, control, pre_run))
    for f in ("requests.jsonl", "permlog.txt", "pluginlog.txt", "snippet.txt", "rc"):
        (work / f).write_text("")
    mode = "control" if control else "fix"
    print(f"[permprobe:{mode}] image={image} scaffold={scaffold} work={work}", flush=True)
    t0 = time.time()
    proc = subprocess.run(["docker", "run", "--rm", "--network", "none", "--name", f"permprobe-{mode}-{int(t0)}",
                           "-v", f"{work}:/cap", "--env", "HOME=/root", "--workdir", "/testbed",
                           image, "bash", "-lc", "bash /cap/run.sh"],
                          text=True, capture_output=True, timeout=timeout + 120)
    (work / "driver.err").write_text(redact(proc.stderr or ""))
    rc_txt = (work / "rc").read_text().strip()
    rc = int(rc_txt) if rc_txt.isdigit() else proc.returncode
    reqs = [json.loads(l) for l in (work / "requests.jsonl").read_text().splitlines() if l.strip()]
    phases = [r["phase"] for r in reqs]
    first_req = min((r["t"] for r in reqs), default=None)
    sub_result = next((r["tool_results"] for r in reqs if r["phase"] == "subagent-done"), None)
    permlog = redact((work / "permlog.txt").read_text(errors="replace"))
    ext = [l for l in permlog.splitlines() if "external_directory" in l and ("evaluated" in l or "asking" in l)]
    snippet = redact((work / "snippet.txt").read_text(errors="replace")).strip().splitlines()
    plog = redact((work / "pluginlog.txt").read_text(errors="replace")).splitlines()
    ext_plugins = [l for l in plog if "internal plugin" not in l]
    res = {"mode": mode, "scaffold": scaffold, "rc": rc, "seconds": round(time.time() - t0, 1), "phases": phases,
           "boot_seconds": round(first_req - t0, 1) if first_req else None,   # docker start → first model request
           "pre_run": pre_run or None,
           "subagent_bash_result": sub_result, "task_tool_offered": any(r.get("has_task_tool") for r in reqs),
           "external_directory_log": ext[-4:], "snippet_lines": snippet,
           "plugin_log": ext_plugins[-6:],
           "external_plugin_loaded": any("loading plugin" in l for l in ext_plugins),
           "dependency_install_failed": any("dependency install failed" in l for l in ext_plugins),
           "completed": rc == 0 and "subagent-done" in phases and "main-done" in phases,
           "hung_after_subagent_bash": rc == 124 and "subagent-bash" in phases and "subagent-done" not in phases}
    print(f"[permprobe:{mode}] rc={rc} in {res['seconds']}s boot={res['boot_seconds']}s phases={phases}", flush=True)
    for l in snippet: print(f"  {l}")
    for l in ext[-4:]: print(f"  log: {l[:220]}")
    for l in ext_plugins[-6:]: print(f"  plugin: {l[:220]}")
    if sub_result: print(f"  subagent bash result: {sub_result[0][:160]!r}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--scaffold", default="opencode", choices=["opencode", "opencode-dcp"])
    ap.add_argument("--image", help="rollout image (default: any swebench-rollout/* present, else build --instance-id)")
    ap.add_argument("--instance-id", default="sympy__sympy-20590")
    ap.add_argument("--context-window", type=int, default=262144)
    ap.add_argument("--mode", default="fix", choices=["fix", "control", "both"])
    ap.add_argument("--timeout", type=int, default=120, help="wall for the opencode run; control must hit it")
    ap.add_argument("--pre-run", default="", help="shell line(s) injected right before `opencode run` (experiments: "
                    "e.g. rewrite the plugin spec in $HOME/.config/opencode/opencode.json)")
    ap.add_argument("--receipt")
    ap.add_argument("--workdir")
    args = ap.parse_args()

    image = args.image
    if not image:
        rc, out, _ = dr.sh("docker", "images", "--format", "{{.Repository}}:{{.Tag}}", check=False, capture=True)
        cands = [l for l in (out or "").splitlines() if l.startswith("swebench-rollout/")]
        image = cands[0] if cands else dr.ensure_rollout_image(args.instance_id)
    pinned = "swebench-rollout-permprobe:latest"   # the janitor reaps swebench-rollout/* under us
    dr.sh("docker", "tag", image, pinned, check=True)
    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="permprobe-"))

    runs = []
    modes = ["control", "fix"] if args.mode == "both" else [args.mode]
    for m in modes:
        runs.append(run_once(pinned, args.scaffold, args.served_name, args.context_window, args.timeout,
                             m == "control", work / m, args.pre_run))
    by = {r["mode"]: r for r in runs}
    ok = True
    if "fix" in by:
        ok &= by["fix"]["completed"]
        if args.scaffold == "opencode-dcp":
            ok &= by["fix"]["external_plugin_loaded"]   # a fast boot without the plugin is not the DCP lane
    if "control" in by:
        ok &= by["control"]["hung_after_subagent_bash"]
    verdict = "PASS" if ok else "FAIL"
    print(f"[permprobe] {verdict}: " + "; ".join(
        f"{r['mode']} {'completed' if r['completed'] else ('HUNG after subagent bash' if r['hung_after_subagent_bash'] else 'rc=%d phases=%s' % (r['rc'], r['phases']))}"
        + (f" boot={r['boot_seconds']}s plugin={'loaded' if r['external_plugin_loaded'] else 'NOT loaded'}"
           if args.scaffold == "opencode-dcp" else "")
        for r in runs))
    if args.receipt:
        Path(args.receipt).write_text(json.dumps({
            "date": time.strftime("%Y-%m-%d %H:%M"), "served_name": args.served_name, "image": image,
            "opencode_permission": dr.OPENCODE_PERMISSION, "probe_cmd": PROBE_CMD, "timeout": args.timeout,
            "verdict": verdict, "runs": runs}, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
