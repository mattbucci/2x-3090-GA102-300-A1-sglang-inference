#!/usr/bin/env python3
"""Scaffold request audit: prove what each scaffold actually sends the server.

Runs every scaffold's real rollout invocation (built by
docker_rollout.build_scaffold_invocation, so the audited command IS the
rolled command) inside a throwaway rollout container whose 127.0.0.1:23334
is a capture endpoint instead of SGLang: no --network=host, no GPU, no
server traffic. The first generation request per scaffold is checked against
the harness policy:

  * thinking at the model's MAXIMUM tier — `reasoning_effort` absent (the
    served template's default applies) or in --allow-effort; never
    `enable_thinking: false` / `thinking.type: disabled`;
  * output budget — max_tokens / max_completion_tokens / max_output_tokens
    absent or >= docker_rollout.OUTPUT_CAP_FLOOR (32000: pi 0.68 / prime
    clamp OUTPUT_BUDGET 32768 to 32000 on the wire);
  * `model` == the served name.

Why: scaffold defaults are a harness input (commit 9c31fff for context
budgets; this is the thinking/output-budget counterpart). pi's default
`reasoning_effort: medium` ran the qwen38 little-coder lanes at half budget,
opencode's packaged `limit.output: 8192` truncated 38/53 xhigh traces into
empty patches. Nothing in the rollout logs shows either.

Usage (run_model_cycle.sh runs it before the lanes; non-zero exit blocks):
  scaffold_request_audit.py --served-name qwen38 --scaffolds "opencode little-coder"
      [--image swebench-rollout/<iid>:latest | --instance-id <iid>]
      [--context-window 262144] [--receipt out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import docker_rollout as dr  # noqa: E402

CAPTURE_SERVER = r'''
import json, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
SERVED, CTX = sys.argv[1], int(sys.argv[2])
LOG = open("/cap/requests.jsonl", "a")
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
        raw = self.rfile.read(n)
        try: body = json.loads(raw)
        except Exception: body = {"_raw": raw[:500].decode(errors="replace")}
        slim = {}
        for k, v in body.items():
            if k == "messages": slim["messages_n"] = len(v)
            elif k == "input": slim["input_n"] = len(v) if isinstance(v, list) else 1
            elif k == "tools": slim["tools_n"] = len(v)
            else: slim[k] = v
        LOG.write(json.dumps({"scaffold": open("/cap/current").read().strip(), "path": self.path,
                              "user_agent": self.headers.get("User-Agent"), "body": slim}) + "\n"); LOG.flush()
        text = "I cannot make changes; the task is complete as-is."
        if self.path.endswith("/chat/completions"):
            if body.get("stream"):
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                def chunk(delta, fin=None, usage=None):
                    d = {"id": "cap", "object": "chat.completion.chunk", "created": int(time.time()), "model": body.get("model"),
                         "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}
                    if usage: d["usage"] = usage
                    self.wfile.write(("data: " + json.dumps(d) + "\n\n").encode())
                chunk({"role": "assistant", "content": text})
                chunk({}, "stop", {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22})
                self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
            else:
                self._json(200, {"id": "cap", "object": "chat.completion", "created": int(time.time()), "model": body.get("model"),
                                 "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22}})
        else:
            self._json(200, {})
HTTPServer(("127.0.0.1", 23334), H).serve_forever()
'''

GEN_PATHS = ("/chat/completions", "/responses", "/completions")
PROBE_PROMPT = "Reply with the single word OK and stop."


def first_run_only(inner: str) -> str:
    """Keep the scaffold's first invocation (up to the `"$PROMPT"` line); the
    cleanup run and diff capture are irrelevant for the audit."""
    lines = inner.splitlines()
    for i, ln in enumerate(lines):
        if '"$PROMPT"' in ln:
            return "\n".join(lines[: i + 1]) + "\n"
    return inner


def build_driver(scaffolds: list[str], served: str, ctx: int, per_timeout: int) -> str:
    parts = ["set +e", "mkdir -p /cap", f"python3 /cap/capsrv.py {shlex.quote(served)} {ctx} &", "sleep 1"]
    for sc in scaffolds:
        envs, inner = dr.build_scaffold_invocation(sc, f"sglang/{served}", served,
                                                  timeout=per_timeout, context_window=ctx)
        kv = [envs[i + 1] for i in range(0, len(envs), 2) if envs[i] == "--env"]
        parts.append(f"echo {sc} > /cap/current")
        parts.append("export HOME=/root")
        parts += [f"export {shlex.quote(e)}" for e in kv]
        parts.append(dr.ACTIVATE_TESTBED.rstrip("\n"))
        parts.append(f"export PROMPT={shlex.quote(PROBE_PROMPT)}")
        parts.append(f'echo "### {sc} start $(date +%T)" >&2')
        parts.append(f"timeout {per_timeout} bash -c {shlex.quote(first_run_only(inner))} "
                     f"> /cap/{sc}.out 2> /cap/{sc}.err")
        parts.append(f'echo "### {sc} rc=$? $(date +%T)" >&2')
    return "\n".join(parts) + "\n"


def check_body(sc: str, path: str, body: dict, served: str, allow_effort: set[str]) -> list[str]:
    bad = []
    if body.get("model") != served:
        bad.append(f"model={body.get('model')!r} != served {served!r}")
    eff = body.get("reasoning_effort")
    if eff is None and isinstance(body.get("reasoning"), dict):
        eff = body["reasoning"].get("effort")
    if eff is not None and eff not in allow_effort:
        bad.append(f"reasoning_effort={eff!r} (allowed: absent or {sorted(allow_effort)})")
    ctk = body.get("chat_template_kwargs") or {}
    for src, d in (("top-level", body), ("chat_template_kwargs", ctk)):
        if d.get("enable_thinking") is False:
            bad.append(f"enable_thinking=false ({src})")
    th = body.get("thinking")
    if isinstance(th, dict) and th.get("type") == "disabled":
        bad.append("thinking.type=disabled")
    for k in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        v = body.get(k)
        if v is not None and int(v) < dr.OUTPUT_CAP_FLOOR:
            bad.append(f"{k}={v} < OUTPUT_CAP_FLOOR {dr.OUTPUT_CAP_FLOOR}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--scaffolds", default="opencode opencode-dcp little-coder little-coder-rtk prime dcode")
    ap.add_argument("--image", help="rollout image to run in (default: any swebench-rollout/* present, else build --instance-id)")
    ap.add_argument("--instance-id", default="sympy__sympy-20590",
                    help="instance whose rollout image to build when none exists (any instance works; only the scaffold installs matter)")
    ap.add_argument("--context-window", type=int, default=262144)
    ap.add_argument("--allow-effort", default="xhigh", help="comma list of reasoning_effort values that count as max (absent is always fine)")
    ap.add_argument("--per-scaffold-timeout", type=int, default=180)
    ap.add_argument("--receipt", help="write the audited bodies + verdicts as JSON here")
    ap.add_argument("--workdir", help="host dir mounted at /cap (default: a temp dir)")
    args = ap.parse_args()

    scaffolds = args.scaffolds.split()
    allow = {e for e in args.allow_effort.split(",") if e}
    image = args.image
    if not image:
        rc, out, _ = dr.sh("docker", "images", "--format", "{{.Repository}}:{{.Tag}}", check=False, capture=True)
        cands = [l for l in (out or "").splitlines() if l.startswith("swebench-rollout/")]
        image = cands[0] if cands else dr.ensure_rollout_image(args.instance_id)
    # the janitor deletes swebench-rollout/* once their prediction lands; pin our copy
    pinned = "swebench-rollout-audit:latest"
    dr.sh("docker", "tag", image, pinned, check=True)

    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="scaffold-audit-"))
    work.mkdir(parents=True, exist_ok=True)
    (work / "capsrv.py").write_text(CAPTURE_SERVER)
    (work / "run_all.sh").write_text(build_driver(scaffolds, args.served_name, args.context_window,
                                                 args.per_scaffold_timeout))
    (work / "requests.jsonl").write_text("")
    (work / "current").write_text("")
    print(f"[audit] image={image} (pinned {pinned}) work={work} scaffolds={scaffolds}", flush=True)
    t0 = time.time()
    cmd = ["docker", "run", "--rm", "--name", f"scaffold-audit-{int(t0)}",
           "-v", f"{work}:/cap", "--env", "HOME=/root", "--workdir", "/testbed",
           pinned, "bash", "-lc", "bash /cap/run_all.sh"]
    proc = subprocess.run(cmd, text=True, capture_output=True,
                          timeout=args.per_scaffold_timeout * len(scaffolds) + 120)
    (work / "driver.err").write_text(proc.stderr or "")
    print(f"[audit] container rc={proc.returncode} in {time.time() - t0:.0f}s", flush=True)

    recs = [json.loads(l) for l in (work / "requests.jsonl").read_text().splitlines() if l.strip()]
    first: dict[str, dict] = {}
    for r in recs:
        if any(r["path"].endswith(p) for p in GEN_PATHS) and r["scaffold"] not in first:
            first[r["scaffold"]] = r
    budget_lines: dict[str, list[str]] = {}
    for sc in scaffolds:
        err = (work / f"{sc}.err").read_text(errors="replace") if (work / f"{sc}.err").exists() else ""
        budget_lines[sc] = [l for l in err.splitlines() if "context budget:" in l or "TRIPWIRE" in l]

    verdicts = {}
    ok = True
    print(f"{'scaffold':18} {'path':22} {'effort':8} {'cap':>6}  verdict")
    for sc in scaffolds:
        r = first.get(sc)
        if not r:
            ok = False
            verdicts[sc] = {"status": "NO_REQUEST", "problems": ["no generation request captured"]}
            print(f"{sc:18} {'-':22} {'-':8} {'-':>6}  FAIL no generation request captured (see {work}/{sc}.err)")
            continue
        b = r["body"]
        probs = check_body(sc, r["path"], b, args.served_name, allow)
        eff = b.get("reasoning_effort")
        if eff is None and isinstance(b.get("reasoning"), dict):
            eff = b["reasoning"].get("effort")
        cap = next((b[k] for k in ("max_tokens", "max_completion_tokens", "max_output_tokens") if b.get(k) is not None), None)
        status = "OK" if not probs else "FAIL"
        ok &= not probs
        verdicts[sc] = {"status": status, "problems": probs, "path": r["path"], "body": b,
                        "budget_lines": budget_lines[sc]}
        print(f"{sc:18} {r['path']:22} {str(eff or '-'):8} {str(cap or '-'):>6}  {status} {'; '.join(probs)}")
    if args.receipt:
        Path(args.receipt).write_text(json.dumps({
            "date": time.strftime("%Y-%m-%d %H:%M"), "served_name": args.served_name, "image": image,
            "context_window": args.context_window, "output_budget": dr.OUTPUT_BUDGET,
            "output_cap_floor": dr.OUTPUT_CAP_FLOOR,
            "allow_effort": sorted(allow), "verdicts": verdicts,
        }, indent=2) + "\n")
    print(f"[audit] {'PASS' if ok else 'FAIL'}: {sum(v['status']=='OK' for v in verdicts.values())}/{len(scaffolds)} scaffolds at policy")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
