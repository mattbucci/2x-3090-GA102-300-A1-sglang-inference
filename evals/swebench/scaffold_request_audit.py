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
  * sampling from the preset — no scaffold-pinned `temperature` (little-coder's
    default model profile injected 0.3 and a 2048/4096-token thinking-budget
    abort until the profile pin — docker_rollout.LC_MODEL_PROFILE);
  * `model` == the served name;
  * when the server needs a credential (SERVE_MODE=docker: secure-launch in
    the OCI image), the request carries `Authorization: Bearer <key>` — the
    per-cycle key serve_backend.sh mints (SWEBENCH_API_KEY_FILE).

Why: scaffold defaults are a harness input (commit 9c31fff for context
budgets; this is the thinking/output-budget counterpart). pi's default
`reasoning_effort: medium` ran the qwen38 little-coder lanes at half budget,
opencode's packaged `limit.output: 8192` truncated 38 of 293 xhigh sessions, 35
of them into empty patches. Nothing in the rollout logs shows either.

Usage (run_model_cycle.sh runs it before the lanes; non-zero exit blocks):
  scaffold_request_audit.py --served-name qwen38 --scaffolds "opencode little-coder"
      [--image swebench-rollout/<iid>:latest | --instance-id <iid>]
      [--context-window 262144] [--receipt out.json]
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

CAPTURE_SERVER = r'''
import hashlib, json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
SERVED, CTX = sys.argv[1], int(sys.argv[2])
EXPECT = os.environ.get("SWEBENCH_API_KEY_EXPECT", "")
LOG = open("/cap/requests.jsonl", "a")
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _auth(self):
        h = self.headers.get("Authorization") or ""
        if not h: return "missing"
        return "ok" if h == "Bearer " + EXPECT else "mismatch"
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
        # first user message, verbatim: proves how the task prompt reached the
        # model (argv re-quoting / trimming / scaffold wrapping are all visible)
        msgs = body.get("messages") if isinstance(body.get("messages"), list) else body.get("input")
        if isinstance(msgs, str): msgs = [{"role": "user", "content": msgs}]
        for m in msgs or []:
            if not (isinstance(m, dict) and m.get("role") == "user"): continue
            c = m.get("content")
            if isinstance(c, list):
                c = "".join(x.get("text") or x.get("input_text") or "" for x in c if isinstance(x, dict))
            if isinstance(c, str):
                slim["user0"] = {"len": len(c), "sha256": hashlib.sha256(c.encode()).hexdigest()[:16],
                                 "text": c if len(c) <= 4000 else c[:2000] + "\n...\n" + c[-2000:]}
            break
        LOG.write(json.dumps({"scaffold": open("/cap/current").read().strip(), "path": self.path,
                              "user_agent": self.headers.get("User-Agent"), "auth": self._auth(),
                              "body": slim}) + "\n"); LOG.flush()
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
PROBE_PROMPT = "Reply with the single word OK and stop. Probe text: run `python -c \"print(1)\"` — it has \"quotes\" and spaces."


def first_run_only(inner: str) -> str:
    """Keep the scaffold's first invocation (up to the line that reads the
    task from PROMPT_FILE); the cleanup run and diff capture are irrelevant
    for the audit."""
    lines = inner.splitlines()
    for i, ln in enumerate(lines):
        if dr.PROMPT_FILE in ln or '"$PROMPT"' in ln:
            return "\n".join(lines[: i + 1]) + "\n"
    return inner


PROMPT_MARKER = "single word OK"
DCP_ID_RE = re.compile(r"\s*<dcp-message-id>[^<]*</dcp-message-id>\s*$")


def check_prompt(sc: str, gens: list[dict]) -> tuple[str, list[str]]:
    """How the task prompt arrived at the model: the first user message of the
    request that carries it must be PROBE_PROMPT verbatim (whitespace-trimmed
    is fine: pi and dcode strip). Catches argv re-quoting (opencode `run`
    wrapped a positional message in `"…"` with inner quotes escaped — every
    opencode cell before 2026-09-19), scaffold wrapping and loss."""
    cand = [r for r in gens if PROMPT_MARKER in ((r["body"].get("user0") or {}).get("text") or "")]
    if not cand:
        return "MISSING", ["task prompt not found in any user message of a generation request"]
    t = cand[0]["body"]["user0"]["text"]
    if sc == "opencode-dcp":  # the DCP plugin tags every user turn; that is the lane under test
        t = DCP_ID_RE.sub("", t)
    if t == PROBE_PROMPT:
        return "verbatim", []
    if t.strip() == PROBE_PROMPT.strip():
        return "verbatim*", []   # * = whitespace-trimmed by the scaffold
    if t.strip().startswith('"') and t.strip().rstrip().endswith('"'):
        return "RE-QUOTED", [f"prompt re-quoted by the scaffold (argv delivery): {t[:70]!r}"]
    if PROBE_PROMPT in t:
        return "WRAPPED", [f"prompt wrapped by the scaffold (+{len(t) - len(PROBE_PROMPT)} chars): {t[:70]!r}"]
    return "ALTERED", [f"prompt altered: {t[:120]!r}"]


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
        parts.append(f"mkdir -p {os.path.dirname(dr.PROMPT_FILE)} && printf '%s' {shlex.quote(PROBE_PROMPT)} > {dr.PROMPT_FILE}")
        parts.append(f'echo "### {sc} start $(date +%T)" >&2')
        parts.append(f"timeout {per_timeout} bash -c {shlex.quote(first_run_only(inner))} "
                     f"> /cap/{sc}.out 2> /cap/{sc}.err &")
        # argv sweep while the scaffold runs: the task must not be on any
        # command line the agent's own shell could `pkill -f` (R9700 2026-09-19)
        parts.append(f"pid=$!; : > /cap/{sc}.ps; while kill -0 $pid 2>/dev/null; do ps -eo args >> /cap/{sc}.ps; sleep 0.3; done; wait $pid")
        parts.append(f'echo "### {sc} rc=$? $(date +%T)" >&2')
    return "\n".join(parts) + "\n"


def check_body(sc: str, path: str, body: dict, served: str, allow_effort: set[str],
               auth: str = "missing") -> list[str]:
    bad = []
    if dr.api_auth_enabled() and auth != "ok":
        bad.append(f"auth={auth} (server key configured; the image's secure-launch would 401)")
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
    # Sampling comes from the preset (--sampling-defaults model): no scaffold
    # may pin a temperature (little-coder's default model profile injected 0.3
    # until the profile pin, 2026-09-13). opencode's `top_p: 1` is tolerated
    # (top_k still comes from generation_config) and shows in the body column.
    if body.get("temperature") is not None:
        bad.append(f"temperature={body['temperature']} (scaffold-pinned sampling)")
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
           "--env", f"SWEBENCH_API_KEY_EXPECT={dr.api_key()}",
           pinned, "bash", "-lc", "bash /cap/run_all.sh"]
    proc = subprocess.run(cmd, text=True, capture_output=True,
                          timeout=args.per_scaffold_timeout * len(scaffolds) + 120)
    (work / "driver.err").write_text(proc.stderr or "")
    print(f"[audit] container rc={proc.returncode} in {time.time() - t0:.0f}s", flush=True)

    recs = [json.loads(l) for l in (work / "requests.jsonl").read_text().splitlines() if l.strip()]
    gens: dict[str, list[dict]] = {}
    for r in recs:
        if any(r["path"].endswith(p) for p in GEN_PATHS):
            gens.setdefault(r["scaffold"], []).append(r)
    # opencode's first generation call is its title request; policy is
    # checked on EVERY generation request, the prompt on the one carrying it
    first = {sc: rs[0] for sc, rs in gens.items()}
    budget_lines: dict[str, list[str]] = {}
    for sc in scaffolds:
        err = (work / f"{sc}.err").read_text(errors="replace") if (work / f"{sc}.err").exists() else ""
        budget_lines[sc] = [l for l in err.splitlines() if "context budget:" in l or "TRIPWIRE" in l]

    verdicts = {}
    ok = True
    print(f"{'scaffold':18} {'path':22} {'effort':8} {'cap':>6} {'auth':8} {'prompt':10} {'argv':8} verdict")
    for sc in scaffolds:
        r = first.get(sc)
        if not r:
            ok = False
            verdicts[sc] = {"status": "NO_REQUEST", "problems": ["no generation request captured"]}
            print(f"{sc:18} {'-':22} {'-':8} {'-':>6} {'-':8} {'-':10} {'-':8} FAIL no generation request captured (see {work}/{sc}.err)")
            continue
        b = r["body"]
        probs = []
        for g in gens[sc]:
            for pb in check_body(sc, g["path"], g["body"], args.served_name, allow, g.get("auth", "missing")):
                if pb not in probs:
                    probs.append(pb)
        prompt_how, pprobs = check_prompt(sc, gens[sc])
        probs += pprobs
        ps_dump = (work / f"{sc}.ps").read_text(errors="replace") if (work / f"{sc}.ps").exists() else ""
        argv_how = "clean" if PROMPT_MARKER not in ps_dump else "EXPOSED"
        if argv_how == "EXPOSED":
            probs.append("task prompt visible on a container argv (the agent's `pkill -f` can hit its own scaffold)")
        elif not ps_dump.strip():
            argv_how = "unswept"
        eff = b.get("reasoning_effort")
        if eff is None and isinstance(b.get("reasoning"), dict):
            eff = b["reasoning"].get("effort")
        cap = next((b[k] for k in ("max_tokens", "max_completion_tokens", "max_output_tokens") if b.get(k) is not None), None)
        status = "OK" if not probs else "FAIL"
        ok &= not probs
        verdicts[sc] = {"status": status, "problems": probs, "path": r["path"], "body": b,
                        "auth": r.get("auth"), "prompt": prompt_how, "argv": argv_how, "n_requests": len(gens[sc]),
                        "user0": next((g["body"]["user0"] for g in gens[sc] if PROMPT_MARKER in ((g["body"].get("user0") or {}).get("text") or "")), None),
                        "budget_lines": budget_lines[sc]}
        print(f"{sc:18} {r['path']:22} {str(eff or '-'):8} {str(cap or '-'):>6} {r.get('auth', '-'):8} {prompt_how:10} {argv_how:8} {status} {'; '.join(probs)}")
    if args.receipt:
        Path(args.receipt).write_text(json.dumps({
            "date": time.strftime("%Y-%m-%d %H:%M"), "served_name": args.served_name, "image": image,
            "context_window": args.context_window, "output_budget": dr.OUTPUT_BUDGET,
            "output_cap_floor": dr.OUTPUT_CAP_FLOOR, "api_auth": dr.api_auth_enabled(),
            "allow_effort": sorted(allow), "verdicts": verdicts,
        }, indent=2) + "\n")
    print(f"[audit] {'PASS' if ok else 'FAIL'}: {sum(v['status']=='OK' for v in verdicts.values())}/{len(scaffolds)} scaffolds at policy")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
