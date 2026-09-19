"""Probe the Responses API (`/v1/responses`) with a function tool.

dcode (deepagents 0.7.x) forces the Responses API for every `openai:*` model,
so it is the only scaffold that touches this endpoint — and v0.5.20 added
`OpenAIServingResponses._validate_model`, which 404s any request whose
`model` is not the served model name ("The model 'X' does not exist").
Chat-completion probes cannot see that class (they never validate `model`),
and a capture-endpoint wire audit cannot either (the failure is server-side).
R9700 lost a whole dcode lane to it (every instance dead in 10 s).

What this checks, in order:
  1. `/v1/models` lists exactly one id; the probe uses it unless --model is
     given (dcode takes the id from /v1/models the same way).
  2. a function-tool request through /v1/responses returns
     status == "completed" and at least one `function_call` output item whose
     `arguments` parse as JSON — reasoning items are allowed in front of it.
  3. a request with a deliberately wrong `model` is rejected with 404
     (proves the validator is live, so a served-name drift will be loud).

Usage:
  python scripts/eval/probe_responses_api.py [--port 23334] [--model ID]
                                            [--api-key K] [--json OUT]
Exit 0 = pass, 1 = fail (any step), 2 = server unreachable.
"""
import argparse
import json
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]
PROMPT = "What is the weather in Paris right now? Use the get_weather tool."


def _req(url, payload=None, api_key=None, timeout=300):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "null")
    except HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            body = json.loads(body)
        except ValueError:
            pass
        return e.code, body


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=23334)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--model", default=None, help="served id; default = the /v1/models entry")
    p.add_argument("--api-key", default=None)
    p.add_argument("--json", default=None, help="write the receipt here")
    args = p.parse_args()
    base = f"http://{args.host}:{args.port}"
    receipt = {"base": base, "steps": {}}

    try:
        st, models = _req(f"{base}/v1/models", api_key=args.api_key, timeout=30)
    except URLError as e:
        print(f"FAIL: server unreachable: {e}")
        return 2
    ids = [m.get("id") for m in (models or {}).get("data", [])] if st == 200 else []
    receipt["steps"]["models"] = {"status": st, "ids": ids}
    if st != 200 or not ids:
        print(f"FAIL: /v1/models status={st} ids={ids}")
        return 1
    model = args.model or ids[0]
    print(f"served ids: {ids} -> probing model={model!r}")

    payload = {
        "model": model,
        "input": [{"role": "user", "content": PROMPT}],
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_output_tokens": 4096,
    }
    t0 = time.time()
    st, resp = _req(f"{base}/v1/responses", payload, args.api_key)
    dt = time.time() - t0
    out = (resp or {}).get("output", []) if isinstance(resp, dict) else []
    kinds = [o.get("type") for o in out]
    calls = [o for o in out if o.get("type") == "function_call"]
    args_ok = False
    if calls:
        try:
            a = json.loads(calls[0].get("arguments") or "{}")
            args_ok = isinstance(a, dict) and "city" in a
        except ValueError:
            args_ok = False
    status = (resp or {}).get("status") if isinstance(resp, dict) else None
    receipt["steps"]["function_call"] = {
        "http": st, "status": status, "output_types": kinds,
        "call": {k: calls[0].get(k) for k in ("name", "arguments")} if calls else None,
        "arguments_ok": args_ok, "seconds": round(dt, 1),
        "error": None if st == 200 else resp,
    }
    ok_call = st == 200 and status == "completed" and bool(calls) and args_ok \
        and calls[0].get("name") == "get_weather"
    print(f"function_call: http={st} status={status} output={kinds} "
          f"call={receipt['steps']['function_call']['call']} ({dt:.1f}s) -> "
          f"{'ok' if ok_call else 'FAIL'}")
    if st != 200:
        print(f"  error body: {json.dumps(resp)[:400]}")

    bad = dict(payload, model=model + "-does-not-exist", max_output_tokens=16)
    st2, resp2 = _req(f"{base}/v1/responses", bad, args.api_key, timeout=60)
    ok_404 = st2 == 404
    receipt["steps"]["wrong_model"] = {"http": st2, "body": resp2 if not ok_404 else "404 as expected"}
    print(f"wrong model: http={st2} -> {'ok (validator live)' if ok_404 else 'FAIL (expected 404)'}")

    receipt["ok"] = ok_call and ok_404
    if args.json:
        with open(args.json, "w") as f:
            json.dump(receipt, f, indent=1)
    print("PASS" if receipt["ok"] else "FAIL")
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
