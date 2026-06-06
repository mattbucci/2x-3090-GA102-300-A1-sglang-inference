#!/usr/bin/env python3
"""probe_256k_tooluse.py — does tool-calling stay valid + correct AT long context?

The single-user 256K-agentic goal is the rig's north star, but SWE-bench Lite
agentic context tops out ~128K, so the 256K regime is otherwise unmeasured. This
synthetic probe fills that gap: at a ladder of TRUE token-context lengths it plants
a needle instruction deep in filler, exposes one tool, and asks the model to call it
with the planted argument. Two things are scored per length:

  1. valid_toolcall  — did the model emit a STRUCTURALLY valid tool call
                       (finish_reason=tool_calls + parseable JSON args)? This is the
                       "tool-call garbling at long context" axis (cf. R9700 dense
                       Qwen3.5 emitting malformed calls at long KV).
  2. correct_action  — were the args RIGHT (id pulled from deep context)? The agentic
                       analogue of needle-in-haystack: retrieval-into-action.

Context length is reported as the server's ACTUAL prompt_tokens (read from usage),
not the char estimate, so the ladder is honest. Needs a server launched with the
preset's --tool-call-parser.

Usage:
    python scripts/eval/probe_256k_tooluse.py --port 23334 --tag qwen36 \
        --lengths 16384,65536,131072,196608,256000 \
        --out benchmarks/quality/tooluse256k-qwen36-v0512.json
"""
import argparse
import json
import time
from pathlib import Path

import requests

NEEDLE_ID = "BANANA42"
# A distinctive instruction line dropped into the middle of the filler.
NEEDLE = (f"\n\n>>> CRITICAL INSTRUCTION: when you call the lookup_record tool, you "
          f"MUST pass id=\"{NEEDLE_ID}\". This is the only valid id. <<<\n\n")
FILLER_UNIT = "The archive contains routine maintenance logs and status reports. " * 64

TOOLS = [{
    "type": "function",
    "function": {
        "name": "lookup_record",
        "description": "Look up an archive record by its id.",
        "parameters": {
            "type": "object",
            "properties": {"id": {"type": "string", "description": "the record id to fetch"}},
            "required": ["id"],
        },
    },
}]

TASK = ("\n\nNow use the lookup_record tool to fetch the record. Use exactly the id "
        "specified in the CRITICAL INSTRUCTION above. Call the tool — do not answer in prose.")


def build_prompt(approx_tokens: int) -> str:
    """~approx_tokens of filler with the needle planted at the midpoint."""
    target_chars = int(approx_tokens * 3.8)  # rough chars/token for this filler
    half = target_chars // 2
    n = (half // len(FILLER_UNIT)) + 1
    block = (FILLER_UNIT * n)[:half]
    return block + NEEDLE + block + TASK


def extract_toolcall(msg: dict):
    """Return (valid, parsed_args_or_None). Handles the structured tool_calls field."""
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return False, None
    fn = (tcs[0] or {}).get("function") or {}
    raw = fn.get("arguments")
    if raw is None:
        return False, None
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return True, None  # emitted a call but args were malformed JSON → garbled
    return True, args


def probe_one(url, approx_tokens, max_tokens=2048, timeout=900):
    prompt = build_prompt(approx_tokens)
    t0 = time.time()
    try:
        r = requests.post(url, json={
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "tools": TOOLS,
            "tool_choice": "auto",
            "max_tokens": max_tokens,
            "temperature": 0,
        }, timeout=max(timeout, approx_tokens // 150)).json()
    except Exception as e:
        return {"approx_tokens": approx_tokens, "error": str(e)[:120]}
    choice = (r.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    finish = choice.get("finish_reason")
    prompt_tokens = (r.get("usage") or {}).get("prompt_tokens")
    valid, args = extract_toolcall(msg)
    correct = bool(args) and str(args.get("id", "")).strip() == NEEDLE_ID
    return {
        "approx_tokens": approx_tokens,
        "actual_prompt_tokens": prompt_tokens,
        "finish_reason": finish,
        "valid_toolcall": valid,
        "correct_action": correct,
        "got_id": (args or {}).get("id") if args else None,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--lengths", default="16384,65536,131072,196608,256000",
                    help="comma-separated approx token context lengths")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    url = f"http://localhost:{args.port}/v1/chat/completions"
    lengths = [int(x) for x in args.lengths.split(",")]
    print(f"256K tool-use probe: {args.tag}")
    print(f"{'approx':>8} {'actual':>8} {'finish':>12} {'valid':>6} {'correct':>8} {'id':>10} {'s':>5}")
    results = []
    for L in lengths:
        res = probe_one(url, L, max_tokens=args.max_tokens)
        results.append(res)
        if "error" in res:
            print(f"{L:>8} {'—':>8} {'ERROR':>12} {res['error']}")
        else:
            print(f"{L:>8} {str(res['actual_prompt_tokens']):>8} {str(res['finish_reason']):>12} "
                  f"{str(res['valid_toolcall']):>6} {str(res['correct_action']):>8} "
                  f"{str(res['got_id']):>10} {res['elapsed_s']:>5}")

    ok = [r for r in results if "error" not in r]
    summary = {
        "tag": args.tag,
        "results": results,
        "valid_rate": round(sum(r["valid_toolcall"] for r in ok) / len(ok), 3) if ok else None,
        "correct_rate": round(sum(r["correct_action"] for r in ok) / len(ok), 3) if ok else None,
        "max_ctx_correct": max([r["actual_prompt_tokens"] for r in ok if r["correct_action"]], default=0),
    }
    print(f"\nvalid_toolcall: {summary['valid_rate']}  correct_action: {summary['correct_rate']}  "
          f"max-ctx-correct: {summary['max_ctx_correct']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
