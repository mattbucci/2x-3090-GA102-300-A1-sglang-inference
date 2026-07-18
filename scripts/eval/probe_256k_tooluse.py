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


# Measured on the actual FILLER_UNIT: qwen3/gemma4 tokenizers all give 6.59
# chars/token (simple common words ≈ 1 token each; the old 3.8 guess under-filled
# every rung to ~58% of its label). Self-calibrated per model from usage after
# each rung, so tekken/other tokenizers converge by rung 2.
CHARS_PER_TOKEN_INIT = 6.6


def build_prompt(approx_tokens: int, depth: float = 0.5,
                 chars_per_token: float = CHARS_PER_TOKEN_INIT) -> str:
    """~approx_tokens of filler with the needle instruction planted at `depth` (0..1
    through the filler) — vary depth to probe lost-in-the-middle tool-calling."""
    target_chars = int(approx_tokens * chars_per_token)
    n = (target_chars // len(FILLER_UNIT)) + 1
    body = (FILLER_UNIT * n)[:target_chars]
    pos = int(len(body) * depth)
    return body[:pos] + NEEDLE + body[pos:] + TASK


def server_context_length(port: int):
    """Read the server's --context-length so deep rungs can be capped instead of 400ing."""
    for ep in ("server_info", "get_server_info"):
        try:
            info = requests.get(f"http://localhost:{port}/{ep}", timeout=10).json()
            ctx = info.get("context_length") or (info.get("model_config") or {}).get("context_len")
            if ctx:
                return int(ctx)
        except Exception:
            continue
    return None


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


FOLLOWUP_SENTINEL = "KIWI77"


def followup_one(url, prompt, assistant_msg, max_tokens=1024, timeout=900):
    """Multi-turn rung: send the model's own tool call back with a synthetic
    RESULT carrying a sentinel fact, and check the final answer uses it.

    Single-turn probes score only the call; if the serving path blanks tool
    responses (template list-content class — the qwen3-ream June mis-verdict),
    calls score 1.0 while agents run blind. This rung closes that blind spot:
    the sentinel is only knowable from the tool response.
    """
    tc = (assistant_msg.get("tool_calls") or [{}])[0]
    clean_assistant = {
        "role": "assistant",
        "content": assistant_msg.get("content") or "",
        "tool_calls": [{
            "id": tc.get("id", "call_0"),
            "type": "function",
            "function": {
                "name": (tc.get("function") or {}).get("name", "lookup_record"),
                "arguments": (tc.get("function") or {}).get("arguments", "{}"),
            },
        }],
    }
    result_text = (f'{{"id": "{NEEDLE_ID}", "status": "ARCHIVED", '
                   f'"access_code": "{FOLLOWUP_SENTINEL}"}}')
    messages = [
        {"role": "user", "content": prompt},
        clean_assistant,
        {"role": "tool", "content": result_text,
         "tool_call_id": clean_assistant["tool_calls"][0]["id"]},
        {"role": "user", "content": "State the record's access_code exactly."},
    ]
    try:
        r = requests.post(url, json={
            "model": "default", "messages": messages, "tools": TOOLS,
            "max_tokens": max_tokens, "temperature": 0,
        }, timeout=timeout).json()
    except Exception as e:
        return {"followup_error": str(e)[:120]}
    if "error" in r:
        return {"followup_error": str(r["error"].get("message", r["error"]))[:120]}
    msg = (r.get("choices") or [{}])[0].get("message") or {}
    text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
    return {"used_tool_response": FOLLOWUP_SENTINEL in text,
            "followup_text": (msg.get("content") or "")[:120]}


def probe_one(url, approx_tokens, max_tokens=2048, timeout=900, depth=0.5,
              chars_per_token=CHARS_PER_TOKEN_INIT, multi_turn=False):
    prompt = build_prompt(approx_tokens, depth, chars_per_token)
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
    if "error" in r:  # e.g. prompt overflowed the server window — caller may retry smaller
        return {"approx_tokens": approx_tokens,
                "error": str(r["error"].get("message", r["error"]))[:120]}
    choice = (r.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    finish = choice.get("finish_reason")
    prompt_tokens = (r.get("usage") or {}).get("prompt_tokens")
    valid, args = extract_toolcall(msg)
    correct = bool(args) and str(args.get("id", "")).strip() == NEEDLE_ID
    res = {
        "approx_tokens": approx_tokens,
        "actual_prompt_tokens": prompt_tokens,
        "finish_reason": finish,
        "valid_toolcall": valid,
        "correct_action": correct,
        "got_id": (args or {}).get("id") if args else None,
        "elapsed_s": round(time.time() - t0, 1),
    }
    if multi_turn and valid:
        res.update(followup_one(url, prompt, msg, timeout=timeout))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23334)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--lengths", default="16384,65536,131072,196608,256000",
                    help="comma-separated approx token context lengths")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--depth", type=float, default=0.5,
                    help="needle depth 0..1 through the filler (sweep externally for lost-in-the-middle)")
    ap.add_argument("--multi-turn", action="store_true",
                    help="after each valid call, feed back a synthetic tool RESULT "
                         "with a sentinel and verify the model uses it (closes the "
                         "response-path blind spot)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    url = f"http://localhost:{args.port}/v1/chat/completions"
    lengths = [int(x) for x in args.lengths.split(",")]
    ctx_len = server_context_length(args.port)
    usable = (ctx_len - args.max_tokens - 512) if ctx_len else None
    if usable:
        capped = [L for L in lengths if L > usable]
        lengths = sorted({min(L, usable) for L in lengths})
        if capped:
            print(f"server context_length={ctx_len}: capped {capped} -> {usable}")
    print(f"256K tool-use probe: {args.tag}")
    print(f"{'approx':>8} {'actual':>8} {'finish':>12} {'valid':>6} {'correct':>8} {'id':>10} {'s':>5}")
    results = []
    cpt = CHARS_PER_TOKEN_INIT
    for L in lengths:
        res = probe_one(url, L, max_tokens=args.max_tokens, depth=args.depth,
                        chars_per_token=cpt, multi_turn=args.multi_turn)
        if "error" in res and usable:  # likely window overflow from cpt overshoot
            cpt *= 0.9
            res = probe_one(url, L, max_tokens=args.max_tokens, depth=args.depth,
                            chars_per_token=cpt, multi_turn=args.multi_turn)
        results.append(res)
        if "error" in res:
            print(f"{L:>8} {'—':>8} {'ERROR':>12} {res['error']}")
        else:
            actual = res["actual_prompt_tokens"]
            if actual:  # converge chars/token from the server's own count
                cpt = max(2.0, min(12.0, cpt * L / actual))
                if actual < 0.95 * L:
                    res["depth_shortfall"] = True
                    print(f"WARN: rung {L} landed at {actual} actual (<95%) — "
                          f"recalibrated to {cpt:.2f} chars/token")
            print(f"{L:>8} {str(actual):>8} {str(res['finish_reason']):>12} "
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
    if args.multi_turn:
        ft = [r for r in ok if "used_tool_response" in r]
        summary["tool_response_used_rate"] = (
            round(sum(r["used_tool_response"] for r in ft) / len(ft), 3) if ft else None)
    print(f"\nvalid_toolcall: {summary['valid_rate']}  correct_action: {summary['correct_rate']}  "
          f"max-ctx-correct: {summary['max_ctx_correct']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
