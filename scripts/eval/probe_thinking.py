"""Probe a thinking-capable model with skip_special_tokens=False to see the
raw structure of the reasoning channel.

The capability validator (validate_capabilities.py) only checks that
reasoning_content is non-empty and that the final answer contains a few
keywords. That can pass on shallow output. This probe asks a multi-step
arithmetic problem and inspects:

  - the raw `<|channel>thought` markers (rendered as `thought\\n` with
    skip_special_tokens off) confirm the model is actually entering the
    thinking channel rather than emitting reasoning-shaped prose
  - intermediate scratch work (e.g. `22`) demonstrates real chain-of-thought
  - final answer (`11`) confirms the reasoning is correct, not just present
  - reasoning_content / content separation confirms the parser sees the
    channel boundary

Usage:
  python scripts/eval/probe_thinking.py [--port PORT] [--model MODEL]

Defaults: port 30000, model "default".
"""
import argparse
import json
from urllib.request import Request, urlopen


PROMPT = (
    "I have 17 apples. I give 5 to my sister, eat 2, and buy 12 more. "
    "Then I give half of what I have to my brother. How many apples do I "
    "have left? Think step by step before answering."
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--model", default="default")
    args = p.parse_args()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 600,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "skip_special_tokens": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    req = Request(
        f"http://localhost:{args.port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = json.loads(urlopen(req, timeout=240).read())

    choice = r["choices"][0]
    msg = choice["message"]
    rc = msg.get("reasoning_content") or ""
    co = msg.get("content") or ""

    print("=" * 60)
    print(f"finish_reason: {choice.get('finish_reason')}")
    print(f"usage: {r.get('usage')}")
    print()
    print("--- reasoning_content ---")
    print(repr(rc)[:1500])
    print()
    print("--- content ---")
    print(repr(co)[:1500])
    print()

    combined = rc + co
    hit_11 = "11" in combined
    hit_22 = "22" in combined
    has_channel_marker = "thought" in rc[:50].lower() or "channel" in rc[:50].lower()

    print(f"correct answer (11) appears   : {hit_11}")
    print(f"intermediate step (22) appears: {hit_22}")
    print(f"reasoning channel marker      : {has_channel_marker}")
    print(f"reasoning_content non-empty   : {bool(rc.strip())}")
    print(f"content non-empty             : {bool(co.strip())}")

    ok = hit_11 and hit_22 and bool(rc.strip()) and bool(co.strip())
    print()
    print("THINKING VERIFIED" if ok else "THINKING DEGRADED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
