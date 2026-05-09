"""Content-aware code-generation probe for coder-family models.

Complements `probe_thinking.py` (thinking-channel structure) and
`probe_vision.py` (image-content recognition). The repo's
validate_capabilities.py:check_basic only asks "What is the capital
of France" — passes on any model that emits "paris" somewhere. For
coder models (coder-30b, coder-reap, coder-reap-25b) we need to
actually verify the model can synthesize working code.

This probe:
  - Sends two algorithmic prompts (paren-balance + interval-merge).
  - Extracts the Python code block from the response.
  - Executes it and runs hand-rolled unit tests.
  - Classifies STRONG (8/8 pass) / PARTIAL (some pass) / FAIL (none).

Validated 2026-05-04 against coder-reap-25b at TP=1 / 4K port 23358 →
8/8 PASS, STRONG verdict.

Usage:
  python scripts/eval/probe_codegen.py [--port PORT] [--model MODEL]

Defaults: port 30000, model "default".
"""
import argparse
import json
import textwrap
from urllib.request import Request, urlopen


PROMPTS = [
    {
        "name": "is_balanced (parens)",
        "fn_name": "is_balanced",
        "prompt": textwrap.dedent("""\
            Write a complete Python function `is_balanced(s)` that takes a
            string of parentheses (just '(' and ')') and returns True if the
            parens are balanced, False otherwise. Just the function — no
            imports, no examples. Use markdown code fences."""),
        "tests": [
            ("()", True),
            ("(())", True),
            ("(()", False),
            (")(", False),
            ("", True),
        ],
    },
    {
        "name": "merge_intervals",
        "fn_name": "merge_intervals",
        "prompt": textwrap.dedent("""\
            Write a complete Python function `merge_intervals(intervals)` that
            takes a list of intervals like [[1,3], [2,6], [8,10], [15,18]]
            and returns the merged non-overlapping intervals as a list. Just
            the function — no imports, no examples. Use markdown code
            fences."""),
        "tests": [
            ([[1, 3], [2, 6], [8, 10], [15, 18]], [[1, 6], [8, 10], [15, 18]]),
            ([[1, 4], [4, 5]], [[1, 5]]),
            ([[1, 4]], [[1, 4]]),
        ],
    },
]


def _call(host_port: str, model: str, prompt: str, max_tokens: int = 600):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
    }
    req = Request(
        f"http://{host_port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = json.loads(urlopen(req, timeout=240).read())
    choice = r["choices"][0]
    msg = choice["message"]
    # Thinking-mode models (qwen3-ream, etc.) emit code into `reasoning_content`
    # while leaving `content` null when finish_reason='stop' fires inside the
    # think channel. Fall back to reasoning_content so the probe can still
    # extract code from those models.
    text = msg.get("content") or msg.get("reasoning_content") or ""
    return choice.get("finish_reason"), text, r.get("usage")


def _extract_code(text: str) -> str:
    """Pull the first ```python ... ``` block (or any ``` block as fallback)."""
    if "```python" in text:
        return text.split("```python", 1)[1].split("```", 1)[0]
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0]
    return text


def _run_one(host_port: str, model: str, spec: dict) -> tuple[int, int, str]:
    """Returns (n_pass, n_total, label_summary)."""
    name = spec["name"]
    fn_name = spec["fn_name"]
    print("=" * 60)
    print(f"PROBE: {name}")
    finish, content, usage = _call(host_port, model, spec["prompt"])
    print(f"finish={finish}  usage={usage}")
    print()
    print("--- response (first 400 chars) ---")
    print(content[:400])
    print()
    code = _extract_code(content).strip()
    if not code:
        print("FAIL: no code extracted")
        return 0, len(spec["tests"]), f"{name}: no code"
    try:
        ns = {}
        exec(code, ns)  # noqa: S102 — controlled local probe
    except Exception as e:
        print(f"FAIL: code did not exec: {e!r}")
        return 0, len(spec["tests"]), f"{name}: exec failed ({type(e).__name__})"
    fn = ns.get(fn_name)
    if fn is None:
        print(f"FAIL: {fn_name} not defined")
        return 0, len(spec["tests"]), f"{name}: fn not defined"
    n_pass = 0
    for inp, want in spec["tests"]:
        try:
            got = fn(*inp) if isinstance(inp, tuple) else fn(inp)
            ok = got == want
            print(f"  {fn_name}({inp!r}) = {got!r}  expected {want!r}  {'OK' if ok else 'FAIL'}")
            if ok:
                n_pass += 1
        except Exception as e:
            print(f"  {fn_name}({inp!r}) raised {e!r}")
    return n_pass, len(spec["tests"]), f"{name}: {n_pass}/{len(spec['tests'])}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--model", default="default")
    args = p.parse_args()

    host_port = f"localhost:{args.port}"
    summaries = []
    total_pass = 0
    total_count = 0
    for spec in PROMPTS:
        n_pass, n_total, summary = _run_one(host_port, args.model, spec)
        summaries.append(summary)
        total_pass += n_pass
        total_count += n_total
        print()

    print("=" * 60)
    for s in summaries:
        print(f"  {s}")
    print(f"OVERALL: {total_pass}/{total_count}")
    print()
    if total_pass == total_count:
        verdict = "STRONG"
        rc = 0
    elif total_pass > 0:
        verdict = "PARTIAL"
        rc = 1
    else:
        verdict = "FAIL"
        rc = 2
    print(f"VERDICT: {verdict}")
    print()
    print("Notes:")
    print("- STRONG = all 8 algorithmic test cases pass; model genuinely")
    print("  synthesizes working code (matches coder-reap-25b baseline).")
    print("- PARTIAL = some tests fail (algorithm mostly right but edge")
    print("  cases broken, or one of the two functions doesn't extract).")
    print("- FAIL = no working code returned (server hang, bad output)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
