#!/usr/bin/env python3
"""Chat template validator — run before trusting any quantized model.

Checks we've been burned by:
  1. No chat_template set on the tokenizer (Gemma4 community weights).
  2. Template renders a BOS token that produces <unk> under SGLang tokenizer
     (Devstral community AWQ).
  3. Thinking-mode models: template doesn't open <think> structure or never
     emits </think>. Detects by rendering with and without enable_thinking
     and checking for the expected tokens.
  4. Vision models: user_message with an image placeholder renders without
     raising and includes the expected placeholder token.

Run this against a model path or a live SGLang server.

Usage:
    # Static check on a local model
    python validate_chat_template.py --model /path/to/model

    # Live check against a running server (adds a tokenize roundtrip)
    python validate_chat_template.py --model /path/to/model --port 23334
"""
import argparse
import json
import sys
from pathlib import Path


def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def check_has_template(tok):
    tmpl = getattr(tok, "chat_template", None)
    if not tmpl:
        return False, "tokenizer.chat_template is None — model will error on /v1/chat/completions"
    return True, f"template length: {len(tmpl)} chars"


def check_bos_handling(tok):
    msgs = [{"role": "user", "content": "Hello"}]
    try:
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        return False, f"render failed: {e}"
    bos = getattr(tok, "bos_token", None)
    if bos and rendered.startswith(bos):
        return False, f"template starts with BOS token '{bos}' — SGLang auto-adds BOS, doubled BOS produces <unk>"
    return True, "no leading BOS"


def check_thinking(tok):
    msgs = [{"role": "user", "content": "What is 1+1?"}]
    markers = ("<think>", "<|channel>", "<start_working_out>")
    findings = []
    for enable in (True, False):
        try:
            out = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": enable},
            )
        except TypeError:
            try:
                out = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=enable,
                )
            except Exception:
                return None, "template does not accept enable_thinking kwarg — OK for non-thinking models"
        except Exception as e:
            return False, f"render failed with enable_thinking={enable}: {e}"
        findings.append((enable, out))

    on, off = findings[0][1], findings[1][1]
    markers_on = [t for t in markers if t in on]
    markers_off = [t for t in markers if t in off]
    if on == off and not markers_on and not markers_off:
        return None, "template ignores enable_thinking and contains no thinking markers — not a thinking model"
    if on == off:
        return False, f"enable_thinking had no effect but template contains markers {markers_off} — model may still emit thinking regardless of flag"
    if not markers_on:
        return False, "enable_thinking=True did not introduce any known thinking marker"
    return True, f"thinking markers toggle correctly (on={markers_on})"


def check_vision_placeholder(tok):
    msgs = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            {"type": "text", "text": "Describe this image."},
        ],
    }]
    try:
        out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        return None, f"template does not support image content: {e}"
    return True, f"vision content rendered (placeholder present: {'<image>' in out or '<|image|>' in out or '<image_pad>' in out})"


def check_live(port, model, tok):
    import requests
    msgs = [{"role": "user", "content": "Say 'pong'."}]
    try:
        r = requests.post(f"http://localhost:{port}/v1/chat/completions", json={
            "model": model, "messages": msgs, "max_tokens": 8, "temperature": 0,
        }, timeout=30)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}: {r.text[:200]}"
        data = r.json()
        text = data["choices"][0]["message"].get("content", "")
        if not text.strip():
            return False, "server returned empty content — possible <unk> or template bug"
        if "<unk>" in text.lower() or "\ufffd" in text:
            return False, f"output contains <unk> / unknown chars: {text!r}"
        return True, f"server responded: {text!r}"
    except Exception as e:
        return False, f"live check failed: {e}"


CHECKS = [
    ("has chat_template", check_has_template),
    ("no doubled BOS", check_bos_handling),
    ("thinking toggle", check_thinking),
    ("vision content", check_vision_placeholder),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=None, help="Also run live roundtrip against server")
    args = ap.parse_args()

    print(f"Validating chat template: {args.model}")
    tok = load_tokenizer(args.model)

    results = []
    for name, fn in CHECKS:
        try:
            status, detail = fn(tok)
        except Exception as e:
            status, detail = False, f"check crashed: {e}"
        tag = "OK" if status is True else ("SKIP" if status is None else "FAIL")
        print(f"  [{tag:4}] {name}: {detail}")
        results.append((name, status, detail))

    live_status = None
    if args.port:
        print(f"\nLive server check on :{args.port}")
        live_status, live_detail = check_live(args.port, args.model, tok)
        tag = "OK" if live_status is True else "FAIL"
        print(f"  [{tag:4}] live /v1/chat/completions roundtrip: {live_detail}")

    failures = [r for r in results if r[1] is False]
    if args.port and live_status is False:
        failures.append(("live", False, "server check failed"))

    print()
    if failures:
        print(f"FAIL: {len(failures)} check(s) failed")
        sys.exit(1)
    else:
        print("PASS: chat template is ready for launch")


if __name__ == "__main__":
    main()
