"""Content-aware vision probe for multimodal models.

Complements `probe_thinking.py` by going beyond the validator's loose
keyword grep. The validator's `check_vision` only requires that the
response mention at least one color word AND one shape word — that
passes on degraded outputs like "scattered red and black pixels"
(Gemma 4's failure mode) where the model hits the keywords without
actually recognizing the image.

This probe:
  - Sends a clear synthetic 256x256 red circle (with optional black
    outline) to the served model.
  - Inspects both `content` and `reasoning_content` fields (handles
    thinking-mode models that emit answers into reasoning_content).
  - Prints classification:
      STRONG  — response says circle/round/dot/ball AND red, no
                degradation keywords (pixel/scatter/gradient).
      DEGRADED — keyword pass but content-degraded (e.g.
                 "scattered red and black pixels", "pixelated
                 gradient", "small red shape").
      FAIL     — no recognition at all.

Usage:
  python scripts/eval/probe_vision.py [--port PORT] [--model MODEL] \
                                      [--no-outline]

Defaults: port 30000, model "default", outline included.
"""
import argparse
import base64
import io
import json
from urllib.request import Request, urlopen

DEGRADATION_KEYWORDS = ("pixel", "scatter", "gradient", "fragment", "specks")
STRONG_SHAPE_KEYWORDS = ("circle", "round", "dot", "ball", "disk", "sphere")
COLOR_KEYWORDS = ("red", "crimson", "scarlet")


def _make_image(outline: bool) -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (256, 256), color="white")
    draw = ImageDraw.Draw(img)
    if outline:
        draw.ellipse([(64, 64), (192, 192)], fill="red", outline="black", width=3)
    else:
        draw.ellipse([(64, 64), (192, 192)], fill="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--model", default="default")
    p.add_argument("--no-outline", action="store_true",
                   help="Render without black outline (cleaner test of fill-color recognition)")
    args = p.parse_args()

    img_bytes = _make_image(outline=not args.no_outline)
    b64 = base64.b64encode(img_bytes).decode()
    data_url = f"data:image/png;base64,{b64}"

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": "Describe this image in one short sentence."},
        ]}],
        "max_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
    }

    req = Request(
        f"http://localhost:{args.port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = json.loads(urlopen(req, timeout=240).read())
    choice = r["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""

    print("=" * 60)
    print(f"finish_reason: {choice.get('finish_reason')}")
    print(f"usage: {r.get('usage')}")
    print()
    print("--- content ---")
    print(repr(content)[:600])
    print()
    print("--- reasoning_content ---")
    print(repr(reasoning)[:600])
    print()

    haystack = (content + " " + reasoning).lower().strip()
    color_hit = any(c in haystack for c in COLOR_KEYWORDS)
    shape_hit = any(s in haystack for s in STRONG_SHAPE_KEYWORDS)
    deg_hits = [k for k in DEGRADATION_KEYWORDS if k in haystack]

    print("=== content checks ===")
    print(f"color hit (red/crimson/...)    : {color_hit}")
    print(f"shape hit (circle/round/...)   : {shape_hit}")
    print(f"degradation keywords found     : {deg_hits or 'none'}")
    print()

    if color_hit and shape_hit and not deg_hits:
        verdict = "STRONG"
    elif color_hit and (shape_hit or not deg_hits):
        verdict = "DEGRADED"
    else:
        verdict = "FAIL"

    print(f"VERDICT: {verdict}")
    print()
    print("Notes:")
    print("- STRONG = real content recognition (matches Qwen-family on Ampere).")
    print("- DEGRADED = keyword grep passes but content reads 'pixels/scatter/")
    print("  gradient' style (Gemma 4 calibration-side issue, task #66).")
    print("- FAIL = vision tower not engaging or model emits unrelated text.")
    return 0 if verdict == "STRONG" else (1 if verdict == "DEGRADED" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
