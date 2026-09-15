#!/usr/bin/env python3
"""Offline unit test — Qwen3CoderDetector streaming parser, orphan tags (patch 063).

No GPU, no server: drives the streaming detector with small chunks and checks
that a ``<parameter=`` / ``</function>`` with NO open ``<function=...>`` is
passed through as text instead of being emitted as an argument delta with
``tool_index=-1`` and no id/name.

Why it matters: SGLang only attaches an ``id`` to the delta that carries the
function name. A -1-index argument delta therefore reaches the client as
``{"index":-1,"id":null,"function":{"name":null,...}}``; the AI SDK
(opencode, and every ``@ai-sdk/openai-compatible`` client) raises
``InvalidResponseDataError: Expected 'id' to be a string.`` and drops the whole
stream — opencode surfaces it as ``UnknownError`` and ends the agent session.
Seen on 10 SWE-bench Lite instances across four Qwen presets (qwen38,
qwen36-ream, qwen35-moe, coder-reap-25b), each time killing the session; the
server log shows ``Tool 'None' is not defined in the tools list.`` The one-shot
parser (``detect_and_parse``) already treats such text as prose, so streaming
and non-streaming disagreed.

Expected: pre-063 cases 2-5 FAIL (a -1 index is emitted); post-063 all PASS.
Run in the serving env:
    source scripts/common.sh && activate_conda
    python scripts/eval/test_qwen3_coder_detector_orphan_tags.py
"""
from __future__ import annotations

import sys

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector

TOOLS = [
    Tool(
        type="function",
        function=Function(
            name="bash",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        ),
    )
]


def drive(chunks):
    det = Qwen3CoderDetector()
    text, calls = "", []
    for c in chunks:
        r = det.parse_streaming_increment(c, TOOLS)
        text += r.normal_text or ""
        calls.extend((x.tool_index, x.name, x.parameters) for x in r.calls)
    return text, calls


WELL_FORMED = "<tool_call>\n<function=bash>\n<parameter=command>ls</parameter>\n</function>\n</tool_call>"

CASES = [
    # (label, chunks, expected_text, expected_calls_or_None (None = any), forbid_neg_index)
    (
        "1 well-formed call still parses",
        [WELL_FORMED],
        "",
        [(0, "bash", ""), (0, None, "{"), (0, None, '"command": "ls"'), (0, None, "}")],
    ),
    (
        "2 prose mentioning </function> is text",
        ["Note the closing ", "</function>", " tag here."],
        "Note the closing </function> tag here.",
        [],
    ),
    (
        "3 prose mentioning <parameter=x> is text",
        ["The template uses <parameter=name>value</parameter> syntax."],
        "The template uses <parameter=name>value</parameter> syntax.",
        [],
    ),
    (
        "4 malformed call (no <function=) emits no -1 index",
        ["<tool_call>\n<parameter=command>ls</parameter>\n</function>\n</tool_call>"],
        None,
        [],
    ),
    (
        "5 prose with orphan tag, then a real call",
        ["see </function> above\n", WELL_FORMED],
        "see </function> above\n",
        [(0, "bash", ""), (0, None, "{"), (0, None, '"command": "ls"'), (0, None, "}")],
    ),
    (
        "6 two calls keep distinct indices",
        [WELL_FORMED, "\n", WELL_FORMED],
        None,
        [
            (0, "bash", ""), (0, None, "{"), (0, None, '"command": "ls"'), (0, None, "}"),
            (1, "bash", ""), (1, None, "{"), (1, None, '"command": "ls"'), (1, None, "}"),
        ],
    ),
]


def main() -> int:
    failures = 0
    for label, chunks, exp_text, exp_calls in CASES:
        text, calls = drive(chunks)
        ok = all(c[0] >= 0 for c in calls)
        if exp_text is not None and text != exp_text:
            ok = False
        if exp_calls is not None and calls != exp_calls:
            ok = False
        print(f"{'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            failures += 1
            print(f"      text={text!r}\n      calls={calls}")
    print(f"{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
