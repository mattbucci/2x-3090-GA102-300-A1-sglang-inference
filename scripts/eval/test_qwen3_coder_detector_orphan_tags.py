#!/usr/bin/env python3
"""Offline unit test — Qwen3CoderDetector streaming parser, orphan tags (patch 063)
and inter-call whitespace (patch 065).

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

Patch 065 (cases 7-8): the ``\n`` between two ``<tool_call>`` blocks is not
normal text once a call has been emitted.  When one increment spans the end of
call N and the start of call N+1 (routine under speculative decoding) the
detector returns a single result and the server sends its ``normal_text``
BEFORE its tool-call deltas, so the client sees a content delta between call
N's name and its closing ``}``.  Block-tracking clients (pi-ai 0.68 /
little-coder 1.1) open a fresh nameless tool call for the orphaned deltas — a
phantom ``{}`` call the agent executes, and with the whole body displaced the
real call goes out with ``{}``.  qwen38 little-coder lane under DSpark: 168 of
1898 multi-call turns, 110/300 instances, 5 argument thefts.  Case 8 replays
every split of a two-call turn through a pi-0.68-style block builder and fails
on any nameless block; the one-shot parser never returned inter-call text.

Expected: pre-063 cases 2-5 FAIL (a -1 index is emitted); pre-065 cases 7-8
FAIL; with both, all PASS.
Run in the serving env:
    source scripts/common.sh && activate_conda
    python scripts/eval/test_qwen3_coder_detector_orphan_tags.py [--detector FILE]
``--detector`` loads the detector class from FILE instead of the installed
tree (test a patched copy before the tree is touched).
"""
from __future__ import annotations

import importlib.util
import sys

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector

if "--detector" in sys.argv:
    _path = sys.argv[sys.argv.index("--detector") + 1]
    _spec = importlib.util.spec_from_file_location(
        "sglang.srt.function_call.qwen3_coder_detector_under_test", _path
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    Qwen3CoderDetector = _mod.Qwen3CoderDetector  # noqa: F811

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
    (
        "7 inter-call / trailing whitespace is not text; pre-call prose is",
        ["Let me look.\n\n", WELL_FORMED + "\n" + WELL_FORMED + "\n"],
        "Let me look.\n\n",
        [
            (0, "bash", ""), (0, None, "{"), (0, None, '"command": "ls"'), (0, None, "}"),
            (1, "bash", ""), (1, None, "{"), (1, None, '"command": "ls"'), (1, None, "}"),
        ],
    ),
]


def pi068_blocks(chunks):
    """Replay a stream through a pi-ai 0.68-style block builder (the server
    yields a result's normal_text before its tool-call deltas; a delta without
    an id that does not continue the current toolCall block opens a nameless
    one).  Returns the toolCall blocks as (name, arguments)."""
    det = Qwen3CoderDetector()
    blocks, cur = [], None
    for c in chunks:
        r = det.parse_streaming_increment(c, TOOLS)
        if r.normal_text:
            if cur is None or cur["type"] != "text":
                cur = {"type": "text"}
                blocks.append(cur)
        for x in r.calls:
            cid = f"call_{x.tool_index}" if x.name else None
            if cur is None or cur["type"] != "toolCall" or (cid and cur["id"] != cid):
                cur = {"type": "toolCall", "id": cid or "", "name": x.name or "", "args": ""}
                blocks.append(cur)
            if x.name:
                cur["name"] = x.name
            cur["args"] += x.parameters or ""
    return [(b["name"], b["args"]) for b in blocks if b["type"] == "toolCall"]


def case_8_every_split() -> bool:
    """8: every 2-way split of a two-call turn builds exactly the two calls."""
    turn = WELL_FORMED + "\n" + WELL_FORMED
    want = [("bash", '{"command": "ls"}')] * 2
    bad = []
    for i in range(1, len(turn)):
        got = pi068_blocks([turn[:i], turn[i:]])
        if got != want:
            bad.append((i, got))
    if bad:
        print(f"      first bad split at {bad[0][0]}: {bad[0][1]}  ({len(bad)} splits)")
    return not bad


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
    ok = case_8_every_split()
    print(f"{'PASS' if ok else 'FAIL'}  8 every split of a two-call turn: no phantom block (pi 0.68 model)")
    failures += 0 if ok else 1
    n = len(CASES) + 1
    print(f"{n - failures}/{n} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
