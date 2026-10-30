#!/usr/bin/env python
"""FLINT_STRICT_TOOLS=1 must not force a scaffold's metadata key (omp's ``i``) before the command.

XGrammar's strict object grammar enforces the schema's declared property order. omp declares ``i`` first and
Flint writes ``command`` first, so the grammar forced ``<arg_key>i</arg_key><arg_value>`` onto the wire and Flint
put the command there (flint:omp 2026-10-28: 7.4 % of bash calls swapped, 47 % of sessions with re-emitted
tags inside the forced value). ``_as_strict`` moves ``TRAILING_KEYS`` to the end of ``properties``; this test
compiles the real grammar (CPU, the served checkpoint's tokenizer) and checks what it accepts.

Run (serving env): FLINT_STRICT_TOOLS=1 python scripts/eval/test_flint_tool_grammar_key_order.py
"""
import os
import sys

os.environ["FLINT_STRICT_TOOLS"] = "1"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "plugins", "flint_sglang"))

import xgrammar as xgr  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from flint_sglang.tool_grammar import _as_strict, build_flint_tool_grammar  # noqa: E402

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.path.expanduser("~"), "AI", "models"))
CKPT = os.environ.get("FLINT_CKPT", os.path.join(MODELS_DIR, "hf-mattbucci", "Flint-AWQ-INT4-g64-gptq"))


def tool(name, props, required):
    return {"type": "function", "function": {"name": name, "description": name,
            "parameters": {"type": "object", "properties": props, "required": required}}}


OMP_BASH = tool("bash", {"i": {"type": "string"}, "command": {"type": "string"}, "timeout": {"type": "number"}},
                ["i", "command"])
OPENCODE_BASH = tool("bash", {"command": {"type": "string"}, "timeout": {"type": "number"},
                              "workdir": {"type": "string"}, "description": {"type": "string"}},
                     ["command", "description"])
CALL = "<tool_call>bash{}</tool_call>"
ARG = "<arg_key>{}</arg_key><arg_value>{}</arg_value>"


def accepts(compiled, text, stop_ids):
    m = xgr.GrammarMatcher(compiled)
    if not m.accept_string(text):
        return False
    return m.is_terminated() or (bool(stop_ids) and m.accept_token(stop_ids[0]))


def main():
    # schema-level: order rewritten, required kept, nothing else touched
    strict = _as_strict(OMP_BASH)["function"]
    assert strict["strict"] is True
    assert list(strict["parameters"]["properties"]) == ["command", "timeout", "i"], strict["parameters"]
    assert strict["parameters"]["required"] == ["i", "command"]
    oc = _as_strict(OPENCODE_BASH)["function"]["parameters"]
    assert list(oc["properties"]) == ["command", "timeout", "workdir", "description"]
    assert _as_strict({"type": "function", "function": {"name": "x"}})["function"]["strict"] is True
    if not os.path.isdir(CKPT):
        print(f"schema checks ok; grammar checks skipped (no checkpoint at {CKPT})")
        return 0
    tok = AutoTokenizer.from_pretrained(CKPT)
    info = xgr.TokenizerInfo.from_huggingface(tok)
    comp = xgr.GrammarCompiler(info)
    stop = info.stop_token_ids
    g_omp = comp.compile_structural_tag(build_flint_tool_grammar([OMP_BASH], "auto", True))
    g_oc = comp.compile_structural_tag(build_flint_tool_grammar([OPENCODE_BASH], "auto", True))
    cmd_then_i = CALL.format(ARG.format("command", "ls") + ARG.format("i", "list files"))
    cmd_timeout_i = CALL.format(ARG.format("command", "ls") + ARG.format("timeout", "30") + ARG.format("i", "list"))
    i_then_cmd = CALL.format(ARG.format("i", "list files") + ARG.format("command", "ls"))
    cmd_only = CALL.format(ARG.format("command", "ls"))
    checks = [
        ("omp: command then i", accepts(g_omp, cmd_then_i, stop), True),
        ("omp: command, timeout, i", accepts(g_omp, cmd_timeout_i, stop), True),
        ("omp: i then command (old forced order)", accepts(g_omp, i_then_cmd, stop), False),
        ("omp: command only (i still required)", accepts(g_omp, cmd_only, stop), False),
        ("opencode: command then description", accepts(g_oc, CALL.format(ARG.format("command", "ls") + ARG.format("description", "list")), stop), True),
        ("opencode: command only (description required)", accepts(g_oc, cmd_only, stop), False),
    ]
    bad = 0
    for name, got, want in checks:
        ok = got == want
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: accept={got} want={want}")
    print(f"{len(checks) - bad}/{len(checks)} grammar checks pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
