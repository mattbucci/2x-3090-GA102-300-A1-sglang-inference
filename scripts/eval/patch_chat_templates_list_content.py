#!/usr/bin/env python3
"""patch_chat_templates_list_content.py — make string-only chat templates render
OpenAI structured content (list-of-parts) instead of silently blanking it.

The Qwen3-Instruct-2507-style template guards every message with

    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}          <-- lists render as EMPTY
    {%- endif %}

OpenAI clients (opencode et al.) may send content as [{"type":"text","text":...}]
for user AND tool messages. With this template the model then sees an EMPTY task
and EMPTY tool responses — it spins on retrieval tools and produces empty diffs.
This is invisible to single-turn tool-call probes (they score the CALL and never
send a RESULT back) and produced the June "qwen3-ream can't code" mis-verdict.

The patch replaces the else-branch with a namespace loop that joins the text
parts. Idempotent; writes a .pre-list-content.bak backup next to the template.

Usage:
    python scripts/eval/patch_chat_templates_list_content.py <model_dir> [...]
    python scripts/eval/patch_chat_templates_list_content.py --scan  # fleet scan+fix
"""
import shutil
import sys
from pathlib import Path

OLD = """    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}"""

NEW = """    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- elif message.content is iterable and message.content is not mapping %}
        {%- set ns_parts = namespace(text='') %}
        {%- for item in message.content %}
            {%- if item is mapping and item.type == 'text' %}
                {%- set ns_parts.text = ns_parts.text + item.text %}
            {%- endif %}
        {%- endfor %}
        {%- set content = ns_parts.text %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}"""

MARKER = "ns_parts = namespace(text='')"


def patch_dir(model_dir: Path) -> str:
    tpl = model_dir / "chat_template.jinja"
    if not tpl.is_file():
        return "no-template"
    text = tpl.read_text()
    if MARKER in text:
        return "already-patched"
    if OLD not in text:
        return "pattern-absent"
    shutil.copy2(tpl, tpl.with_suffix(".jinja.pre-list-content.bak"))
    tpl.write_text(text.replace(OLD, NEW, 1))
    return "PATCHED"


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    if args == ["--scan"]:
        import os
        root = Path(os.environ.get("MODELS_DIR", str(Path.home() / "AI/models")))
        dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    else:
        dirs = [Path(a) for a in args]
    for d in dirs:
        result = patch_dir(d)
        if result != "no-template":
            print(f"{d.name}: {result}")


if __name__ == "__main__":
    main()
