#!/usr/bin/env python3
"""Make chat templates accept the OpenAI `developer` role as `system`.

Why: little-coder (pi-ai backend) and other newer OpenAI-compat scaffolds send the
system prompt with role `developer` (the o1-era replacement for `system`). Templates
that validate roles and `raise_exception('Unexpected message role.')` on anything
outside {system,user,assistant,tool} — Qwen3.5/3.6 family — then 400 on the FIRST
turn, so the scaffold exits the rollout in ~3 s with an empty diff (root-caused
2026-06-12, cross-team with R9700; see patches/README + bakeoff footnote †). Lenient
templates (Qwen3-Coder, Qwen3-30B-Instruct) silently DROP the developer message
instead of raising, so they "work" but lose the scaffold's system prompt.

Fix (matches the gemma4 template's existing `role in ['system','developer']` form):
turn every `<expr>.role == 'system'` role check into `<expr>.role in ['system',
'developer']`, so a developer message is rendered exactly where a system message
would be — at every site, including the one guarding the unknown-role raise.

Idempotent: re-running is a no-op once a template already uses the `in [...]` form.
Backs up the original to `<file>.pre-developer-role.bak` on first patch.

Usage:
  python scripts/eval/patch_chat_templates_developer_role.py            # default fleet
  python scripts/eval/patch_chat_templates_developer_role.py FILE...    # explicit files
"""
import re
import shutil
import sys
from pathlib import Path

# `<dot-expr>.role == '<role>'`  /  `... == "<role>"` — only the system role.
_PAT = re.compile(r"""\.role\s*==\s*(['"])system\1""")


def patch_text(text: str):
    n = [0]

    def repl(m):
        q = m.group(1)
        n[0] += 1
        return f".role in [{q}system{q}, {q}developer{q}]"

    return _PAT.sub(repl, text), n[0]


def patch_file(path: Path) -> str:
    if not path.is_file():
        return f"SKIP (missing)  {path}"
    text = path.read_text()
    if "developer" in text and "== " not in re.sub(_PAT, "", text):
        pass  # fallthrough; the count below is the real idempotency signal
    new, n = patch_text(text)
    if n == 0:
        already = "developer" in text
        return f"OK (no system== checks; {'already developer-aware' if already else 'lenient/raise-free'})  {path}"
    bak = path.with_suffix(path.suffix + ".pre-developer-role.bak")
    if not bak.exists():
        shutil.copy2(path, bak)
    path.write_text(new)
    return f"PATCHED ({n} role check(s) -> system|developer)  {path}"


def default_targets():
    repo = Path(__file__).resolve().parents[2]
    models = Path("/data/models/hf-mattbucci")
    names = [
        "Qwen3.6-35B-A3B-AWQ", "Qwen3.6-REAM-A3B-AWQ", "Qwen3.6-27B-AWQ",
        "Qwen3.5-28B-A3B-REAP-AWQ", "Qwen3-Coder-30B-A3B-AWQ",
        "Qwen3-Coder-30B-A3B-REAM-AWQ", "Qwen3-Coder-REAP-25B-A3B-AWQ",
        "Qwen3-30B-Instruct-2507-REAM-AWQ",
    ]
    targets = [models / n / "chat_template.jinja" for n in names]
    targets += sorted((repo / "scripts").glob("*chat_template*.jinja"))
    return targets


def main():
    targets = [Path(a) for a in sys.argv[1:]] or default_targets()
    for t in targets:
        print(patch_file(t))


if __name__ == "__main__":
    main()
