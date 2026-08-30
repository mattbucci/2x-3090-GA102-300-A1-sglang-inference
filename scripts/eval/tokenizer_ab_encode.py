#!/usr/bin/env python3
"""Tokenizer A/B gate for stack flips (learned at the v0.5.15 flip, patch 057).

A transformers/sglang bump can silently re-route a tokenizer family to a
different backend (tx 5.12 sent tekken.json Mistral checkpoints to
MistralCommonBackend, which never parses special tokens from text -> sglang's
render-then-encode chat path fed `[INST]`/`[TOOL_CALLS]` as plain text ->
needle 0.0 / HE halved / dead tool calls while boot + basic probes stayed
green). Boot smoke cannot see this; only token ids can.

Run the SAME command in the OLD and NEW serving env, then diff the JSONs:

    conda run -n sglang-v0517 python scripts/eval/tokenizer_ab_encode.py -o /tmp/tok-v0517.json
    conda run -n sglang-v0518 python scripts/eval/tokenizer_ab_encode.py -o /tmp/tok-v0518.json
    python scripts/eval/tokenizer_ab_encode.py --compare /tmp/tok-v0517.json /tmp/tok-v0518.json

Per family it records: backend class, the ids of a special-token-bearing
probe string (each family's control tokens MUST map to single ids), and the
ids of a rendered chat with a tool call + tool result (through the same
`get_tokenizer` path launch.sh's server uses, with the preset's chat
template where launch.sh overrides it). Exit 1 on any id mismatch.
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("MODELS_DIR", str(Path.home() / "AI/models")))

# preset -> (model dir, chat-template override used by launch.sh or None, probe string)
FAMILIES = {
    "devstral": (
        MODELS / "hf-mattbucci/Devstral-Small-2-24B-AWQ",
        REPO / "scripts/devstral2_chat_template.jinja",
        "<s>[SYSTEM_PROMPT]sys[/SYSTEM_PROMPT][INST]hi[/INST][TOOL_CALLS]x</s>",
    ),
    "qwen36": (
        MODELS / "hf-mattbucci/Qwen3.6-35B-A3B-AWQ",
        None,
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\nt\n</think>\n<tool_call>\nx\n</tool_call><|im_end|>",
    ),
    "qwen3-ream": (
        MODELS / "Qwen3-30B-Instruct-2507-REAM-AWQ",
        None,
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<tool_call>\nx\n</tool_call><|im_end|>",
    ),
    "coder-30b-eval": (
        MODELS / "hf-mattbucci/Qwen3-Coder-30B-A3B-AWQ",
        None,
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<tool_call>\n<function=f>\n</function>\n</tool_call><|im_end|>",
    ),
    "gemma4-31b": (
        MODELS / "hf-mattbucci/gemma-4-31B-AWQ",
        REPO / "scripts/gemma4_chat_template.jinja",
        "<bos><start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n<|channel>thought\n<channel|>ok<end_of_turn>",
    ),
    "nemotron3-omni": (
        MODELS / "hf-mattbucci/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-AWQ",
        None,
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\nt\n</think>\n<tool_call>\nx\n</tool_call><|im_end|>",
    ),
}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
}]
MESSAGES = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "Open README.md and summarize it."},
    {"role": "assistant", "content": "", "tool_calls": [{
        "id": "call_0", "type": "function",
        # dict, not a JSON string: Qwen/Nemotron templates iterate `arguments|items`,
        # Mistral/Gemma templates `tojson` it — a dict renders on every family.
        "function": {"name": "read_file", "arguments": {"path": "README.md"}},
    }]},
    {"role": "tool", "content": "# Title\nHello.", "tool_call_id": "call_0", "name": "read_file"},
]


def load_tokenizer(path):
    from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer
    return get_tokenizer(str(path), trust_remote_code=True)


def encode_family(name, model, template, probe):
    out = {"model": str(model)}
    if not model.exists():
        out["skipped"] = "model dir missing"
        return out
    tok = load_tokenizer(model)
    out["backend"] = type(tok).__name__
    out["probe_ids"] = tok.encode(probe, add_special_tokens=False)
    kw = {}
    if template is not None and template.exists():
        kw["chat_template"] = template.read_text()
        out["chat_template"] = template.name
    try:
        rendered = tok.apply_chat_template(MESSAGES, tools=TOOLS, tokenize=False, add_generation_prompt=True, **kw)
        out["chat_ids"] = tok.encode(rendered, add_special_tokens=False)
        out["chat_len"] = len(out["chat_ids"])
    except Exception as e:  # template may reject the tool schema; record, don't hide
        out["chat_error"] = f"{type(e).__name__}: {e}"[:300]
    return out


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    bad = 0
    for name in sorted(set(a) | set(b)):
        ra, rb = a.get(name), b.get(name)
        if not ra or not rb or "skipped" in ra or "skipped" in rb:
            print(f"{name:16s} SKIP (missing on one side)"); continue
        notes = []
        if ra.get("backend") != rb.get("backend"):
            notes.append(f"backend {ra.get('backend')} -> {rb.get('backend')}")
        for key in ("probe_ids", "chat_ids"):
            if key in ra or key in rb:
                if ra.get(key) != rb.get(key):
                    notes.append(f"{key} DIFFER ({len(ra.get(key) or [])} vs {len(rb.get(key) or [])} ids)")
        if ra.get("chat_error") != rb.get("chat_error"):
            notes.append(f"chat_error changed: {ra.get('chat_error')!r} -> {rb.get('chat_error')!r}")
        status = "MISMATCH" if notes else "identical"
        bad += bool(notes)
        print(f"{name:16s} {status:9s} backend={rb.get('backend')} probe={len(rb.get('probe_ids', []))} chat={rb.get('chat_len')}  {'; '.join(notes)}")
    print("RESULT:", "FAIL" if bad else "PASS", f"({bad} mismatching families)")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", help="write per-family encode receipts (JSON)")
    ap.add_argument("--families", default=",".join(FAMILIES), help="comma list (default: all)")
    ap.add_argument("--compare", nargs=2, metavar=("OLD_JSON", "NEW_JSON"))
    args = ap.parse_args()
    if args.compare:
        sys.exit(compare(*args.compare))
    import sglang, transformers
    res = {"_env": {"sglang": sglang.__version__, "transformers": transformers.__version__,
                    "python": sys.executable}}
    for name in args.families.split(","):
        model, template, probe = FAMILIES[name]
        res[name] = encode_family(name, model, template, probe)
        r = res[name]
        print(f"{name:16s} {r.get('backend', r.get('skipped')):28s} probe={len(r.get('probe_ids', []))} chat={r.get('chat_len', r.get('chat_error'))}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
