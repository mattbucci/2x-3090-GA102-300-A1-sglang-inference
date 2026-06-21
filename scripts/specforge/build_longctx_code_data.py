#!/usr/bin/env python
"""Build a LONG-CONTEXT code-instruct dataset for EAGLE3 draft training.

WHY THIS EXISTS
---------------
SpecForge's trainer TRUNCATES samples to --max-length (preprocessing.py:268,
truncation=True) — it does NOT pack. So `--max-length 16384` is a no-op unless
the individual training samples are themselves ~16K tokens long. Plain
code-instruct datasets (OpenCodeInstruct, magicoder, opc) are single Q&A pairs
of ~200-400 tokens. Training a draft on those yields a 2K-regime draft, which
the R9700 amendment warns NETS A SLOWDOWN at long context.

This script streams short code Q&A pairs and GREEDILY PACKS them into multi-turn
conversations of ~`--target-tokens` tokens each, so every emitted sample fills
the long-context window. The Devstral chat template renders these as
`[INST] q1 [/INST] a1</s>[INST] q2 [/INST] a2</s>...` and the GeneralParser
loss-masks EVERY assistant span — so packing also maximizes loss coverage.

Streaming (not full download) keeps disk/time bounded: OpenCodeInstruct is ~5M
rows; we stop once we have enough.

Output schema (SpecForge jsonl): {"id": str, "conversations": [{role, content}...]}
"""
import argparse
import hashlib
import json
import sys

from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="/data/models/Devstral-Small-2-24B-AWQ-textonly")
    ap.add_argument("--dataset", default="nvidia/OpenCodeInstruct")
    ap.add_argument("--output", required=True)
    ap.add_argument("--target-tokens", type=int, default=15500,
                    help="approx content tokens per packed conversation (leave headroom under --max-length)")
    ap.add_argument("--num-sequences", type=int, default=2000,
                    help="how many packed long conversations to emit")
    ap.add_argument("--turn-overhead", type=int, default=10,
                    help="approx template tokens per turn ([INST]/[/INST]/</s>)")
    ap.add_argument("--max-source", type=int, default=2_000_000,
                    help="hard cap on source rows streamed (safety)")
    ap.add_argument("--min-turn-tokens", type=int, default=24,
                    help="skip degenerate tiny samples")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = load_dataset(args.dataset, split="train", streaming=True)

    out = open(args.output, "w")
    emitted = 0
    seen = 0
    cur_turns = []      # list of {"role","content"}
    cur_tokens = 0
    tok_total = 0

    def flush():
        nonlocal cur_turns, cur_tokens, emitted, tok_total
        if len(cur_turns) >= 2:  # need at least one full user+assistant pair
            rid = hashlib.md5(("".join(t["content"] for t in cur_turns)).encode()).hexdigest()
            out.write(json.dumps({"id": rid, "conversations": cur_turns}) + "\n")
            emitted += 1
            tok_total += cur_tokens
            if emitted % 50 == 0:
                out.flush()
                print(f"[build] emitted={emitted}/{args.num_sequences} "
                      f"seen={seen} avg_tok={tok_total // max(emitted,1)}", flush=True)
        cur_turns = []
        cur_tokens = 0

    for row in ds:
        if emitted >= args.num_sequences or seen >= args.max_source:
            break
        seen += 1
        q = (row.get("input") or row.get("instruction") or "").strip()
        a = (row.get("output") or row.get("response") or "").strip()
        if not q or not a:
            continue
        # count content tokens (no special tokens) + per-turn template overhead
        n = len(tok(q, add_special_tokens=False).input_ids) \
            + len(tok(a, add_special_tokens=False).input_ids) \
            + 2 * args.turn_overhead
        if n < args.min_turn_tokens:
            continue
        # if a single pair is already huge, emit it on its own (truncation will cap it)
        if n >= args.target_tokens and not cur_turns:
            cur_turns = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
            cur_tokens = n
            flush()
            continue
        # would overflow -> flush current, start fresh with this pair
        if cur_tokens + n > args.target_tokens and cur_turns:
            flush()
        cur_turns.append({"role": "user", "content": q})
        cur_turns.append({"role": "assistant", "content": a})
        cur_tokens += n

    flush()  # tail
    out.close()
    print(f"[build] DONE emitted={emitted} seen={seen} avg_tok={tok_total // max(emitted,1)} "
          f"-> {args.output}", flush=True)
    if emitted < args.num_sequences:
        print(f"[build] WARNING: only {emitted} < requested {args.num_sequences} "
              f"(source exhausted or max-source hit)", file=sys.stderr)


if __name__ == "__main__":
    main()
