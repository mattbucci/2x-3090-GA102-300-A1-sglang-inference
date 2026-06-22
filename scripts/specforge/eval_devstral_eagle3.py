#!/usr/bin/env python
"""Measure the REAL accept length + coherence of a trained Devstral EAGLE3 draft.

Serves the (text-only) Devstral AWQ target + the EAGLE3 draft via sglang's offline
Engine with speculative decoding, runs coding prompts, and reports the average
accept length (the speedup driver) and a coherence eyeball. This is the "test that
it works" gate: ship if accept_len clears ~3.5 (or report the honest number).

Accept length is read from each output's meta_info (sglang reports
`spec_verify_ct` / completion tokens; avg accept len = completion_tokens /
spec_verify_ct when spec is on).

Usage:
  python eval_devstral_eagle3.py --draft <ckpt_dir> --num-steps 3 [--target <path>]
"""
import argparse
import time


CODING_PROMPTS = [
    "Write a Python function `merge_intervals(intervals)` that merges overlapping "
    "intervals. Include docstring, type hints, and 3 doctests. Then explain the "
    "time complexity.",
    "Implement an LRU cache class in Python using a doubly linked list and a dict. "
    "Provide get/put in O(1), with full type hints and a short usage example.",
    "Refactor this into idiomatic async Python with proper error handling:\n"
    "def fetch_all(urls):\n    out=[]\n    for u in urls:\n        r=requests.get(u)\n"
    "        out.append(r.json())\n    return out",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="/data/models/Devstral-Small-2-24B-AWQ-textonly")
    ap.add_argument("--draft", required=True)
    ap.add_argument("--num-steps", type=int, default=3)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--num-draft-tokens", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--context-length", type=int, default=65536)
    args = ap.parse_args()

    import sglang as sgl

    print(f"[eval] building Engine: target={args.target}\n        draft={args.draft}\n"
          f"        num_steps={args.num_steps} topk={args.topk} draft_tokens={args.num_draft_tokens}",
          flush=True)
    engine = sgl.Engine(
        model_path=args.target,
        quantization="awq_marlin",
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path=args.draft,
        speculative_num_steps=args.num_steps,
        speculative_eagle_topk=args.topk,
        speculative_num_draft_tokens=args.num_draft_tokens,
        speculative_draft_model_quantization="unquant",
        mem_fraction_static=0.78,
        context_length=args.context_length,
        tp_size=2,
        cuda_graph_max_bs=1,
        disable_radix_cache=True,
        # v0.5.13: overlap-spec-v2 (default-on for EAGLE3) pulls in a TVM-FFI JIT
        # kernel that mis-probes for ROCm arch on NVIDIA -> crash. Force the
        # synchronous (non-overlap) spec-v2 path, which avoids that JIT.
        disable_overlap_schedule=True,
    )

    sp = {"temperature": 0.0, "max_new_tokens": args.max_new}
    accepts = []
    for i, p in enumerate(CODING_PROMPTS):
        t0 = time.time()
        out = engine.generate(p, sp)
        dt = time.time() - t0
        mi = out.get("meta_info", {})
        comp = mi.get("completion_tokens", 0)
        verify = mi.get("spec_verify_ct", 0) or 0
        acc_len = (comp / verify) if verify else float("nan")
        accepts.append(acc_len)
        text = out.get("text", "")
        print(f"\n===== PROMPT {i} =====", flush=True)
        print(f"  completion_tokens={comp} spec_verify_ct={verify} "
              f"accept_len={acc_len:.2f} tok/s={comp/dt:.1f} ({dt:.1f}s)", flush=True)
        print(f"  coherence head: {text[:240]!r}", flush=True)

    valid = [a for a in accepts if a == a]  # drop nan
    if valid:
        print(f"\n[eval] MEAN accept_len = {sum(valid)/len(valid):.2f} over {len(valid)} prompts "
              f"(num_steps={args.num_steps}, ceiling={args.num_steps + 1})", flush=True)
    else:
        print("\n[eval] WARNING: no spec_verify_ct reported — spec may not be active.", flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()
