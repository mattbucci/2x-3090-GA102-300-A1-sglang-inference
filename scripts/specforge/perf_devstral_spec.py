#!/usr/bin/env python
"""A/B the Devstral EAGLE3 draft: pure-DECODE tok/s WITHOUT spec vs WITH spec, on the
same target + prompts, at short and ~16K context.

Decode is isolated from prefill via the delta method: time a short generation (N0
tokens) and a long one (N1) on the SAME prompt; decode tok/s = (N1-N0)/(t1-t0). The
shared prefill cancels, so the number is the steady-state decode rate (not contaminated
by the prefill of a long-context prompt — the bug in v1). ignore_eos forces the full
decode window; a warmup pass absorbs first-request graph capture.

Usage: python perf_devstral_spec.py --target <path> --draft <path> [--num-steps 3]
"""
import argparse
import time

PROMPTS = [
    "Write a Python LRU cache class (dict + doubly linked list, O(1) get/put, type hints).",
    "Implement Dijkstra shortest path in Python with a heap, with a docstring and example.",
    "Write a Python function to merge overlapping intervals, with doctests.",
]
FILLER = (
    "\n# --- utils ---\n"
    "def _validate(cfg: dict) -> None:\n"
    "    for k in ('host','port','timeout'):\n"
    "        if k not in cfg: raise ValueError(f'missing {k}')\n"
    "class Service:\n"
    "    def __init__(self, cfg): self.cfg=cfg; _validate(cfg)\n"
    "    def run(self): return sum(i*i for i in range(self.cfg['timeout']))\n"
)
N0, N1 = 16, 272  # decode window = N1-N0 = 256 tokens


def build(target, draft, num_steps, spec):
    import sglang as sgl
    # 32768 = the draft's max_position_embeddings; the spec engine caps context to it.
    kw = dict(model_path=target, quantization="awq_marlin", mem_fraction_static=0.80,
              context_length=32768, tp_size=2, cuda_graph_max_bs=1,
              disable_radix_cache=True, disable_overlap_schedule=True)
    if spec:
        kw.update(speculative_algorithm="EAGLE3", speculative_draft_model_path=draft,
                  speculative_num_steps=num_steps, speculative_eagle_topk=4,
                  speculative_num_draft_tokens=8,
                  speculative_draft_model_quantization="unquant")
    return sgl.Engine(**kw)


def decode_toks(engine, prompt):
    """Pure decode tok/s via the short/long delta on one prompt. Returns (tok_s, accept_len)."""
    def gen(n):
        sp = {"temperature": 0.0, "max_new_tokens": n, "ignore_eos": True}
        t = time.time(); o = engine.generate(prompt, sp); dt = time.time() - t
        mi = o.get("meta_info", {})
        return dt, mi.get("completion_tokens", 0), (mi.get("spec_verify_ct", 0) or 0)
    t0, c0, _ = gen(N0)
    t1, c1, v1 = gen(N1)
    dtok = (c1 - c0); dt = (t1 - t0)
    toks = dtok / dt if dt > 0 else 0
    # accept_len over the long generation's spec verifies
    acc = (c1 / v1) if v1 else float("nan")
    return toks, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--num-steps", type=int, default=3)
    args = ap.parse_args()
    ctxs = [("short", ""), ("~16K", FILLER * 200 + "\n\nNow answer:\n")]
    res = {}
    for spec in (False, True):
        lab = "spec" if spec else "nospec"
        print(f"\n######## {lab} ########", flush=True)
        eng = build(args.target, args.draft, args.num_steps, spec)
        eng.generate("warmup", {"temperature": 0.0, "max_new_tokens": 8, "ignore_eos": True})
        for cname, pre in ctxs:
            ts, accs = [], []
            for p in PROMPTS:
                tk, ac = decode_toks(eng, pre + p)
                ts.append(tk); accs.append(ac)
            avg = sum(ts) / len(ts)
            aa = [a for a in accs if a == a]
            amean = sum(aa) / len(aa) if aa else float("nan")
            res[(lab, cname)] = (avg, amean)
            print(f"  {lab}/{cname}: decode {avg:.1f} tok/s"
                  + (f"  accept_len {amean:.2f}" if aa else ""), flush=True)
        eng.shutdown(); time.sleep(2)
    print("\n======== DEVSTRAL EAGLE3 SPEEDUP (decode tok/s) ========", flush=True)
    for cname, _ in ctxs:
        ns = res.get(("nospec", cname), (0,))[0]
        sp, acc = res.get(("spec", cname), (0, float("nan")))
        if ns and sp:
            print(f"  {cname:>6}:  no-spec {ns:6.1f}  ->  +EAGLE3 {sp:6.1f} tok/s  "
                  f"=  {sp/ns:.2f}x   (accept_len {acc:.2f})", flush=True)


if __name__ == "__main__":
    main()
