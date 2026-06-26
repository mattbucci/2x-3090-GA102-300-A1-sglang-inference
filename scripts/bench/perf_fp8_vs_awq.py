#!/usr/bin/env python
"""How does our 2x RTX 3090 (sm_86, Ampere) handle FP8 vs AWQ-int4 for single-user
decode? Controlled same-weights A/B on a standard MoE (Qwen3-Coder-30B-A3B-REAP):
the AWQ-int4 ship vs runtime FP8 (--quantization fp8 over the BF16 base).

Ampere has NO FP8 tensor cores (those start at Ada sm_89 / Hopper sm_90), so the
question is two-fold and this script answers both empirically:
  (1) Does FP8 even compile/boot on sm_86 for a 96-expert MoE? (try/except the build)
  (2) If it boots, is it faster or slower than AWQ-int4 at single-user decode?
      Decode is weight-memory-bound at M=1, and FP8 weights are 8-bit vs AWQ's 4-bit
      = ~2x the bytes read per step, so the a-priori expectation is FP8 is SLOWER.

Pure-decode tok/s via the short/long delta (prefill cancels), warmup, ignore_eos,
temp 0 — same method as perf_devstral_spec.py.

Usage: python perf_fp8_vs_awq.py --awq <awq_dir> --bf16 <bf16_dir> [--ctx 16384]
Writes a JSON summary to --out (default benchmarks/fp8-vs-awq-coder-reap.json).
"""
import argparse
import json
import os
import time
import traceback

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


def build(model_path, quant, ctx):
    import sglang as sgl
    kw = dict(model_path=model_path, quantization=quant, mem_fraction_static=0.82,
              context_length=ctx, tp_size=2, cuda_graph_max_bs=1,
              disable_radix_cache=True, disable_overlap_schedule=True,
              disable_piecewise_cuda_graph=True)  # MoE-marlin TP regression guard
    return sgl.Engine(**kw)


def decode_toks(engine, prompt):
    """Pure decode tok/s via the short/long delta on one prompt."""
    def gen(n):
        sp = {"temperature": 0.0, "max_new_tokens": n, "ignore_eos": True}
        t = time.time(); o = engine.generate(prompt, sp); dt = time.time() - t
        return dt, o.get("meta_info", {}).get("completion_tokens", 0)
    t0, c0 = gen(N0)
    t1, c1 = gen(N1)
    dt = (t1 - t0)
    return (c1 - c0) / dt if dt > 0 else 0.0


def measure(model_path, quant, ctx, label):
    """Build + measure one quant config. Returns dict; never raises (captures the
    build failure as the result, since 'FP8 won't compile on sm_86' is the finding)."""
    print(f"\n######## {label} :: quant={quant} :: {model_path} ########", flush=True)
    rec = {"label": label, "quant": quant, "model": model_path, "ctx": ctx}
    t_build = time.time()
    try:
        eng = build(model_path, quant, ctx)
    except Exception as e:
        rec["boot"] = "FAILED"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback_tail"] = "".join(traceback.format_exc()).splitlines()[-6:]
        print(f"  !! BUILD FAILED ({label}): {rec['error']}", flush=True)
        return rec
    rec["boot"] = "OK"
    rec["build_sec"] = round(time.time() - t_build, 1)
    try:
        eng.generate("warmup", {"temperature": 0.0, "max_new_tokens": 8, "ignore_eos": True})
        # FILLER ~= 80 tok/rep; leave room for the prompt (~40 tok) + N1 completion.
        reps = max(1, (ctx - N1 - 400) // 80)
        ctxs = [("short", ""), (f"~{ctx//1024}K", FILLER * reps + "\n\nNow answer:\n")]
        for cname, pre in ctxs:
            ts = [decode_toks(eng, pre + p) for p in PROMPTS]
            avg = sum(ts) / len(ts)
            rec[cname] = round(avg, 1)
            print(f"  {label}/{cname}: decode {avg:.1f} tok/s", flush=True)
    except Exception as e:
        rec["measure_error"] = f"{type(e).__name__}: {e}"
        print(f"  !! MEASURE ERROR ({label}): {rec['measure_error']}", flush=True)
    finally:
        try:
            eng.shutdown()
        except Exception:
            pass
        time.sleep(3)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--awq", required=True)
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "benchmarks", "fp8-vs-awq-coder-reap.json")
    summary = {
        "hardware": "2x RTX 3090 (GA102, sm_86, Ampere — no FP8 tensor cores)",
        "engine": "SGLang v0.5.13.post1 TP=2, cuda graphs, disable_piecewise",
        "method": "pure-decode tok/s via short/long delta, ignore_eos, temp 0, 3 coding prompts",
        "note": ("Same-weights A/B on Qwen3-Coder-30B-A3B-REAP (96-expert MoE): AWQ-int4 "
                 "ship vs runtime FP8 (--quantization fp8 over BF16 base). FP8 weights are "
                 "8-bit vs AWQ 4-bit -> ~2x bytes/step at M=1 weight-bound decode."),
        "results": [],
    }

    def flush():
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)

    # AWQ-int4 first (known-good baseline on the same weights). Flush BEFORE the
    # FP8 attempt: FP8 MoE on sm_86 SIGQUITs the scheduler subprocess (uncatchable
    # in-process), so persisting AWQ first guarantees it survives the crash.
    summary["results"].append(measure(args.awq, "awq_marlin", args.ctx, "AWQ-int4"))
    flush()
    # Pre-record the FP8 attempt so the receipt shows it was tested even if the
    # process is killed by the child SIGQUIT before measure() can return.
    summary["results"].append({
        "label": "FP8-runtime", "quant": "fp8", "model": args.bf16,
        "boot": "PENDING (process killed if fused-MoE JIT rejects fp8e4nv on sm_86)",
    })
    flush()
    # Runtime FP8 over the BF16 base (the sm_86 question).
    summary["results"][-1] = measure(args.bf16, "fp8", args.ctx, "FP8-runtime")
    flush()
    results = summary["results"]
    print("\n======== FP8 vs AWQ (decode tok/s, sm_86) ========", flush=True)
    for r in results:
        if r.get("boot") == "OK":
            print(f"  {r['label']:>12}: short {r.get('short','?')}  "
                  f"~{args.ctx//1024}K {r.get(f'~{args.ctx//1024}K','?')} tok/s", flush=True)
        else:
            print(f"  {r['label']:>12}: BOOT FAILED — {r.get('error','')}", flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
