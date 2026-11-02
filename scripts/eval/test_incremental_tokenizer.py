#!/usr/bin/env python
"""Exactness + speed of the incremental tokenizer (3090 patch 072) on real chat templates, CPU only.

For each tokenizer: a growing multi-turn conversation (haystack text as user / assistant / tool turns) rendered
through the model's chat template; at each turn the incremental encode must equal apply_chat_template(tokenize=True)
(= encode(rendered, add_special_tokens=False)), and the /generate mode must equal tokenizer(text). Also an edited-tail
case (last user message rewritten) and a random-cut fuzz over texts stuffed with special tokens.
    python scripts/eval/test_incremental_tokenizer.py --out benchmarks/profiling/incremental-tokenizer-2026-11-01.json
"""
import argparse, json, os, random, sys, time
os.environ["SGLANG_INCREMENTAL_TOKENIZE"] = "1"
from transformers import AutoTokenizer
from sglang.srt.managers.incremental_tokenizer import IncrementalTokenizer

PRESETS = {"flint": "/data/models/hf-mattbucci/Flint-AWQ-INT4-g64-gptq", "qwen38": "/data/models/hf-mattbucci/Qwen3.8-27B-AWQ",
           "gemma4-31b": "/data/models/hf-mattbucci/gemma-4-31B-AWQ"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--presets", default="flint,qwen38,gemma4-31b"); ap.add_argument("--target-tokens", type=int, default=250000)
    ap.add_argument("--chunk-chars", type=int, default=6000); ap.add_argument("--out", default=None)
    ap.add_argument("--paths", default="", help="extra name=path pairs (comma separated) to test beyond the built-in presets")
    a = ap.parse_args()
    for kv in filter(None, a.paths.split(",")):
        k, v = kv.split("=", 1); PRESETS[k] = v
    hay = open(os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "quality", "haystack-platypus.txt")).read()
    while len(hay) < 3_000_000: hay += hay
    results = []; fails = 0
    for name in a.presets.split(","):
        tok = AutoTokenizer.from_pretrained(PRESETS[name]); inc = IncrementalTokenizer(tok, verify=False)
        row = {"preset": name, "enabled": inc.enabled, "special_prefix_ok": inc.special_prefix_ok, "turns": []}
        if not inc.enabled:
            print(f"{name}: incremental tokenizer DISABLED by the exactness check (safe fallback, not a failure)"); results.append(row); continue
        msgs = [{"role": "system", "content": "You are a careful assistant. " + hay[:2000]}]
        pos = 2000; turn = 0; n_tok = 0; mism = 0; checkpoints = []
        while n_tok < a.target_tokens:
            role = "user" if turn % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": hay[pos:pos + a.chunk_chars]}); pos += a.chunk_chars; turn += 1
            if role == "assistant":
                continue  # measure on user turns (the request shape)
            try:
                t0 = time.perf_counter(); rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True); t_render = time.perf_counter() - t0
                t0 = time.perf_counter(); full = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False); t_full = time.perf_counter() - t0
            except Exception as e:
                print(f"{name}: chat template failed at turn {turn}: {str(e)[:120]}"); break
            t0 = time.perf_counter(); got = inc.encode(rendered, add_special_tokens=False); t_inc = time.perf_counter() - t0
            n_tok = len(full)
            if got != full:
                mism += 1
                i = next((k for k in range(min(len(got), len(full))) if got[k] != full[k]), min(len(got), len(full)))
                print(f"{name} turn {turn}: MISMATCH at token {i} of {len(full)} (inc {len(got)})")
            if turn in (1, 3, 7, 15, 31) or n_tok >= a.target_tokens or turn % 20 == 1:
                checkpoints.append({"turn": turn, "tokens": n_tok, "render_ms": round(t_render * 1e3, 1), "full_ms": round(t_full * 1e3, 1), "inc_ms": round(t_inc * 1e3, 1)})
                print(f"{name:10} turn {turn:3} {n_tok:>7} tok  render {t_render*1e3:6.1f} ms  full encode {t_full*1e3:7.1f} ms  incremental {t_inc*1e3:6.1f} ms  {'ok' if got == full else 'MISMATCH'}", flush=True)
        # edited tail: rewrite the last user message
        msgs[-1] = {"role": "user", "content": "Actually, ignore that. " + hay[pos:pos + 3000]}
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True); full = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False)
        t0 = time.perf_counter(); got = inc.encode(rendered, add_special_tokens=False); t_edit = time.perf_counter() - t0
        edit_ok = got == full; mism += 0 if edit_ok else 1
        print(f"{name:10} edited last turn: {'ok' if edit_ok else 'MISMATCH'} in {t_edit*1e3:.1f} ms (hits {inc.stats['hits']}/{inc.stats['calls']})")
        # /generate mode (add_special_tokens=True) on growing raw text
        gen_ok = True
        if inc.special_prefix_ok:
            base = hay[:400_000]
            for cut in (100_000, 250_000, 400_000):
                txt = base[:cut]; got = inc.encode(txt, add_special_tokens=True); full = tok(txt)["input_ids"]
                gen_ok = gen_ok and got == full
            print(f"{name:10} /generate mode (add_special_tokens=True): {'ok' if gen_ok else 'MISMATCH'}")
        mism += 0 if gen_ok else 1
        # fuzz: random texts stuffed with special tokens, random appends / cuts
        specials = [str(t) for t in tok.added_tokens_decoder.values()][:40]; words = hay.split()[:5000]; rng = random.Random(0); fuzz_bad = 0
        inc2 = IncrementalTokenizer(tok, verify=False)
        prev = ""
        for i in range(300):
            if prev and rng.random() < 0.7:
                cut = rng.randint(max(0, len(prev) - 2000), len(prev)); txt = prev[:cut]
            else:
                txt = ""
            txt += " ".join(rng.choice(specials) if rng.random() < 0.05 else rng.choice(words) for _ in range(rng.randint(2000, 4000)))
            if rng.random() < 0.3: txt += rng.choice(specials)
            got = inc2.encode(txt, add_special_tokens=False); full = tok(txt, add_special_tokens=False)["input_ids"]
            fuzz_bad += 0 if got == full else 1; prev = txt
        print(f"{name:10} fuzz 300 random append/cut texts with specials: {300 - fuzz_bad}/300 exact (hits {inc2.stats['hits']})")
        mism += fuzz_bad
        row.update({"turns": checkpoints, "mismatches": mism, "edit_ok": edit_ok, "generate_mode_ok": gen_ok, "fuzz_exact": 300 - fuzz_bad})
        results.append(row); fails += 1 if mism else 0
    if a.out:
        json.dump({"timestamp": time.strftime("%Y-%m-%d %H:%M"), "results": results}, open(a.out, "w"), indent=1); print("wrote", a.out)
    print("RESULT:", "FAIL" if fails else "PASS"); return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
