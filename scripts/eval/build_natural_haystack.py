#!/usr/bin/env python3
"""Build a natural-text haystack for `needle_band_probe.py --haystack`.

The stock needle filler ("The quick brown fox jumps over the lazy dog. " x N)
is a degenerate input for models with NoPE full-attention layers (Flint: 14 of
49 layers carry no positional signal, so every filler key is near-identical and
a 29K-copy haystack becomes a repetition attractor — 0/5 cells at >=131K@0.5
under greedy, sampled AND a flushed cache, while the same model retrieves at
131K from natural text).  This writes a plain-text corpus tiled from
Open-Platypus (instruction + output per doc) with any doc mentioning the
needle vocabulary dropped, so the retrieval verdict measures the model and not
the filler.

Usage:
  python3 scripts/eval/build_natural_haystack.py [--out PATH] [--min-chars N]
Default output: benchmarks/quality/haystack-platypus.txt (git-ignored; ~28 MB).
Chars/token is tokenizer-dependent — measure it on the served model's tokenizer
over the slice you will actually send (Flint: 3.43 on the first 400K chars but
3.22 over the 800K-char slice a 250K-token cell needs; a 1 % over-estimate at
250K is a 400 Bad Request on a 262,144 window).
"""
import argparse, glob, os, re, sys

NEEDLE_WORDS = re.compile(r"banana|password|secret", re.I)


def main():
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(repo, "benchmarks", "quality", "haystack-platypus.txt"))
    ap.add_argument("--min-chars", type=int, default=25_000_000,
                    help="warn if the corpus is shorter than this (needle_eval tiles it, so shorter still works)")
    ap.add_argument("--parquet", default=None,
                    help="Open-Platypus train parquet; default = the HF hub cache snapshot")
    a = ap.parse_args()
    pq = a.parquet or next(iter(sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--garage-bAInd--Open-Platypus/snapshots/*/data/*.parquet")))), None)
    if not pq:
        sys.exit("Open-Platypus parquet not found; `hf download garage-bAInd/Open-Platypus --repo-type dataset` "
                 "or pass --parquet")
    import pyarrow.parquet as pqt
    t = pqt.read_table(pq, columns=["instruction", "output"]).to_pydict()
    docs, dropped = [], 0
    for ins, out in zip(t["instruction"], t["output"]):
        d = (ins or "").strip() + "\n\n" + (out or "").strip()
        if NEEDLE_WORDS.search(d):
            dropped += 1
            continue
        docs.append(d)
    text = "\n\n".join(docs) + "\n\n"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(text)
    print(f"wrote {a.out}: {len(text):,} chars, {len(docs):,} docs ({dropped} dropped for needle vocabulary)")
    if len(text) < a.min_chars:
        print(f"WARNING: corpus shorter than {a.min_chars:,} chars; needle_eval tiles it", file=sys.stderr)


if __name__ == "__main__":
    main()
