#!/usr/bin/env python3
"""Minimal quality eval: MMLU + HumanEval against SGLang OpenAI-compatible API.

No heavy dependencies — just datasets + requests. Runs a small sample from
each benchmark and reports accuracy.

Usage:
    # Start model server first, then:
    python scripts/eval/eval_quality.py --port 23334 --mmlu-samples 100 --humaneval-samples 20

    # Compare models:
    python scripts/eval/eval_quality.py --port 23334 --tag "REAM-30B"
    python scripts/eval/eval_quality.py --port 23334 --tag "Coder-30B"
    python scripts/eval/eval_quality.py --port 23334 --tag "REAP-28B"
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset


def mmlu_eval(url, n_samples=100, max_workers=8):
    """Run MMLU multiple-choice eval."""
    ds = load_dataset("cais/mmlu", "all", split="test")
    # Sample diverse subjects
    subjects = list(set(ds["subject"]))
    per_subject = max(1, n_samples // len(subjects))
    samples = []
    for subj in subjects:
        subj_items = [x for x in ds if x["subject"] == subj][:per_subject]
        samples.extend(subj_items)
    samples = samples[:n_samples]

    correct = 0
    total = 0
    choices_map = ["A", "B", "C", "D"]

    def eval_one(item):
        q = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]  # 0-3
        correct_letter = choices_map[answer_idx]

        prompt = f"Question: {q}\n"
        for i, c in enumerate(choices):
            prompt += f"{choices_map[i]}. {c}\n"
        prompt += "\nAnswer with just the letter (A, B, C, or D):"

        try:
            r = requests.post(url, json={
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0,
            }, timeout=60).json()
            content = r["choices"][0]["message"]["content"] or ""
            # Strip thinking tags
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            # Extract LAST letter A-D (after thinking)
            matches = re.findall(r"[ABCD]", content.upper())
            match = type("M", (), {"group": lambda s, *a: matches[-1]})() if matches else None
            predicted = match.group(0) if match else ""
            return predicted == correct_letter
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(eval_one, s) for s in samples]
        for f in as_completed(futures):
            total += 1
            if f.result():
                correct += 1

    accuracy = correct / total if total > 0 else 0
    return {"name": "MMLU", "correct": correct, "total": total, "accuracy": accuracy}


def humaneval_eval(url, n_samples=20, max_workers=4):
    """Run HumanEval code generation eval (pass@1)."""
    ds = load_dataset("openai/openai_humaneval", split="test")
    samples = list(ds)[:n_samples]

    passed = 0
    total = 0

    def eval_one(item):
        prompt = item["prompt"]
        test_code = item["test"]
        entry_point = item["entry_point"]

        try:
            r = requests.post(url.replace("/chat/completions", "/completions"), json={
                "prompt": prompt,
                "max_tokens": 512,
                "temperature": 0,
                "stop": ["\ndef ", "\nclass ", "\n#", "\nif __name__"],
            }, timeout=60).json()
            completion = r["choices"][0]["text"]
            # Strip thinking tags that some models produce
            completion = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL)
            completion = re.sub(r"<think>.*", "", completion, flags=re.DOTALL)
            full_code = prompt + completion

            # Try to execute
            exec_globals = {}
            exec(full_code + "\n" + test_code, exec_globals)
            exec_globals["check"](exec_globals[entry_point])
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(eval_one, s) for s in samples]
        for f in as_completed(futures):
            total += 1
            if f.result():
                passed += 1

    pass_rate = passed / total if total > 0 else 0
    return {"name": "HumanEval", "passed": passed, "total": total, "pass_rate": pass_rate}


def needle_in_haystack(url, context_lengths=[1024, 4096, 16384, 65536]):
    """Simple needle-in-a-haystack test at various context lengths."""
    filler = "The quick brown fox jumps over the lazy dog. " * 100
    needle = "The secret password is: BANANA42."
    results = []

    for ctx in context_lengths:
        # Build haystack
        n_filler = ctx // 50  # ~50 chars per filler sentence
        mid = n_filler // 2
        haystack = (filler * mid)[:ctx * 2] + "\n" + needle + "\n" + (filler * mid)[:ctx * 2]
        prompt = haystack[:ctx * 4] + "\n\nWhat is the secret password mentioned above? Answer with just the password."

        try:
            r = requests.post(url, json={
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 20,
                "temperature": 0,
            }, timeout=120).json()
            content = r["choices"][0]["message"]["content"]
            found = "BANANA42" in content
            results.append({"context": ctx, "found": found})
        except Exception as e:
            results.append({"context": ctx, "found": False, "error": str(e)[:50]})

    return {"name": "Needle-in-Haystack", "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=23334)
    parser.add_argument("--tag", type=str, default="model")
    parser.add_argument("--mmlu-samples", type=int, default=100)
    parser.add_argument("--humaneval-samples", type=int, default=20)
    parser.add_argument("--needle", action="store_true", help="Run needle-in-haystack")
    parser.add_argument("--needle-lengths", type=str, default="1024,4096,16384,65536",
                        help="Comma-separated context lengths for needle test")
    args = parser.parse_args()

    chat_url = f"http://localhost:{args.port}/v1/chat/completions"

    print(f"{'=' * 50}")
    print(f"Quality Eval: {args.tag}")
    print(f"Endpoint: localhost:{args.port}")
    print(f"{'=' * 50}")

    # MMLU
    print(f"\nRunning MMLU ({args.mmlu_samples} samples)...")
    t0 = time.time()
    mmlu = mmlu_eval(chat_url, n_samples=args.mmlu_samples)
    print(f"  MMLU: {mmlu['correct']}/{mmlu['total']} = {mmlu['accuracy']:.1%} ({time.time()-t0:.0f}s)")

    # HumanEval
    print(f"\nRunning HumanEval ({args.humaneval_samples} samples)...")
    t0 = time.time()
    he = humaneval_eval(chat_url, n_samples=args.humaneval_samples)
    print(f"  HumanEval: {he['passed']}/{he['total']} = {he['pass_rate']:.1%} ({time.time()-t0:.0f}s)")

    # Needle
    if args.needle:
        lengths = [int(x) for x in args.needle_lengths.split(",")]
        print(f"\nRunning Needle-in-Haystack ({lengths})...")
        t0 = time.time()
        niah = needle_in_haystack(chat_url, context_lengths=lengths)
        for r in niah["results"]:
            status = "✓" if r["found"] else "✗"
            print(f"  {r['context']:>6d} tokens: {status}")
        print(f"  ({time.time()-t0:.0f}s)")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Summary: {args.tag}")
    print(f"  MMLU:      {mmlu['accuracy']:.1%} ({mmlu['correct']}/{mmlu['total']})")
    print(f"  HumanEval: {he['pass_rate']:.1%} ({he['passed']}/{he['total']})")
    if args.needle:
        found = sum(1 for r in niah["results"] if r["found"])
        print(f"  Needle:    {found}/{len(niah['results'])}")
    print(f"{'=' * 50}")

    # Save results
    results = {
        "tag": args.tag,
        "mmlu": mmlu,
        "humaneval": he,
    }
    if args.needle:
        results["needle"] = niah
    outfile = f"eval_{args.tag.replace(' ', '_')}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
