#!/usr/bin/env python3
"""Sanity-check AWQ safetensors for degenerate weights and scales.

Checks BOTH `*.scales` and `*.qweight` tensors, because the two failure
modes are independent:

  scales=0  (v2 disaster, Gemma-4-26B drop_images=False, 21B-REAP-v2):
    llmcompressor silently skipped quantization (degenerate Hessian) but
    the layer still got saved as `qweight + scales=0 + qzeros`. At
    inference the layer dequantizes to zero, propagates as zeros into
    the forward pass, and produces NaN logits downstream.

  qweight=0 (v3 disaster, 21B-REAP-v3 2026-05-08):
    GPTQ Hessian was *also* degenerate but produced non-zero scales with
    quantized-to-zero qweight (rare-expert under-calibration: 512 samples
    × top-k=8 ≈ 4K activations split across 128 experts × N layers means
    rare experts got ~0 activations, weight quantized to 0). Validator
    sees 1/4 PASS with empty `(reasoning)` placeholder content. Server
    log diagnostic shows `expert0_nonzero=False expert0_first4=[0,0,0,0]`.
    Audit script (.scales-only) MISSES this because the scales are fine.

Per-tensor flags:
  - all-zero       → ALL elements are zero (catastrophic)
  - majority-zero  → >50% zero (suspicious, rare-expert under-cal pattern)
  - any-NaN / Inf  → numerical blowup
  - all-tiny       → abs_max < 1e-8 (scales) or extreme outliers

Usage:
    python scripts/eval/check_awq_scales.py <model-dir-or-shard>
    python scripts/eval/check_awq_scales.py --hf mattbucci/Qwen3-VL-32B-AWQ
    python scripts/eval/check_awq_scales.py <path> --skip-qweight  # legacy

Exit code: 0 if clean, 1 if any scale or qweight tensor failed a check.

The HF mode Range-fetches the safetensors header to enumerate tensor
names + shapes + dtypes without downloading the full weights, then for
any flagged tensor does a targeted Range-fetch of just that tensor
to confirm the values.  RAM-safe; doesn't load the full model.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch  # for bf16-tolerant scale loading via safetensors framework="pt"


def _check_scale_tensor(name: str, arr: np.ndarray) -> list[str]:
    """Return list of human-readable failure messages for a scale tensor."""
    issues: list[str] = []

    # Coerce bf16 → float32 for nan/inf checks (numpy doesn't have native bf16)
    if arr.dtype == np.uint16:
        # bfloat16 stored as raw uint16 — reinterpret to fp32
        as_fp32 = (arr.astype(np.uint32) << 16).view(np.float32)
    elif arr.dtype in (np.float16, np.float32, np.float64):
        as_fp32 = arr.astype(np.float32, copy=False)
    else:
        return [f"[unexpected scales dtype {arr.dtype}]"]

    n = as_fp32.size
    if n == 0:
        issues.append(f"empty tensor (shape {arr.shape})")
        return issues

    n_nan = int(np.isnan(as_fp32).sum())
    n_inf = int(np.isinf(as_fp32).sum())
    n_zero = int((as_fp32 == 0).sum())
    abs_arr = np.abs(as_fp32)
    abs_min = float(abs_arr.min())
    abs_max = float(abs_arr.max())
    abs_mean = float(abs_arr.mean())

    if n_nan > 0:
        issues.append(f"{n_nan}/{n} NaN")
    if n_inf > 0:
        issues.append(f"{n_inf}/{n} Inf")
    if n_zero == n:
        issues.append(f"ALL-ZERO scales (n={n}, shape={tuple(arr.shape)})")
    elif n_zero / n > 0.5:
        issues.append(f"{100*n_zero/n:.1f}% zero scales (suspicious)")
    if abs_max < 1e-8 and n_zero != n:
        issues.append(f"all scales tiny: abs_max={abs_max:.2e}")
    elif abs_max > 1e6:
        issues.append(f"scale outlier: abs_max={abs_max:.2e}")

    return issues


def _check_qweight_tensor(name: str, arr: np.ndarray) -> list[str]:
    """Return list of human-readable failure messages for a qweight tensor.

    qweight is int32 packed (8 4-bit values per int32). int32 == 0 means
    all 8 packed 4-bit values are 0, i.e. the underlying weight is zero
    in that block. Per-element `==0` is sufficient because hitting all
    zeros across an entire packed-int32 block is the disaster signal.
    """
    issues: list[str] = []
    n = arr.size
    if n == 0:
        issues.append(f"empty tensor (shape {arr.shape})")
        return issues

    # int32 packed ints; np handles == 0 directly. Coerce to int64 to be
    # safe across platforms where dtype reads might come back as int32 vs
    # uint32.
    n_zero = int((arr == 0).sum())

    if n_zero == n:
        issues.append(f"ALL-ZERO qweight (n={n}, shape={tuple(arr.shape)})")
    elif n_zero / n > 0.5:
        issues.append(f"{100*n_zero/n:.1f}% zero qweight (rare-expert under-cal pattern)")
    return issues


def check_local(path: Path, skip_qweight: bool = False) -> tuple[int, int, int, list[tuple[str, str, list[str]]]]:
    """Return (scale_count, qweight_count, fail_count, failures[(file, name, issues)])."""
    from safetensors import safe_open

    if path.is_file() and path.suffix == ".safetensors":
        files = [path]
    else:
        files = sorted(path.glob("*.safetensors"))
    if not files:
        print(f"[error] no .safetensors files found at {path}", file=sys.stderr)
        return 0, 0, 0, []

    scale_count = 0
    qweight_count = 0
    failures: list[tuple[str, str, list[str]]] = []

    # Use framework="pt" for bf16 support — safetensors np backend raises
    # TypeError("data type 'bfloat16' not understood") on bf16 tensors which
    # are common in CT-format AWQ scales (e.g. gemma-4-26B-A4B-it-AWQ-4bit
    # ships scales in bf16). Torch handles bf16 natively; we cast to float
    # for the all-zero / NaN / Inf checks below.
    for f in files:
        with safe_open(str(f), framework="pt") as h:
            for k in h.keys():
                if k.endswith(".scales") or k.endswith(".weight_scale"):
                    t_pt = h.get_tensor(k)
                    t = t_pt.float().cpu().numpy()
                    scale_count += 1
                    issues = _check_scale_tensor(k, t)
                    if issues:
                        failures.append((f.name, k, issues))
                elif (not skip_qweight) and k.endswith(".qweight"):
                    t_pt = h.get_tensor(k)
                    # qweight is int32; numpy handles natively
                    t = t_pt.cpu().numpy()
                    qweight_count += 1
                    issues = _check_qweight_tensor(k, t)
                    if issues:
                        failures.append((f.name, k, issues))

    return scale_count, qweight_count, len(failures), failures


def _hf_token() -> str | None:
    p = Path("~/.secrets/hf_token").expanduser()
    if p.exists():
        return p.read_text().strip()
    return os.environ.get("HF_TOKEN")


def _hf_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    h = {}
    tok = _hf_token()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    if extra:
        h.update(extra)
    return h


def _hf_resolve(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def _hf_range_get(url: str, start: int, length: int) -> bytes:
    req = urllib.request.Request(
        url, headers=_hf_headers({"Range": f"bytes={start}-{start+length-1}"})
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _list_repo_files(repo: str) -> list[str]:
    url = f"https://huggingface.co/api/models/{repo}/tree/main"
    req = urllib.request.Request(url, headers=_hf_headers())
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return [d["path"] for d in data if d.get("type") == "file"]


def check_hf(repo: str, skip_qweight: bool = False) -> tuple[int, int, int, list[tuple[str, str, list[str]]]]:
    """Range-fetch each safetensors header + scale/qweight tensor data."""
    files = [f for f in _list_repo_files(repo) if f.endswith(".safetensors")]
    if not files:
        print(f"[error] no safetensors in HF repo {repo}", file=sys.stderr)
        return 0, 0, 0, []

    scale_count = 0
    qweight_count = 0
    failures: list[tuple[str, str, list[str]]] = []

    dtype_map = {
        "F32": (np.float32, 4),
        "F16": (np.float16, 2),
        "BF16": (np.uint16, 2),  # treat as raw uint16, _check_scale_tensor handles it
        "F64": (np.float64, 8),
        "I32": (np.int32, 4),
        "U32": (np.uint32, 4),
    }

    for fname in files:
        url = _hf_resolve(repo, fname)
        # safetensors header: first 8 bytes = u64 little-endian length, then JSON
        head = _hf_range_get(url, 0, 8)
        hdr_len = struct.unpack("<Q", head)[0]
        hdr_bytes = _hf_range_get(url, 8, hdr_len)
        hdr = json.loads(hdr_bytes)

        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            is_scale = name.endswith(".scales") or name.endswith(".weight_scale")
            is_qweight = name.endswith(".qweight")
            if not (is_scale or (is_qweight and not skip_qweight)):
                continue
            dtype = info.get("dtype", "")
            if dtype not in dtype_map:
                failures.append((fname, name, [f"unknown dtype {dtype}"]))
                continue
            np_dtype, elem_size = dtype_map[dtype]
            shape = info.get("shape", [])
            n_elem = 1
            for d in shape:
                n_elem *= d
            data_offsets = info.get("data_offsets", [0, 0])
            byte_start = 8 + hdr_len + data_offsets[0]
            byte_len = data_offsets[1] - data_offsets[0]
            if byte_len != n_elem * elem_size:
                failures.append((fname, name, [f"byte/elem mismatch: {byte_len} vs {n_elem*elem_size}"]))
                continue
            raw = _hf_range_get(url, byte_start, byte_len)
            arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
            if is_scale:
                scale_count += 1
                issues = _check_scale_tensor(name, arr)
            else:
                qweight_count += 1
                issues = _check_qweight_tensor(name, arr)
            if issues:
                failures.append((fname, name, issues))

    return scale_count, qweight_count, len(failures), failures


def main():
    ap = argparse.ArgumentParser(description="AWQ scales + qweight sanity check")
    ap.add_argument("path", nargs="?", help="local model dir OR single .safetensors file")
    ap.add_argument("--hf", help="HF repo (e.g. mattbucci/Qwen3-VL-32B-AWQ)")
    ap.add_argument("--skip-qweight", action="store_true",
                    help="legacy mode: only audit .scales (skip new qweight check)")
    args = ap.parse_args()

    if args.hf:
        print(f"=== HF repo {args.hf} ===")
        scale_count, qweight_count, fail_count, failures = check_hf(args.hf, skip_qweight=args.skip_qweight)
    elif args.path:
        path = Path(args.path).expanduser().resolve()
        if not path.exists():
            print(f"[error] {path} does not exist", file=sys.stderr)
            sys.exit(1)
        print(f"=== local {path} ===")
        scale_count, qweight_count, fail_count, failures = check_local(path, skip_qweight=args.skip_qweight)
    else:
        ap.print_usage()
        sys.exit(2)

    if scale_count == 0 and qweight_count == 0:
        # Not an AWQ build (BF16 base, FP8, full-precision checkpoint, etc.)
        print("[info] no *.scales / *.qweight tensors found — not an AWQ build (skipping audit)")
        sys.exit(0)

    summary_parts = [f"{scale_count} *.scales"]
    if qweight_count:
        summary_parts.append(f"{qweight_count} *.qweight")
    print(f"Scanned {' + '.join(summary_parts)} tensors, {fail_count} flagged.")
    if failures:
        for fname, name, issues in failures:
            print(f"  [FAIL] {fname}::{name}")
            for i in issues:
                print(f"         - {i}")
        sys.exit(1)
    else:
        print("All scales + qweight clean.")
        sys.exit(0)


if __name__ == "__main__":
    main()
