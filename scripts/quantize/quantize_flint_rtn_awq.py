#!/usr/bin/env python3
"""Flint (RecursiveSI/flint) BF16 -> native AWQ-Marlin W4A16, group 64, data-free MSE-RTN.

Why not convert the vendor's MLX int4 export: MLX affine quantization stores
`w = s*q + b` with a FREE fp16 bias per group; -b/s is uniformly fractional on
the real tensors (mean |frac| 0.25, max 0.5 — measured on q_proj, down_proj,
shared_expert.up_proj, switch_mlp.gate_proj), so there is no exact AWQ
`w = s*(q - z)` representation with an integer zero point. Rounding z would add a
correlated per-group offset of up to s/2 on every weight (~doubles the weight
error energy). The vendor export is itself data-free (`scale_rule: mse`, no
calibration), so re-quantizing the BF16 base with the same class of search —
per-group MSE over the clipping range, this time on the integer-zero-point grid —
is the parity route. No llmcompressor path exists (transformers has no Flint
modeling file), so this is a plain tensor-by-tensor conversion, not a calibration.

What gets INT4 (group 64, asymmetric, integer zero point; matches the vendor's
INT4 coverage except embeddings/lm_head, which stay BF16 here):
  model.layers.N.self_attn.{q,k,v,o}_proj.weight
  model.layers.N.mlp.{gate,up,down}_proj.weight              (dense layers 0,1)
  model.layers.N.mlp.shared_expert.{gate,up,down}_proj.weight
  model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight     (256 x 47 layers)
Kept BF16: mlp.gate (router; the plugin builds it with quant_config=None),
embed_tokens, lm_head, all norms, self_attn.softmax_offset (learned sinks).

Output layout = AutoAWQ GEMM (what SGLang's AWQ-Marlin loader reads):
  <name>.qweight  int32 [in, out/8]   nibble i of a word holds column 8j+[0,2,4,6,1,3,5,7][i]
  <name>.qzeros   int32 [in/64, out/8] same packing
  <name>.scales   fp16  [in/64, out]
`--selftest` checks the packer bit-for-bit against AutoAWQ's WQLinear_GEMM (run it
in the awq-quant env) and against SGLang's unpack_cols + undo-interleave (sglang env).

Marlin shape constraints at TP=2 were checked up front: every per-rank k is a
multiple of 128 and every per-rank n a multiple of 64 (kv per rank 256, expert
down_proj per-rank k 256, shared 512, dense 3072).

Usage (serving env has torch+CUDA; one GPU is plenty, ~minutes):
  PYTHONNOUSERSITE=1 python scripts/quantize/quantize_flint_rtn_awq.py \
      --src /data/models/hf-3rdparty/flint-bf16 \
      --dst /data/models/hf-mattbucci/Flint-AWQ-INT4-g64 --device cuda:0
Then: scripts/eval/check_awq_scales.py <dst> --base <src>  (non-zero exit = do not ship).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import struct
import time
from collections import OrderedDict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

W_BIT = 4
MAX_Q = (1 << W_BIT) - 1
PACK = 32 // W_BIT
AWQ_INTERLEAVE = (0, 2, 4, 6, 1, 3, 5, 7)
QUANT_RE = re.compile(
    r"^model\.layers\.\d+\."
    r"(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj|mlp\.shared_expert\.(gate|up|down)_proj"
    r"|mlp\.experts\.\d+\.(gate|up|down)_proj)\.weight$"
)
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def modules_to_not_convert(quant_names, kept_names):
    r"""Entries for quantization_config.modules_to_not_convert covering exactly
    the quantizable tensors in `kept_names` (a subset of `quant_names`).

    SGLang's skip test (`_module_path_match`) matches an entry on dotted-path
    boundaries anywhere in the module prefix, and the fused qkv_proj /
    gate_up_proj prefixes never contain the per-shard names -- so name the
    parent module when every quantizable child under it is kept. A name is
    collapsed to its layer-stripped form (e.g. ``self_attn``) only when it is
    kept in EVERY layer that has it; a keep that is layer-specific (e.g. the
    dense MLPs of layers 0/1 under ``layers\.[01]\.mlp``) is emitted fully
    qualified (``model.layers.0.mlp``), because the stripped ``mlp`` would
    also match every ``mlp.experts`` and the loader would build BF16 experts
    (23 GiB per rank -> OOM at boot). Found 2026-10-22 on the calibrated
    experts-only build; the first such checkpoints had never been served.
    """
    kept = {n[: -len(".weight")] for n in kept_names}
    by_parent: dict[str, set[str]] = {}
    for n in quant_names:
        mod = n[: -len(".weight")]
        by_parent.setdefault(mod.rsplit(".", 1)[0], set()).add(mod)
    entries: set[str] = set()
    for parent, children in by_parent.items():
        if children <= kept:
            entries.add(parent)
        else:
            entries |= children & kept
    # every module path in the model (all dotted ancestors of every quantizable
    # tensor), so that ``mlp`` of a MoE layer counts as an instance of ``mlp``
    # even though its quantizable Linears sit two levels down
    instances: dict[str, set[str]] = {}
    for n in quant_names:
        parts = n[: -len(".weight")].split(".")
        for i in range(1, len(parts) + 1):
            m = ".".join(parts[:i])
            instances.setdefault(LAYER_RE.sub("", m), set()).add(m)
    groups: dict[str, set[str]] = {}
    for e in entries:
        groups.setdefault(LAYER_RE.sub("", e), set()).add(e)
    out: set[str] = set()
    for stripped, fulls in groups.items():
        if fulls == instances[stripped]:
            out.add(stripped)
        else:
            out |= fulls
    return sorted(out)


# ----------------------------------------------------------------------------- quantizer
def search_groups(W: torch.Tensor, group: int, grid: int, shrink_min: float, norm: float):
    """Per (row, group) MSE search over the clipping range on the integer-zero-point grid.

    W [out, in] float32. Returns scales [out, in/g] fp16, zeros [out, in/g] int32,
    q [out, in] uint8 — with `q` rounded against the fp16 scale the kernel will use.
    """
    out, inn = W.shape
    Wg = W.view(out, inn // group, group)
    wmax = Wg.amax(-1, keepdim=True)
    wmin = Wg.amin(-1, keepdim=True)
    best_err = torch.full_like(wmax, float("inf"))
    best_s = torch.zeros_like(wmax)
    best_z = torch.zeros_like(wmax)
    for p in torch.linspace(1.0, shrink_min, grid, device=W.device):
        mx, mn = wmax * p, wmin * p
        s = ((mx - mn) / MAX_Q).clamp(min=1e-5).half().float()  # the scale as stored
        z = (-mn / s).round().clamp(0, MAX_Q)
        q = (Wg / s + z).round().clamp(0, MAX_Q)
        err = ((q - z) * s - Wg).abs().pow(norm).sum(-1, keepdim=True)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_s = torch.where(better, s, best_s)
        best_z = torch.where(better, z, best_z)
    q = (Wg / best_s + best_z).round().clamp(0, MAX_Q)
    return (
        best_s.squeeze(-1).half(),
        best_z.squeeze(-1).to(torch.int32),
        q.view(out, inn).to(torch.uint8),
    )


def pack_cols(x: torch.Tensor) -> torch.Tensor:
    """x [rows, n] ints in 0..15 -> int32 [rows, n/8] with the AWQ nibble interleave."""
    rows, n = x.shape
    assert n % PACK == 0, x.shape
    x = x.to(torch.int64).view(rows, n // PACK, PACK)
    packed = torch.zeros(rows, n // PACK, dtype=torch.int64, device=x.device)
    for i, col in enumerate(AWQ_INTERLEAVE):
        packed |= x[:, :, col] << (W_BIT * i)
    packed = torch.where(packed >= (1 << 31), packed - (1 << 32), packed)  # two's-complement int32
    return packed.to(torch.int32)


def pack_awq(scales: torch.Tensor, zeros: torch.Tensor, q: torch.Tensor):
    """[out, ...] tensors from search_groups -> AWQ GEMM {qweight, qzeros, scales} ([in, ...])."""
    return {
        "qweight": pack_cols(q.t().contiguous()),
        "qzeros": pack_cols(zeros.t().contiguous()),
        "scales": scales.t().contiguous(),
    }


def quantize_linear(W: torch.Tensor, group: int, grid: int, shrink_min: float, norm: float, device):
    out, inn = W.shape
    assert inn % group == 0 and out % PACK == 0, f"bad shape {tuple(W.shape)} @g{group}"
    s, z, q = search_groups(W.to(device=device, dtype=torch.float32), group, grid, shrink_min, norm)
    packed = pack_awq(s, z, q)
    return {k: v.cpu().contiguous() for k, v in packed.items()}


def unpack_awq(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor):
    """Reference reader (same algebra as SGLang's unpack_cols + undo-interleave) -> W [out, in] fp32."""
    def unpack_cols(packed):
        rows, nw = packed.shape
        p = packed.to(torch.int64) & 0xFFFFFFFF
        nib = torch.stack([(p >> (W_BIT * i)) & MAX_Q for i in range(PACK)], dim=-1)  # [rows, nw, 8]
        undo = torch.argsort(torch.tensor(AWQ_INTERLEAVE))
        return nib[:, :, undo].reshape(rows, nw * PACK)
    q = unpack_cols(qweight).float()          # [in, out]
    z = unpack_cols(qzeros).float()           # [in/g, out]
    g = q.shape[0] // z.shape[0]
    deq = (q.view(-1, g, q.shape[1]) - z[:, None, :]) * scales.float()[:, None, :]
    return deq.reshape(q.shape).t().contiguous()


# ----------------------------------------------------------------------------- self-test
def selftest(group: int, device: str) -> None:
    torch.manual_seed(0)
    W = torch.randn(256, 512, dtype=torch.bfloat16) * 0.02
    W[:, :64] *= 0  # a dead group: scale floor path, must dequantize to exactly 0
    s, z, q = search_groups(W.to(device, torch.float32), group, 24, 0.7, 2.4)
    packed = {k: v.cpu() for k, v in pack_awq(s, z, q).items()}
    deq = unpack_awq(**packed)
    ref = ((q.float() - z.float().repeat_interleave(group, dim=1)) * s.float().repeat_interleave(group, dim=1)).cpu()
    assert torch.equal(deq, ref), "pack/unpack round trip mismatch"
    assert torch.all(deq[:, :64] == 0), "dead group did not dequantize to 0"
    err = (deq - W.float()).pow(2).mean().sqrt() / W.float().pow(2).mean().sqrt()
    print(f"selftest: round trip exact; relative RMS error {err:.4f} (g{group}, 4-bit)")
    try:
        from awq.modules.linear.gemm import WQLinear_GEMM  # awq-quant env only
    except ImportError:
        print("selftest: AutoAWQ not importable here — WQLinear_GEMM bitwise check skipped (run in awq-quant)")
    else:
        lin = torch.nn.Linear(W.shape[1], W.shape[0], bias=False)
        lin.weight.data = ref.clone()  # on-grid weights: AutoAWQ's intweight == q exactly
        wq = WQLinear_GEMM.from_linear(
            lin, W_BIT, group, scales=s.cpu().float().t().contiguous(), zeros=z.cpu().float().t().contiguous()
        )
        assert torch.equal(wq.qweight.to(torch.int32), packed["qweight"]), "qweight differs from AutoAWQ"
        assert torch.equal(wq.qzeros.to(torch.int32), packed["qzeros"]), "qzeros differs from AutoAWQ"
        assert torch.equal(wq.scales.half(), packed["scales"]), "scales differ from AutoAWQ"
        print("selftest: bitwise identical to AutoAWQ WQLinear_GEMM")
    try:
        from sglang.srt.layers.quantization.utils import unpack_cols as sgl_unpack  # sglang env only
    except ImportError:
        print("selftest: sglang not importable here — unpack_cols check skipped (run in the serving env)")
    else:
        import numpy as np
        size_k, size_n = W.shape[1], W.shape[0]
        qq = sgl_unpack(packed["qweight"], W_BIT, size_k, size_n)
        undo = np.argsort(np.array(AWQ_INTERLEAVE))
        qq = qq.reshape(-1, PACK)[:, undo].reshape(size_k, size_n)
        assert torch.equal(qq.to(torch.int64).cpu(), q.t().to(torch.int64).cpu()), "SGLang unpack_cols disagrees"
        print("selftest: SGLang unpack_cols + undo-interleave reproduces q")


# ----------------------------------------------------------------------------- driver
def read_headers(src: str) -> "OrderedDict[str, tuple[str, list[int], str]]":
    names: "OrderedDict[str, tuple[str, list[int], str]]" = OrderedDict()
    for fn in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        with open(fn, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                names[k] = (v["dtype"], v["shape"], os.path.basename(fn))
    return names


def layer_key(name: str):
    m = LAYER_RE.match(name)
    return (int(m.group(1)) if m else 10**6, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.join(os.environ.get("MODELS_DIR", "/data/models"), "hf-3rdparty/flint-bf16"))
    ap.add_argument("--dst", default=os.path.join(os.environ.get("MODELS_DIR", "/data/models"), "hf-mattbucci/Flint-AWQ-INT4-g64"))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--grid", type=int, default=24, help="clipping-range candidates per group (1 = plain min-max RTN)")
    ap.add_argument("--shrink-min", type=float, default=0.7, help="smallest range multiplier searched")
    ap.add_argument("--norm", type=float, default=2.4, help="error norm power (GPTQ convention)")
    ap.add_argument("--keep-bf16", default="", help="extra regex of quantizable names to leave BF16 (e.g. 'self_attn')")
    ap.add_argument("--shard-gb", type=float, default=5.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="replace model files already in --dst")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    if args.selftest:
        selftest(args.group_size, args.device)
        return

    keep_re = re.compile(args.keep_bf16) if args.keep_bf16 else None
    names = read_headers(args.src)
    plan_q = [n for n in names if QUANT_RE.match(n) and not (keep_re and keep_re.search(n))]
    plan_q_set = set(plan_q)
    plan_k = [n for n in names if n not in plan_q_set]
    # Quantizable tensors held back by --keep-bf16 must also be named in
    # modules_to_not_convert, or the AWQ loader builds Marlin layers for them and
    # finds no qweight (rule + the layer-specific trap: modules_to_not_convert()).
    kept_q = [n for n in names if QUANT_RE.match(n) and n not in plan_q_set]
    not_convert = modules_to_not_convert([x for x in names if QUANT_RE.match(x)], kept_q)
    q_params = sum(math.prod(names[n][1]) for n in plan_q)
    k_params = sum(math.prod(names[n][1]) for n in plan_k)
    print(f"src {args.src}\ndst {args.dst}\ndevice {args.device}  g{args.group_size}  grid {args.grid}  shrink>={args.shrink_min}  norm {args.norm}")
    print(f"quantize {len(plan_q)} tensors / {q_params/1e9:.2f} G params; keep BF16 {len(plan_k)} tensors / {k_params/1e9:.2f} G params")
    est = q_params * (0.5 + 2.5 / args.group_size) + k_params * 2
    print(f"estimated output {est/2**30:.1f} GiB")
    if not_convert:
        print(f"modules_to_not_convert += {not_convert}")
    if args.dry_run:
        for n in plan_k:
            print("  keep", n, names[n][0], names[n][1])
        return

    os.makedirs(args.dst, exist_ok=True)
    stale = glob.glob(os.path.join(args.dst, "model*.safetensors*"))
    if stale and not args.overwrite:
        raise SystemExit(f"{args.dst} already holds {len(stale)} model files; pass --overwrite to replace them")
    for fn in stale:
        os.remove(fn)
    for fn in glob.glob(os.path.join(args.src, "*")):
        b = os.path.basename(fn)
        if os.path.isfile(fn) and not b.endswith((".safetensors", ".safetensors.index.json")) and b != "config.json":
            shutil.copy2(fn, os.path.join(args.dst, b))
    cfg = json.load(open(os.path.join(args.src, "config.json")))
    cfg.pop("quantization", None)
    cfg.pop("mlx_export", None)
    cfg["quantization_config"] = {
        "quant_method": "awq",
        "bits": W_BIT,
        "group_size": args.group_size,
        "zero_point": True,
        "version": "gemm",
        "modules_to_not_convert": ["lm_head", "embed_tokens", *not_convert],
        "producer": {
            "script": "scripts/quantize/quantize_flint_rtn_awq.py",
            "method": "data-free per-group MSE search (clipping grid) on the integer-zero-point grid, from the BF16 base",
            "grid": args.grid, "shrink_min": args.shrink_min, "norm": args.norm,
            "kept_bf16": sorted({re.sub(r"\.\d+\.", ".N.", n) for n in plan_k}),
        },
    }
    json.dump(cfg, open(os.path.join(args.dst, "config.json"), "w"), indent=2)

    handles: dict[str, object] = {}

    def get(name):
        shard = names[name][2]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(args.src, shard), framework="pt")
        return handles[shard].get_tensor(name)

    shard_bytes = int(args.shard_gb * 2**30)
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    out_size = 0
    shard_idx = 0
    weight_map: dict[str, str] = {}
    total = 0

    def flush():
        nonlocal out, out_size, shard_idx
        if not out:
            return
        shard_idx += 1
        fn = f"model-{shard_idx:05d}.safetensors"
        save_file(out, os.path.join(args.dst, fn), metadata={"format": "pt"})
        # safetensors writes shards 0600 (tempfile + rename); the docker-served
        # stack reads /models as UID 10001 and would fail the boot with a
        # misleading FileNotFoundError on the first shard (2026-10-22 smoke).
        os.chmod(os.path.join(args.dst, fn), 0o644)
        for k in out:
            weight_map[k] = fn
        print(f"  wrote {fn} ({out_size/2**30:.2f} GiB, {len(out)} tensors)", flush=True)
        out, out_size = OrderedDict(), 0

    t0 = time.time()
    order = sorted(names, key=layer_key)
    n_q = 0
    for i, name in enumerate(order):
        W = get(name)
        if name in plan_q_set:
            packed = quantize_linear(W, args.group_size, args.grid, args.shrink_min, args.norm, args.device)
            base = name[: -len(".weight")]
            for suffix, t in packed.items():
                out[f"{base}.{suffix}"] = t
                out_size += t.numel() * t.element_size()
            n_q += 1
        else:
            t = W.to(torch.bfloat16).contiguous()
            out[name] = t
            out_size += t.numel() * t.element_size()
        total += 1
        if out_size >= shard_bytes:
            flush()
        if n_q and n_q % 2000 == 0 and name in plan_q_set:
            print(f"  {n_q}/{len(plan_q)} quantized, {time.time()-t0:.0f}s", flush=True)
    flush()
    for f in handles.values():
        del f
    total_size = 0
    for fn in glob.glob(os.path.join(args.dst, "model-*.safetensors")):
        total_size += os.path.getsize(fn)
    json.dump(
        {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(weight_map.items()))},
        open(os.path.join(args.dst, "model.safetensors.index.json"), "w"),
        indent=2,
    )
    print(f"done: {total} tensors ({n_q} quantized) in {time.time()-t0:.0f}s -> {args.dst} ({total_size/2**30:.1f} GiB)")


if __name__ == "__main__":
    main()
