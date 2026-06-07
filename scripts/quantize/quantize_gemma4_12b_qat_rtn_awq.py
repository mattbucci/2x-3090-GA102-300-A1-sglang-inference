#!/usr/bin/env python3
"""Gemma 4 *unified* 12B AWQ-Marlin W4A16 from the QAT-unquantized base — RTN pack.

The base `google/gemma-4-12B-it-qat-q4_0-unquantized` is QAT-trained (weights already
sit on a 4-bit grid), so a data-free RTN re-quantization to AWQ group-128 is
near-lossless — no calibration corpus or model load needed. We also CAN'T load the
arch the normal way: `gemma4_unified` needs transformers 5.10.dev but our quant env
ships 5.5.4 (no `gemma4_unified`). So we pack straight from the safetensors tensors
using AutoAWQ's tested primitives (`pseudo_quantize_tensor` + `WQLinear_GEMM.from_linear`),
which emit the canonical AWQ-GEMM format SGLang's awq_marlin loader repacks at load.

Quantize ONLY the text-decoder Linears (q/k/v/o_proj, mlp gate/up/down_proj under
language_model.layers). Keep BF16 (modality + numerics preservation per CLAUDE.md):
  - vision_embedder.* / embed_vision.* / embed_audio.*  (encoder-free mm embedders)
  - every *_norm / norm (RMSNorm)
  - embed_tokens / lm_head (tied)
attention_k_eq_v means full-attention layers ship only k_proj (no v_proj) — handled
naturally since we quantize whatever *_proj.weight tensors exist.

Usage (CPU, ~10 min, no GPU — safe alongside nothing-else):
    /home/letsrtfm/miniforge3/envs/awq-quant/bin/python \
        scripts/quantize/quantize_gemma4_12b_qat_rtn_awq.py
Env: SRC (QAT base dir), OUT (AWQ output dir), GROUP_SIZE (128).
Next: scripts/eval/check_awq_scales.py <OUT>  then boot via launch.sh.
"""
from __future__ import annotations
import os, json, glob, shutil, re, sys
import torch
import torch.nn as nn
from collections import OrderedDict, defaultdict
from safetensors import safe_open
from safetensors.torch import save_file

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU pack, no GPU contention

SRC = os.environ.get("SRC", "/data/models/gemma-4-12B-it-qat-q4_0-unquantized")
OUT = os.environ.get("OUT", "/data/models/gemma-4-12B-it-AWQ")
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "128"))
W_BIT = 4
SHARD_BYTES = int(os.environ.get("SHARD_GB", "5")) * (1024**3)

from awq.modules.linear.gemm import WQLinear_GEMM
from awq.quantize.quantizer import AwqQuantizer

# minimal shim to call the (otherwise instance-bound) RTN tensor quantizer
class _Q:
    group_size = GROUP_SIZE
    w_bit = W_BIT
    zero_point = True
_q = _Q()
pseudo = AwqQuantizer.pseudo_quantize_tensor.__get__(_q, _Q)

# Which tensors to quantize: text-decoder projections only.
PROJ_RE = re.compile(r"language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$")
KEEP_BF16_HINTS = ("vision_embedder", "embed_vision", "embed_audio")

def is_quant_target(key: str) -> bool:
    if any(h in key for h in KEEP_BF16_HINTS):
        return False
    return bool(PROJ_RE.search(key))

print(f"SRC: {SRC}\nOUT: {OUT}\ngroup_size={GROUP_SIZE} w_bit={W_BIT}")
os.makedirs(OUT, exist_ok=True)

# --- copy all non-weight files (config handled separately below) ---
for fn in glob.glob(f"{SRC}/*"):
    b = os.path.basename(fn)
    if b.endswith(".safetensors") or b.endswith(".safetensors.index.json") or b == "config.json":
        continue
    if os.path.isfile(fn):
        shutil.copy2(fn, os.path.join(OUT, b)); print(f"  copied {b}")

# --- open source shards ---
shards = sorted(glob.glob(f"{SRC}/model*.safetensors"))
idx_path = f"{SRC}/model.safetensors.index.json"
if os.path.exists(idx_path):
    wmap = json.load(open(idx_path))["weight_map"]
else:
    wmap = {}
    for sp in shards:
        with safe_open(sp, framework="pt") as f:
            for k in f.keys(): wmap[k] = os.path.basename(sp)
handles = {}
def get(key):
    s = wmap[key]
    if s not in handles: handles[s] = safe_open(os.path.join(SRC, s), framework="pt")
    return handles[s].get_tensor(key)

all_keys = list(wmap.keys())
n_q = n_keep = 0
not_convert = set()
out_tensors = OrderedDict()

for key in all_keys:
    if is_quant_target(key):
        W = get(key).to(torch.float32)            # [out, in]
        out_f, in_f = W.shape
        if in_f % GROUP_SIZE != 0 or out_f % 8 != 0:
            # can't group/pack cleanly -> keep BF16 and record
            out_tensors[key] = get(key)
            not_convert.add(key.rsplit(".", 1)[0])
            n_keep += 1
            print(f"  KEEP-BF16 (indivisible {out_f}x{in_f}) {key}")
            continue
        _, scales, zeros = pseudo(W)              # pseudo returns [out, in/G]
        lin = nn.Linear(in_f, out_f, bias=False)
        lin.weight.data = W
        # from_linear indexes scales/zeros by group along dim0 -> needs [in/G, out]
        wq = WQLinear_GEMM.from_linear(
            lin, W_BIT, GROUP_SIZE,
            scales=scales.t().contiguous(), zeros=zeros.t().contiguous(),
        )
        base = key[: -len(".weight")]
        out_tensors[f"{base}.qweight"] = wq.qweight.to(torch.int32).contiguous()
        out_tensors[f"{base}.qzeros"]  = wq.qzeros.to(torch.int32).contiguous()
        out_tensors[f"{base}.scales"]  = wq.scales.to(torch.float16).contiguous()
        n_q += 1
        if n_q % 50 == 0: print(f"  quantized {n_q} linears...")
        del W, wq, lin
    else:
        out_tensors[key] = get(key)               # BF16 passthrough
        if any(h in key for h in KEEP_BF16_HINTS) or key.endswith((".weight", ".bias")):
            if "_proj." not in key:
                not_convert.add(key.rsplit(".", 1)[0])
        n_keep += 1

print(f"quantized {n_q} linears, kept {n_keep} tensors BF16")

# --- shard + write ---
shard_id, cur, cur_bytes, index = 1, OrderedDict(), 0, {}
def flush(last=False):
    global shard_id, cur, cur_bytes
    if not cur: return
    name = f"model-{shard_id:05d}-of-XXXXX.safetensors"
    save_file(cur, os.path.join(OUT, name), metadata={"format": "pt"})
    for k in cur: index[k] = name
    print(f"  wrote {name} ({cur_bytes/1e9:.2f} GB, {len(cur)} tensors)")
    shard_id += 1; cur = OrderedDict(); cur_bytes = 0
for k, t in out_tensors.items():
    nb = t.numel() * t.element_size()
    if cur_bytes + nb > SHARD_BYTES and cur:
        flush()
    cur[k] = t; cur_bytes += nb
flush()
# rename shards to final count
total = shard_id - 1
final_index = {}
for k, name in index.items():
    sid = int(name.split("-")[1])
    final = f"model-{sid:05d}-of-{total:05d}.safetensors"
    final_index[k] = final
for sid in range(1, total + 1):
    src = os.path.join(OUT, f"model-{sid:05d}-of-XXXXX.safetensors")
    dst = os.path.join(OUT, f"model-{sid:05d}-of-{total:05d}.safetensors")
    if os.path.exists(src): os.rename(src, dst)
json.dump({"metadata": {"total_size": sum(t.numel()*t.element_size() for t in out_tensors.values())},
           "weight_map": final_index},
          open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=2)

# --- config.json with awq quantization_config ---
cfg = json.load(open(f"{SRC}/config.json"))
cfg["quantization_config"] = {
    "quant_method": "awq",
    "bits": W_BIT,
    "group_size": GROUP_SIZE,
    "zero_point": True,
    "version": "gemm",
    "modules_to_not_convert": sorted({
        "vision_embedder", "embed_vision", "embed_audio", "lm_head",
    }),
}
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)
print(f"\nDONE. {total} shard(s) -> {OUT}")
print("Next: scripts/eval/check_awq_scales.py", OUT)
