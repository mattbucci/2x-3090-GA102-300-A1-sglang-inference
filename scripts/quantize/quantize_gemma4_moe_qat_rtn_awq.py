#!/usr/bin/env python3
"""Gemma 4 26B MoE (parallel dense+MoE) AWQ-Marlin W4A16 from the QAT base — RTN pack.

The 26B is QAT-trained (google/gemma-4-26B-A4B-it-qat-q4_0-unquantized), so RTN to AWQ
is near-lossless AND — unlike GPTQ — QAT conditions EVERY expert, not just the hot ones,
so the rare-expert under-calibration that plagues GPTQ MoE simply doesn't apply. Data-free.

Layout (verified): 30 layers, each a PARALLEL dense+MoE block. Per layer:
  - attention q/k/v/o_proj                          (2-D)  -> AWQ gs=32
  - dense mlp.{gate,up,down}_proj                    (2-D)  -> AWQ gs=32
  - experts.gate_up_proj  [128, 1408, 2816]  (FUSED 3-D)   -> per-expert unfused AWQ
      gate = rows[0:704], up = rows[704:1408]
  - experts.down_proj     [128, 2816, 704]   (FUSED 3-D)   -> per-expert unfused AWQ
  - router.{proj.weight,scale,per_expert_scale}            -> BF16
  - vision_tower / embed_vision / norms / embed_tokens     -> BF16
Output experts match the shipped mattbucci/gemma-4-26B-AWQ format:
  experts.{e}.{gate,up,down}_proj.{qweight,qzeros,scales}  (gs=32).
GROUP_SIZE MUST be 32 (experts.down_proj in=704 is /32 not /128).

Usage (CPU, no GPU):
  SRC=/data/models/gemma-4-26B-A4B-it-qat-q4_0-unquantized OUT=/data/models/gemma-4-26B-AWQ-QAT \
    /home/letsrtfm/miniforge3/envs/awq-quant/bin/python scripts/quantize/quantize_gemma4_moe_qat_rtn_awq.py
"""
from __future__ import annotations
import os, json, glob, shutil, re, sys
import torch, torch.nn as nn
from collections import OrderedDict
from safetensors import safe_open
from safetensors.torch import save_file

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
SRC = os.environ.get("SRC", "/data/models/gemma-4-26B-A4B-it-qat-q4_0-unquantized")
OUT = os.environ.get("OUT", "/data/models/gemma-4-26B-AWQ-QAT")
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "32"))   # experts.down_proj in=704 -> must be 32
W_BIT = 4
SHARD_BYTES = int(os.environ.get("SHARD_GB", "5")) * (1024**3)

from awq.modules.linear.gemm import WQLinear_GEMM
from awq.quantize.quantizer import AwqQuantizer
class _Q:
    group_size = GROUP_SIZE; w_bit = W_BIT; zero_point = True
pseudo = AwqQuantizer.pseudo_quantize_tensor.__get__(_Q(), _Q)

def pack_linear(W):
    """W: [out, in] (any dtype) -> {qweight,qzeros,scales} AWQ-GEMM (gs=GROUP_SIZE)."""
    W = W.to(torch.float32)
    out_f, in_f = W.shape
    assert in_f % GROUP_SIZE == 0 and out_f % 8 == 0, f"bad shape {out_f}x{in_f} @gs{GROUP_SIZE}"
    _, scales, zeros = pseudo(W)                       # [out, in/G]
    lin = nn.Linear(in_f, out_f, bias=False); lin.weight.data = W
    wq = WQLinear_GEMM.from_linear(lin, W_BIT, GROUP_SIZE,
                                   scales=scales.t().contiguous(), zeros=zeros.t().contiguous())
    return {"qweight": wq.qweight.to(torch.int32).contiguous(),
            "qzeros":  wq.qzeros.to(torch.int32).contiguous(),
            "scales":  wq.scales.to(torch.float16).contiguous()}

# 2-D quant targets: attention + dense MLP projections (NOT experts/router/vision).
PROJ_RE = re.compile(r"language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$")
# NOTE: the fused expert tensors are stored as nn.Parameter (no `.weight` suffix):
#   model.language_model.layers.N.experts.gate_up_proj   [E, 2*inter, hidden]
#   model.language_model.layers.N.experts.down_proj      [E, hidden, inter]
EXPERTS_GATEUP_RE = re.compile(r"language_model\.layers\.\d+\.experts\.gate_up_proj$")
EXPERTS_DOWN_RE   = re.compile(r"language_model\.layers\.\d+\.experts\.down_proj$")
KEEP_BF16_HINTS = ("vision_tower", "embed_vision", "multi_modal_projector", "router", "audio_tower")

print(f"SRC: {SRC}\nOUT: {OUT}\ngroup_size={GROUP_SIZE}")
os.makedirs(OUT, exist_ok=True)

# copy non-weight files (config handled separately)
for fn in glob.glob(f"{SRC}/*"):
    b = os.path.basename(fn)
    if b.endswith(".safetensors") or b.endswith(".safetensors.index.json") or b == "config.json":
        continue
    if os.path.isfile(fn):
        shutil.copy2(fn, os.path.join(OUT, b))

idx = glob.glob(f"{SRC}/*.safetensors.index.json")
if idx:
    wmap = json.load(open(idx[0]))["weight_map"]
else:
    wmap = {}
    for sp in glob.glob(f"{SRC}/model*.safetensors"):
        with safe_open(sp, framework="pt") as f:
            for k in f.keys(): wmap[k] = os.path.basename(sp)
handles = {}
def get(key):
    s = wmap[key]
    if s not in handles: handles[s] = safe_open(os.path.join(SRC, s), framework="pt")
    return handles[s].get_tensor(key)

out_tensors = OrderedDict()
n_proj = n_exp = n_keep = 0

for key in list(wmap.keys()):
    if any(h in key for h in KEEP_BF16_HINTS):
        out_tensors[key] = get(key); n_keep += 1; continue

    m_gu = EXPERTS_GATEUP_RE.search(key)
    m_dn = EXPERTS_DOWN_RE.search(key)
    if PROJ_RE.search(key):
        base = key[:-len(".weight")]
        for t, v in pack_linear(get(key)).items():
            out_tensors[f"{base}.{t}"] = v
        n_proj += 1
    elif m_gu:
        prefix = key.rsplit(".", 1)[0]                  # model...layers.N.experts (keep model.)
        fused = get(key)                                # [E, 2*inter, hidden]
        E = fused.shape[0]; half = fused.shape[1] // 2
        for e in range(E):
            we = fused[e]                               # [2*inter, hidden]
            for name, w in (("gate_proj", we[:half]), ("up_proj", we[half:])):
                for t, v in pack_linear(w).items():
                    out_tensors[f"{prefix}.{e}.{name}.{t}"] = v
            n_exp += 1
        del fused
    elif m_dn:
        prefix = key.rsplit(".", 1)[0]                  # model...layers.N.experts (keep model.)
        fused = get(key)                                # [E, hidden, inter]
        for e in range(fused.shape[0]):
            for t, v in pack_linear(fused[e]).items():
                out_tensors[f"{prefix}.{e}.down_proj.{t}"] = v
        del fused
    else:
        out_tensors[key] = get(key); n_keep += 1
    if (n_proj + n_exp) and (n_proj + n_exp) % 200 == 0:
        print(f"  packed {n_proj} dense + {n_exp} expert-projs...")

print(f"packed {n_proj} dense/attn linears + {n_exp} expert-proj-sets, kept {n_keep} BF16")

# shard + write
shard_id, cur, cur_bytes, index = 1, OrderedDict(), 0, {}
def flush():
    global shard_id, cur, cur_bytes
    if not cur: return
    name = f"model-{shard_id:05d}-of-XXXXX.safetensors"
    save_file(cur, os.path.join(OUT, name), metadata={"format": "pt"})
    for k in cur: index[k] = name
    print(f"  wrote {name} ({cur_bytes/1e9:.2f} GB)")
    shard_id += 1; cur = OrderedDict(); cur_bytes = 0
for k, t in out_tensors.items():
    nb = t.numel() * t.element_size()
    if cur_bytes + nb > SHARD_BYTES and cur: flush()
    cur[k] = t; cur_bytes += nb
flush()
total = shard_id - 1
final_index = {k: f"model-{int(n.split('-')[1]):05d}-of-{total:05d}.safetensors" for k, n in index.items()}
for sid in range(1, total + 1):
    s = os.path.join(OUT, f"model-{sid:05d}-of-XXXXX.safetensors")
    if os.path.exists(s): os.rename(s, os.path.join(OUT, f"model-{sid:05d}-of-{total:05d}.safetensors"))
json.dump({"metadata": {"total_size": sum(t.numel()*t.element_size() for t in out_tensors.values())},
           "weight_map": final_index},
          open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=2)

cfg = json.load(open(f"{SRC}/config.json"))
cfg["quantization_config"] = {
    "quant_method": "awq", "bits": W_BIT, "group_size": GROUP_SIZE,
    "zero_point": True, "version": "gemm",
    "modules_to_not_convert": ["model.vision_tower", "model.embed_vision"],
}
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)
print(f"\nDONE. {total} shard(s) -> {OUT}\nNext: scripts/eval/check_awq_scales.py {OUT}")
