#!/usr/bin/env python3
"""Convert Qwen3.5 MoE REAP compressed-tensors to AWQ format for SGLang.

Handles two format issues:
1. Weight prefix: strips 'model.language_model.' -> 'model.'
   (VL model saved with language_model prefix, text-only load expects model.*)
2. Expert format: fuses per-expert weights into [E, K, N] tensors
   (CT stores experts.0.gate_proj.weight_packed, SGLang needs experts.w13_qweight[E,K,N])

Also fixes config: adds norm_topk_prob, fixes architectures to ForCausalLM.

Usage:
    python scripts/quantize/convert_qwen35_moe_reap_ct_to_awq.py
"""
import glob
import json
import os
import re
import shutil
from collections import OrderedDict, defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = os.path.expanduser("~/AI/models/Qwen3.5-28B-A3B-REAP-CT")
DST = os.path.expanduser("~/AI/models/Qwen3.5-28B-A3B-REAP-AWQ")

W_BIT = 4
PACK_FACTOR = 32 // W_BIT  # 8 values per int32
AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def unpack_int32_to_4bit(packed: torch.Tensor) -> torch.Tensor:
    """Unpack int32 tensor with 8x4-bit values (sequential) to int8."""
    unpacked = []
    for i in range(PACK_FACTOR):
        unpacked.append((packed >> (i * W_BIT)) & 0xF)
    return torch.stack(unpacked, dim=-1).reshape(*packed.shape[:-1], -1).to(torch.int8)


def pack_4bit_to_int32_awq(values: torch.Tensor) -> torch.Tensor:
    """Pack int8 tensor (0-15) into int32 with AWQ interleaved order."""
    assert values.shape[-1] % PACK_FACTOR == 0
    grouped = values.reshape(*values.shape[:-1], -1, PACK_FACTOR)
    packed = torch.zeros(*grouped.shape[:-1], dtype=torch.int32, device=values.device)
    for i in range(PACK_FACTOR):
        packed |= (grouped[..., i].to(torch.int32) & 0xF) << (AWQ_PACK_ORDER[i] * W_BIT)
    return packed


def convert_weight(packed: torch.Tensor, scale: torch.Tensor, group_size: int):
    """Convert one CT quantized weight to AWQ format.

    CT: weight_packed [out, in//8] int32 sequential, weight_scale [out, in//G] float
    AWQ: qweight [in, out//8] int32 interleaved, scales [in//G, out] fp16, qzeros [in//G, out//8] int32
    """
    out_features = packed.shape[0]
    in_features = packed.shape[1] * PACK_FACTOR

    # Unpack → transpose → repack with AWQ order
    unpacked = unpack_int32_to_4bit(packed)  # [out, in]
    unpacked_t = unpacked.T.contiguous()  # [in, out]
    qweight = pack_4bit_to_int32_awq(unpacked_t)  # [in, out//8]

    # Transpose scales, clamp to FP16
    scales = scale.T.contiguous().clamp(-65504, 65504).to(torch.float16)  # [in//G, out]

    # Symmetric qzeros (zero_point = 8)
    num_groups = in_features // group_size
    num_out_packed = out_features // PACK_FACTOR
    zp_val = torch.tensor([8], dtype=torch.int32)
    qzeros = torch.zeros((num_groups, num_out_packed), dtype=torch.int32)
    for i in range(PACK_FACTOR):
        qzeros |= (zp_val << (AWQ_PACK_ORDER[i] * W_BIT))

    return qweight, scales, qzeros


def quantize_bf16_to_awq(weight: torch.Tensor, group_size: int):
    """RTN quantize a BF16 weight to AWQ INT4 for non-quantized layers."""
    out_features, in_features = weight.shape
    w = weight.float()
    num_groups = in_features // group_size
    w_grouped = w.reshape(out_features, num_groups, group_size)
    w_max = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    scale_vals = w_max / 7.0
    q = torch.round(w_grouped / scale_vals).clamp(-8, 7).to(torch.int8) + 8
    q = q.reshape(out_features, in_features)
    q_t = q.T.contiguous()
    qweight = pack_4bit_to_int32_awq(q_t)
    scales = scale_vals.squeeze(-1).T.contiguous().clamp(-65504, 65504).to(torch.float16)
    num_out_packed = out_features // PACK_FACTOR
    zp_val = torch.tensor([8], dtype=torch.int32)
    qzeros = torch.zeros((num_groups, num_out_packed), dtype=torch.int32)
    for i in range(PACK_FACTOR):
        qzeros |= (zp_val << (AWQ_PACK_ORDER[i] * W_BIT))
    return qweight, scales, qzeros


def main():
    print(f"Source: {SRC}")
    print(f"Output: {DST}")
    os.makedirs(DST, exist_ok=True)

    # Load config
    with open(os.path.join(SRC, "config.json")) as f:
        config = json.load(f)

    qconfig = config.get("quantization_config", {})
    cg = qconfig.get("config_groups", {}).get("group_0", {})
    group_size = cg.get("weights", {}).get("group_size", 128)
    num_experts = config.get("num_experts", 0)
    print(f"Group size: {group_size}, Experts: {num_experts}")

    # Load all weights
    shard_files = sorted(glob.glob(os.path.join(SRC, "model*.safetensors")))
    if not shard_files:
        shard_files = [os.path.join(SRC, "model.safetensors")]

    all_weights = {}
    for sf_path in shard_files:
        print(f"Loading {os.path.basename(sf_path)}...")
        with safe_open(sf_path, framework="pt") as sf:
            for key in sf.keys():
                all_weights[key] = sf.get_tensor(key)

    print(f"Loaded {len(all_weights)} tensors")

    # Strip language_model prefix
    stripped = {}
    for k, v in all_weights.items():
        new_k = k.replace("model.language_model.", "model.")
        stripped[new_k] = v
    all_weights = stripped
    print(f"Stripped language_model prefix")

    # Identify expert layers and group them
    # Pattern: model.layers.X.mlp.experts.Y.{gate,up,down}_proj.weight_{packed,scale,shape}
    expert_pattern = re.compile(
        r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight_(packed|scale|shape)"
    )

    # Group expert weights by layer
    expert_data = defaultdict(lambda: defaultdict(dict))  # layer -> expert_id -> proj -> {packed, scale}
    non_expert_keys = []

    for k in sorted(all_weights.keys()):
        m = expert_pattern.match(k)
        if m:
            layer_idx, expert_idx, proj, wtype = m.groups()
            if wtype != "shape":  # skip shape metadata
                expert_data[int(layer_idx)][int(expert_idx)][f"{proj}_{wtype}"] = all_weights[k]
        else:
            non_expert_keys.append(k)

    print(f"Found {len(expert_data)} layers with experts")
    print(f"Non-expert tensors: {len(non_expert_keys)}")

    # Build output
    output = OrderedDict()

    # Process non-expert weights: convert CT packed to AWQ format
    for k in non_expert_keys:
        if k.endswith(".weight_shape"):
            continue  # skip metadata
        if k.endswith(".weight_packed"):
            base = k.replace(".weight_packed", "")
            scale_key = base + ".weight_scale"
            if scale_key in all_weights:
                # Quantized layer — convert to AWQ
                packed = all_weights[k]
                scale = all_weights[scale_key]
                out_features = packed.shape[0]
                in_features = packed.shape[1] * PACK_FACTOR
                # Skip tiny layers where output dim < pack_factor (e.g. shared_expert_gate)
                if out_features < PACK_FACTOR or in_features < PACK_FACTOR:
                    # Dequantize to fp16 instead
                    unpacked = unpack_int32_to_4bit(packed)
                    scale_exp = scale.float().repeat_interleave(group_size, dim=1)
                    if scale_exp.shape[1] > in_features:
                        scale_exp = scale_exp[:, :in_features]
                    w_float = (unpacked.float() - 8.0) * scale_exp
                    output[base + ".weight"] = w_float.to(torch.float16)
                    print(f"  Dequantized (too small for AWQ): {base} [{packed.shape} -> {w_float.shape}]")
                else:
                    qw, sc, qz = convert_weight(packed, scale, group_size)
                    output[base + ".qweight"] = qw
                    output[base + ".scales"] = sc
                    output[base + ".qzeros"] = qz
                    print(f"  Converted: {base} [{packed.shape} -> qw{qw.shape}]")
            else:
                print(f"  WARNING: {k} has no scale, passing through")
                output[k] = all_weights[k]
        elif k.endswith(".weight_scale"):
            continue  # handled with packed
        elif k.endswith(".weight"):
            # Unquantized weight (DeltaNet, embeddings, etc) — keep as-is
            output[k] = all_weights[k]
        else:
            output[k] = all_weights[k]

    # Process expert weights: per-expert AWQ format
    # SGLang's FusedMoE.make_expert_params_mapping expects per-expert names:
    #   experts.{eid}.gate_proj.qweight, experts.{eid}.up_proj.qweight, etc.
    # The weight_loader maps these to internal w13/w2 tensors with TP sharding.
    for layer_idx in sorted(expert_data.keys()):
        experts = expert_data[layer_idx]
        num_exp = max(experts.keys()) + 1
        prefix = f"model.layers.{layer_idx}.mlp.experts"

        for eid in range(num_exp):
            if eid not in experts:
                print(f"  WARNING: layer {layer_idx} missing expert {eid}, skipping")
                continue

            e = experts[eid]

            # Gate proj
            gqw, gsc, gqz = convert_weight(e["gate_proj_packed"], e["gate_proj_scale"], group_size)
            output[f"{prefix}.{eid}.gate_proj.qweight"] = gqw
            output[f"{prefix}.{eid}.gate_proj.scales"] = gsc
            output[f"{prefix}.{eid}.gate_proj.qzeros"] = gqz

            # Up proj
            uqw, usc, uqz = convert_weight(e["up_proj_packed"], e["up_proj_scale"], group_size)
            output[f"{prefix}.{eid}.up_proj.qweight"] = uqw
            output[f"{prefix}.{eid}.up_proj.scales"] = usc
            output[f"{prefix}.{eid}.up_proj.qzeros"] = uqz

            # Down proj
            dqw, dsc, dqz = convert_weight(e["down_proj_packed"], e["down_proj_scale"], group_size)
            output[f"{prefix}.{eid}.down_proj.qweight"] = dqw
            output[f"{prefix}.{eid}.down_proj.scales"] = dsc
            output[f"{prefix}.{eid}.down_proj.qzeros"] = dqz

        print(f"  Layer {layer_idx}: {num_exp} experts, per-expert AWQ "
              f"gate={gqw.shape} down={dqw.shape}")

    # Save weights
    print(f"\nSaving {len(output)} tensors to {DST}...")
    save_file(output, os.path.join(DST, "model.safetensors"))

    # Fix config
    config["architectures"] = ["Qwen3_5MoeForCausalLM"]
    config["model_type"] = "qwen3_5_moe_text"
    config.setdefault("norm_topk_prob", True)
    # Replace CT quant config with AWQ
    config["quantization_config"] = {
        "quant_method": "awq",
        "bits": 4,
        "group_size": group_size,
        "zero_point": True,
        "version": "gemm",
        "modules_to_not_convert": ["lm_head", "visual", "mlp.gate"]
    }
    with open(os.path.join(DST, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Copy tokenizer files
    for pattern in ["tokenizer*", "*.jinja", "generation_config.json", "special_tokens_map.json"]:
        for fname in glob.glob(os.path.join(SRC, pattern)):
            dst_path = os.path.join(DST, os.path.basename(fname))
            if not os.path.exists(dst_path):
                shutil.copy2(fname, dst_path)

    print(f"\nDone! AWQ model saved to {DST}")
    total_size = os.path.getsize(os.path.join(DST, "model.safetensors"))
    print(f"Model size: {total_size / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
