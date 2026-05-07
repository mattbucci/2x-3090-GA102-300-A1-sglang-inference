#!/usr/bin/env python3
"""Static verification that patch 028's per-expert AWQ mapping handles
realistic R9700 mattbucci/gemma-4-26B-AWQ checkpoint key naming.

Replicates upstream v0.5.11 FusedMoE.make_expert_params_mapping verbatim
so the test runs without an installed SGLang / GPU. If SGLang's logic ever
changes upstream, regenerate this from
python/sglang/srt/layers/moe/fused_moe_triton/layer.py.

Usage:
    python scripts/test/test_gemma4_per_expert_mapping.py

Exit non-zero on any unmatched key.  Used as a pre-flight before relying on
patch 028 for a per-expert AWQ Gemma 4 load.
"""
from __future__ import annotations

import sys


def make_expert_params_mapping(
    ckpt_gate_proj_name: str,
    ckpt_down_proj_name: str,
    ckpt_up_proj_name: str,
    num_experts: int,
):
    return [
        (
            (
                "experts.w13_"
                if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                else "experts.w2_"
            ),
            f"experts.{expert_id}.{weight_name}.",
            expert_id,
            shard_id,
        )
        for expert_id in range(num_experts)
        for shard_id, weight_name in [
            ("w1", ckpt_gate_proj_name),
            ("w2", ckpt_down_proj_name),
            ("w3", ckpt_up_proj_name),
        ]
    ]


def insert_moe_prefix(name: str) -> str:
    """Mirror gemma4_mm.py:load_weights line 747-748:
    `if .experts. in name and .moe.experts. not in name: insert .moe.`
    R9700's HF mirror keys lack `.moe.` so the loader inserts it before
    matching against expert mappings.
    """
    if ".experts." in name and ".moe.experts." not in name:
        return name.replace(".experts.", ".moe.experts.")
    return name


def assert_matches(num_experts: int, sample_keys: list[str]) -> None:
    mapping = make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=num_experts,
    )
    expected = num_experts * 3
    assert len(mapping) == expected, f"mapping size {len(mapping)} != {expected}"

    print(f"[INFO] {num_experts} experts × 3 projs = {expected} mapping tuples")

    matches = 0
    for raw_key in sample_keys:
        # Step 1: loader's .moe. insertion (gemma4_mm.py:747-748).
        key = insert_moe_prefix(raw_key)
        # Step 2: per-expert matcher (patch 028).
        matched = False
        for param_name, weight_name, expert_id, shard_id in mapping:
            if weight_name in key:
                remapped = key.replace(weight_name, param_name)
                print(
                    f"  {raw_key:75s} -> {remapped}  (shard={shard_id} e={expert_id})"
                )
                matches += 1
                matched = True
                break
        if not matched:
            raise AssertionError(f"unmatched key: {raw_key}")
    assert matches == len(sample_keys), f"only matched {matches}/{len(sample_keys)}"
    print(f"[PASS] {matches}/{len(sample_keys)} keys matched + remapped")


# Gemma 4 26B has 128 experts.  Real keys sampled from the R9700 HF mirror
# `mattbucci/gemma-4-26B-AWQ` `model.safetensors.index.json` (Range-fetched
# 2026-05-08; total 35923 keys, of which ~34560 are per-expert MoE).  These
# are the pre-`.moe.`-insertion form the loader actually receives — patch 028
# relies on the existing `.experts.` → `.moe.experts.` rename at line 747-748
# happening first.
GEMMA4_26B_KEYS = [
    "model.language_model.layers.0.experts.0.down_proj.qweight",
    "model.language_model.layers.0.experts.0.down_proj.qzeros",
    "model.language_model.layers.0.experts.0.down_proj.scales",
    "model.language_model.layers.0.experts.0.gate_proj.qweight",
    "model.language_model.layers.0.experts.0.up_proj.qweight",
    "model.language_model.layers.0.experts.10.down_proj.qweight",
    "model.language_model.layers.0.experts.100.down_proj.qweight",
    "model.language_model.layers.5.experts.42.gate_proj.qweight",
    "model.language_model.layers.29.experts.127.down_proj.scales",
]

# Gemma 4 21B-REAP variant has fewer experts post-pruning (Cerebras REAP at
# ~80% retention from 128 → ~103). Verify smaller-num_experts also works.
GEMMA4_21B_REAP_KEYS = [
    "language_model.layers.0.moe.experts.0.gate_proj.qweight",
    "language_model.layers.10.moe.experts.50.up_proj.scales",
    "language_model.layers.20.moe.experts.102.down_proj.qzeros",
]


def assert_no_match(num_experts: int, fused_keys: list[str]) -> None:
    """Negative test: per-expert mapping must NOT match fused-format keys
    (e.g., `experts.gate_up_proj` from HF transformers checkpoints).
    Patch 028 is layered: fused keys MUST fall through the per-expert
    matcher unmatched so the original fused path handles them.
    """
    mapping = make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=num_experts,
    )
    for raw_key in fused_keys:
        key = insert_moe_prefix(raw_key)
        for param_name, weight_name, expert_id, shard_id in mapping:
            if weight_name in key:
                raise AssertionError(
                    f"FUSED KEY MATCHED PER-EXPERT MAPPING (regression): "
                    f"key={raw_key} weight_name={weight_name}"
                )
        print(f"  {raw_key:75s} OK (no per-expert match — falls through to fused path)")
    print(f"[PASS] {len(fused_keys)}/{len(fused_keys)} fused keys correctly bypass per-expert path")


# HF transformers fused-experts checkpoints carry these keys (one tensor
# per layer × all experts on output dim).  Patch 028 must NOT match these
# in the per-expert matcher; the fused-mapping path handles them.
GEMMA4_FUSED_KEYS = [
    "model.language_model.layers.0.experts.gate_up_proj",
    "model.language_model.layers.0.experts.down_proj",
    "model.language_model.layers.5.experts.gate_up_proj",
    "model.language_model.layers.29.experts.down_proj",
]


def main() -> int:
    print("=== Gemma 4 26B (128 experts) — per-expert source ===")
    assert_matches(128, GEMMA4_26B_KEYS)
    print()
    print("=== Gemma 4 21B-REAP (103 experts) — per-expert source ===")
    assert_matches(103, GEMMA4_21B_REAP_KEYS)
    print()
    print("=== Negative test: HF fused source must bypass per-expert matcher ===")
    assert_no_match(128, GEMMA4_FUSED_KEYS)
    print()
    print("All checks passed — patch 028 mapping is correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
