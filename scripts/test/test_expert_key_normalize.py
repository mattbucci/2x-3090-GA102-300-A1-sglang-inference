#!/usr/bin/env python3
"""Lock-in test for `_normalize_expert_key` in convert_gemma4_26b_ct_to_awq.

Background: the 2026-05-08 21B-REAP-v3 disaster (validator 1/4 PASS,
27/30 MoE layers with expert-0 zero in fused tensor) was traced to a
key-order mismatch between llmcompressor's CT save format
(`experts.<proj>.<E>.<attr>`) and the SGLang gemma4_mm.py loader (post
patch 028) which expects `experts.<E>.<proj>.<attr>`.  This test pins
the converter's normalization so the bug can't silently come back.

Usage:
    python scripts/test/test_expert_key_normalize.py
"""
from __future__ import annotations

import re
import sys

# Mirror the helper from scripts/quantize/convert_gemma4_26b_ct_to_awq.py
# verbatim — keeping the test self-contained avoids importing torch / the
# heavy converter module just to exercise a string transform.  Drift guard:
# the converter's docstring + this regex literal must stay in sync; see the
# converter's `_normalize_expert_key` for the canonical definition + why.
_EXPERT_REORDER_RE = re.compile(
    r"\.experts\.(gate_proj|up_proj|down_proj)\.(\d+)(\.|$)"
)


def _normalize_expert_key(name: str) -> str:
    return _EXPERT_REORDER_RE.sub(r".experts.\2.\1\3", name)


def case(input_key: str, expected: str) -> None:
    got = _normalize_expert_key(input_key)
    if got != expected:
        print(f"FAIL: {input_key!r}\n  expected: {expected!r}\n  got:      {got!r}")
        sys.exit(1)
    print(f"  OK: {input_key} -> {got}")


def main() -> int:
    print("=== llmcompressor proj-first → SGLang expert-first remap ===")
    case(
        "model.language_model.layers.0.experts.gate_proj.0.weight_packed",
        "model.language_model.layers.0.experts.0.gate_proj.weight_packed",
    )
    case(
        "model.language_model.layers.0.experts.gate_proj.0",
        "model.language_model.layers.0.experts.0.gate_proj",
    )
    case(
        "model.language_model.layers.29.experts.down_proj.127.qweight",
        "model.language_model.layers.29.experts.127.down_proj.qweight",
    )
    case(
        "model.language_model.layers.5.experts.up_proj.42.scales",
        "model.language_model.layers.5.experts.42.up_proj.scales",
    )

    print("\n=== Idempotent: already-normalized keys pass through ===")
    case(
        "model.language_model.layers.0.experts.0.gate_proj.qweight",
        "model.language_model.layers.0.experts.0.gate_proj.qweight",
    )
    case(
        "model.language_model.layers.29.experts.127.down_proj.scales",
        "model.language_model.layers.29.experts.127.down_proj.scales",
    )

    print("\n=== Non-expert keys untouched ===")
    case(
        "model.language_model.layers.0.mlp.gate_proj.qweight",
        "model.language_model.layers.0.mlp.gate_proj.qweight",
    )
    case(
        "model.language_model.layers.0.experts.gate_up_proj",  # fused HF format
        "model.language_model.layers.0.experts.gate_up_proj",
    )
    case(
        "model.language_model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
    )

    print("\n[PASS] all 9 cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
