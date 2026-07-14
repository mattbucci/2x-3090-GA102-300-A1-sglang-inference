# Triton MoE config tuning for nemotron3-omni (sm_86) — NULL (2026-07-14)

Port of R9700's model-aware MoE tuning method (their +5.1%/+11.2% on gfx1201). Target: the one
3090 preset on the tunable Triton MoE path — `nemotron3-omni` (moe_wna16, E=128, per-rank N=464,
int4_w4a16, group 64). Exhaustive 640-config search at M=1 (decode) + M=512 (prefill anchor) via
`benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`, merged config installed to
`configs/triton_3_6_0/`, A/B on the standard instrument (TP=2 @262144, M=1 fresh prefill).

| arm | 2K | 131K | 250K | TTFT@250K |
|---|---:|---:|---:|---:|
| heuristic defaults | 102.5 tok/s | 98.2 | 98.1 | 728 ms |
| tuned (M=1+512 merged) | 101.7 tok/s | 97.5 | 97.5 | 846 ms |

**Verdict: NULL, marginally negative (decode −0.6…−0.8%, TTFT +16…33%).** The tuned config was
uninstalled. Interpretation: sglang's heuristic default configs for this shape are already
at-or-near optimal on Ampere at M=1 — R9700's win came from gfx1201 defaults being poor, not from
headroom that exists on every arch. Their arch-specificity hedge is confirmed for the third and
final transferable-looking lever of their campaign. The full default-grid tune was also mis-sized
for the question (7h in, 10h+ projected — killed): for M=1-mission presets, tune {1, chunked-prefill}
only.

Harness fixes required to tune this model at all (in the serving tree's
`benchmark/kernels/fused_moe_triton/common_utils.py`, upstream-PR candidates):
1. composite AVLM configs (`llm_config` nesting, no top-level `hidden_size`) — descend before
   `text_config`; add `NemotronH_Nano_Omni_Reasoning_V3` arch branch (E=`n_routed_experts`,
   topk=`num_experts_per_tok`, N=`moe_intermediate_size`);
2. AWQ-style flat `quantization_config` (`bits`/`group_size`) — derive `block_shape=[0, group_size]`
   (the script only knew fp8 `weight_block_size` and CT `config_groups`).

Receipts: `nemotron-moe-tune-null-{baseline,tuned}-2026-07-14.json`.
