# nemotron3-omni: where the v0.5.23 decode gap sits (issue #5)

Torch-profiler traces of 179 decode steps on each stack (v0.5.22 env vs v0.5.23 env, same checkpoint, same preset), plus one v0.5.23 boot with a launch-pointer dump.

| | v0.5.22 | v0.5.23 |
|---|---|---|
| GPU span per decode step | 10.40 ms | 11.31 ms |
| routed MoE gate/up launch, rank 0, sum over 23 layers | 3.31 ms | 5.03 ms |
| same, rank 1 | 4.51 ms | 4.12 ms |
| routed MoE down launch (both ranks) | 1.53 / 1.56 ms | 1.53 / 1.55 ms |

- The whole gap is the gate/up launch of `fused_moe_kernel_gptq_awq` on rank 0. Mamba, attention and the down launch are identical.
- The launch time is quantized: 124 µs + k × 37 µs, k = 0..6. It changes per token within one layer.
- It is not the kernel source (env-gated inactive-expert check A/B: null, tripwire receipts `benchmarks/regression/nemotron3-omni-2026-11-14--icheck-*.json`), not the launch arguments (shapes, dtypes, strides, alignment, config identical on both stacks), not weight placement (pointer dump: every layer has the same alignment pattern), and not the routed experts (the level of the same launch on the two TP ranks is uncorrelated, Pearson 0.00, although both ranks route to the same experts).
- It is also not proportional to overlap time with the other stream (`level-vs-overlap.txt`).
- What runs concurrently: upstream's `_forward_core_shared_routed_overlap` puts the routed experts on an alternate stream while the shared expert runs on the main stream. The shared expert is on the AWQ dequantize + cuBLAS path (`dequantize_weights` 46 launches per step, about 1 ms), not Marlin.
- v0.5.22 rank 1 already sat at the slow level mix; v0.5.22 rank 0 was the outlier on the fast side. Step time follows the slower rank per layer, so the regression is rank 0 losing its fast phase.

**Serial path A/B (patch 090, `SGLANG_NEMOTRON_MOE_STREAM_OVERLAP`), two interleaved tripwire boots per arm, tok/s at 1K / 32K / 262K:**

| arm | boot 1 | boot 2 |
|---|---|---|
| overlap (upstream) | 91.5 / 93.8 / 89.8 | 92.4 / 94.9 / 89.6 |
| serial | 115.9 / 114.9 / 107.0 | 116.3 / 114.7 / 106.8 |

Receipts: `benchmarks/regression/nemotron3-omni-2026-11-14--ovl-*.json`. The stream overlap was the cost on both stacks; v0.5.22 only hid part of it on rank 0.

Untested: shared expert on Marlin instead of dequantize + cuBLAS (about 1 ms per step).

Files: `anatomy-*.txt` (step anatomy per stack), `per-layer-per-rank.txt`, `level-vs-overlap.txt`, `level-cross-rank.txt`, and the scripts that produced them.
