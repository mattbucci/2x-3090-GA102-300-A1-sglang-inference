# Where the GPU time goes on a real agentic workload (qwen38, 2×3090, 2026-09-10)

**Question.** The bake-off lanes drive the production stack with real agentic-coding traffic for
days at a time. Instead of only watching pass rates, use that traffic to answer: *what is the
2×3090 actually bottlenecked on, and which decode/prefill levers are still worth anything?*

**Setup.** `qwen38` (Qwen3.8-27B AWQ INT4 g128, 64 layers = 48 GatedDeltaNet + 16 full-attention,
vocab 248,320), SGLang v0.5.18, TP=2 over the NVLink bridge (NCCL 2.29.7, 4 channels P2P/IPC,
custom allreduce off — sm_86 graph-capture receipt `allreduce-accel-null-2026-06-15.md`), flashinfer
attention, fp8_e4m3 KV, cuda graphs bs=1, chunked prefill 8192, `--max-running-requests 1`.
Traffic = the little-coder-rtk / opencode-dcp SWE-bench Lite lanes (2.9 days, 14.6K requests) —
nothing was added to the box; every measurement below is passive except the 165 s power-cap A/B
and one 40-step torch-profiler capture (`POST /start_profile`, `activities ["CPU","GPU"]`,
`num_steps 40`, one chrome trace per TP rank).

Instruments: `/metrics` (Prometheus histograms), server-log `Decode batch … gen throughput`,
`nvidia-smi dmon -s pucvmt` (note: its PCIe/clock columns are 20 ms samples, not 1 s averages),
`nvidia-smi nvlink -gt d` counters, `nvidia-smi -q -d PERFORMANCE`, and the trace analyzer
[`scripts/bench/trace_step_anatomy.py`](../scripts/bench/trace_step_anatomy.py) (per-step sweep-line
attribution — it charges *exposed* time, so kernels hidden behind another stream are not
over-counted).

## 1. Workload shape (14,582 requests)

| metric | p50 | p90 | p99 | mean |
|---|---|---|---|---|
| prompt tokens | 28.6K | 75.6K | 137K | 35.7K |
| **uncached** prompt tokens (94% prefix-cache hit) | 450 | 5.6K | 27.5K | 2.15K |
| generated tokens | 276 | 1.8K | 5.8K | 665 |
| TTFT (streaming) | 0.49 s | 3.7 s | 19.5 s | 1.8 s |
| inter-token latency | 16.5 ms | 19.3 ms | 19.9 ms | 15.5 ms |
| queue time | 0.6 ms | 4 ms | 11.5 s | 0.37 s |

Typical turn: a few hundred new tokens on a 10–30K cached conversation, ~1 s of decode emitting a
tool call, ~0.5–1 s of GPU idle while the scaffold runs the tool, repeat. The server has a request
in flight **69.5%** of wall time; the other 30% is the client thinking — not a server lever.

**Server time budget** (sums of the latency histograms): decode (ITL) **151,796 s = 86%**, prefill
forward **18,905 s = 11%** (31.4M uncached tokens → 1,660 tok/s average; small turns are
overhead-dominated), queueing 5,362 s = 3% (p99 11.5 s — a new turn arriving while an aborted
predecessor is still being drained). **Decode is the whole game for this workload.**

## 2. Power and clocks

The 260 W cap (`gpu-cooling.service`) **binds ~100% of the time in both phases** (`pviol` 99–100%,
SM 1,500–1,560 MHz; memory stays at the P2 9,501 MHz). Lifting both cards to 350 W for 165 s
(clocks 1,830–1,875 MHz, pviol 0): decode **+5.5%** (server-log gen throughput), prefill **+6%** (chunk timing). A +35% power budget buys
~5% → decode is memory/latency-bound, not clock-bound. **Rejected**: the DDR5 DIMMs already sit
at 55.5–56.2 °C against a 55.0 °C ALARM HIGH and the cooling profile is load-bearing for
multi-day lanes. Reverted to 260 W (both cards verified).

## 3. Decode step anatomy (bs=1, 14.6K context, GPU span 15.4 ms/step, both ranks identical ±2%)

Exposed time per step (sweep line over TP0's 34 steady steps; kernel-sum is 16.4 ms because the
GDN `in_proj_ba` gemv runs on the alt stream under the Marlin qkvz GEMM):

| component | ms/step | share | kernels/step | note |
|---|---|---|---|---|
| Marlin INT4 GEMMs (alone) | 7.32 | 47.5% | 256 | 6.32 GB/rank of INT4 weights+scales → **727 GB/s = 78% of 936 GB/s** (80% of the P2-clock ceiling) |
| Marlin + `in_proj_ba` gemv overlapped | 1.38 | 9.0% | — | the 48 cuBLAS `gemvx` calls (32.9 µs each for a 48×5120 fp16 matrix — a bad heuristic, 15 GB/s) are 90% hidden by the dual-stream path |
| fp16 gemv exposed | 1.78 | 11.6% | 49 | **lm_head 1.47 ms** (1.27 GB/rank at **863 GB/s = 92% of peak** — already at roofline in fp16) + ~0.3 ms of `in_proj_ba` tails |
| NCCL allreduce (RING_LL) | 1.80 | 11.7% | 129 | 2/layer + embedding; **13.6 µs per 10 KB call** — the TP latency tax |
| flashinfer attention (fp8 KV, tensor-core decode) | 1.04 | 6.7% | 16 + 16 merge | 58 µs/layer at 14.6K — latency-bound here; grows to ~+3.2 ms by 140K (68 → 56 tok/s), consistent with the fp8 KV read |
| GDN recurrent + conv update | 0.73 | 4.7% | 48 × 4 | 12.4 µs recurrent kernel (786 KB state/layer) |
| fused-add RMSNorm / gated norm | 0.54 | 3.5% | 176 | 3.2 µs each — launch-floor kernels |
| idle gaps between graph nodes | 0.44 | 2.8% | ~990 nodes | ~0.45 µs per kernel |
| act / misc | 0.38 | 2.4% | — | |

CPU step wall is 1.1 ms — fully overlapped; the scheduler is not in the way at bs=1.

**Roofline check.** Pure streaming floor per step at 936 GB/s = INT4 weights 6.32 GB + fp16 lm_head
1.27 GB + KV@14.6K 0.24 GB + GDN states 0.04 GB = 7.9 GB → 8.4 ms. Actual 15.4 ms = **55% of the
bandwidth roofline**; the gap is NCCL latency (1.8), Marlin tail/launch inefficiency (~1.9),
attention latency (~0.8), small-kernel floors + gaps (~1.4), gemv tails (~0.3).

## 4. Prefill anatomy (one 5,830-token chunk on a ~9K cached prefix: 3.70 s = 1,576 tok/s)

| component | ms | share | note |
|---|---|---|---|
| Marlin INT4 GEMMs (M=5830 tiles) | 2,408 | 65% | 142 TFLOP/rank → **59 TFLOPS ≈ 90% of the clock-adjusted fp16 tensor peak** (71 TFLOPS at 1.695 GHz boost → ~64 at the capped 1.53 GHz). Yet +20% SM clock (350 W) bought only +6%, so the kernel is co-limited by re-reading the 60 MB M×K activation tile per N-tile through L2/DRAM (~0.4 TB per chunk). No serving-side lever either way. |
| flashinfer ragged prefill attention (fp8 KV, hd 256) | 772 | 21% | ~18 TFLOPS ≈ 28% of peak — the sm_86 prefill kernel is the weak spot (known: `attn-roofline-sm86-2026-07-15.md`) |
| NCCL allreduce | 345 | 9.3% | 129 × 59.7 MB at **22 GB/s** — NCCL's tuner keeps **RING_LL** even for 60 MB messages (LL128 is unavailable on GeForce NVLink); Simple would run ~2× faster but costs decode's 13 µs small-message floor |
| GDN chunked kernels / norms / act / misc | 165 | 4.5% | |

## 5. Levers, ranked by exposed time on THIS workload

| # | lever | ceiling | status |
|---|---|---|---|
| 1 | **NGRAM speculative decoding** (draft-free, depth-safe: 2.6× on copy-heavy spans, `ngram-copyheavy-at-depth-2026-06-15.md`). Agentic output is copy-heavy (file contents → edits, tool-call JSON) and bs=1 verify of k tokens is nearly free when 57% of the step is weight streaming. | **1.3–2×** decode | **Blocked** on the DeltaNet spec-verify conv1d dtype assert — README Tooling item 3. Now the highest-value serving item for the hybrids. |
| 2 | **lm_head INT8 (or INT4) via Marlin** — the only fp16 weight that matters (1.27 GB/rank, 9.6% of the step). Per-channel W8 is lossless in practice; calibration-device recipe change or a load-time W8 patch. | −0.7 ms (INT8, −4.6%) / −1.1 ms (INT4, −7%) | untried; quality gate = needle + HE + tool probe |
| 3 | **Custom / one-shot allreduce** (13.6 µs → ~6 µs × 129) | −0.9 ms (−6%) | blocked: sm_86 graph capture (`allreduce-accel-null-2026-06-15.md`) |
| 4 | Power cap 350 W | +5.5% | rejected (DIMM thermals) |
| 5 | `in_proj_ba` fp16 gemv → a real kernel (cuBLAS picks a 33 µs split-K gemvx for 491 KB) | −0.3 ms exposed (−2%) | cheap patch, low value — dual-stream already hides 90% |
| 6 | P0 memory clock (9,501 → 9,751 MHz) | ≤ +2.6% BW, ~+1.5% decode | unverified on GeForce (`-lmc`) |
| 7 | Prefill NCCL protocol (Simple for ≥1 MB messages) | −4.7% prefill = −0.5% e2e | needs a tuner plugin; env `NCCL_PROTO` is global and would tax decode |
| 8 | W4A8 Marlin for prefill (INT8 tensor cores, 2× fp16 peak) | ≤ −40% prefill = −4% e2e | activation quant → quality-gated; sm_80 kernel exists in vLLM lineage |
| — | Marlin decode kernel, attention kernel, scheduler/CPU | <3% each | at or near roofline; no lever |

## 6. Open item — 3–4 GB/s of PCIe *reads* during decode

During decode (never during prefill) each GPU pulls **3.1 GB/s (rank 0) / 4.3 GB/s (rank 1) over
PCIe** with only ~0.5 GB/s outbound — a ~7:1 rx:tx ratio is the signature of GPU-initiated reads
of host memory (256 B completions vs small request TLPs), ≈ 50–70 MB per decode step. NVLink
carries only ~0.6 GB/s each way (that is NCCL's LL traffic, accounted for). The torch trace shows
0.24 MB of HtoD memcpy over 4.3 s, so the reads are zero-copy accesses from inside the
cuda-graph decode path, not `.to(cuda)` copies. Candidates: NCCL's host-resident work FIFO /
LL flag polling, cuda-graph launch descriptors, a host-mapped buffer in the flashinfer decode
plan or the mamba radix-cache tracker. Decisive tests (need the GPUs, so after the cycle):
60 s with `--disable-cuda-graph` and with `NCCL_P2P_LEVEL`/`NCCL_WORK_FIFO_DEPTH` variations
under `dmon`; an `nsys --gpu-metrics-device` capture would attribute it directly (nsys is not
installed). It is off the critical path today (CPU wall 1.1 ms vs 15.4 ms GPU step) but it is
25–30% of a Gen4 x8 link for no known purpose.

## Reproduce

```bash
curl -s :23334/metrics > metrics.txt                      # workload histograms
curl -s -X POST :23334/start_profile -H 'Content-Type: application/json' \
  -d '{"output_dir":"/tmp/prof","num_steps":40,"activities":["CPU","GPU"],"with_stack":false,"record_shapes":true}'
scripts/bench/trace_step_anatomy.py /tmp/prof/*TP-0*.trace.json.gz --skip 5
nvidia-smi dmon -s pucvmt -d 1 -o T -c 120                # pviol / clocks / PCIe (20 ms samples)
```
