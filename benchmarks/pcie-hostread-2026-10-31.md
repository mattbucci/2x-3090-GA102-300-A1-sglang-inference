# PCIe host reads during decode — decisive tests (2026-10-31)

**Question** (qwen38 agentic profile 2026-09-10 §6): each GPU pulls 3–4 GB/s over PCIe *inbound* during decode with
~0.5 GB/s outbound — the signature of GPU-initiated reads of host memory, ≈ 50–70 MB per decode step, 25–30 % of a
Gen4 x8 link, for no known purpose. Candidates named then: NCCL's host-resident work FIFO / LL flag polling,
cuda-graph launch descriptors, a host-mapped buffer in the flashinfer decode plan or the mamba radix-cache tracker.

**Instrument:** `qwen38` served as shipped (TP=2 / 262144, fp8 KV, INT8 head, 300 W), one `/generate` request with a
1K prompt and 4,000 forced tokens (`ignore_eos`) ≈ 52 s of pure single-user decode, `nvidia-smi dmon -s t -d 1`
(rxpci / txpci, 1 s samples) around it; "busy" = seconds with rx > 200 MB/s. Driver `/var/tmp/pcie-hostread/run.sh`,
raw `dmon-<variant>.csv` alongside.

## Round 1 — server knobs

| variant | GPU 0 rx mean / max | GPU 1 rx mean / max | tx mean | 4,000 tokens in |
|---|---:|---:|---:|---:|
| baseline (decode CUDA graphs on) | **4,166 / 7,901 MB/s** | **4,136 / 8,717 MB/s** | ~580 MB/s | 52.3 s |
| `--cuda-graph-backend-decode disabled` | **< 200 MB/s every second** | **< 200 MB/s every second** | — | 208.9 s |
| `NCCL_PROTO=Simple` | 3,931 / 8,701 | 3,880 / 8,491 | ~550 | 60.2 s |
| `NCCL_WORK_FIFO_DEPTH=64` | 4,210 / 7,822 | 3,974 / 8,843 | ~575 | 52.3 s |
| `NCCL_P2P_DISABLE=1` | 4,101 / 7,967 | 4,044 / 8,711 | ~570 | 52.2 s |

**Read:** the reads are a property of the **graph-replay decode path**, not of NCCL. Graphs off removes them
entirely — and not proportionally: eager decode is 4× slower, so the same per-step read volume would still have
shown as ~1 GB/s. None of the three NCCL knobs moves them (the LL-vs-Simple and FIFO-depth hypotheses are dead; P2P
off neither slows decode nor adds host traffic here, so the allreduce is not on the host path either way). Per step at
76 tok/s the baseline is ≈ 54 MB of host reads per GPU — far more than launch descriptors; something captured in the
decode graph dereferences host-mapped (zero-copy) memory every replay.

## Round 2 — model families (same load, decode CUDA graphs on)

| preset | arch / backend | GPU 0 rx mean | GPU 1 rx mean | 4,000 tokens in | per step per GPU |
|---|---|---:|---:|---:|---:|
| devstral | dense, flashinfer, no GDN | < 200 MB/s (one busy second at start) | same | 42.4 s | ~0 |
| coder-30b | MoE, flashinfer, no GDN | < 200 MB/s every second | same | 18.8 s | 0 |
| flint | MoE, **triton** attention, no GDN | < 200 MB/s every second | same | 25.0 s | 0 |
| qwen36 | **GDN** + MoE | **2,532 MB/s** | **2,586 MB/s** | 16.9 s | ≈ 11 MB |
| qwen38 (round 1) | **GDN** + dense | **4,166 MB/s** | **4,136 MB/s** | 52.3 s | ≈ 54 MB |

**Read:** the host reads belong to the **GDN (gated DeltaNet) decode path** — no dense, MoE, flashinfer or triton
attention preset without GDN layers shows them, and the two GDN presets show them in proportion to their recurrent
state: qwen38 has 48 GDN layers × 48 value heads × 128 × 128 at the fp32 state default ≈ 75 MB per rank per token,
qwen36 a quarter of that. The state pool itself is a device buffer (`UnifiedKVPool._raw`, `torch.empty(..., device)`),
so the reads are not the state's home; something the captured GDN decode step dereferences lives in host-mapped memory
and is touched at state scale every replay. Round 3 narrows it: SSM state in bf16 (halves the state bytes), the fused
per-slot conv ops off, qwen36-dense (GDN without MoE), and the GDN kernels pinned to triton (today's dispatcher already
reports `decode=TritonGDNKernel`, with a logged fallback from the fused decode projection / Conv1D path).

## Round 3 — GDN-side knobs (qwen38 unless noted; decode CUDA graphs on)

| variant | GPU 0 rx mean | GPU 1 rx mean | 4,000 tokens in | note |
|---|---:|---:|---:|---|
| `SGLANG_MAMBA_SSM_DTYPE=bfloat16` (state pool 0.63 → 0.32 GB, verified in the boot log) | 4,337 | 4,450 | 51.6 s | **unchanged — the reads are not the state bytes** |
| `SGLANG_DISABLE_FUSED_MAMBA_SLOT_OPS=1` | 4,157 | 4,005 | 52.0 s | unchanged |
| `qwen36-dense` (dense + GDN, same 5,120 × 48-layer geometry as qwen38) | 4,028 | 4,132 | 52.1 s | identical to qwen38 — MoE vs dense irrelevant |
| `--linear-attn-{decode,prefill}-backend triton` | 4,200 | 4,019 | 52.2 s | unchanged (triton was already the dispatched decode kernel) |
| `SGLANG_MAMBA_CONV_DTYPE=float16` | 3,942 | 4,070 | 51.7 s | unchanged reads; **side finding:** the fp16 conv state lets the fused "GDN decode QKVZ/BA unpack + indexed Conv1D" path engage (the default bf16 conv state mismatches the fp16 weights and falls back) — ~1 % faster here, measured properly below |
| `--cuda-graph-backend-decode piecewise` | — | — | — | invalid value on this build (choices `full`, `breakable`, `disabled`); `breakable` in round 4 |

**Where this leaves the open item:** the host reads are (1) specific to the GDN decode step, (2) present only under CUDA-graph replay, (3) proportional to hidden size × GDN layer count (≈ 1.1 MB per GDN layer per step at hidden 5,120; ≈ 0.37 MB at 2,048), and (4) independent of the SSM / conv state dtype, the slot-copy kernel, the GDN kernel backend, and every NCCL knob. The state pool, the int8 checkpoint pool and the per-forward index tables are all device-resident or proper H2D copies, so the remaining candidates are a host-mapped buffer the captured GDN decode kernels dereference (graph-replay-only code path: packed decode / causal-conv1d update) or the graph's own replay of a pinned staging buffer. Per the 2026-09-10 profile it is off the critical path (CPU wall 1.1 ms vs 15.4 ms GPU step), and no variant moved decode speed with it, so it stays a diagnostic finding: attribute it with an `nsys --gpu-metrics-device` capture (not installed) when a GDN preset is next on the bench; graphs off is not a trade anyone would take (4× slower).

## Round 4

`--cuda-graph-backend-decode breakable`: the server crashed at boot on this build (child failed during init; log `/var/tmp/pcie-hostread/serve-r4-graph-breakable.log`), so the breakable backend is not a discriminator here.

## Side lever measured — conv state in the model dtype (fused GDN decode path)

Every GDN preset here runs fp16 weights with the default bf16 conv state, so the fused "GDN decode QKVZ/BA unpack +
indexed Conv1D" path logs `QKVZ, Conv1D state, and weight dtypes must match` and falls back on all of qwen36 /
qwen35-moe / qwen36-dense / qwen38. `SGLANG_MAMBA_CONV_DTYPE=float16` engages it:

| preset | 1K | 32K | 261,916 |
|---|---:|---:|---:|
| qwen38 | 77.0 → **78.1** (+1.4 %) | 72.3 → **73.2** (+1.2 %) | 53.9 → **54.5** (+1.1 %) |
| qwen36-dense | 77.1 → **78.0** (+1.2 %) | 72.4 → **73.2** (+1.1 %) | 53.9 → **54.5** (+1.1 %) |

(baselines = today's INT8-head level; `regression/{qwen38,qwen36-dense}-2026-10-31-conv-fp16.json`.) Also visible in
both runs: the KV pool is 682,923 tokens, up from 652K, because the INT8 head freed 0.64 GB per card. Quality gate
(fleet harness MMLU + HumanEval, sampled needle at 131K / 250K) on qwen38 and qwen36 decides whether launch.sh makes
the conv state follow the model dtype for every preset.

**Gate (fleet harness, MMLU 60 + HumanEval 30, `--mc-budget 2560`; sampled needle at 131K / 250K × 0.1 / 0.5 / 0.9):**
qwen38 fp16 conv state **55/57, 29/30, needle 6/6** (bf16 conv state, same day: 52–53/57, 29/30, 12/12);
qwen36 **53/57, 29/30** (bf16: 53–54/57, 29/30). Parity → `launch.sh` now sets `SGLANG_MAMBA_CONV_DTYPE` to the
model dtype for every preset (env override still wins); tripwire baselines re-saved for the four GDN presets as shipped
and the serving image rebuilt (receipts below).

**Tripwire baselines re-saved as shipped (INT8 head + matched conv state), 1K / 32K / 261,916 actual:** qwen36 241.0 / 213.2 / 135.1
(morning bf16-head 223.7 / 198.4 / 128.9), qwen35-moe 239.8 / 219.8 / 165.3 (222.2 / 205.8 / 156.2), qwen36-dense 78.0 / 73.2 / 54.4
(73.0 / 69.0 / 51.9), qwen38 78.0 / 73.2 / 54.4 (72.9 / 68.7 / 51.8). Serving image rebuilt on it: `50ef7b838d73`, smoke PASS on
qwen36-ream (`benchmarks/quality/qwen36-ream/docker-serving-smoke-2026-10-31c.json`).

