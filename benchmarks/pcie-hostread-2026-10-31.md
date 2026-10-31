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

## Round 2 — what the graph reads (model families, backends)

Pending: `devstral` (dense, flashinfer, no GDN), `coder-30b` (MoE, no GDN), `qwen36` (GDN + MoE), `flint` (triton
attention backend), and `qwen38` under the piecewise graph backend — if the reads track the GDN presets the
mamba/GDN state bookkeeping is the suspect; if they track flashinfer they are the decode plan; if every preset shows
them it is generic to graph replay on this stack.
