# Triton extend attention: fp32 vs bf16 softmax·V dot (patch 011) on Ampere — 2026-11-02

Patch 011 (`triton-attention-fp32`) keeps the fp32 softmax weights un-downcast for the P·V dot on every
platform (upstream downcasts `p` to the KV dtype before the dot; upstream's own comment records that
downcast as the accuracy loss, and the patch has carried it since v0.5.11 for RDNA4 / SM12 long-context
quality). On Ampere the fp32 operand means the `tl.dot` runs on the TF32 / FMA path instead of the bf16
tensor-core path for half of every attention tile. `SGLANG_EXTEND_ATTN_FP32_PV=0` restores upstream's
bf16 P·V (fp32 accumulation either way); default unchanged (fp32).

Arms: served preset as shipped, fresh prefill on the natural prompt (`bench_long_context.py
--prompt-source natural`, cache flushed per point), greedy 131K needle on the natural haystack, the 256K
reasoning probe at 131K / 256K, MMLU-57. Chain `/var/tmp/route-ship/chain38.sh`.

| Preset | arm | TTFT 32K (33.9K actual) | TTFT 131K (133K actual) | needle 131K | reasoning 131K / 256K | MMLU-57 |
|---|---|--:|--:|:--:|:--:|:--:|
| `flint` | fp32 (shipped) | 3.62 s | 32.98 s | 3/3 | stock filler 0 / 0 (repetition attractor, see below) | 48 |
| `flint` | bf16 P·V | **3.35 s (−7.5 %)** | **28.78 s (−12.7 %)** | 3/3 | stock filler 0 / 0 (same) | 51 |
| `gemma4-31b` | fp32 (shipped) | 17.74 s | 146.70 s | 3/3 | 100 % / 100 % | 55 |
| `gemma4-31b` | bf16 P·V | 18.67 s (+5.2 %) | **138.12 s (−5.8 %)** | 3/3 | 100 % / 100 % | 53 |

Receipts: `flint-{fp32,bf16}.json`, `gemma4-31b-{fp32,bf16}.json` (this dir);
`benchmarks/quality/flint-pv-*-needle-2026-11-02.json`, `gemma4-31b-pv-*-needle-2026-11-02.json`,
`reason256k-{flint,gemma4-31b}-pv-{fp32,bf16}-2026-11-02.json`, `{flint,gemma4-31b}-pv-{fp32,bf16}-2026-11-02.json` (MMLU).

**Flint and the stock-filler reasoning probe:** both arms score 0/3 at 131K and 256K on
`probe_256k_quality.py`'s repetitive filler — every cell runs to the 3,000-token cap regurgitating the
filler sentence (`answer_tail` in the bf16 receipt), the NoPE repetition attractor the needle probe already
avoids with `--haystack`. The probe now takes `--haystack` (natural text, 3.1 chars/token on Flint's
tokenizer); the natural-haystack re-probe of both Flint arms at 32K / 131K / 240K is chain39
(`reason256k-flint-pv-{fp32,bf16}-hay-2026-11-02.json`).

**Natural-haystack reasoning probe on Flint (chain39, `--haystack benchmarks/quality/haystack-platypus.txt --chars-per-token 3.1`, lengths 32768 / 131072 / 240000 → 29.4K / 118.5K / 219.0K actual):**

| arm | 29K | 119K | 219K | cells |
|---|:--:|:--:|:--:|---|
| fp32 (shipped) | 3/3 | 2/3 | 1/3 | 119K aggregate and 219K aggregate run to the 3,000-token cap (thinking drifts into the haystack's own problems); 219K vartrack mis-reads VAR_A (5 instead of 37) |
| bf16 P·V | 2/3 | 2/3 | 1/3 | same misses as fp32 plus the 29K multikey, which ran to the cap drifting into a haystack fraction problem; 119K and 219K identical cell-for-cell |

The 29K multikey flip is a single greedy cell on a thinking model that drifts; chain41 repeats the identical
task instance (3× at 32K, 2× at 65K) on each arm to read it against run-to-run variance before the Flint
preset decides (`reason256k-flint-pv-{fp32,bf16}-hayrep-2026-11-02.json`).

**Repeat sample (chain41 + chain42, identical task instances; the first row at each length is a fresh fill, later rows hit the prefix cache — same protocol on both arms):**

| arm | 29K cells | 58K | 119K | 219K | all |
|---|:--:|:--:|:--:|:--:|:--:|
| fp32 | 24/30 | 4/6 | 5/12 | 1/3 | **34/51** |
| bf16 | 21/30 | 3/6 | 9/12 | 1/3 | **34/51** |

The flipping cells (vartrack at 29K / 58K, multikey and aggregate at 119K) flip on the fp32 arm alone across
identical repeats; neither arm is consistently ahead and the totals are equal. **Verdict: bf16 P·V is at
parity on Flint → `flint` preset default (`SGLANG_EXTEND_ATTN_FP32_PV=0`, with the 128×128 tile): fresh
prefill 3.62 → 2.97 s @32K, 32.98 → 23.40 s @131K (−18 % / −29 %), decode unchanged. `gemma4-31b` stays
fp32 (mixed speed at parity).** Combined-arm receipts: `flint-bf16-128-128-8.json`, `flint-bf16-128-64-8.json`
(128×64 + bf16 = 3.03 s / 25.9 s). Repeat receipts: `benchmarks/quality/reason256k-flint-pv-{fp32,bf16}-hayrep{,2}-2026-11-02.json`.
