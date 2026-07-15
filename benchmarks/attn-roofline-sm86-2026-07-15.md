# 256K decode-attention bandwidth on sm_86 — ~72% of roofline, lever CLOSED (2026-07-15)

R9700 measured their Triton decode kernel at ~21% of bandwidth roofline (flat ~275 GB/s),
promising ~4.7× attention headroom. Replication on our stack, derived from fixed-instrument
serving measurements + exact checkpoint geometry (no kernel isolation needed at this margin):

- coder-30b (Qwen3-Coder-30B-A3B, 48L, 4 KV heads, head_dim 128, fp8_e4m3 KV, TP=2):
  KV read per decode step per rank @255K = 261,120 tok × 48L × 2(K+V) × 2 heads × 128 × 1B = **6.42 GB**
- Measured TPOT (fixed instrument, server-verified depths): **4.98 ms @2K → 14.53 ms @255K**
  → depth-attributable cost ≈ 9.55 ms → achieved **≈ 670 GB/s per GPU ≈ 72% of the 936 GB/s roofline**
- Perfect-roofline attention (6.86 ms) would yield TPOT ≈ 11.8 ms ≈ 85 tok/s — **≤ +23% ceiling**

**Verdict:** the sm_86 Triton decode-attention path is already bandwidth-decent at depth; the
R9700 finding is another gfx1201-specific deficiency, not universal headroom (third such result
in their campaign after the MoE-config and RMSNorm levers). A kernel-engineering effort for a
≤23%-on-one-preset-class ceiling is below the bar while the EAGLE3 cross-team deliverable is
queued. Revisit only if R9700's kernel work produces a portable technique.

Cross-checks: earlier kv-splits sweep flat (8/16/32); nemotron/qwen36 depth curves consistent
with high attention BW-efficiency (their flatness/decay match architecture, not kernel waste).
Derivation inputs: `benchmarks/coder-30b-awq/results.json` (2026-07-14 re-sweep), checkpoint
config.json.
