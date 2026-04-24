# llmcompressor patches (3090)

llmcompressor is installed via pip in the `quant` conda env — we don't vendor the source tree yet. This directory holds standalone patched copies of files that need fixes for our calibration targets. TODO: vendor llmcompressor at a pinned commit like R9700 does (`components/llmcompressor/`) and convert these into proper numbered `.patch` files.

## `qwen3_vl_moe.py.patched`

Replaces `site-packages/llmcompressor/modeling/qwen3_vl_moe.py` to work against transformers ≥5.5 + Qwen3-VL-30B-A3B. Two fixes in one file:

1. **`top_k` moved onto the router.** `Qwen3VLMoeTextSparseMoeBlock` no longer has a direct `top_k` attribute; it lives on `block.gate` (a `Qwen3VLMoeTextTopKRouter`, which pre-softmaxes its output). Wrapper now reads `original.gate.top_k` and computes raw router logits via `F.linear(x, original.gate.weight)` so the existing softmax+topk path keeps working.
2. **MoE experts sized by `moe_intermediate_size`, not `intermediate_size`.** For Qwen3-VL-30B, `moe_intermediate_size=768` but `intermediate_size=6144` (dense). The unfuse path was creating per-expert MLPs at the dense size, causing `mat1 and mat2 shapes cannot be multiplied (N×2048 and 1536×768)` at first forward. Fix passes the `intermediate_size` kwarg on `Qwen3VLMoeTextMLP(config, intermediate_size=moe_intermediate_size)`.

### Apply

```bash
cp llmcompressor-patches/qwen3_vl_moe.py.patched \
   $(python -c 'import llmcompressor; import os; print(os.path.join(os.path.dirname(llmcompressor.__file__), "modeling/qwen3_vl_moe.py"))')
find $(python -c 'import llmcompressor; import os; print(os.path.dirname(llmcompressor.__file__))') -name "qwen3_vl_moe.cpython*.pyc" -delete
```

Then restart any running calibration. Run this before `quantize_qwen3vl_30b_moe_thinking_vision.py`.
