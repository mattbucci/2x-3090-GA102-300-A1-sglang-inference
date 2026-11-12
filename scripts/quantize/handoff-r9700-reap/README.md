# R9700 handoff — unified Qwen3_5Moe REAP tooling (3090-G → R97-F)

Three files are shared byte-identical between the two repos; `install.sh` copies
them and checks `SHA256SUMS`:

| 3090 path | R9700 path |
|---|---|
| `patches/qwen3_5moe_unfused_experts.py` | `ream-patches/qwen3_5moe_unfused_experts.py` |
| `scripts/quantize/run_reap.py` | `scripts/quantize/run_reap.py` |
| `scripts/quantize/test_qwen3_5moe_unfuse.py` | `scripts/quantize/test_qwen3_5moe_unfuse.py` |

`run_reap.py` and the test resolve the helper directory by file presence
(`ream-patches/` first, then `patches/`), so neither repo edits them.

## What changed against the old donor

- `pick_model_class(config)`: composite configs (`vision_config`) load through
  `AutoModelForImageTextToText` at every load, so the vision tower survives.
- `read_expert_counts(config)` / `set_expert_counts(config, keep)`: counts are
  read from and written to `text_config` (the Qwen3.6-35B top level has none).
- Config-only preflight before any weight load; `--preflight` exits after a
  header-only scan of the shard key layout.
- `reap_report.json` (survivors + saliency) is written before any save work;
  `--survivors-from <report>` skips the observation pass.
- `--save-mode stream`: slices the safetensors shards directly — router rows and
  fused expert stacks per layer — with peak RAM of one shard. `auto` selects it
  when the checkpoint exceeds host RAM minus 14 GiB (67 GiB BF16 on a 61 GB
  host → stream). Non-weight files (chat template, tokenizer, image and video
  preprocessor configs) are copied verbatim. It raises rather than write a
  pruned config over unsliced weights.
- The unfuse helper gained a fused-attribute bridge: under transformers 5.5.4
  `from_pretrained` never calls `load_state_dict`, so the old pre-hook did not
  fire and a real load died with `AttributeError: 'Qwen3_5MoeExpertsUnfused'
  object has no attribute 'down_proj'`. The old 7-case test never called
  `from_pretrained`; cases 13–14 now do, text-only and composite.

## Install and gates

```bash
scripts/quantize/handoff-r9700-reap/install.sh ~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference
cd ~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference
REAP_ENV=nemo-quant scripts/quantize/reap_gates.sh B                                   # miniature, incl. the slicer tripwire
REAP_ENV=nemo-quant scripts/quantize/reap_gates.sh C ~/AI/models/Qwen3.6-35B-A3B-BF16 192
```

Manual edits on the R9700 side: `run_reap.sh` default `REAP_ENV` (`vllm` no
longer exists there → `nemo-quant`), and a helper row in `ream-patches/README.md`.

## Prune

```bash
REAP_ENV=nemo-quant setsid ./scripts/quantize/run_reap.sh \
    --model ~/AI/models/Qwen3.6-35B-A3B-BF16 \
    --save-path ~/AI/models/Qwen3.6-35B-A3B-REAP-BF16 \
    --keep-experts 192 --max-memory-per-gpu 28GiB
REAP_ENV=nemo-quant scripts/quantize/reap_gates.sh D ~/AI/models/Qwen3.6-35B-A3B-BF16 ~/AI/models/Qwen3.6-35B-A3B-REAP-BF16 192
```

`--max-memory-per-gpu` defaults to 20GiB (24 GB cards); pass 28GiB on 32 GB
cards. If the save stage fails, rerun with
`--survivors-from <save-path>/reap_report.json` — no second observation pass.

## Receipts on the 3090 box (transformers 5.5.4, torch 2.11.0+cu130, CPU)

- Gate A/B: 15 cases pass; stream vs reload save is bit-equal on every tensor
  with logits Δ = 0, text-only and composite.
- Gate C on the real `Qwen3.6-35B-A3B-BF16`: `Qwen3_5MoeConfig →
  AutoModelForImageTextToText`, 256 / 8, 40 layers with router + expert keys,
  save mode `stream`.
- Gate D and the `--survivors-from` resume: exercised end to end on a tiny
  composite checkpoint. The full 35B prune has not been run on either box.
