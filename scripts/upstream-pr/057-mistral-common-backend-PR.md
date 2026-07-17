# Upstream PR draft — prefer HF backend over MistralCommonBackend when tokenizer.json ships (our patch 057)

**Status: PREPARED, NOT OPENED** — user green-light required.

## Bug status on main (verified 2026-07-16, `27ad9d1`)
Live. Main's `hf_transformers/mistral_utils.py` (`patch_mistral_common_tokenizer`)
already shims `_text_to_ids` for the three **Pixtral image markers** (`[IMG]`,
`[IMG_BREAK]`, `[IMG_END]`) — proof upstream knows MistralCommon never parses
special tokens from text — but the shim does not cover the instruct-control
tokens (`[INST]`, `[/INST]`, `[TOOL_CALLS]`, `[AVAILABLE_TOOLS]`,
`[TOOL_RESULTS]`, `[SYSTEM_PROMPT]`), so sglang's render-then-encode chat path
still encodes them as literal text on any Mistral-family checkpoint that
transformers ≥5.12 routes to `MistralCommonBackend`.

## PR title
```
[Bugfix] Tokenizer: prefer the HF backend over MistralCommonBackend when the checkpoint ships tokenizer.json (render-then-encode corrupts control tokens)
```

## PR body
```
### Motivation

transformers >=5.12 resolves AutoTokenizer for Mistral-family checkpoints that
ship tekken.json to `MistralCommonBackend` — even when the checkpoint ALSO
ships a valid HF tokenizer.json. MistralCommonBackend never parses special
tokens out of plain text (mistral-common is a request-native tokenizer), so
SGLang's render-then-encode chat path silently encodes `[INST]`, `[/INST]`,
`[TOOL_CALLS]`, `[AVAILABLE_TOOLS]` as literal character sequences.

Nothing crashes. The failure is a smoke-invisible quality collapse: boot is
green, short prompts answer fine, but instruct formatting is destroyed —
we measured needle-recall 0.0, HumanEval halved, and dead tool-calls on
Devstral-Small AWQ the day we bumped transformers (fine on tx 5.11 with the
identical checkpoint). `patch_mistral_common_tokenizer` already special-cases
this exact mechanism for the Pixtral image markers ([IMG]/[IMG_BREAK]/[IMG_END])
— the instruct-control tokens have no such shim.

### Modification

In `get_tokenizer`, after `_auto_tokenizer_from_pretrained` returns: if the
loaded tokenizer is a `MistralCommonBackend` AND the checkpoint ships a
tokenizer.json, reload preferring the HF backend (`fix_mistral_regex=False`
routes transformers away from the mistral-common resolution), and skip
`_resolve_tokenizers_backend` for the rerouted result (it rejects the kwarg).
Checkpoints that ship ONLY tekken.json keep MistralCommonBackend (there the
native mistral-common request path is the correct contract).

### Validation

Devstral-Small AWQ on transformers 5.12.1: without this, needle 0.0 / HE ~24%
/ tool-calls never parsed; with it, needle 3/3 at its 131K pool, HE 48%,
tool-use probe 1.0/1.0 (usage-verified depths). A/B token-id comparison of a
chat-rendered prompt across both backends shows the control tokens encoding to
single special ids vs character runs. In production on our stack since the
v0.5.15 rebase.
```

## Reference diff
Our `patches/057-mistral-common-backend-optout.patch` (against v0.5.15's
`python/sglang/srt/utils/hf_transformers/tokenizer.py`; main's file is a
refactor of the same flow — regenerate hunks at open time; insertion point is
directly after the `_auto_tokenizer_from_pretrained` call in `get_tokenizer`).

## Checklist before opening
- [ ] Fork + branch `fix/mistral-common-backend-optout`
- [ ] Regenerate hunks against main's `tokenizer.py` (structure verified
      compatible 2026-07-16; `_auto_tokenizer_from_pretrained` at ~L163)
- [ ] Detection helper: main already has tekken/tokenizer.json presence
      helpers in `mistral_utils.py` — reuse, don't duplicate
- [ ] A/B encode test: consider contributing a unit test to
      `test/registered/unit/utils/test_hf_transformers.py` (they have the file)
- [ ] Run their pre-commit
