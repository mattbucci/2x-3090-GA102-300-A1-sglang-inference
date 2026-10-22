"""SGLang grammar backend for Flint: XGrammar, with Flint's optional reasoning before every constraint.

Flint decides on each turn whether to reason. With SGLang's own backends a constraint (a response format,
regex, EBNF grammar or structural tag, the ``flint`` tool-call grammar included) either applies from the first
token (Flint cannot reason) or, when a request declares thinking, only after a ``</think>`` (nothing is
constrained when Flint answers directly). This backend compiles each constraint after an optional
``<think>...</think>`` block instead. Structural tags in SGLang's legacy format (``structures`` / ``triggers``)
are used as they are. The ``flint`` reasoning parser moves that block out of ``content``; the plugin refuses to
start this backend without it.
"""

import json

from flint_sglang.tool_grammar import reasoning_then
from pydantic import ValidationError
from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject, InvalidGrammarObject
from sglang.srt.constrained.utils import is_legacy_structural_tag
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.srt.server_args import ServerArgs
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from xgrammar.structural_tag import Format, GrammarFormat, JSONSchemaFormat, RegexFormat, StructuralTag

GRAMMAR_BACKEND_NAME = "flint"


class FlintGrammarBackend(XGrammarGrammarBackend):
    """XGrammar backend that lets Flint reason before a json_schema, regex, ebnf or structural-tag constraint."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.tokenizer = tokenizer

    def dispatch_json(self, key_string: str) -> BaseGrammarObject:
        schema = True if key_string == "$$ANY$$" else json.loads(key_string)
        return self._after_reasoning(JSONSchemaFormat(json_schema=schema))

    def dispatch_regex(self, key_string: str) -> BaseGrammarObject:
        return self._after_reasoning(RegexFormat(pattern=key_string))

    def dispatch_ebnf(self, key_string: str) -> BaseGrammarObject:
        return self._after_reasoning(GrammarFormat(grammar=key_string))

    def dispatch_structural_tag(self, key_string: str) -> BaseGrammarObject:
        try:
            tag = json.loads(key_string)
            if is_legacy_structural_tag(tag):
                return super().dispatch_structural_tag(key_string)
            self._sanitize_structural_format(tag["format"])
            constraint = StructuralTag.model_validate(tag).format
        except (json.JSONDecodeError, AssertionError, ValidationError) as error:
            return InvalidGrammarObject(str(error))
        return self._after_reasoning(constraint)

    def _after_reasoning(self, constraint: Format) -> BaseGrammarObject:
        return super().dispatch_structural_tag(reasoning_then(self.tokenizer, constraint).model_dump_json())


def create_grammar_backend(
    server_args: ServerArgs, tokenizer: PreTrainedTokenizerBase, vocab_size: int, eos_token_ids: set[int] | None
) -> FlintGrammarBackend:
    """Factory for SGLang's grammar backend registry (``--grammar-backend flint``)."""
    return FlintGrammarBackend(
        tokenizer,
        vocab_size=vocab_size,
        model_eos_token_ids=list(eos_token_ids) if eos_token_ids else None,
        any_whitespace=not server_args.constrained_json_disable_any_whitespace,
    )
