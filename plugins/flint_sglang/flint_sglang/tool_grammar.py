"""XGrammar structural tags that constrain a Flint assistant turn to Flint's chat protocol.

``build_flint_tool_grammar`` describes what Flint writes after its reasoning in a turn with tools: text without
protocol markers, then tool calls in Flint's format (``<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>
</tool_call>``) to the request's tools: any number with ``tool_choice="auto"``, at least one with ``"required"``,
exactly one call of the named tool with a named choice, and at most one (exactly one with ``"required"``) when the
request sets ``parallel_tool_calls`` to false. The tool calls reuse XGrammar's built-in structural tag for this
format: argument values follow a tool's JSON schema when the tool sets ``strict: true`` and are
syntactically constrained but schema-free otherwise, as XGrammar defines it. ``reasoning_then`` puts Flint's
optional ``<think>...</think>`` block before a constraint; the ``flint`` grammar backend does that for every
constraint, the tool grammar included.

The grammar does not name the token that ends the turn. SGLang gives XGrammar the model's EOS tokens as
stop tokens, which XGrammar allows wherever the grammar can end and never as grammar tokens, so Flint's
end-of-turn tokens (``<|observation|>`` after tool calls, ``<|user|>`` or end-of-text otherwise) close
the turn as stop tokens.
"""

from typing import Any, Literal

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from xgrammar.builtin_structural_tag import get_model_structural_tag
from xgrammar.structural_tag import (
    AnyTextFormat,
    Format,
    OptionalFormat,
    SequenceFormat,
    StructuralTag,
    TagFormat,
    TagsWithSeparatorFormat,
    TokenFormat,
)

PROTOCOL_MARKERS = (
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|system|>",
    "<|developer|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
)
STOP_MARKERS = ("<|end_of_text|>", "<|user|>", "<|observation|>")


def build_flint_tool_grammar(
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Literal["auto", "required"] | dict[str, Any] = "auto",
    parallel_tool_calls: bool = True,
) -> StructuralTag:
    """Build the structural tag for what Flint writes after its reasoning in a turn with tools.

    Args:
        tools: OpenAI-style function tool definitions, as SGLang's ``Tool.model_dump()`` returns them.
            Without tools, the turn is text.
        tool_choice: ``"auto"``, ``"required"`` or a named choice
            (``{"type": "function", "function": {"name": ...}}``).
        parallel_tool_calls: The request's ``parallel_tool_calls``; False allows a single call.

    Returns:
        The structural tag, ready for ``GrammarCompiler.compile_structural_tag``.
    """
    elements: list[Format] = [AnyTextFormat(excludes=list(PROTOCOL_MARKERS))]
    if tools:
        # XGrammar's "required" is one or more calls; "auto" makes them optional after the text
        calls_choice = "required" if tool_choice == "auto" else tool_choice
        calls = get_model_structural_tag("glm_4_7", tools=tools, tool_choice=calls_choice, reasoning=False).format
        if not parallel_tool_calls and isinstance(calls, TagsWithSeparatorFormat):
            calls = calls.model_copy(update={"stop_after_first": True})
        elements.append(OptionalFormat(content=calls) if tool_choice == "auto" else calls)
    return StructuralTag(format=SequenceFormat(elements=elements))


def reasoning_then(tokenizer: PreTrainedTokenizerBase, constraint: Format) -> StructuralTag:
    """Flint's optional reasoning block followed by ``constraint``.

    Raises:
        ValueError: If ``<think>`` or ``</think>`` is not a dedicated token.
    """
    return StructuralTag(format=SequenceFormat(elements=[OptionalFormat(content=_reasoning(tokenizer)), constraint]))


def _reasoning(tokenizer: PreTrainedTokenizerBase) -> TagFormat:
    return TagFormat(
        begin=TokenFormat(token=_token_id(tokenizer, "<think>")),
        content=AnyTextFormat(excludes=["</think>", *STOP_MARKERS]),
        end=TokenFormat(token=_token_id(tokenizer, "</think>")),
    )


def _token_id(tokenizer: PreTrainedTokenizerBase, marker: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(marker)
    if (
        not isinstance(token_id, int)
        or tokenizer.convert_ids_to_tokens(token_id) != marker
        or tokenizer.encode(marker, add_special_tokens=False) != [token_id]
    ):
        raise ValueError(f"Flint grammar requires {marker!r} to be a dedicated vocabulary token")
    return token_id
