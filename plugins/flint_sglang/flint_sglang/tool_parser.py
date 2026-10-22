"""SGLang tool-call parser for Flint: SGLang's parsing of Flint's tool-call format plus Flint's turn grammar.

The main additions are SGLang's structural-tag hooks: every request with tools (``tool_choice`` ``auto``,
``required`` or a named function) is constrained by ``build_flint_tool_grammar``; the ``flint`` grammar backend
lets Flint reason before it. As in OpenAI's API, a tool's argument schema is enforced only when the tool is sent with
``strict: true`` (SGLang fills an omitted ``strict`` with False); other tools keep the call format but not the
schema.

v0.5.22 port notes: ``FunctionCallParser.get_structure_constraint`` now passes ``parallel_tool_calls`` to the
detector's structural-tag hooks directly, so both hooks take it as a keyword (the vendor's ``pass_parallel_tool_calls``
plugin hook is kept as a no-cost fallback for the attribute path). The same method also short-circuits any
``Glm47MoeDetector`` whose ``parses_required_natively()`` is true into a GLM-4.7 full-assistant EBNF built with GLM's
special tokens — wrong for Flint — so the detector pins that to ``False``.
"""

import re
from typing import Literal

from flint_sglang.tool_grammar import build_flint_tool_grammar
from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from xgrammar import StructuralTag

TOOL_CALL_PARSER_NAME = "flint"


class FlintDetector(Glm47MoeDetector):
    """Tool-call detector for Flint's format that constrains every tool choice to Flint's turn grammar."""

    parallel_tool_calls: bool = True  # the request's; fallback when a caller omits the keyword

    def __init__(self) -> None:
        super().__init__()
        # Never take the GLM-4.7 full-assistant EBNF path in get_structure_constraint.
        self.use_full_assistant_constraint = False
        # XGrammar's tool-call tag allows whitespace after a tool's name. SGLang strips it when streaming; strip it
        # here too, before the name is used to type the arguments.
        self.func_detail_regex = re.compile(r"<tool_call>(.*?)\s*(<arg_key>.*?)?</tool_call>", re.DOTALL)

    def parses_required_natively(self) -> bool:
        return False

    def supports_structural_tag(self) -> bool:
        return True

    def get_auto_tool_call_structural_tag(
        self,
        tools: list[Tool] | None = None,
        thinking_mode: bool = False,
        parallel_tool_calls: bool | None = None,
    ) -> StructuralTag:
        return self.get_structural_tag(tools, thinking_mode=thinking_mode, parallel_tool_calls=parallel_tool_calls)

    def get_structural_tag(
        self,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | Literal["auto", "required"] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool | None = None,
    ) -> StructuralTag:
        del thinking_mode  # the flint grammar backend puts the optional <think> block before every constraint
        if parallel_tool_calls is None:
            parallel_tool_calls = self.parallel_tool_calls
        choice = tool_choice.model_dump() if isinstance(tool_choice, ToolChoice) else tool_choice
        return build_flint_tool_grammar([tool.model_dump() for tool in tools or []], choice, parallel_tool_calls)


def pass_parallel_tool_calls(parser: FunctionCallParser, *args, parallel_tool_calls: bool = True, **kwargs) -> None:
    """``BEFORE`` hook on ``FunctionCallParser.get_structure_constraint``: give Flint's detector the request's
    ``parallel_tool_calls`` (fallback for callers that reach the structural-tag hooks without the keyword)."""
    if isinstance(parser.detector, FlintDetector):
        parser.detector.parallel_tool_calls = parallel_tool_calls
