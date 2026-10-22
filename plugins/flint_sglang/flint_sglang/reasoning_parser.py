"""SGLang reasoning parser for Flint: ``<think>...</think>`` parsing that never assumes a turn starts in reasoning.

Flint decides on each turn whether to reason, and its chat template has no thinking switch. SGLang's other
``<think>`` parsers let a request (or a parser default) declare that the turn starts inside reasoning; for Flint
that would put a turn without ``<think>`` into ``reasoning_content`` and hold back the tool-call grammar until a
``</think>`` that never comes. This parser reads ``<think>`` blocks only where Flint writes them, ignores
``thinking`` / ``enable_thinking`` in ``chat_template_kwargs``, and keeps a ``<tool_call>`` written inside the
reasoning as reasoning (Flint always closes ``<think>`` before it calls a tool).
"""

from sglang.srt.parser.reasoning_parser import BaseReasoningFormatDetector

REASONING_PARSER_NAME = "flint"


class FlintReasoningDetector(BaseReasoningFormatDetector):
    """SGLang's ``<think>...</think>`` detector that never treats a request as thinking."""

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            # SGLang treats a missing default as "not thinking", whatever the request's chat_template_kwargs say
            reasoning_default=None,
            force_nonempty_content=force_nonempty_content,
        )
