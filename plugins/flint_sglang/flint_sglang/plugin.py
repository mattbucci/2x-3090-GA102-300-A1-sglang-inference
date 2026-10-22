"""SGLang general plugin that makes Flint checkpoints loadable without ``--trust-remote-code``.

SGLang runs ``sglang.srt.plugins`` entry points in every process before it parses the model
config, so registering ``FlintConfig`` here lets ``AutoConfig`` resolve ``model_type: flint``.
The model class itself is found through ``SGLANG_EXTERNAL_MODEL_PACKAGE``. The plugin also adds
the ``flint`` tool-call parser, reasoning parser and grammar backend; SGLang loads plugins before it
parses ``--tool-call-parser``, ``--reasoning-parser`` and ``--grammar-backend``, and ``check_flags``
refuses combinations of those that would misparse or silently degrade Flint's output.

v0.5.22 port: unchanged API surface (``GRAMMAR_BACKEND_REGISTRY`` factory signature, ``ToolCallParserEnum`` /
``DetectorMap`` dicts, ``add_grammar_backend_choices``, ``HookRegistry`` targets and ``ServerArgs.check_server_args``
all verified against the v0.5.22 tree). The ``pass_parallel_tool_calls`` hook is now a fallback only — see
``tool_parser.py``.
"""

import os
from collections.abc import Callable

from flint_sglang.configuration_flint import FlintConfig
from flint_sglang.grammar_backend import GRAMMAR_BACKEND_NAME, create_grammar_backend
from flint_sglang.reasoning_parser import REASONING_PARSER_NAME, FlintReasoningDetector
from flint_sglang.tool_parser import TOOL_CALL_PARSER_NAME, FlintDetector, pass_parallel_tool_calls
from sglang.srt.constrained.base_grammar_backend import GRAMMAR_BACKEND_REGISTRY
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.server_args import ServerArgs, add_grammar_backend_choices
from transformers import AutoConfig

EXTERNAL_MODEL_PACKAGE_ENV = "SGLANG_EXTERNAL_MODEL_PACKAGE"
CUDA_NORM_ENV = "FLASHINFER_USE_CUDA_NORM"


def register() -> None:
    """Register the Flint config, model package, parsers and grammar backend with SGLang.

    Also selects FlashInfer's CUDA RMSNorm kernels, which Flint was evaluated with. SGLang loads
    plugins before it spawns model workers, so the workers inherit this before importing FlashInfer.
    """
    AutoConfig.register(FlintConfig.model_type, FlintConfig, exist_ok=True)
    os.environ.setdefault(EXTERNAL_MODEL_PACKAGE_ENV, "flint_sglang")
    os.environ.setdefault(CUDA_NORM_ENV, "1")
    _register(FunctionCallParser.ToolCallParserEnum, TOOL_CALL_PARSER_NAME, FlintDetector)
    _register(ReasoningParser.DetectorMap, REASONING_PARSER_NAME, FlintReasoningDetector)
    _register(GRAMMAR_BACKEND_REGISTRY, GRAMMAR_BACKEND_NAME, create_grammar_backend)
    add_grammar_backend_choices([GRAMMAR_BACKEND_NAME])
    HookRegistry.register(
        "sglang.srt.function_call.function_call_parser.FunctionCallParser.get_structure_constraint",
        pass_parallel_tool_calls,
        HookType.BEFORE,
    )
    HookRegistry.register("sglang.srt.server_args.ServerArgs.check_server_args", check_flags, HookType.BEFORE)


def check_flags(server_args: ServerArgs) -> None:
    """``BEFORE`` hook on ``ServerArgs.check_server_args``, which SGLang runs before it loads the model.

    The ``flint`` grammar backend lets Flint reason before every constraint, so it needs the ``flint`` reasoning
    parser, and the ``flint`` tool-call grammar leaves that reasoning to the backend.

    Raises:
        RuntimeError: For a combination of flags Flint's parsers and grammar backend don't support.
    """
    flint_backend = server_args.grammar_backend == GRAMMAR_BACKEND_NAME
    if flint_backend and server_args.reasoning_parser != REASONING_PARSER_NAME:
        raise RuntimeError(
            f"--grammar-backend flint requires --reasoning-parser flint (got {server_args.reasoning_parser}): it lets "
            "Flint reason before a constraint, and that reasoning must be parsed as Flint writes it."
        )
    if server_args.tool_call_parser == TOOL_CALL_PARSER_NAME and not flint_backend:
        raise RuntimeError(
            f"--tool-call-parser flint requires --grammar-backend flint (got {server_args.grammar_backend}): "
            "without it, Flint can't reason before calling a tool."
        )
    if flint_backend and server_args.enable_strict_thinking:
        raise RuntimeError(
            "--enable-strict-thinking is not supported with --grammar-backend flint: Flint's grammar already keeps "
            "stop and tool-call tokens out of its reasoning, and there is no thinking budget."
        )
    if flint_backend and server_args.constrained_json_disable_any_whitespace:
        raise RuntimeError(
            "--constrained-json-disable-any-whitespace is not supported with --grammar-backend flint, whose JSON "
            "constraints allow any whitespace."
        )


def _register(registry: dict[str, Callable], name: str, entry: Callable) -> None:
    registered = registry.setdefault(name, entry)
    if registered is not entry:
        raise RuntimeError(f"SGLang already has a {name!r} entry: {registered}")
