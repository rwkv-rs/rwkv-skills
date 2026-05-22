from __future__ import annotations

"""Shared function-calling protocol helpers."""

from .benchmarks import (
    AGENT_METRIC_KEYS,
    FUNCTION_CALLING_BENCHMARK_SPECS,
    FUNCTION_CALLING_DATASET_PREFIXES,
    FUNCTION_CALLING_EXPLICIT_ONLY_JOBS,
    FUNCTION_CALLING_JOB_NAMES,
    FunctionCallingBenchmarkSpec,
    function_calling_benchmark_spec,
    is_function_calling_job,
)
from .action import ToolAction, tool_action_payload
from .events import FunctionCallEvent, event_payloads
from .parser import ToolCallParseResult, parse_json_tool_calls
from .payload import (
    FunctionCallRunStats,
    build_agent_completion_payload,
    build_one_step_completion_payload,
    extract_function_call_text,
    extract_stats,
)
from .schema import normalize_tool_schema, normalize_tool_schemas
from .score import FunctionCallScore
from .types import ParseError, ToolCall, ToolSchema

__all__ = [
    "FunctionCallEvent",
    "FunctionCallRunStats",
    "FunctionCallScore",
    "FunctionCallingBenchmarkSpec",
    "AGENT_METRIC_KEYS",
    "FUNCTION_CALLING_BENCHMARK_SPECS",
    "FUNCTION_CALLING_DATASET_PREFIXES",
    "FUNCTION_CALLING_EXPLICIT_ONLY_JOBS",
    "FUNCTION_CALLING_JOB_NAMES",
    "ParseError",
    "ToolAction",
    "ToolCall",
    "ToolCallParseResult",
    "ToolSchema",
    "build_agent_completion_payload",
    "build_one_step_completion_payload",
    "event_payloads",
    "extract_function_call_text",
    "extract_stats",
    "function_calling_benchmark_spec",
    "is_function_calling_job",
    "normalize_tool_schema",
    "normalize_tool_schemas",
    "parse_json_tool_calls",
    "tool_action_payload",
]
