from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.eval.datasets.data_struct.function_call import FunctionCallTaskRecord


@dataclass(frozen=True, slots=True)
class OneStepScoreResult:
    passed: bool
    fail_reason: str
    reference: str = ""
    details: dict[str, Any] = field(default_factory=dict)


def score_one_step_prediction(
    record: FunctionCallTaskRecord | None,
    prediction: str,
) -> OneStepScoreResult:
    if record is None:
        return OneStepScoreResult(False, "missing_record")
    if not record.expected_tool_calls:
        return OneStepScoreResult(False, "missing_expected_tool_calls")
    if _looks_like_template_leak(prediction):
        return OneStepScoreResult(False, "decision stage leaked internal template/control tokens")

    from src.eval.function_calling.one_step.simple_tool_call import decode_simple_tool_call_response

    parse_error: str | None = None
    decoded_calls: list[dict[str, Any]] = []
    try:
        decoded_calls = decode_simple_tool_call_response(prediction)
    except Exception as exc:  # noqa: BLE001
        parse_error = str(exc)
    if uses_bfcl_exec_scorer(record):
        from src.eval.function_calling.one_step.bfcl_exec import evaluate_bfcl_executable_calls

        result = evaluate_bfcl_executable_calls(record, decoded_calls, parse_error=parse_error)
    elif uses_apibank_official_scorer(record):
        from src.eval.function_calling.one_step.apibank import evaluate_apibank_official_calls

        result = evaluate_apibank_official_calls(record, decoded_calls, parse_error=parse_error)
    elif uses_toolalpaca_official_scorer(record):
        from src.eval.function_calling.one_step.toolalpaca import evaluate_toolalpaca_official_calls

        result = evaluate_toolalpaca_official_calls(record, decoded_calls, parse_error=parse_error)
    else:
        from src.eval.function_calling.one_step.simple_tool_call import (
            evaluate_simple_tool_calls,
            normalize_simple_tool_call_manifest_row,
        )

        simple_record = normalize_simple_tool_call_manifest_row(
            {
                "task_id": record.task_id,
                "instruction": record.instruction,
                "tools": record.tools,
                "expected_tool_calls": record.expected_tool_calls,
                "metadata": record.metadata,
            },
            index=0,
        )
        result = evaluate_simple_tool_calls(simple_record, decoded_calls, parse_error=parse_error)
    reference = json.dumps(
        result.details.get("expected_tool_calls") or [],
        ensure_ascii=False,
        sort_keys=True,
    )
    return OneStepScoreResult(
        passed=bool(result.is_passed),
        fail_reason=result.fail_reason,
        reference=reference,
        details=dict(result.details),
    )


def uses_bfcl_exec_scorer(record: FunctionCallTaskRecord) -> bool:
    scorer_type = str((record.scorer or {}).get("type") or "")
    if scorer_type == "bfcl_exec":
        return True
    category = str((record.metadata or {}).get("category") or "")
    if category in {"exec_simple", "exec_multiple"}:
        return True
    return str(record.task_id or "").startswith(("exec_simple_", "exec_multiple_"))


def uses_toolalpaca_official_scorer(record: FunctionCallTaskRecord) -> bool:
    scorer_type = str((record.scorer or {}).get("type") or "")
    if scorer_type == "toolalpaca_official":
        return True
    source_format = str((record.metadata or {}).get("source_format") or "")
    return source_format == "official_toolalpaca" and str(record.task_id or "").startswith("toolalpaca_")


def uses_apibank_official_scorer(record: FunctionCallTaskRecord) -> bool:
    scorer_type = str((record.scorer or {}).get("type") or "")
    if scorer_type == "apibank_official":
        return True
    source_format = str((record.metadata or {}).get("source_format") or "")
    return source_format == "official_apibank" and str(record.task_id or "").startswith("apibank_")


def record_env_type(record: FunctionCallTaskRecord | None) -> str:
    if record is None:
        return "unknown"
    env = record.env or {}
    return str(env.get("type") or "simple_tool_call")


_TEMPLATE_LEAK_MARKERS = (
    "<system message>",
    "</system message>",
    "<assistant>",
    "</assistant>",
    "<user_input>",
    "</user_input>",
)


def _looks_like_template_leak(text: str) -> bool:
    lowered = str(text or "").lower()
    if not lowered:
        return False
    if "<system message>" in lowered and "you are a helpful assistant" in lowered:
        return True
    marker_hits = sum(lowered.count(marker) for marker in _TEMPLATE_LEAK_MARKERS)
    return marker_hits >= 3


__all__ = [
    "OneStepScoreResult",
    "record_env_type",
    "score_one_step_prediction",
    "uses_apibank_official_scorer",
    "uses_bfcl_exec_scorer",
    "uses_toolalpaca_official_scorer",
]
