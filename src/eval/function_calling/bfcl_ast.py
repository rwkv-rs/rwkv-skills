from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from src.eval.function_calling.bfcl_exec import bfcl_official_ast_checker_status
from src.eval.function_calling.runner_common import ResolvedFunctionCallingRun
from src.eval.function_calling.simple_tool_call import (
    SimpleToolCallEvaluation,
    SimpleToolCallRecord,
    evaluate_simple_tool_calls,
    load_simple_tool_call_manifest_records,
    _expectation_payload,
)

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext

_BFCL_OFFICIAL_MODEL_NAME = "gorilla-openfunctions-v2"


def evaluate_bfcl_ast_calls(
    record: SimpleToolCallRecord,
    decoded_calls: Sequence[Mapping[str, Any]],
    *,
    parse_error: str | None = None,
) -> SimpleToolCallEvaluation:
    if str(record.metadata.get("source_format") or "") != "official_bfcl_v4_ast":
        return evaluate_simple_tool_calls(record, decoded_calls, parse_error=parse_error)

    actual = _official_model_output(decoded_calls)
    expected = [_expectation_payload(item) for item in record.expected_tool_calls]
    details: dict[str, Any] = {
        "expected_tool_calls": expected,
        "decoded_tool_calls": [
            {"name": str(item.get("name") or ""), "arguments": dict(item.get("arguments") or {})}
            for item in decoded_calls
        ],
        "tool_count_ok": len(decoded_calls) == len(record.expected_tool_calls),
        "call_matches": [],
        "parse_error": parse_error or "",
        "bfcl_official_ast": {
            "source": "gorilla/berkeley-function-call-leaderboard",
            "official_root": _record_official_root(record),
            "test_category": _record_test_category(record),
            "language": _record_language(record),
            "model_output": actual,
            "possible_answer": _record_possible_answer(record),
        },
    }
    if parse_error:
        details["bfcl_official_ast"]["skipped"] = "parse_error"
        return SimpleToolCallEvaluation(0.0, False, parse_error, details)

    try:
        checker_result = run_bfcl_official_ast_checker(record, actual)
    except Exception as exc:  # noqa: BLE001
        details["bfcl_official_ast"]["checker_error"] = str(exc)
        return SimpleToolCallEvaluation(0.0, False, f"bfcl_official_ast_checker_error:{exc}", details)

    details["bfcl_official_ast"]["checker_result"] = checker_result
    is_passed = bool(checker_result.get("valid"))
    fail_reason = "" if is_passed else _official_checker_failure(checker_result)
    return SimpleToolCallEvaluation(
        reward=1.0 if is_passed else 0.0,
        is_passed=is_passed,
        fail_reason=fail_reason,
        details=details,
    )


def run_bfcl_official_ast_checker(
    record: SimpleToolCallRecord,
    model_output: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root = _record_official_root(record)
    ast_checker, language_enum = _load_official_ast_checker(root)
    language_name = _record_language(record).upper()
    try:
        language = getattr(language_enum, language_name)
    except AttributeError as exc:
        raise RuntimeError(f"unsupported BFCL AST language: {language_name.lower()}") from exc
    result = ast_checker(
        _record_function_description(record),
        list(model_output),
        _record_possible_answer(record),
        language,
        _record_test_category(record),
        str(record.metadata.get("bfcl_official_model_name") or _BFCL_OFFICIAL_MODEL_NAME),
    )
    return dict(result) if isinstance(result, Mapping) else {"valid": False, "error": [str(result)]}


def preflight_bfcl_ast_runtime(
    records: Sequence[SimpleToolCallRecord],
    *,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    official_records = [record for record in records if str(record.metadata.get("source_format") or "") == "official_bfcl_v4_ast"]
    for record in official_records:
        if not _record_function_description(record):
            errors.append(f"missing_bfcl_official_function:{record.task_id}")
        if not _record_possible_answer(record):
            errors.append(f"missing_bfcl_official_ground_truth:{record.task_id}")

    roots = sorted({_record_official_root(record) for record in official_records})
    for root in roots:
        status = bfcl_official_ast_checker_status(root)
        if status.available:
            continue
        if status.missing_dependencies:
            errors.append(f"missing_bfcl_ast_dependencies:{','.join(status.missing_dependencies)}")
        if status.import_error:
            errors.append(f"bfcl_ast_import_error:{status.import_error}")
        if not status.missing_dependencies and not status.import_error:
            errors.append(f"bfcl_ast_unavailable:{status.official_root}")

    report = {"ok": not errors, "errors": errors, "checked_roots": roots}
    if errors and raise_on_error:
        raise RuntimeError("BFCL official AST runtime preflight failed: " + "; ".join(errors))
    return report


def _run_bfcl_ast(
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    *,
    run_context: "RunContext | None" = None,
) -> int:
    if not bool(getattr(args, "skip_runtime_preflight", False)):
        preflight_bfcl_ast_runtime(load_simple_tool_call_manifest_records(run.dataset_path))
    from src.eval.function_calling.simple_tool_call import _run_simple_tool_call

    return _run_simple_tool_call(
        args,
        run,
        default_job_name="function_bfcl_ast",
        evaluator=evaluate_bfcl_ast_calls,
        run_context=run_context,
    )


def _official_model_output(decoded_calls: Sequence[Mapping[str, Any]]) -> list[dict[str, dict[str, Any]]]:
    result: list[dict[str, dict[str, Any]]] = []
    for call in decoded_calls:
        name = str(call.get("name") or call.get("tool_name") or "").strip()
        arguments = call.get("arguments")
        result.append({name: dict(arguments) if isinstance(arguments, Mapping) else {}})
    return result


def _record_function_description(record: SimpleToolCallRecord) -> list[dict[str, Any]]:
    raw = record.metadata.get("bfcl_official_function")
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    return [_officialize_tool_schema(tool) for tool in record.tools]


def _record_possible_answer(record: SimpleToolCallRecord) -> list[dict[str, Any]]:
    raw = record.metadata.get("bfcl_official_ground_truth")
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    result: list[dict[str, Any]] = []
    for expected in record.expected_tool_calls:
        result.append({expected.name: {key: list(value) for key, value in expected.argument_options.items()}})
    return result


def _record_test_category(record: SimpleToolCallRecord) -> str:
    category = str(record.metadata.get("category") or "").strip()
    if category:
        return category
    return record.task_id.rsplit("_", 1)[0]


def _record_language(record: SimpleToolCallRecord) -> str:
    value = str(record.metadata.get("bfcl_official_language") or "python").strip().lower()
    if value in {"js", "javascript"}:
        return "javascript"
    if value in {"java"}:
        return "java"
    return "python"


def _record_official_root(record: SimpleToolCallRecord) -> str:
    return str(record.metadata.get("official_root") or "").strip()


def _officialize_tool_schema(tool: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(tool)
    parameters = payload.get("parameters")
    if isinstance(parameters, Mapping):
        parameters = dict(parameters)
        if str(parameters.get("type") or "").lower() == "object":
            parameters["type"] = "dict"
        payload["parameters"] = parameters
    return payload


def _official_checker_failure(result: Mapping[str, Any]) -> str:
    error_type = str(result.get("error_type") or "bfcl_official_ast_mismatch")
    errors = result.get("error")
    if isinstance(errors, list) and errors:
        return f"{error_type}:{errors[0]}"
    if errors:
        return f"{error_type}:{errors}"
    return error_type


def _load_official_ast_checker(root: str) -> tuple[Callable[..., Any], Any]:
    status = bfcl_official_ast_checker_status(root or None)
    if not status.available:
        parts: list[str] = []
        if status.missing_dependencies:
            parts.append("missing dependencies: " + ", ".join(status.missing_dependencies))
        if status.import_error:
            parts.append(status.import_error)
        if not parts:
            parts.append(f"missing official BFCL checker under {status.official_root}")
        raise RuntimeError("; ".join(parts))
    with _official_import_context(status.official_root):
        from bfcl_eval.constants.enums import Language
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

    return ast_checker, Language


@contextmanager
def _official_import_context(root: str):
    added = False
    if root and root not in sys.path:
        sys.path.insert(0, root)
        added = True
    try:
        yield
    finally:
        if added:
            try:
                sys.path.remove(root)
            except ValueError:
                pass


__all__ = [
    "evaluate_bfcl_ast_calls",
    "preflight_bfcl_ast_runtime",
    "run_bfcl_official_ast_checker",
    "_run_bfcl_ast",
]
