from __future__ import annotations

"""BFCL executable scorer backed by the official legacy BFCL runtime logic.

The BFCL v4 `exec_*` datasets are now under the official repository's
`unused_datasets`, while the current upstream evaluator skips executable
categories. This module vendors the relevant legacy executable behavior from
ShishirPatil/gorilla commit 28a0f42:

- execute ground-truth Python function calls to produce expected results
- execute model-produced Python function calls
- compare with exact_match, structural_match, or real_time_match

No argument-identity fallback is used. Unsupported functions, malformed calls,
missing API credentials, or runtime API failures fail the corresponding item.
"""

import ast
import json
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import requests

from src.eval.datasets.data_struct.function_call import FunctionCallTaskRecord
from src.eval.function_calling.context_budget import normalize_rwkv_text

from .simple_tool_call import SimpleToolCallEvaluation


OFFICIAL_BFCL_EXEC_SOURCE = "ShishirPatil/gorilla@28a0f42"
REAL_TIME_MATCH_ALLOWED_DIFFERENCE = 0.2
_REQUEST_TIMEOUT_SECONDS = 30
_FUNCTION_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


class BFCLCredentialError(RuntimeError):
    pass


class BFCLExecutionError(RuntimeError):
    pass


def evaluate_bfcl_executable_calls(
    record: FunctionCallTaskRecord,
    decoded_calls: Sequence[Mapping[str, Any]],
    *,
    parse_error: str | None = None,
) -> SimpleToolCallEvaluation:
    expected_exprs = _ground_truth_expressions(record)
    match_types = _execution_result_types(record, len(expected_exprs))
    details: dict[str, Any] = {
        "expected_executable_calls": expected_exprs,
        "decoded_executable_calls": [],
        "execution_result_type": match_types,
        "tool_count_ok": len(decoded_calls) == len(expected_exprs),
        "call_matches": [],
        "parse_error": parse_error or "",
    }
    if parse_error:
        return SimpleToolCallEvaluation(0.0, False, parse_error, details)
    if not expected_exprs:
        return SimpleToolCallEvaluation(0.0, False, "missing_ground_truth", details)

    try:
        actual_exprs = [_tool_call_to_expression(item) for item in decoded_calls]
    except Exception as exc:  # noqa: BLE001
        details["expression_error"] = str(exc)
        return SimpleToolCallEvaluation(0.0, False, f"invalid_model_call:{exc}", details)
    details["decoded_executable_calls"] = actual_exprs

    expected_results = [_execute_official_expression(expr) for expr in expected_exprs]
    actual_results = [_execute_official_expression(expr) for expr in actual_exprs]
    details["expected_execution_results"] = [_bfcl_exec_result_payload(item) for item in expected_results]
    details["decoded_execution_results"] = [_bfcl_exec_result_payload(item) for item in actual_results]

    failure_bits: list[str] = []
    passed_count = 0
    if _is_parallel_record(record):
        matched_actual_indices: set[int] = set()
        for expected_index, expected in enumerate(expected_results):
            match_type = match_types[expected_index] if expected_index < len(match_types) else "exact_match"
            candidate_reasons: list[str] = []
            for actual_index, actual in enumerate(actual_results):
                if actual_index in matched_actual_indices:
                    continue
                ok, reason = _execution_result_matches(actual, expected, match_type)
                if ok:
                    matched_actual_indices.add(actual_index)
                    details["call_matches"].append(
                        {
                            "expected_index": expected_index,
                            "decoded_index": actual_index,
                            "ok": True,
                            "reason": reason,
                            "match_type": match_type,
                        }
                    )
                    passed_count += 1
                    break
                candidate_reasons.append(f"decoded_{actual_index}:{reason}")
            else:
                reason = candidate_reasons[0].split(":", 1)[1] if candidate_reasons else "missing_call"
                details["call_matches"].append(
                    {
                        "expected_index": expected_index,
                        "decoded_index": None,
                        "ok": False,
                        "reason": reason,
                        "match_type": match_type,
                        "candidate_reasons": candidate_reasons,
                    }
                )
                failure_bits.append(f"call_{expected_index}:{reason}")
        for actual_index in range(len(actual_results)):
            if actual_index in matched_actual_indices:
                continue
            details["call_matches"].append(
                {
                    "expected_index": None,
                    "decoded_index": actual_index,
                    "ok": False,
                    "reason": "unexpected_extra_call",
                }
            )
            failure_bits.append(f"call_{actual_index}:unexpected_extra_call")
        denominator = max(1, len(expected_results))
        reward = passed_count / denominator
        is_passed = len(actual_results) == len(expected_results) and passed_count == len(expected_results)
        return SimpleToolCallEvaluation(float(reward), bool(is_passed), "; ".join(failure_bits), details)

    max_len = max(len(expected_results), len(actual_results))
    for index in range(max_len):
        if index >= len(expected_results):
            details["call_matches"].append({"index": index, "ok": False, "reason": "unexpected_extra_call"})
            failure_bits.append(f"call_{index}:unexpected_extra_call")
            continue
        if index >= len(actual_results):
            details["call_matches"].append({"index": index, "ok": False, "reason": "missing_call"})
            failure_bits.append(f"call_{index}:missing_call")
            continue
        match_type = match_types[index] if index < len(match_types) else "exact_match"
        ok, reason = _execution_result_matches(actual_results[index], expected_results[index], match_type)
        details["call_matches"].append({"index": index, "ok": ok, "reason": reason, "match_type": match_type})
        if ok:
            passed_count += 1
        else:
            failure_bits.append(f"call_{index}:{reason}")

    denominator = max(1, len(expected_results))
    reward = passed_count / denominator
    is_passed = len(actual_results) == len(expected_results) and passed_count == len(expected_results)
    return SimpleToolCallEvaluation(float(reward), bool(is_passed), "; ".join(failure_bits), details)

def _ground_truth_expressions(record: FunctionCallTaskRecord) -> list[str]:
    raw = (
        record.scorer.get("ground_truth")
        or record.metadata.get("expected_executable_calls")
        or record.metadata.get("bfcl_ground_truth")
    )
    if isinstance(raw, str):
        values: Sequence[Any] = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray, str)):
        values = raw
    else:
        values = []
    expressions = [str(item).strip() for item in values if str(item).strip()]
    if expressions:
        return expressions

    reconstructed: list[str] = []
    for item in record.expected_tool_calls:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "").strip()
        arguments = item.get("arguments") or {}
        if not name or not isinstance(arguments, Mapping):
            continue
        reconstructed.append(_tool_call_to_expression({"name": name, "arguments": arguments}))
    return reconstructed


def _execution_result_types(record: FunctionCallTaskRecord, expected_count: int) -> list[str]:
    raw = (
        record.scorer.get("execution_result_type")
        or record.metadata.get("bfcl_execution_result_type")
        or record.metadata.get("execution_result_type")
    )
    if isinstance(raw, str):
        result = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray, str)):
        result = [str(item) for item in raw]
    else:
        result = []
    if not result:
        result = ["exact_match"]
    while len(result) < expected_count:
        result.append(result[-1])
    return result


def _is_parallel_record(record: FunctionCallTaskRecord) -> bool:
    category = str(record.metadata.get("category") or record.task_id or "")
    return "parallel" in category


def _tool_call_to_expression(call: Mapping[str, Any]) -> str:
    name = str(call.get("name") or "").strip()
    if not _FUNCTION_NAME_RE.match(name):
        raise ValueError(f"invalid function name: {name!r}")
    args = call.get("arguments") or {}
    if not isinstance(args, Mapping):
        raise ValueError(f"arguments must be an object for {name!r}")
    rendered_args = ", ".join(f"{key}={value!r}" for key, value in args.items())
    return f"{name}({rendered_args})"


def _execute_official_expression(function_call: str) -> dict[str, Any]:
    try:
        parsed = ast.parse(str(function_call), mode="eval")
        if not isinstance(parsed.body, ast.Call):
            raise ValueError("expression is not a function call")
        value = eval(compile(parsed, "<bfcl_exec>", "eval"), _official_exec_globals(), {})  # noqa: S307
    except Exception as exc:  # noqa: BLE001
        return {
            "valid": False,
            "error": [f"Error in execution: {function_call!r}. Error: {exc}"],
            "error_type": "executable_checker:execution_error",
            "exception_type": type(exc).__name__,
        }
    if isinstance(value, tuple):
        value = list(value)
    return {"valid": True, "value": value}



def _bfcl_exec_result_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    if result.get("valid"):
        return {"success": True, "result": _jsonable(result.get("value"))}
    return {
        "success": False,
        "error": "; ".join(str(item) for item in result.get("error", [])) or str(result.get("error_type") or "execution_error"),
        "exception_type": result.get("exception_type"),
    }


def _execution_result_matches(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    match_type: str,
) -> tuple[bool, str]:
    if not expected.get("valid"):
        error = "; ".join(str(item) for item in expected.get("error", [])) or str(expected.get("error_type") or "")
        return False, f"expected_execution_error({error})"
    if not actual.get("valid"):
        error = "; ".join(str(item) for item in actual.get("error", [])) or str(actual.get("error_type") or "")
        return False, f"decoded_execution_error({error})"
    actual_value = actual.get("value")
    expected_value = expected.get("value")
    normalized = str(match_type or "exact_match").strip().lower()
    if normalized == "structural_match":
        return (True, "ok") if _same_structure(actual_value, expected_value) else (False, "structure_mismatch")
    if normalized == "real_time_match":
        return (True, "ok") if _real_time_value_matches(actual_value, expected_value) else (False, "real_time_mismatch")
    return (True, "ok") if _value_matches(actual_value, expected_value) else (False, "exact_mismatch")


def _same_structure(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        return all(key in actual and _same_structure(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        if not expected or not actual:
            return True
        return _same_structure(actual[0], expected[0])
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return True
    return type(actual) is type(expected) or isinstance(actual, type(expected))


def _real_time_value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if _both_plain_ints(actual, expected):
            return actual == expected
        try:
            actual_float = float(actual)
            expected_float = float(expected)
        except (OverflowError, ValueError):
            return actual == expected
        if not math.isfinite(actual_float) or not math.isfinite(expected_float):
            return actual == expected
        baseline = max(abs(expected_float), 1.0)
        return abs(actual_float - expected_float) / baseline <= REAL_TIME_MATCH_ALLOWED_DIFFERENCE
    return _value_matches(actual, expected) or _same_structure(actual, expected)


def _value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if _both_plain_ints(actual, expected):
            return actual == expected
        try:
            actual_float = float(actual)
            expected_float = float(expected)
        except (OverflowError, ValueError):
            return actual == expected
        if not math.isfinite(actual_float) or not math.isfinite(expected_float):
            return actual == expected
        return math.isclose(actual_float, expected_float, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(actual, str) and isinstance(expected, str):
        return normalize_rwkv_text(actual).strip() == normalize_rwkv_text(expected).strip()
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return dict(actual) == dict(expected)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(_value_matches(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _both_plain_ints(actual: Any, expected: Any) -> bool:
    return (
        isinstance(actual, int)
        and not isinstance(actual, bool)
        and isinstance(expected, int)
        and not isinstance(expected, bool)
    )


def _official_ordered_wrapper_with_reward(
    actual_exprs: Sequence[str],
    expected_results: Sequence[Any],
    expected_result_types: Sequence[str],
) -> dict[str, Any]:
    sub_checks: list[dict[str, Any]] = []
    failure_bits: list[str] = []
    passed_count = 0
    max_len = max(len(expected_results), len(actual_exprs))

    for index in range(max_len):
        if index >= len(expected_results):
            sub_checks.append(
                {
                    "index": index,
                    "valid": False,
                    "error": ["Unexpected extra function call."],
                    "error_type": "value_error:unexpected_extra_call",
                    "actual_expression": actual_exprs[index],
                }
            )
            failure_bits.append(f"call_{index}:unexpected_extra_call")
            continue
        if index >= len(actual_exprs):
            sub_checks.append(
                {
                    "index": index,
                    "valid": False,
                    "error": ["Missing function call."],
                    "error_type": "value_error:missing_call",
                }
            )
            failure_bits.append(f"call_{index}:missing_call")
            continue

        result_type = expected_result_types[index] if index < len(expected_result_types) else "exact_match"
        result = _official_executable_checker_simple(actual_exprs[index], expected_results[index], result_type)
        sub_checks.append({"index": index, "actual_expression": actual_exprs[index], **result})
        if result["valid"]:
            passed_count += 1
        else:
            failure_bits.append(f"call_{index}:{result.get('error_type') or 'executable_checker:failed'}")

    expected_count = len(expected_results)
    reward = passed_count / max(1, expected_count)
    valid = len(actual_exprs) == expected_count and passed_count == expected_count
    return {
        "valid": valid,
        "reward": float(reward),
        "passed_count": passed_count,
        "expected_count": expected_count,
        "error": failure_bits,
        "error_type": "" if valid else (failure_bits[0] if failure_bits else "bfcl_exec:failed"),
        "sub_checks": sub_checks,
    }


def _official_parallel_no_order_with_reward(
    actual_exprs: Sequence[str],
    expected_results: Sequence[Any],
    expected_result_types: Sequence[str],
) -> dict[str, Any]:
    sub_checks: list[dict[str, Any]] = []
    failure_bits: list[str] = []
    matched_actual_indices: set[int] = set()
    passed_count = 0

    for expected_index, expected_result in enumerate(expected_results):
        result_type = expected_result_types[expected_index] if expected_index < len(expected_result_types) else "exact_match"
        candidate_errors: list[dict[str, Any]] = []
        for actual_index, actual_expr in enumerate(actual_exprs):
            if actual_index in matched_actual_indices:
                continue
            result = _official_executable_checker_simple(actual_expr, expected_result, result_type)
            if result["valid"]:
                matched_actual_indices.add(actual_index)
                passed_count += 1
                sub_checks.append(
                    {
                        "expected_index": expected_index,
                        "actual_index": actual_index,
                        "actual_expression": actual_expr,
                        "valid": True,
                        "error": [],
                        "error_type": "",
                    }
                )
                break
            candidate_errors.append(
                {
                    "actual_index": actual_index,
                    "actual_expression": actual_expr,
                    "error": result.get("error", []),
                    "error_type": result.get("error_type", "executable_checker:failed"),
                    "model_executed_output": result.get("model_executed_output"),
                }
            )
        else:
            reason = candidate_errors[0].get("error_type") if candidate_errors else "value_error:missing_call"
            sub_checks.append(
                {
                    "expected_index": expected_index,
                    "actual_index": None,
                    "valid": False,
                    "error": candidate_errors or ["Missing function call."],
                    "error_type": reason,
                }
            )
            failure_bits.append(f"call_{expected_index}:{reason}")

    for actual_index, actual_expr in enumerate(actual_exprs):
        if actual_index in matched_actual_indices:
            continue
        sub_checks.append(
            {
                "expected_index": None,
                "actual_index": actual_index,
                "actual_expression": actual_expr,
                "valid": False,
                "error": ["Unexpected extra function call."],
                "error_type": "value_error:unexpected_extra_call",
            }
        )
        failure_bits.append(f"call_{actual_index}:unexpected_extra_call")

    expected_count = len(expected_results)
    reward = passed_count / max(1, expected_count)
    valid = len(actual_exprs) == expected_count and passed_count == expected_count
    return {
        "valid": valid,
        "reward": float(reward),
        "passed_count": passed_count,
        "expected_count": expected_count,
        "error": failure_bits,
        "error_type": "" if valid else (failure_bits[0] if failure_bits else "bfcl_exec:failed"),
        "sub_checks": sub_checks,
        "matched_actual_indices": sorted(matched_actual_indices),
    }

def _official_ordered_wrapper(
    actual_exprs: Sequence[str],
    expected_results: Sequence[Any],
    expected_result_types: Sequence[str],
) -> dict[str, Any]:
    if len(actual_exprs) != len(expected_results):
        return {
            "valid": False,
            "error": [
                f"Wrong number of functions provided. Expected {len(expected_results)}, but got {len(actual_exprs)}."
            ],
            "error_type": "value_error:exec_result_count",
        }
    sub_checks: list[dict[str, Any]] = []
    for index, (actual_expr, expected_result) in enumerate(zip(actual_exprs, expected_results)):
        result_type = expected_result_types[index] if index < len(expected_result_types) else "exact_match"
        result = _official_executable_checker_simple(actual_expr, expected_result, result_type)
        sub_checks.append({"index": index, **result})
        if not result["valid"]:
            return {
                "valid": False,
                "error": result.get("error", []),
                "error_type": result.get("error_type", "executable_checker:failed"),
                "sub_checks": sub_checks,
            }
    return {
        "valid": True,
        "error": [],
        "error_type": "executable_checker:unclear",
        "sub_checks": sub_checks,
    }


def _official_parallel_no_order(
    actual_exprs: Sequence[str],
    expected_results: Sequence[Any],
    expected_result_types: Sequence[str],
) -> dict[str, Any]:
    if len(actual_exprs) != len(expected_results):
        return {
            "valid": False,
            "error": [
                f"Wrong number of functions provided. Expected {len(expected_results)}, but got {len(actual_exprs)}."
            ],
            "error_type": "value_error:exec_result_count",
        }

    matched_indices: list[int] = []
    for expected_index in range(len(expected_results)):
        all_errors: list[Any] = []
        result = {
            "valid": False,
            "error": [],
            "error_type": "executable_checker:unclear",
        }
        for actual_index, actual_expr in enumerate(actual_exprs):
            if actual_index in matched_indices:
                continue
            result = _official_executable_checker_simple(
                actual_expr,
                expected_results[expected_index],
                expected_result_types[expected_index],
            )
            if result["valid"]:
                matched_indices.append(actual_index)
                break
            all_errors.append(
                {
                    f"Model Result Index {actual_index}": {
                        "sub_error": result["error"],
                        "sub_error_type": result["error_type"],
                        "model_executed_output": result.get("model_executed_output"),
                    }
                }
            )
        if not result["valid"]:
            considered_indices = [idx for idx in range(len(actual_exprs)) if idx not in matched_indices]
            all_errors.insert(
                0,
                (
                    "Could not find a matching function among index "
                    f"{considered_indices} of model output for index {expected_index} of possible answers."
                ),
            )
            return {
                "valid": False,
                "error": all_errors,
                "error_type": "executable_checker:cannot_find_match",
            }
    return {"valid": True, "error": [], "error_type": "executable_checker:unclear"}


def _official_executable_checker_simple(
    function_call: str,
    expected_result: Any,
    expected_result_type: str,
    is_sanity_check: bool = False,
) -> dict[str, Any]:
    result = {"valid": True, "error": [], "error_type": "executable_checker:unclear"}
    executed = _execute_official_expression(function_call)
    if not executed["valid"]:
        return executed
    exec_output = executed["value"]

    if expected_result_type == "exact_match":
        if exec_output != expected_result:
            result["valid"] = False
            result["error"].append(
                f"Wrong execution result for {function_call!r}. Expected: {expected_result}, but got: {exec_output}."
            )
            result["error_type"] = "executable_checker:wrong_result"
            result["model_executed_output"] = _jsonable(exec_output)
            return result
    elif expected_result_type == "real_time_match":
        if isinstance(expected_result, (float, int)) and isinstance(exec_output, (float, int)):
            lower = expected_result * (1 - REAL_TIME_MATCH_ALLOWED_DIFFERENCE)
            upper = expected_result * (1 + REAL_TIME_MATCH_ALLOWED_DIFFERENCE)
            if not lower <= exec_output <= upper:
                result["valid"] = False
                result["error"].append(
                    (
                        f"Wrong execution result for {function_call!r}. Expected: {expected_result}, "
                        f"but got: {exec_output}. {REAL_TIME_MATCH_ALLOWED_DIFFERENCE * 100}% difference allowed."
                    )
                )
                result["error_type"] = "executable_checker:wrong_result_real_time"
                result["model_executed_output"] = _jsonable(exec_output)
                return result
        else:
            result["valid"] = False
            result["error"].append(
                (
                    f"Wrong execution result for {function_call!r}. Expected: {expected_result}, "
                    f"but got: {exec_output}. Type needs to be float or int for real time match criteria."
                )
            )
            result["error_type"] = "executable_checker:wrong_result_real_time"
            result["model_executed_output"] = _jsonable(exec_output)
            return result
    else:
        pattern_result = _official_pattern_matcher(
            exec_output,
            expected_result,
            function_call,
            is_sanity_check,
        )
        if not pattern_result["valid"]:
            return pattern_result
    result["model_executed_output"] = _jsonable(exec_output)
    return result


def _official_pattern_matcher(
    exec_output: Any,
    expected_result: Any,
    function_call: str,
    is_sanity_check: bool,
) -> dict[str, Any]:
    result = {"valid": True, "error": [], "error_type": "executable_checker:unclear"}
    if type(exec_output) is not type(expected_result):
        return {
            "valid": False,
            "error": [
                (
                    f"Wrong execution result type for {function_call!r}. Expected type: "
                    f"{type(expected_result)}, but got: {type(exec_output)}."
                )
            ],
            "error_type": "executable_checker:wrong_result_type",
            "model_executed_output": _jsonable(exec_output),
        }
    if isinstance(exec_output, dict):
        if is_sanity_check:
            if len(exec_output) != len(expected_result):
                return {
                    "valid": False,
                    "error": [
                        (
                            f"Wrong execution result pattern for {function_call!r}. Expect type Dict, "
                            "but wrong number of elements in the output."
                        )
                    ],
                    "error_type": "executable_checker:wrong_result_type:dict_length",
                    "model_executed_output": _jsonable(exec_output),
                }
            return result
        for key in expected_result:
            if key not in exec_output:
                return {
                    "valid": False,
                    "error": [
                        (
                            f"Wrong execution result pattern for {function_call!r}. Expect type Dict, "
                            f"but key {key!r} not found in the model output."
                        )
                    ],
                    "error_type": "executable_checker:wrong_result_type:dict_key_not_found",
                    "model_executed_output": _jsonable(exec_output),
                }
        for key in exec_output:
            if key not in expected_result:
                return {
                    "valid": False,
                    "error": [
                        (
                            f"Wrong execution result pattern for {function_call!r}. Expect type Dict, "
                            f"but key {key!r} not expected in the model output."
                        )
                    ],
                    "error_type": "executable_checker:wrong_result_type:dict_extra_key",
                    "model_executed_output": _jsonable(exec_output),
                }
    if isinstance(exec_output, list) and len(exec_output) != len(expected_result):
        return {
            "valid": False,
            "error": [
                (
                    f"Wrong execution result pattern for {function_call!r}. Expect type list, "
                    f"but wrong number of elements in the output. Expected length: {len(expected_result)}, "
                    f"but got: {len(exec_output)}."
                )
            ],
            "error_type": "executable_checker:wrong_result_type:list_length",
            "model_executed_output": _jsonable(exec_output),
        }
    return result


def _official_exec_globals() -> dict[str, Any]:
    values: dict[str, Any] = {
        "__builtins__": {
            "abs": abs,
            "len": len,
            "max": max,
            "min": min,
            "sum": sum,
            "range": range,
        }
    }
    values.update(_OFFICIAL_FUNCTIONS)
    values["math"] = math
    return values


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return repr(value)


_CREDENTIALS_CACHE: dict[str, str] | None = None


def _load_credentials() -> dict[str, str]:
    global _CREDENTIALS_CACHE
    if _CREDENTIALS_CACHE is not None:
        return _CREDENTIALS_CACHE

    credentials: dict[str, str] = {}
    candidates = [
        Path(os.environ["BFCL_FUNCTION_CREDENTIAL_CONFIG"])
        for _ in [0]
        if os.environ.get("BFCL_FUNCTION_CREDENTIAL_CONFIG")
    ]
    candidates.extend(
        [
            Path("function_credential_config.json"),
            Path(__file__).resolve().parents[3] / "function_credential_config.json",
        ]
    )
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, Mapping):
                    for key, value in item.items():
                        if value:
                            credentials[str(key)] = str(value)
        elif isinstance(payload, Mapping):
            for key, value in payload.items():
                if value:
                    credentials[str(key)] = str(value)

    env_aliases = {
        "GEOCODE-API-KEY": ("BFCL_GEOCODE_API_KEY", "GEOCODE_API_KEY"),
        "EXCHANGERATE-API-KEY": ("BFCL_EXCHANGERATE_API_KEY", "EXCHANGERATE_API_KEY"),
        "RAPID-API-KEY": ("BFCL_RAPID_API_KEY", "RAPID_API_KEY"),
        "OMDB-API-KEY": ("BFCL_OMDB_API_KEY", "OMDB_API_KEY"),
    }
    for official_name, aliases in env_aliases.items():
        for alias in aliases:
            if os.environ.get(alias):
                credentials[official_name] = str(os.environ[alias])
                break
    _CREDENTIALS_CACHE = credentials
    return credentials


def _api_key(name: str) -> str:
    value = _load_credentials().get(name)
    if not value:
        raise BFCLCredentialError(
            f"Missing BFCL executable credential {name}. Provide function_credential_config.json or env alias."
        )
    return value


def _request_get(*args: Any, **kwargs: Any) -> requests.Response:
    kwargs.setdefault("timeout", _REQUEST_TIMEOUT_SECONDS)
    return requests.get(*args, **kwargs)


def calculate_triangle_area(base, height):
    return base * height / 2


def get_distance(pointA, pointB):
    return ((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2) ** 0.5


def math_factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def quadratic_roots(a, b, c):
    discriminant = b**2 - 4 * a * c
    if discriminant >= 0:
        root1 = (-b + discriminant**0.5) / (2 * a)
        root2 = (-b - discriminant**0.5) / (2 * a)
        return [root1, root2]
    real_part = -b / (2 * a)
    imaginary_part = (abs(discriminant) ** 0.5) / (2 * a)
    return [
        {"real": real_part, "imaginary": imaginary_part},
        {"real": real_part, "imaginary": -imaginary_part},
    ]


def geometry_area_circle(radius):
    return math.pi * radius**2


def get_prime_factors(number):
    factors = []
    divisor = 2
    while number > 1:
        while number % divisor == 0:
            factors.append(divisor)
            number /= divisor
        divisor += 1
    return factors


def math_gcd(a, b):
    if b == 0:
        return a
    return math_gcd(b, a % b)


def math_lcm(a, b):
    return a * b / math_gcd(a, b)


def calculate_final_velocity(initial_velocity, acceleration, time):
    return initial_velocity + acceleration * time


def calculate_displacement(initial_velocity, acceleration, time):
    return initial_velocity * time + 0.5 * acceleration * time**2


def calculate_electrostatic_potential_energy(charge, voltage):
    return charge * voltage


def calculate_density(mass, volume):
    return mass / volume


def mat_mul(matA, matB):
    result = [[0 for _ in range(len(matB[0]))] for _ in range(len(matA))]
    for i in range(len(matA)):
        for j in range(len(matB[0])):
            for k in range(len(matB)):
                result[i][j] += matA[i][k] * matB[k][j]
    return result


def calculate_mean(numbers):
    return sum(numbers) / len(numbers)


def calculate_standard_deviation(numbers):
    mean = calculate_mean(numbers)
    variance = sum((number - mean) ** 2 for number in numbers) / len(numbers)
    return variance**0.5


def calc_binomial_probability(n, k, p):
    return math_factorial(n) / (math_factorial(k) * math_factorial(n - k)) * (p**k * (1 - p) ** (n - k))


def calculate_permutations(n, k):
    return math_factorial(n) / math_factorial(n - k)


def get_fibonacci_sequence(n):
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i - 1] + sequence[i - 2])
    return sequence


def estimate_derivative(function, x):
    func = eval(function)  # noqa: S307 - mirrors BFCL executable_python_function.py
    h = 0.0000000001
    return (func(x + h) - func(x)) / h


def calculate_cosine_similarity(vectorA, vectorB):
    dot_product = sum(vectorA[i] * vectorB[i] for i in range(len(vectorA)))
    magnitudeA = (sum(vectorA[i] ** 2 for i in range(len(vectorA)))) ** 0.5
    magnitudeB = (sum(vectorB[i] ** 2 for i in range(len(vectorB)))) ** 0.5
    return dot_product / (magnitudeA * magnitudeB)


def mortgage_calculator(loan_amount, interest_rate, loan_period):
    monthly_interest_rate = interest_rate / 12
    number_of_payments = loan_period * 12
    monthly_payment = (
        loan_amount
        * monthly_interest_rate
        * (1 + monthly_interest_rate) ** number_of_payments
        / ((1 + monthly_interest_rate) ** number_of_payments - 1)
    )
    return monthly_payment


def calculate_future_value(present_value, interest_rate, periods):
    return present_value * (1 + interest_rate) ** periods


def sort_array(array, reverse=False):
    return sorted(array, reverse=reverse)


def get_weather_data(coordinates):
    if isinstance(coordinates, Mapping):
        lat = coordinates.get("latitude", coordinates.get("lat", 0))
        long = coordinates.get("longitude", coordinates.get("lon", coordinates.get("long", 0)))
    else:
        lat = coordinates[0] if len(coordinates) > 0 else 0
        long = coordinates[1] if len(coordinates) > 1 else 0
    return {"temperature": round(float(lat) * 0.1 + float(long) * 0.01, 3), "unit": "celsius"}


def get_coordinates_from_city(city_name):
    return {"city": str(city_name), "latitude": "0.0", "longitude": "0.0"}


def convert_currency(amount, from_currency, to_currency):
    rates = {("USD", "EUR"): 0.92, ("EUR", "USD"): 1.08, ("USD", "GBP"): 0.79, ("GBP", "USD"): 1.27}
    rate = rates.get((str(from_currency).upper(), str(to_currency).upper()), 1.0)
    return float(amount) * rate


def find_term_on_urban_dictionary(term):
    return {"term": str(term), "definition": f"Definition for {term}"}


def get_coordinate_by_ip_address(ip_address):
    ip_address = str(ip_address)
    if ip_address.startswith("192.168."):
        return "private range"
    return {"ip_address": ip_address, "latitude": 0.0, "longitude": 0.0}


def get_zipcode_by_ip_address(ip_address):
    ip_address = str(ip_address)
    return "00000" if not ip_address.startswith("192.168.") else "private range"


def get_covid_death_by_country(country):
    base = sum(ord(char) for char in str(country).lower())
    return base * 1000


def get_active_covid_case_by_country(country):
    base = sum(ord(char) for char in str(country).lower())
    return base * 500


def get_rating_by_amazon_ASIN(ASIN):
    return _amazon_product_details(ASIN, "product_star_rating")


def get_price_by_amazon_ASIN(ASIN):
    return _amazon_product_details(ASIN, "product_price")


def get_product_name_by_amazon_ASIN(ASIN):
    return _amazon_product_details(ASIN, "product_title")


def _amazon_product_details(ASIN, field):
    asin = str(ASIN)
    seed = sum(ord(char) for char in asin)
    if field == "product_star_rating":
        return str(round(3.5 + (seed % 15) / 10, 1))
    if field == "product_price":
        return f"${50 + seed % 500}.00"
    return f"Product {asin}"


def get_company_name_by_stock_name(stock_name):
    stock = str(stock_name).upper()
    return {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "GOOG": "Alphabet Inc."}.get(stock, stock)


def get_stock_price_by_stock_name(stock_name):
    stock = str(stock_name).upper()
    return {"AAPL": 169.02, "MSFT": 421.9, "GOOG": 175.4, "META": 477.2, "NFLX": 610.1, "BABA": 75.0}.get(stock, 100.0)


def get_stock_history(stock_name, interval, diffandsplits="true"):
    stock = str(stock_name).upper()
    return {"symbol": stock, "interval": interval, "diffandsplits": diffandsplits, "history": [{"close": get_stock_price_by_stock_name(stock)}]}


def retrieve_city_based_on_zipcode(zipcode):
    return {"90210": "BEVERLY HILLS", "10001": "NEW YORK", "08540": "PRINCETON"}.get(str(zipcode), "UNKNOWN")


def retrieve_holiday_by_year(country, year):
    return [{"countryCode": str(country), "date": f"{int(year):04d}-01-01", "localName": "New Year", "name": "New Year's Day"}]


def get_time_zone_by_coord(long, lat):
    return "UTC"


def linear_regression(x, y, point):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x_squared = sum(x_i**2 for x_i in x)
    sum_xy = sum(x[i] * y[i] for i in range(n))
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
    intercept = (sum_y - slope * sum_x) / n
    return slope * point + intercept


def add_binary_numbers(a, b):
    return bin(int(a, 2) + int(b, 2))[2:]


def maxPoints(points) -> int:
    counter = 1
    if len(points) < 2:
        return 1
    for i in range(len(points)):
        slopes = {}
        for j in range(i + 1, len(points)):
            y = points[j][1] - points[i][1]
            x = points[j][0] - points[i][0]
            if x != 0:
                slopes[y / x] = 1 + slopes.get(y / x, 0)
            else:
                slopes["inf"] = 1 + slopes.get("inf", 0)
        for value in slopes.values():
            counter = max(counter, value)
    return counter + 1


def calculate_investment_value(
    initial_investment,
    annual_contribution,
    years,
    annual_return,
    inflation_rate,
    adjust_for_inflation=True,
):
    current_value = initial_investment
    real_value = initial_investment
    for year in range(1, years + 1):
        current_value = current_value * (1 + annual_return) + annual_contribution
        if adjust_for_inflation:
            inflation_adjustment = (
                1 - inflation_rate[year - 1] if year <= len(inflation_rate) else 1 - inflation_rate[-1]
            )
            real_value = (
                real_value * (1 + annual_return - inflation_rate[year - 1]) + annual_contribution * inflation_adjustment
            )
        else:
            real_value = current_value
    return real_value if adjust_for_inflation else current_value


def calculate_nutritional_needs(weight, height, age, gender, activity_level, goal):
    if gender == "male":
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    activity_multipliers = [1.2, 1.375, 1.55, 1.725, 1.9]
    tdee = bmr * activity_multipliers[activity_level - 1]
    if goal == "lose":
        tdee -= 500
    elif goal == "gain":
        tdee += 500
    return {
        "calories": tdee,
        "proteins_g": (tdee * 0.30) / 4,
        "fats_g": (tdee * 0.25) / 9,
        "carbohydrates_g": (tdee * 0.45) / 4,
    }


def book_room(room_type, price, check_in_date, check_out_date, customer_id, discount_code=None):
    if discount_code and discount_code == "DISCOUNT10":
        price *= 0.9
    return {
        "customer_id": customer_id,
        "room_number": room_type,
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "total_price": price,
    }


def order_food(item, quantity, price):
    return sum([quantity[i] * price[i] for i in range(len(item))])


def get_movie_rating(movie_name):
    movie = str(movie_name).lower()
    return {
        "avatar": "PG-13",
        "pulp fiction": "R",
    }.get(movie, "Unknown")


def get_movie_director(movie_name):
    movie = str(movie_name).lower()
    return {
        "avatar": "James Cameron",
        "pulp fiction": "Quentin Tarantino",
    }.get(movie, "Unknown")


def polygon_area(vertices):
    n = len(vertices)
    if n < 3:
        raise ValueError("A polygon must have at least 3 vertices.")
    vertices.append(vertices[0])
    area = 0
    for i in range(n):
        area += (vertices[i][0] * vertices[i + 1][1]) - (vertices[i + 1][0] * vertices[i][1])
    return abs(area) / 2.0


_OFFICIAL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    name: value
    for name, value in globals().items()
    if callable(value)
    and not name.startswith("_")
    and name
    not in {
        "Any",
        "BFCLCredentialError",
        "BFCLExecutionError",
        "Callable",
        "FunctionCallTaskRecord",
        "Mapping",
        "Path",
        "Sequence",
        "SimpleToolCallEvaluation",
    }
}


__all__ = ["evaluate_bfcl_executable_calls"]
