from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urljoin

import requests

from src.eval.function_calling.runner_common import ResolvedFunctionCallingRun
from src.eval.function_calling.simple_tool_call import (
    SimpleToolCallEvaluation,
    SimpleToolCallRecord,
    ToolCallExpectation,
    load_simple_tool_call_manifest_records,
    _run_simple_tool_call,
)
from src.eval.function_calling.toolalpaca_source import (
    _TOOLALPACA_OPTIONAL_KEY,
    _TOOLALPACA_REF_KEY,
    load_toolalpaca_api_info_from_source,
)

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext

_HTTP_BODY_METHODS = {"post", "put", "patch"}
_TOOLALPACA_SECRET_PREFIX = "__toolalpaca_secret__:"
_TOOLALPACA_AUTH_ENV_BY_API = {
    "apilayer weatherstack": {
        "access_key": ("TOOLALPACA_WEATHERSTACK_API_KEY", "WEATHERSTACK_API_KEY"),
    },
    "wolframalpha": {
        "appid": (
            "TOOLALPACA_WOLFRAMALPHA_APP_ID",
            "WOLFRAMALPHA_APP_ID",
            "WOLFRAM_ALPHA_APP_ID",
            "WOLFRAM_APP_ID",
        ),
    },
    "currencybeacon": {
        "api_key": ("TOOLALPACA_CURRENCYBEACON_API_KEY", "CURRENCYBEACON_API_KEY", "CURRENCY_BEACON_API_KEY"),
    },
}


@dataclass(frozen=True, slots=True)
class ToolAlpacaActionResult:
    action: str
    action_input: dict[str, Any]
    success: bool
    optional: bool = False
    request: dict[str, Any] = field(default_factory=dict)
    response: Any = None
    error: str | None = None
    status_code: int | None = None


@dataclass(frozen=True, slots=True)
class ToolAlpacaPreflightReport:
    ok: bool
    checked_backends: tuple[str, ...]
    simulator_url: str
    errors: tuple[str, ...] = ()


def _run_toolalpaca(
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    *,
    run_context: "RunContext | None" = None,
) -> int:
    if not bool(getattr(args, "skip_runtime_preflight", False)):
        records = load_simple_tool_call_manifest_records(run.dataset_path)
        sample_limit = args.max_samples if getattr(args, "max_samples", None) else None
        if sample_limit is not None and int(sample_limit) > 0:
            records = records[: int(sample_limit)]
        preflight_toolalpaca_environment(records)
    return _run_simple_tool_call(
        args,
        run,
        default_job_name="function_toolalpaca",
        evaluator=evaluate_toolalpaca_actions,
        run_context=run_context,
    )


def evaluate_toolalpaca_actions(
    record: SimpleToolCallRecord,
    decoded_calls: Sequence[Mapping[str, Any]],
    *,
    parse_error: str | None = None,
    sandbox: "ToolAlpacaSandbox | None" = None,
) -> SimpleToolCallEvaluation:
    expected_calls = [_expectation_to_call(item) for item in record.expected_tool_calls]
    actual_calls = [
        {"name": str(item.get("name") or ""), "arguments": dict(item.get("arguments") or {})}
        for item in decoded_calls
    ]
    sandbox = sandbox or _default_toolalpaca_sandbox(record)
    details: dict[str, Any] = {
        "execution_mode": getattr(sandbox, "execution_mode", "local_toolalpaca_sandbox"),
        "expected_tool_calls": expected_calls,
        "decoded_tool_calls": actual_calls,
        "call_matches": [],
        "parse_error": parse_error or "",
    }
    if parse_error:
        return SimpleToolCallEvaluation(
            reward=0.0,
            is_passed=False,
            fail_reason=parse_error,
            details=details,
        )

    expected_results = sandbox.execute_sequence(record, expected_calls)
    actual_results = sandbox.execute_sequence(record, actual_calls)
    details["expected_execution_results"] = [_result_payload(item) for item in expected_results]
    details["decoded_execution_results"] = [_result_payload(item) for item in actual_results]
    sandbox_error = _blocking_sandbox_error(expected_results, actual_results)
    if sandbox_error:
        return SimpleToolCallEvaluation(
            reward=0.0,
            is_passed=False,
            fail_reason=sandbox_error,
            details=details,
        )

    required_expected = [item for item in expected_results if not item.optional]
    denominator = max(1, len(required_expected))
    passed_count = 0
    failure_bits: list[str] = []
    actual_index = 0
    for expected_index, expected in enumerate(expected_results):
        if actual_index >= len(actual_results):
            if expected.optional:
                details["call_matches"].append(
                    {
                        "expected_index": expected_index,
                        "decoded_index": None,
                        "ok": True,
                        "reason": "optional_skipped",
                    }
                )
                continue
            details["call_matches"].append(
                {
                    "expected_index": expected_index,
                    "decoded_index": None,
                    "ok": False,
                    "reason": "missing_call",
                }
            )
            failure_bits.append(f"call_{expected_index}:missing_call")
            continue

        actual = actual_results[actual_index]
        ok, reason = _execution_matches(actual, expected)
        if ok:
            if not expected.optional:
                passed_count += 1
            details["call_matches"].append(
                {
                    "expected_index": expected_index,
                    "decoded_index": actual_index,
                    "ok": True,
                    "reason": reason,
                }
            )
            actual_index += 1
            continue
        if expected.optional:
            details["call_matches"].append(
                {
                    "expected_index": expected_index,
                    "decoded_index": actual_index,
                    "ok": True,
                    "reason": "optional_skipped",
                    "candidate_reason": reason,
                }
            )
            continue
        details["call_matches"].append(
            {
                "expected_index": expected_index,
                "decoded_index": actual_index,
                "ok": False,
                "reason": reason,
            }
        )
        failure_bits.append(f"call_{expected_index}:{reason}")
        actual_index += 1

    while actual_index < len(actual_results):
        details["call_matches"].append(
            {
                "expected_index": None,
                "decoded_index": actual_index,
                "ok": False,
                "reason": "unexpected_extra_call",
            }
        )
        failure_bits.append(f"call_{actual_index}:unexpected_extra_call")
        actual_index += 1

    reward = passed_count / denominator
    is_passed = passed_count == len(required_expected) and not failure_bits
    if not expected_results:
        is_passed = len(actual_results) == 0
        reward = 1.0 if is_passed else 0.0
    return SimpleToolCallEvaluation(
        reward=float(reward),
        is_passed=bool(is_passed),
        fail_reason="; ".join(failure_bits),
        details=details,
    )


def preflight_toolalpaca_environment(
    records: Sequence[SimpleToolCallRecord],
    *,
    timeout_s: float = 3.0,
    raise_on_error: bool = True,
) -> ToolAlpacaPreflightReport:
    backends = sorted({_toolalpaca_backend_for_record(record) for record in records})
    errors: list[str] = []
    simulator_url = _toolalpaca_simulator_url()
    if "toolalpaca_simulator" in backends:
        try:
            response = requests.get(simulator_url, timeout=max(0.5, float(timeout_s)))
            if int(response.status_code) >= 500:
                errors.append(f"toolalpaca_simulator_unhealthy:{simulator_url}:http_{response.status_code}")
        except requests.RequestException as exc:
            errors.append(f"toolalpaca_simulator_unreachable:{simulator_url}:{exc}")
    if "toolalpaca_real_http" in backends:
        for record in records:
            if _toolalpaca_backend_for_record(record) != "toolalpaca_real_http":
                continue
            api_name = str(record.metadata.get("api_name") or "").strip()
            if _toolalpaca_record_server_url(record) == "":
                errors.append(f"toolalpaca_real_missing_server_url:{record.task_id}")
            for param_name, env_names in _TOOLALPACA_AUTH_ENV_BY_API.get(api_name.lower(), {}).items():
                if not any(os.environ.get(name) for name in env_names):
                    errors.append(
                        f"toolalpaca_real_missing_auth:{api_name}:{param_name}:set_one_of={','.join(env_names)}"
                    )
    report = ToolAlpacaPreflightReport(
        ok=not errors,
        checked_backends=tuple(backends),
        simulator_url=simulator_url,
        errors=tuple(errors),
    )
    if errors and raise_on_error:
        raise RuntimeError("ToolAlpaca runtime preflight failed: " + "; ".join(errors))
    return report


class ToolAlpacaSandbox:
    execution_mode = "local_toolalpaca_sandbox"

    def execute_sequence(
        self,
        record: SimpleToolCallRecord,
        calls: Sequence[Mapping[str, Any]],
    ) -> list[ToolAlpacaActionResult]:
        results: list[ToolAlpacaActionResult] = []
        for call in calls:
            results.append(self.execute_call(record, call, history=results))
        return results

    def execute_call(
        self,
        record: SimpleToolCallRecord,
        call: Mapping[str, Any],
        *,
        history: Sequence[ToolAlpacaActionResult],
    ) -> ToolAlpacaActionResult:
        normalized = _normalize_toolalpaca_call(call, history)
        if normalized is None:
            return ToolAlpacaActionResult(
                action=str(call.get("name") or call.get("Action") or "").strip(),
                action_input={},
                success=False,
                error="arguments_not_object",
            )
        action, resolved_arguments, optional = normalized

        tools_by_name = {str(tool.get("name") or ""): dict(tool) for tool in record.tools}
        tool = tools_by_name.get(action)
        if tool is None and action == "getDetails":
            tool = _get_details_tool_schema()
        if tool is None:
            request = {
                "action": action,
                "method": "",
                "path": "",
                "path_params": {},
                "query": {},
                "body": dict(_json_safe(resolved_arguments)),
                "headers": {},
                "cookies": {},
                "builtin": False,
            }
            response = _synthetic_toolalpaca_response(action, request, history)
            return ToolAlpacaActionResult(
                action=action,
                action_input=dict(_json_safe(resolved_arguments)),
                success=True,
                optional=optional,
                request=request,
                response=response,
            )

        try:
            request = _build_toolalpaca_request(record, tool, dict(resolved_arguments))
        except ValueError as exc:
            return ToolAlpacaActionResult(
                action=action,
                action_input=dict(_json_safe(resolved_arguments)),
                success=False,
                optional=optional,
                error=str(exc),
            )
        response = _synthetic_toolalpaca_response(action, request, history)
        return ToolAlpacaActionResult(
            action=action,
            action_input=dict(_json_safe(resolved_arguments)),
            success=True,
            optional=optional,
            request=request,
            response=response,
        )


class ToolAlpacaHttpSandbox(ToolAlpacaSandbox):
    def __init__(
        self,
        *,
        simulator_url: str | None = None,
        real_http: bool = False,
        timeout_s: float | None = None,
    ) -> None:
        self.simulator_url = (simulator_url or "").rstrip("/")
        self.real_http = bool(real_http)
        self.timeout_s = float(timeout_s or os.environ.get("TOOLALPACA_HTTP_TIMEOUT_S") or 30.0)
        self.execution_mode = "official_toolalpaca_real_http" if self.real_http else "official_toolalpaca_simulator"

    def execute_call(
        self,
        record: SimpleToolCallRecord,
        call: Mapping[str, Any],
        *,
        history: Sequence[ToolAlpacaActionResult],
    ) -> ToolAlpacaActionResult:
        normalized = _normalize_toolalpaca_call(call, history)
        if normalized is None:
            return ToolAlpacaActionResult(
                action=str(call.get("name") or call.get("Action") or "").strip(),
                action_input={},
                success=False,
                error="arguments_not_object",
            )
        action, resolved_arguments, optional = normalized
        tools_by_name = {str(tool.get("name") or ""): dict(tool) for tool in record.tools}
        tool = tools_by_name.get(action)
        if tool is None and action == "getDetails":
            tool = _get_details_tool_schema()
        if tool is None:
            return ToolAlpacaActionResult(
                action=action,
                action_input=dict(_json_safe(resolved_arguments)),
                success=False,
                optional=optional,
                error=f"unknown_tool:{action}",
            )
        try:
            request = _build_toolalpaca_request(record, tool, dict(resolved_arguments))
        except ValueError as exc:
            return ToolAlpacaActionResult(
                action=action,
                action_input=dict(_json_safe(resolved_arguments)),
                success=False,
                optional=optional,
                error=str(exc),
            )
        if request.get("builtin"):
            response = _synthetic_toolalpaca_response(action, request, history)
            return ToolAlpacaActionResult(
                action=action,
                action_input=dict(_json_safe(resolved_arguments)),
                success=True,
                optional=optional,
                request=request,
                response=response,
                status_code=200,
            )
        return self._execute_http_request(
            record,
            action=action,
            action_input=dict(_json_safe(resolved_arguments)),
            optional=optional,
            request=request,
        )

    def _execute_http_request(
        self,
        record: SimpleToolCallRecord,
        *,
        action: str,
        action_input: dict[str, Any],
        optional: bool,
        request: Mapping[str, Any],
    ) -> ToolAlpacaActionResult:
        url = self._request_url(record, request)
        if not url:
            return ToolAlpacaActionResult(
                action=action,
                action_input=action_input,
                success=False,
                optional=optional,
                request=dict(request),
                error="toolalpaca_sandbox_unavailable:missing_base_url",
            )
        method = str(request.get("method") or "get").upper()
        try:
            outbound_query = _resolve_toolalpaca_secret_placeholders(
                dict(request.get("query") or {}),
                api_name=str(request.get("api_name") or record.metadata.get("api_name") or ""),
            )
            outbound_body = _resolve_toolalpaca_secret_placeholders(
                dict(request.get("body") or {}),
                api_name=str(request.get("api_name") or record.metadata.get("api_name") or ""),
            )
            outbound_headers = _resolve_toolalpaca_secret_placeholders(
                dict(request.get("headers") or {}),
                api_name=str(request.get("api_name") or record.metadata.get("api_name") or ""),
            )
            outbound_cookies = _resolve_toolalpaca_secret_placeholders(
                dict(request.get("cookies") or {}),
                api_name=str(request.get("api_name") or record.metadata.get("api_name") or ""),
            )
        except ValueError as exc:
            return ToolAlpacaActionResult(
                action=action,
                action_input=action_input,
                success=False,
                optional=optional,
                request=dict(request),
                error=str(exc),
            )
        try:
            response = requests.request(
                method,
                url,
                params=outbound_query,
                json=(outbound_body or None),
                headers=outbound_headers,
                cookies=outbound_cookies,
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            return ToolAlpacaActionResult(
                action=action,
                action_input=action_input,
                success=False,
                optional=optional,
                request=dict(request),
                error=f"toolalpaca_sandbox_unavailable:{exc}",
            )
        response_payload = _http_response_payload(response)
        success = 200 <= int(response.status_code) < 300
        return ToolAlpacaActionResult(
            action=action,
            action_input=action_input,
            success=success,
            optional=optional,
            request=dict(request),
            response=response_payload,
            error=None if success else f"http_status_{response.status_code}",
            status_code=int(response.status_code),
        )

    def _request_url(self, record: SimpleToolCallRecord, request: Mapping[str, Any]) -> str:
        path = str(request.get("path") or "")
        if self.real_http:
            server_url = str(request.get("server_url") or record.metadata.get("api_server_url") or "").strip()
            return _join_url(server_url, path) if server_url else ""
        base_url = self.simulator_url or _toolalpaca_simulator_url()
        api_name = str(request.get("api_name") or record.metadata.get("api_name") or "").strip()
        if not base_url or not api_name:
            return ""
        return _join_url(f"{base_url.rstrip('/')}/{quote(api_name, safe='')}", path)


def _build_toolalpaca_request(
    record: SimpleToolCallRecord,
    tool: Mapping[str, Any],
    arguments: dict[str, Any],
) -> dict[str, Any]:
    name = str(tool.get("name") or "").strip()
    metadata = tool.get("metadata") if isinstance(tool.get("metadata"), Mapping) else {}
    metadata = dict(metadata or {})
    parameters = tool.get("parameters") if isinstance(tool.get("parameters"), Mapping) else {}
    parameters = dict(parameters or {})
    properties = parameters.get("properties") if isinstance(parameters.get("properties"), Mapping) else {}
    required = {str(item) for item in _coerce_list(parameters.get("required"))}
    if metadata.get("tool_type") == "toolalpaca_builtin" or name == "getDetails":
        return _build_builtin_request(name, arguments, properties, required)

    operation = _load_operation(record, metadata)
    if operation:
        op_required, op_properties = _operation_parameter_schema(operation)
        required.update(op_required)
        properties = {**op_properties, **dict(properties)}

    canonical_arguments: dict[str, Any] = {}
    unknown_arguments: dict[str, Any] = {}
    api_name = str(metadata.get("api_name") or record.metadata.get("api_name") or "")
    for key, value in arguments.items():
        if key in {_TOOLALPACA_OPTIONAL_KEY}:
            continue
        property_name = _resolve_property_name(str(key), properties)
        schema = properties.get(property_name) if property_name else None
        if schema is None:
            unknown_arguments[str(key)] = _json_safe(value)
            continue
        if _is_absent(value) and property_name not in required:
            continue
        canonical_arguments[str(property_name)] = _coerce_argument_value(str(property_name), value, schema)
    _inject_toolalpaca_auth_placeholders(api_name, canonical_arguments, required, properties)
    missing = sorted(key for key in required if key not in canonical_arguments or _is_absent(canonical_arguments[key]))
    path = str(metadata.get("path") or "")
    required_path_arguments = [key for key in missing if f"{{{key}}}" in path]
    if required_path_arguments:
        raise ValueError(f"missing_required_arguments({', '.join(required_path_arguments)})")
    if missing and not canonical_arguments:
        raise ValueError(f"missing_required_arguments({', '.join(missing)})")

    method = str(metadata.get("method") or "get").lower()
    param_locations = _operation_param_locations(operation)
    path_params: dict[str, Any] = {}
    query: dict[str, Any] = {}
    headers: dict[str, Any] = {}
    cookies: dict[str, Any] = {}
    body: dict[str, Any] = {}
    for key, value in canonical_arguments.items():
        location = param_locations.get(key)
        if location == "path" or f"{{{key}}}" in path:
            path_params[key] = value
        elif location == "header":
            headers[key] = value
        elif location == "cookie":
            cookies[key] = value
        elif location == "query" or method not in _HTTP_BODY_METHODS:
            query[key] = value
        else:
            body[key] = value

    rendered_path = path
    for key, value in path_params.items():
        rendered_path = rendered_path.replace(f"{{{key}}}", str(value))
    unresolved_path_params = re.findall(r"\{([^{}]+)\}", rendered_path)
    if unresolved_path_params:
        raise ValueError(f"missing_path_arguments({', '.join(sorted(unresolved_path_params))})")

    return _json_safe(
        {
            "action": name,
            "method": method,
            "path": rendered_path,
            "path_template": path,
            "path_params": path_params,
            "query": _drop_absent_values(query),
            "body": _drop_absent_values(body),
            "headers": _drop_absent_values(headers),
            "cookies": _drop_absent_values(cookies),
            "ignored_arguments": unknown_arguments,
            "builtin": False,
            "api_name": api_name,
            "server_url": str(metadata.get("server_url") or record.metadata.get("api_server_url") or ""),
        }
    )


def _build_builtin_request(
    name: str,
    arguments: Mapping[str, Any],
    properties: Mapping[str, Any],
    required: set[str],
) -> dict[str, Any]:
    canonical_arguments: dict[str, Any] = {}
    for key, value in arguments.items():
        if key not in properties:
            continue
        if _is_absent(value) and key not in required:
            continue
        canonical_arguments[str(key)] = _coerce_argument_value(str(key), value, properties[key])
    missing = sorted(key for key in required if key not in canonical_arguments or _is_absent(canonical_arguments[key]))
    if missing:
        raise ValueError(f"missing_required_arguments({', '.join(missing)})")
    return _json_safe(
        {
            "action": name,
            "method": "",
            "path": "",
            "path_template": "",
            "path_params": {},
            "query": {},
            "body": canonical_arguments,
            "headers": {},
            "cookies": {},
            "ignored_arguments": {},
            "builtin": True,
            "api_name": "",
            "server_url": "",
        }
    )


def _load_operation(record: SimpleToolCallRecord, metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    operation = metadata.get("operation")
    if isinstance(operation, Mapping):
        return operation
    path = str(metadata.get("path") or "")
    method = str(metadata.get("method") or "").lower()
    if not path or not method:
        return {}
    source_path = record.metadata.get("source_path")
    if source_path is None:
        return {}
    api_info = load_toolalpaca_api_info_from_source(
        str(source_path),
        api_index=record.metadata.get("api_index"),
        api_name=str(record.metadata.get("api_name") or ""),
    )
    documentation = str(api_info.get("Documentation") or "") if isinstance(api_info, Mapping) else ""
    if not documentation:
        return {}
    spec = _load_openapi_spec(documentation)
    paths = spec.get("paths") if isinstance(spec.get("paths"), Mapping) else {}
    path_doc = paths.get(path) if isinstance(paths, Mapping) else {}
    operation = path_doc.get(method) if isinstance(path_doc, Mapping) else {}
    return operation if isinstance(operation, Mapping) else {}


@lru_cache(maxsize=32)
def _load_openapi_spec(documentation: str) -> Mapping[str, Any]:
    try:
        spec = json.loads(documentation)
    except json.JSONDecodeError:
        return {}
    return spec if isinstance(spec, Mapping) else {}


def _operation_parameter_schema(operation: Mapping[str, Any]) -> tuple[set[str], dict[str, Any]]:
    required: set[str] = set()
    properties: dict[str, Any] = {}
    for param_doc in _coerce_list(operation.get("parameters")):
        if not isinstance(param_doc, Mapping):
            continue
        name = str(param_doc.get("name") or "")
        if not name:
            continue
        schema = param_doc.get("schema") if isinstance(param_doc.get("schema"), Mapping) else {}
        merged = {**dict(schema or {}), "description": str(param_doc.get("description") or "")}
        properties[name] = merged
        if param_doc.get("required"):
            required.add(name)
    body_schema = _request_body_schema(operation)
    body_props = body_schema.get("properties") if isinstance(body_schema.get("properties"), Mapping) else {}
    for key, schema in body_props.items():
        properties[str(key)] = dict(schema) if isinstance(schema, Mapping) else {}
    required.update(str(item) for item in _coerce_list(body_schema.get("required")))
    return required, properties


def _operation_param_locations(operation: Mapping[str, Any]) -> dict[str, str]:
    locations: dict[str, str] = {}
    for param_doc in _coerce_list(operation.get("parameters")):
        if isinstance(param_doc, Mapping) and param_doc.get("name"):
            locations[str(param_doc.get("name"))] = str(param_doc.get("in") or "query")
    return locations


def _resolve_property_name(key: str, properties: Mapping[str, Any]) -> str | None:
    if key in properties:
        return key
    lowered = key.lower()
    for candidate in properties.keys():
        if str(candidate).lower() == lowered:
            return str(candidate)
    if f"{key}s" in properties:
        return f"{key}s"
    if key.endswith("s") and key[:-1] in properties:
        return key[:-1]
    for candidate in properties.keys():
        candidate_text = str(candidate)
        if candidate_text.lower() == f"{lowered}s":
            return candidate_text
        if lowered.endswith("s") and candidate_text.lower() == lowered[:-1]:
            return candidate_text
    return None


def _request_body_schema(operation: Mapping[str, Any]) -> Mapping[str, Any]:
    request_body = operation.get("requestBody") if isinstance(operation.get("requestBody"), Mapping) else {}
    content = request_body.get("content") if isinstance(request_body.get("content"), Mapping) else {}
    json_content = content.get("application/json") if isinstance(content.get("application/json"), Mapping) else {}
    schema = json_content.get("schema") if isinstance(json_content.get("schema"), Mapping) else {}
    return schema if isinstance(schema, Mapping) else {}


def _coerce_argument_value(key: str, value: Any, schema: Any) -> Any:
    if not isinstance(schema, Mapping):
        return _json_safe(value)
    expected_type = str(schema.get("type") or "").lower()
    if expected_type == "integer":
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"argument_type_error({key}:integer)") from None
    elif expected_type == "number":
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"argument_type_error({key}:number)") from None
    elif expected_type == "boolean":
        value = _coerce_bool(key, value)
    elif expected_type == "array":
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = value
            value = parsed
        if not isinstance(value, list):
            raise ValueError(f"argument_type_error({key}:array)")
    elif expected_type == "object":
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = value
            value = parsed
        if not isinstance(value, Mapping):
            raise ValueError(f"argument_type_error({key}:object)")
        value = dict(value)
    elif expected_type == "string" and not isinstance(value, str):
        value = str(value)

    enum = _schema_enum(schema)
    if enum and value not in enum and not _is_absent(value):
        raise ValueError(f"argument_enum_error({key})")
    return _json_safe(value)


def _coerce_bool(key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    raise ValueError(f"argument_type_error({key}:boolean)")


def _schema_enum(schema: Mapping[str, Any]) -> list[Any]:
    raw_enum = schema.get("enum")
    if isinstance(raw_enum, list):
        return list(raw_enum)
    description = str(schema.get("description") or "")
    match = re.search(r"One of:\s*\[([^\]]+)\]", description, re.IGNORECASE)
    if not match:
        return []
    return [item.strip().strip("\"'") for item in match.group(1).split(",")]


def _resolve_toolalpaca_value(value: Any, history: Sequence[ToolAlpacaActionResult]) -> Any:
    if isinstance(value, Mapping):
        if _TOOLALPACA_REF_KEY in value:
            return _resolve_toolalpaca_ref(str(value.get(_TOOLALPACA_REF_KEY) or ""), history)
        return {str(key): _resolve_toolalpaca_value(item, history) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_toolalpaca_value(item, history) for item in value]
    if isinstance(value, str):
        match = re.fullmatch(r"\$\{([^{}]+)\}", value.strip())
        if match:
            return _resolve_toolalpaca_ref(match.group(1).strip(), history)
    return value


def _resolve_toolalpaca_ref(ref: str, history: Sequence[ToolAlpacaActionResult]) -> Any:
    text = " ".join(str(ref).strip().split())
    if not text:
        return ""
    lowered = text.lower()
    if " from " in lowered:
        prefix, source = re.split(r"\s+from\s+", text, maxsplit=1, flags=re.IGNORECASE)
        field_name = prefix.strip()
        source_name = source.strip()
        for result in reversed(history):
            if result.action == source_name and result.success:
                extracted = _extract_response_field(result.response, field_name)
                if extracted is not None:
                    return extracted
                return _synthetic_ref_value(field_name, source_name, result.request)
        return _synthetic_ref_value(field_name, source_name, {})
    if "end date" in lowered:
        return "2023-01-31"
    if "start date" in lowered:
        return "2023-01-01"
    if "year" in lowered:
        return 2023
    if "permission" in lowered:
        return "read"
    if lowered == "string":
        return "string"
    return _synthetic_ref_value(text, "context", {})


def _extract_response_field(response: Any, field_name: str) -> Any:
    if isinstance(response, Mapping):
        if field_name in response:
            return response[field_name]
        normalized_target = _slug(field_name)
        for key, value in response.items():
            if _slug(str(key)) == normalized_target:
                return value
            nested = _extract_response_field(value, field_name)
            if nested is not None:
                return nested
    if isinstance(response, list):
        for item in response:
            nested = _extract_response_field(item, field_name)
            if nested is not None:
                return nested
    return None


def _synthetic_toolalpaca_response(
    action: str,
    request: Mapping[str, Any],
    history: Sequence[ToolAlpacaActionResult],
) -> dict[str, Any]:
    seed = _stable_seed({"action": action, "request": _comparable_request(request)})
    response: dict[str, Any] = {
        "ok": True,
        "action": action,
        "id": _stable_int(seed, "id"),
        "resultId": _stable_id(seed, "result"),
    }
    for field_name in _likely_response_fields(action, request, history):
        response[field_name] = _synthetic_ref_value(field_name, action, request)
    return _json_safe(response)


def _likely_response_fields(
    action: str,
    request: Mapping[str, Any],
    history: Sequence[ToolAlpacaActionResult],
) -> list[str]:
    fields = {
        "id",
        "resultId",
        "userId",
        "accessToken",
        "animeId",
        "ip",
        "style",
        "format",
        "category",
        "username",
        "holidayId",
        "symbol",
        "dashboardId",
        "sourceId",
        "targetId",
    }
    for container_name in ("query", "body", "path_params"):
        container = request.get(container_name)
        if isinstance(container, Mapping):
            fields.update(str(key) for key in container)
    for result in history:
        if isinstance(result.response, Mapping):
            fields.update(str(key) for key in result.response.keys())
    if action.startswith("search"):
        fields.add(f"{_slug(action.removeprefix('search'))}Id")
    if action.startswith("create"):
        fields.add(f"{_slug(action.removeprefix('create'))}Id")
    return sorted(fields)


def _synthetic_ref_value(field_name: str, source_name: str, request: Mapping[str, Any] | None) -> Any:
    field_slug = _slug(field_name)
    seed = _stable_seed({"field": field_name, "source": source_name, "request": _comparable_request(request or {})})
    if "date" in field_slug:
        return "2023-01-31" if field_slug.startswith("end") else "2023-01-01"
    if field_slug.endswith("year") or field_slug == "year":
        return 2023
    if field_slug.endswith("id") or field_slug == "id":
        return _stable_int(seed, field_slug)
    if field_slug in {"ip"}:
        return f"192.0.2.{_stable_int(seed, field_slug) % 250 + 1}"
    if field_slug in {"symbol"}:
        return "BTC/USD"
    if field_slug in {"format"}:
        return "png"
    if field_slug in {"style"}:
        return "minimal"
    if field_slug in {"category"}:
        return "general"
    if field_slug in {"permission", "permissions", "somepermission"}:
        return "read"
    return _stable_id(seed, field_slug)


def _execution_matches(actual: ToolAlpacaActionResult, expected: ToolAlpacaActionResult) -> tuple[bool, str]:
    if actual.action != expected.action:
        return False, f"name_mismatch(expected={expected.action}, actual={actual.action})"
    if actual.success != expected.success:
        return False, f"execution_status_mismatch(expected={expected.success}, actual={actual.success})"
    if not expected.success:
        return (actual.error == expected.error, "ok" if actual.error == expected.error else "execution_error_mismatch")
    if _comparable_request(actual.request) != _comparable_request(expected.request):
        return False, "request_mismatch"
    return True, "ok"


def _comparable_request(request: Mapping[str, Any]) -> dict[str, Any]:
    return _json_safe(
        {
            "action": request.get("action", ""),
            "method": request.get("method", ""),
            "path": request.get("path", ""),
            "path_params": _drop_absent_values(request.get("path_params", {})),
            "query": _drop_absent_values(request.get("query", {})),
            "body": _drop_absent_values(request.get("body", {})),
            "headers": _drop_absent_values(request.get("headers", {})),
            "cookies": _drop_absent_values(request.get("cookies", {})),
            "builtin": bool(request.get("builtin", False)),
        }
    )


def _result_payload(result: ToolAlpacaActionResult) -> dict[str, Any]:
    return {
        "Action": result.action,
        "Action_Input": _clean_action_arguments(result.action_input),
        "success": bool(result.success),
        "optional": bool(result.optional),
        "request": result.request,
        "response": result.response,
        "error": result.error or "",
        "status_code": result.status_code,
    }


def _normalize_toolalpaca_call(
    call: Mapping[str, Any],
    history: Sequence[ToolAlpacaActionResult],
) -> tuple[str, dict[str, Any], bool] | None:
    action = str(call.get("name") or call.get("Action") or "").strip()
    raw_arguments = call.get("arguments", call.get("Action_Input", {}))
    if isinstance(raw_arguments, str):
        try:
            raw_arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            raw_arguments = {}
    if not isinstance(raw_arguments, Mapping):
        return None
    arguments = dict(raw_arguments)
    optional = _truthy(arguments.pop(_TOOLALPACA_OPTIONAL_KEY, False))
    if action.lower().startswith("[optional]"):
        optional = True
        action = action.split("]", 1)[-1].strip()
    resolved_arguments = _resolve_toolalpaca_value(arguments, history)
    if not isinstance(resolved_arguments, Mapping):
        resolved_arguments = {}
    return action, dict(resolved_arguments), optional


def _default_toolalpaca_sandbox(record: SimpleToolCallRecord) -> ToolAlpacaSandbox:
    backend = _toolalpaca_backend_for_record(record)
    if backend == "toolalpaca_simulator":
        return ToolAlpacaHttpSandbox(simulator_url=_toolalpaca_simulator_url())
    if backend == "toolalpaca_real_http":
        return ToolAlpacaHttpSandbox(real_http=True)
    return ToolAlpacaSandbox()


def _toolalpaca_backend_for_record(record: SimpleToolCallRecord) -> str:
    backend = str(record.metadata.get("execution_backend") or "").strip().lower()
    if not backend and record.task_id.startswith("toolalpaca_eval_simulated__"):
        backend = "toolalpaca_simulator"
    elif not backend and record.task_id.startswith("toolalpaca_eval_real__"):
        backend = "toolalpaca_real_http"
    return backend or "toolalpaca_synthetic"


def _toolalpaca_record_server_url(record: SimpleToolCallRecord) -> str:
    if str(record.metadata.get("api_server_url") or "").strip():
        return str(record.metadata.get("api_server_url") or "").strip()
    for tool in record.tools:
        metadata = tool.get("metadata") if isinstance(tool.get("metadata"), Mapping) else {}
        if isinstance(metadata, Mapping) and str(metadata.get("server_url") or "").strip():
            return str(metadata.get("server_url") or "").strip()
    return ""


def _inject_toolalpaca_auth_placeholders(
    api_name: str,
    arguments: dict[str, Any],
    required: set[str],
    properties: Mapping[str, Any],
) -> None:
    auth_params = _TOOLALPACA_AUTH_ENV_BY_API.get(api_name.strip().lower(), {})
    for param_name in auth_params:
        if param_name in required or param_name in properties:
            arguments[param_name] = _toolalpaca_secret_placeholder(api_name, param_name)


def _toolalpaca_secret_placeholder(api_name: str, param_name: str) -> str:
    return f"{_TOOLALPACA_SECRET_PREFIX}{_slug(api_name)}:{param_name}"


def _resolve_toolalpaca_secret_placeholders(value: Any, *, api_name: str) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _resolve_toolalpaca_secret_placeholders(item, api_name=api_name)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_toolalpaca_secret_placeholders(item, api_name=api_name) for item in value]
    if isinstance(value, str) and value.startswith(_TOOLALPACA_SECRET_PREFIX):
        param_name = value.rsplit(":", 1)[-1]
        secret = _toolalpaca_auth_secret(api_name, param_name)
        if not secret:
            env_names = ", ".join(_toolalpaca_auth_env_names(api_name, param_name))
            raise ValueError(f"toolalpaca_auth_missing:{api_name}:{param_name}: set one of {env_names}")
        return secret
    return value


def _toolalpaca_auth_secret(api_name: str, param_name: str) -> str:
    for env_name in _toolalpaca_auth_env_names(api_name, param_name):
        value = os.environ.get(env_name)
        if value:
            return value
    return ""


def _toolalpaca_auth_env_names(api_name: str, param_name: str) -> tuple[str, ...]:
    return _TOOLALPACA_AUTH_ENV_BY_API.get(api_name.strip().lower(), {}).get(param_name, ())


def _toolalpaca_simulator_url() -> str:
    return (os.environ.get("TOOLALPACA_SIMULATOR_URL") or "http://127.0.0.1:5678").rstrip("/")


def _blocking_sandbox_error(
    expected_results: Sequence[ToolAlpacaActionResult],
    actual_results: Sequence[ToolAlpacaActionResult],
) -> str:
    for result in [*expected_results, *actual_results]:
        error = str(result.error or "")
        if error.startswith("toolalpaca_sandbox_unavailable:"):
            return error
    return ""


def _http_response_payload(response: requests.Response) -> Any:
    content_type = response.headers.get("Content-Type", "")
    if "json" in content_type.lower():
        try:
            return response.json()
        except ValueError:
            pass
    text = response.text
    try:
        return json.loads(text)
    except ValueError:
        return {"response": text}


def _join_url(base_url: str, path: str) -> str:
    if not base_url:
        return ""
    return urljoin(f"{base_url.rstrip('/')}/", path.lstrip("/"))


def _expectation_to_call(expectation: ToolCallExpectation) -> dict[str, Any]:
    return {"name": expectation.name, "arguments": dict(expectation.arguments)}


def _clean_action_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in arguments.items() if key != _TOOLALPACA_OPTIONAL_KEY}


def _get_details_tool_schema() -> dict[str, Any]:
    return {
        "name": "getDetails",
        "description": "Ask the user for missing details.",
        "parameters": {
            "type": "object",
            "properties": {"Question": {"type": "string", "description": "Required. String."}},
            "required": ["Question"],
        },
        "metadata": {"tool_type": "toolalpaca_builtin"},
    }


def _drop_absent_values(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _drop_absent_values(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not _is_absent(item)
        }
    if isinstance(value, list):
        return [_drop_absent_values(item) for item in value if not _is_absent(item)]
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _is_absent(value: Any) -> bool:
    return value is None or value == "" or value == {} or value == []


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _coerce_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    return []


def _stable_seed(payload: Any) -> str:
    return hashlib.sha256(json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _stable_int(seed: str, salt: str) -> int:
    digest = hashlib.sha256(f"{seed}:{salt}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 900000 + 1000


def _stable_id(seed: str, salt: str) -> str:
    digest = hashlib.sha256(f"{seed}:{salt}".encode("utf-8")).hexdigest()
    return f"{_slug(salt)}_{digest[:10]}"


def _slug(value: str) -> str:
    rendered = []
    for char in str(value):
        rendered.append(char.lower() if char.isalnum() else "_")
    return "_".join(part for part in "".join(rendered).split("_") if part) or "value"


__all__ = [
    "ToolAlpacaActionResult",
    "ToolAlpacaHttpSandbox",
    "ToolAlpacaPreflightReport",
    "ToolAlpacaSandbox",
    "evaluate_toolalpaca_actions",
    "preflight_toolalpaca_environment",
    "_run_toolalpaca",
]
