from __future__ import annotations

"""ToolAlpaca official-format adapter and scorer.

The RWKV evaluator keeps a local JSON output contract:

    {"name": "tool_name", "arguments": {...}}

ToolAlpaca's official scripts use Action / Action_Input steps, execute the
selected API function, then ask a GPT judge to compare the process and final
response against the standard answer.  This module keeps RWKV's local output
format and converts it at the scorer boundary.
"""

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import requests
from dotenv import load_dotenv

from .simple_tool_call import SimpleToolCallEvaluation

if TYPE_CHECKING:
    from src.eval.datasets.data_struct.function_call import FunctionCallTaskRecord

load_dotenv()

OFFICIAL_TOOLALPACA_SOURCE = "tangqiaoyu/ToolAlpaca@main"

_AUTH_ENV_CANDIDATES: dict[tuple[str, str], tuple[str, ...]] = {
    ("apilayer weatherstack", "access_key"): (
        "TOOLALPACA_WEATHERSTACK_ACCESS_KEY",
        "WEATHERSTACK_ACCESS_KEY",
    ),
    ("WolframAlpha", "appid"): (
        "TOOLALPACA_WOLFRAMALPHA_APPID",
        "WOLFRAMALPHA_APPID",
        "WOLFRAM_ALPHA_APPID",
    ),
    ("CurrencyBeacon", "api_key"): (
        "TOOLALPACA_CURRENCYBEACON_API_KEY",
        "CURRENCYBEACON_API_KEY",
    ),
}
_PLACEHOLDER_AUTH_VALUES = {"", "***", "REDACTED", "REDACTED_API_KEY", "null", "none"}

_OFFICIAL_EVALUATION_TEMPLATE = """Given the documentation of a REST API and a task instruction, I need you to evaluate whether the solution provided by my AI assistant aligns with the standard answer. Follow these guidelines:
1. You need to assess both the process and final response of the AI assistant's solution.
2. For the process, refer to the standard answer:
- The standard answer only includes function names and parameters, while the AI assistant's solution also includes function returns.
Therefore, it is acceptable to adjust the call situation based on the function return, such as retrying when the function errors, calling function `getDetails` for more information, calling function `retrievalDataFromFile` when function's return is too long.
- Random calls to unrelated functions are not allowed.
- The solution must contain all the steps in the standard answer.
- The necessary parameters of the function need to be consistent with the standard answer.
Parameters not mentioned in the instruction can be inconsistent.
3. You need to comprehensively judge whether the final response of the solution accurately summarizes the entire call process and provides a reasonable response to the initial instruction.
4. You need to first analyze the entire solution according to the guidelines, then give your answer.
Your output should adhere to the format:
## Analysis
{some analysis}
## Results
Process Correctness: one of [Yes, No, Uncertain]
Final Response Correctness: one of [Yes, No, Uncertain]
## Documentation
${documentation}
## Task Instruction
${instruction}
## Standard Answer
${standard}
## AI Assistant's Solution
${solution}
## Analysis
"""


@dataclass(frozen=True, slots=True)
class ToolAlpacaOfficialAction:
    action: str
    action_input: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolAlpacaExecutionStep:
    action: ToolAlpacaOfficialAction
    observation: str


def evaluate_toolalpaca_official_calls(
    record: FunctionCallTaskRecord,
    decoded_calls: Sequence[Mapping[str, Any]],
    *,
    parse_error: str | None = None,
) -> SimpleToolCallEvaluation:
    details: dict[str, Any] = {
        "official_toolalpaca_source": OFFICIAL_TOOLALPACA_SOURCE,
        "expected_tool_calls": list(record.expected_tool_calls or []),
        "decoded_tool_calls": _sanitize_calls(decoded_calls),
        "official_actions": [],
        "execution_steps": [],
        "judge": {},
    }
    if parse_error:
        details["parse_error"] = parse_error
        return SimpleToolCallEvaluation(0.0, False, parse_error, details)

    actions = local_calls_to_official_actions(decoded_calls)
    details["official_actions"] = [_official_action_payload(item) for item in actions]

    try:
        execution_steps = execute_toolalpaca_actions(record, actions)
    except Exception as exc:  # noqa: BLE001
        details["execution_error"] = str(exc)
        return SimpleToolCallEvaluation(0.0, False, f"toolalpaca_official:execution_failed:{exc}", details)
    details["execution_steps"] = [_execution_step_payload(item) for item in execution_steps]

    try:
        judge = judge_toolalpaca_solution(record, execution_steps)
    except Exception as exc:  # noqa: BLE001
        details["judge_error"] = str(exc)
        return SimpleToolCallEvaluation(0.0, False, f"toolalpaca_official:judge_failed:{exc}", details)
    details["judge"] = judge
    process_ok = str(judge.get("process_correctness") or "").strip().lower() == "yes"
    passed = process_ok
    fail_reason = "" if passed else f"toolalpaca_official:process_{judge.get('process_correctness') or 'missing'}"
    return SimpleToolCallEvaluation(1.0 if passed else 0.0, passed, fail_reason, details)


def local_calls_to_official_actions(calls: Sequence[Mapping[str, Any]]) -> list[ToolAlpacaOfficialAction]:
    actions: list[ToolAlpacaOfficialAction] = []
    for call in calls:
        name = str(call.get("name") or "").strip()
        arguments = call.get("arguments")
        if not isinstance(arguments, Mapping):
            arguments = {}
        actions.append(ToolAlpacaOfficialAction(action=name, action_input=dict(arguments)))
    return actions


def execute_toolalpaca_actions(
    record: FunctionCallTaskRecord,
    actions: Sequence[ToolAlpacaOfficialAction],
) -> list[ToolAlpacaExecutionStep]:
    metadata = dict(record.metadata or {})
    openapi_spec = _load_openapi_spec(metadata)
    function_projection = metadata.get("toolalpaca_function_projection")
    if not isinstance(function_projection, Mapping):
        raise ValueError("missing toolalpaca_function_projection metadata")
    api_name = str(metadata.get("api_name") or metadata.get("toolalpaca_api_name") or "").strip()
    if not api_name:
        raise ValueError("missing ToolAlpaca api_name metadata")
    dataset_kind = str(metadata.get("toolalpaca_dataset") or "").strip().lower()
    base_url = None
    if dataset_kind == "simulated":
        simulator_url = os.environ.get("TOOLALPACA_SIMULATOR_URL", "http://127.0.0.1:5678").rstrip("/")
        base_url = f"{simulator_url}/{api_name}"

    timeout = float(os.environ.get("TOOLALPACA_REQUEST_TIMEOUT", "30"))
    steps: list[ToolAlpacaExecutionStep] = []
    for action in actions:
        projected = function_projection.get(action.action)
        if not isinstance(projected, Sequence) or isinstance(projected, (str, bytes, bytearray)) or len(projected) < 2:
            observation = f"`{action.action}` is not a valid action."
            steps.append(ToolAlpacaExecutionStep(action=action, observation=observation))
            continue
        path = str(projected[0])
        method = str(projected[1]).lower()
        input_params = _inject_authentication(api_name, action.action_input, metadata)
        try:
            response = _call_api_function(
                input_params=input_params,
                openapi_spec=openapi_spec,
                path=path,
                method=method,
                base_url=base_url,
                timeout=timeout,
            )
            observation = f"Status Code: {response.status_code}. Response: {response.text}"
            if not 200 <= int(response.status_code) < 300:
                observation += (
                    ". You should choose one of: (1) change the input and retry; "
                    "(2) return the 'Final Answer' and explain what happened; "
                    "(You must choose this one when the error occurs more than 3 times.) "
                    "(3) call another function."
                )
        except Exception as exc:  # noqa: BLE001
            observation = str(exc)
        steps.append(ToolAlpacaExecutionStep(action=action, observation=observation))
    return steps


def judge_toolalpaca_solution(
    record: FunctionCallTaskRecord,
    execution_steps: Sequence[ToolAlpacaExecutionStep],
) -> dict[str, Any]:
    load_dotenv()
    api_key = os.environ.get("JUDGE_API_KEY")
    if not api_key:
        raise ValueError("toolalpaca official judge requires JUDGE_API_KEY")
    model = os.environ.get("JUDGE_MODEL")
    if not model:
        raise ValueError("toolalpaca official judge requires JUDGE_MODEL")
    prompt = build_toolalpaca_judge_prompt(record, execution_steps)
    output = _judge_chat_completion(
        prompt,
        api_key=api_key,
        model=model,
        temperature=float(os.environ.get("JUDGE_TEMPERATURE", "0.2")),
    )
    return {
        "prompt": prompt,
        "output": output,
        "process_correctness": _extract_judge_label(output, "Process Correctness"),
        "final_response_correctness": _extract_judge_label(output, "Final Response Correctness"),
    }


def build_toolalpaca_judge_prompt(
    record: FunctionCallTaskRecord,
    execution_steps: Sequence[ToolAlpacaExecutionStep],
) -> str:
    metadata = dict(record.metadata or {})
    final_response = str(metadata.get("toolalpaca_final_response") or "No final response was generated.")
    return (
        _OFFICIAL_EVALUATION_TEMPLATE.replace("${documentation}", str(metadata.get("toolalpaca_nl_documentation") or ""))
        .replace("${instruction}", str(record.instruction or ""))
        .replace("${standard}", _render_standard_answer(record.expected_tool_calls or []))
        .replace("${solution}", _render_solution(execution_steps, final_response=final_response))
    )


def _load_openapi_spec(metadata: Mapping[str, Any]) -> dict[str, Any]:
    raw = metadata.get("toolalpaca_documentation")
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        parsed = json.loads(raw)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    raise ValueError("missing ToolAlpaca OpenAPI documentation metadata")


def _call_api_function(
    *,
    input_params: Mapping[str, Any],
    openapi_spec: Mapping[str, Any],
    path: str,
    method: str,
    base_url: str | None,
    timeout: float,
) -> requests.Response:
    function_doc = openapi_spec["paths"][path][method]
    params: dict[str, dict[str, Any]] = {"query": {}, "header": {}, "path": {}, "cookie": {}}
    required_params: set[tuple[str, str]] = set()
    for param_doc in function_doc.get("parameters", []):
        if not isinstance(param_doc, Mapping):
            continue
        param_name = str(param_doc.get("name") or "")
        param_in = str(param_doc.get("in") or "query")
        if param_doc.get("required"):
            required_params.add((param_in, param_name))
        if param_name in input_params:
            required_params.discard((param_in, param_name))
            params.setdefault(param_in, {})[param_name] = input_params[param_name]
    body_data: dict[str, Any] | None = None
    required_body_params: set[str] = set()
    if isinstance(function_doc.get("requestBody"), Mapping):
        body_data = {}
        request_body = function_doc["requestBody"]
        schema = request_body.get("content", {}).get("application/json", {}).get("schema", {})
        if isinstance(schema, Mapping) and isinstance(schema.get("properties"), Mapping):
            required_body_params = set(str(item) for item in schema.get("required", []))
            for property_name in schema["properties"]:
                if property_name in input_params:
                    body_data[str(property_name)] = input_params[property_name]
                    required_body_params.discard(str(property_name))
    if required_params or required_body_params:
        missing = [item[1] for item in sorted(required_params)] + sorted(required_body_params)
        raise ValueError(f"Missing required parameters: {', '.join(missing)}. You need to change the input and try again.")
    resolved_base_url = base_url or str((openapi_spec.get("servers") or [{}])[0].get("url") or "")
    url = f"{resolved_base_url.rstrip('/')}{path.format(**params['path'])}"
    headers = {"Content-Type": "application/json"}
    headers.update(params.get("header") or {})
    return requests.request(
        method=method.upper(),
        url=url,
        params=params.get("query") or {},
        json=body_data,
        headers=headers,
        cookies=params.get("cookie") or {},
        timeout=timeout,
    )


def _inject_authentication(
    api_name: str,
    arguments: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    params = dict(arguments)
    auth = metadata.get("toolalpaca_authentication")
    if not isinstance(auth, Mapping):
        return params
    for key, raw_value in auth.items():
        key_text = str(key)
        env_value = _auth_env_value(api_name, key_text)
        if env_value:
            params[key_text] = env_value
            continue
        if _is_placeholder_auth(raw_value):
            continue
        if key_text not in params or _is_placeholder_auth(params.get(key_text)):
            params[key_text] = raw_value
    return params


def _auth_env_value(api_name: str, key: str) -> str:
    for candidate in _AUTH_ENV_CANDIDATES.get((api_name, key), ()):
        value = os.environ.get(candidate)
        if value:
            return value
    generic = os.environ.get(f"TOOLALPACA_{_env_token(api_name)}_{_env_token(key)}")
    return generic or ""


def _env_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value).upper()).strip("_")


def _is_placeholder_auth(value: Any) -> bool:
    return str(value or "").strip() in _PLACEHOLDER_AUTH_VALUES


def _render_standard_answer(expected_tool_calls: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for index, call in enumerate(expected_tool_calls, start=1):
        name = str(call.get("name") or call.get("Action") or "")
        arguments = call.get("arguments", call.get("Action_Input", {}))
        if isinstance(arguments, str):
            arguments_text = arguments
        else:
            arguments_text = json.dumps(_sanitize_value(arguments), ensure_ascii=False, sort_keys=True)
        parts.append(f"{index}. Function: {name}\nParameters: {arguments_text}")
    return "\n".join(parts)


def _render_solution(
    execution_steps: Sequence[ToolAlpacaExecutionStep],
    *,
    final_response: str,
) -> str:
    parts: list[str] = []
    for index, step in enumerate(execution_steps, start=1):
        parts.append(
            f"{index}. Function: {step.action.action}\n"
            f"Parameters: {json.dumps(_sanitize_value(step.action.action_input), ensure_ascii=False, sort_keys=True)}\n"
            f"Returns: {step.observation}"
        )
    parts.append(f"{len(execution_steps) + 1}. Final Response: {final_response}")
    return "\n".join(parts)


def _judge_chat_completion(prompt: str, *, api_key: str, model: str, temperature: float) -> str:
    load_dotenv()
    base_url = str(os.environ.get("JUDGE_BASE_URL") or "").rstrip("/")
    if not base_url:
        raise ValueError("toolalpaca official judge requires JUDGE_BASE_URL")
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    max_tokens = os.environ.get("JUDGE_MAX_TOKENS")
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(os.environ.get("JUDGE_TIMEOUT", "120"))) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"judge request failed: HTTP {exc.code}: {body[:500]}") from exc
    data = json.loads(raw)
    return str(data["choices"][0]["message"]["content"])


def _extract_judge_label(text: str, label: str) -> str:
    match = re.search(rf"{re.escape(label)}:\s*(Yes|No|Uncertain)", text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _official_action_payload(action: ToolAlpacaOfficialAction) -> dict[str, Any]:
    return {
        "Action": action.action,
        "Action_Input": json.dumps(_sanitize_value(action.action_input), ensure_ascii=False, sort_keys=True),
    }


def _execution_step_payload(step: ToolAlpacaExecutionStep) -> dict[str, Any]:
    return {
        "Action": step.action.action,
        "Action_Input": json.dumps(_sanitize_value(step.action.action_input), ensure_ascii=False, sort_keys=True),
        "Observation": _sanitize_text(step.observation),
    }


def _sanitize_calls(calls: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    sanitized = []
    for call in calls:
        sanitized.append(
            {
                "name": str(call.get("name") or ""),
                "arguments": _sanitize_value(call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}),
            }
        )
    return sanitized


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): ("<redacted>" if _looks_secret_key(str(key)) else _sanitize_value(val)) for key, val in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def _sanitize_text(text: str) -> str:
    return str(text or "")[:4000]


def _looks_secret_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in {"api_key", "access_key", "appid"} or "token" in lowered or "secret" in lowered


__all__ = [
    "OFFICIAL_TOOLALPACA_SOURCE",
    "ToolAlpacaExecutionStep",
    "ToolAlpacaOfficialAction",
    "build_toolalpaca_judge_prompt",
    "evaluate_toolalpaca_official_calls",
    "execute_toolalpaca_actions",
    "judge_toolalpaca_solution",
    "local_calls_to_official_actions",
]
