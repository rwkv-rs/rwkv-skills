from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import CoTMode
from src.eval.evaluating import TaskRunSignalGuard
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.eval.execution_plan import build_attempt_keys, plan_attempt_count
from src.eval.field_common import build_plan_task_details
from src.eval.function_calling.common import (
    build_partial_eval_flusher,
    build_pending_attempts,
    finalize_function_calling_run,
    prepare_function_calling_run,
    repeat_probe_entries,
)
from src.eval.function_calling.context_budget import DEFAULT_HISTORY_MAX_CHARS, normalize_rwkv_text, truncate_text
from src.eval.function_calling.runner_common import (
    ResolvedFunctionCallingRun,
    _looks_like_template_leak,
    _resolve_function_calling_plan,
    _resolve_job_name,
)
from src.eval.function_calling.rwkv_prompt import (
    JSON_CALL_STOP_SUFFIXES,
    build_rwkv_json_call_prompt,
    coerce_json_function_call_payloads,
    extract_json_call_value_text,
)
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import make_eval_payload, normalize_sampling_config_by_stage

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext

_MAX_TOOL_DESCRIPTION_CHARS = 700
_MAX_TOOL_SCHEMA_CHARS = 1200


@dataclass(frozen=True, slots=True)
class ToolCallExpectation:
    name: str
    arguments: dict[str, Any]
    argument_options: dict[str, tuple[Any, ...]]


@dataclass(frozen=True, slots=True)
class SimpleToolCallRecord:
    task_id: str
    instruction: str
    tools: tuple[dict[str, Any], ...]
    expected_tool_calls: tuple[ToolCallExpectation, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SimpleToolCallEvaluation:
    reward: float
    is_passed: bool
    fail_reason: str
    details: dict[str, Any]


def load_simple_tool_call_manifest_records(path: str | Path) -> list[SimpleToolCallRecord]:
    target = Path(path)
    records: list[SimpleToolCallRecord] = []
    with target.open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            records.append(normalize_simple_tool_call_manifest_row(payload, index=index, source_path=target))
    return records


def normalize_simple_tool_call_manifest_row(
    payload: Mapping[str, Any],
    *,
    index: int,
    source_path: str | Path | None = None,
) -> SimpleToolCallRecord:
    task_id = str(payload.get("task_id") or payload.get("id") or f"tool_call_{index:04d}")
    instruction = str(payload.get("instruction") or payload.get("question") or "").strip()
    if not instruction:
        raise ValueError(f"simple tool-call row {task_id!r} is missing instruction")
    metadata = dict(payload.get("metadata") or {})
    if source_path is not None:
        metadata.setdefault("manifest_path", str(Path(source_path)))
    return SimpleToolCallRecord(
        task_id=task_id,
        instruction=instruction,
        tools=tuple(_normalize_tool_schema(tool) for tool in _coerce_list(payload.get("tools"))),
        expected_tool_calls=tuple(
            _normalize_tool_expectation(item) for item in _coerce_list(payload.get("expected_tool_calls"))
        ),
        metadata=metadata,
    )


def load_bfcl_ast_rows_from_sources(
    question_path: str | Path,
    possible_answer_path: str | Path,
    *,
    category: str,
) -> list[dict[str, Any]]:
    questions = _read_json_or_jsonl_items(Path(question_path))
    answer_lookup = {
        str(item.get("id") or item.get("task_id") or ""): item
        for item in _read_json_or_jsonl_items(Path(possible_answer_path))
        if isinstance(item, Mapping)
    }
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(questions):
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id") or item.get("task_id") or f"{category}_{index}")
        answer = answer_lookup.get(task_id)
        if answer is None:
            raise ValueError(f"missing BFCL possible-answer entry for {task_id}")
        instruction = _render_bfcl_question(item.get("question"))
        if not instruction:
            raise ValueError(f"BFCL row {task_id!r} is missing question content")
        rows.append(
            {
                "task_id": task_id,
                "instruction": instruction,
                "tools": [_normalize_tool_schema(tool) for tool in _coerce_list(item.get("function"))],
                "expected_tool_calls": _normalize_bfcl_ground_truth_calls(answer.get("ground_truth")),
                "metadata": {
                    "source_format": "official_bfcl_v4_ast",
                    "category": category,
                    "source_path": str(Path(question_path)),
                    "possible_answer_path": str(Path(possible_answer_path)),
                    "execution_result_type": _coerce_list(answer.get("execution_result_type")),
                },
            }
        )
    return rows


def load_toolalpaca_rows_from_source(path: str | Path, *, dataset_name: str) -> list[dict[str, Any]]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"ToolAlpaca source must be a JSON array: {source}")

    rows: list[dict[str, Any]] = []
    for api_index, api_info in enumerate(payload):
        if not isinstance(api_info, Mapping):
            continue
        api_name = str(api_info.get("Name") or api_info.get("API") or f"api_{api_index}")
        instructions = _coerce_list(api_info.get("Instructions"))
        golden_answers = _coerce_list(api_info.get("Golden_Answers"))
        tools = _toolalpaca_tools(api_info)
        for question_index, instruction in enumerate(instructions):
            if question_index >= len(golden_answers):
                continue
            instruction_text = str(instruction or "").strip()
            if not instruction_text:
                continue
            rows.append(
                {
                    "task_id": f"{dataset_name}__{_slug(api_name)}_{question_index:03d}",
                    "instruction": instruction_text,
                    "tools": tools,
                    "expected_tool_calls": _normalize_toolalpaca_golden_answer(golden_answers[question_index]),
                    "metadata": {
                        "source_format": "official_toolalpaca",
                        "api_name": api_name,
                        "api_index": api_index,
                        "question_index": question_index,
                        "source_path": str(source),
                    },
                }
            )
    return rows


def build_simple_tool_call_prompt(record: SimpleToolCallRecord, *, history_max_chars: int) -> str:
    system_prompt = normalize_rwkv_text(
        "\n".join(
            [
                "Tools:",
                _render_tool_catalog(record.tools),
                "Return only a JSON function call.",
                'The JSON shape is {"name":"tool_name","arguments":{...}}.',
                "If multiple tool calls are required, return a JSON array of those objects in execution order.",
                "Use only listed tool names.",
            ]
        )
    )
    return build_rwkv_json_call_prompt(
        system_prompt,
        [{"role": "user", "content": normalize_rwkv_text(record.instruction)}],
        history_max_chars=history_max_chars,
    )


def decode_simple_tool_call_response(response: str) -> list[dict[str, Any]]:
    candidate = extract_json_call_value_text(response)
    payload = json.loads(candidate)
    if payload == []:
        return []
    calls = coerce_json_function_call_payloads(payload, context_label="tool-call selection")
    return [{"name": str(call["name"]), "arguments": dict(call.get("arguments") or {})} for call in calls]


def evaluate_simple_tool_calls(
    record: SimpleToolCallRecord,
    decoded_calls: Sequence[Mapping[str, Any]],
    *,
    parse_error: str | None = None,
) -> SimpleToolCallEvaluation:
    expected = list(record.expected_tool_calls)
    actual = [
        {"name": str(item.get("name") or ""), "arguments": dict(item.get("arguments") or {})}
        for item in decoded_calls
    ]
    details: dict[str, Any] = {
        "expected_tool_calls": [_expectation_payload(item) for item in expected],
        "decoded_tool_calls": actual,
        "tool_count_ok": len(actual) == len(expected),
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

    max_len = max(len(expected), len(actual))
    passed_count = 0
    failure_bits: list[str] = []
    for index in range(max_len):
        if index >= len(expected):
            details["call_matches"].append({"index": index, "ok": False, "reason": "unexpected_extra_call"})
            failure_bits.append(f"call_{index}:unexpected_extra_call")
            continue
        if index >= len(actual):
            details["call_matches"].append({"index": index, "ok": False, "reason": "missing_call"})
            failure_bits.append(f"call_{index}:missing_call")
            continue
        ok, reason = _call_matches_expectation(actual[index], expected[index])
        details["call_matches"].append({"index": index, "ok": ok, "reason": reason})
        if ok:
            passed_count += 1
        else:
            failure_bits.append(f"call_{index}:{reason}")

    denominator = max(1, len(expected))
    reward = passed_count / denominator
    is_passed = len(actual) == len(expected) and passed_count == len(expected)
    if not expected:
        is_passed = len(actual) == 0
        reward = 1.0 if is_passed else 0.0
    return SimpleToolCallEvaluation(
        reward=float(reward),
        is_passed=bool(is_passed),
        fail_reason="; ".join(failure_bits),
        details=details,
    )


def _simple_tool_call_completion_to_eval_payload(payload: dict[str, object]) -> dict[str, object]:
    agent_result = payload.get("agent_result")
    if not isinstance(agent_result, dict):
        agent_result = {}
    agent_info = payload.get("agent_info")
    if not isinstance(agent_info, dict):
        agent_info = {}
    passed = bool(agent_result.get("is_passed", False))
    reason = str(agent_info.get("fail_reason") or agent_result.get("error") or "")
    return make_eval_payload(
        payload,
        is_passed=passed,
        fail_reason=reason if not passed else "",
        answer=json.dumps(agent_info.get("decoded_tool_calls") or [], ensure_ascii=False, sort_keys=True),
        ref_answer=json.dumps(agent_info.get("expected_tool_calls") or [], ensure_ascii=False, sort_keys=True),
    )


def _run_simple_tool_call(
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    *,
    default_job_name: str,
    run_context: "RunContext | None" = None,
) -> int:
    records = load_simple_tool_call_manifest_records(run.dataset_path)
    if args.max_samples and args.max_samples > 0:
        records = records[: int(args.max_samples)]
    if not records:
        raise ValueError("simple tool-call manifest is empty")

    plan = _resolve_function_calling_plan(run.dataset_slug, len(records), avg_ks=args.avg_k)
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    tool_sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="tool",
        fallback_templates="instruction_following_default",
    )
    if tool_sampling is None:
        raise ValueError(f"missing sampling config for dataset={run.dataset_slug}, model={run.model_name}")
    tool_sampling = tool_sampling.clamp(max(1, int(args.decision_max_tokens or 768)))
    sampling_payload = normalize_sampling_config_by_stage([(1, tool_sampling)])
    history_max_chars = max(0, int(args.history_max_chars or DEFAULT_HISTORY_MAX_CHARS))
    batch_size = max(1, int(args.batch_size or 16))
    selected_entries = [(int(index), records[int(index)]) for index in plan.sample_indices]

    if args.probe_only:
        repeated = repeat_probe_entries(selected_entries, batch_size=batch_size)
        prompts = [
            build_simple_tool_call_prompt(record, history_max_chars=history_max_chars)
            for _index, record in repeated
        ]
        run.engine.generate(
            prompts,
            sampling=tool_sampling,
            batch_size=len(prompts),
            progress_desc="ToolCall-Probe",
            prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES) for _ in prompts],
            prompt_seeds=[sample_repeat_seed(index, 0, stage=1) for index, _record in repeated],
        )
        print(f"probe-only run completed: {len(prompts)} prompt(s)")
        return 0

    job_name = _resolve_job_name(default_job_name, run_context=run_context)
    ctx = prepare_function_calling_run(
        dataset_slug=str(run.dataset_slug),
        model_name=run.model_name,
        job_name=job_name,
        attempt_keys=attempt_keys,
        expected_attempt_count=plan_attempt_count(plan, max_pass_k=1),
        sampling_payload=sampling_payload,
        avg_k=plan.avg_k,
        effective_sample_count=plan.effective_sample_count,
        db_write_queue=int(args.db_write_queue or 32),
        run_context=run_context,
    )
    runtime = ctx.runtime
    writer = ctx.writer
    flush_partial = build_partial_eval_flusher(
        ctx=ctx,
        completion_to_eval=_simple_tool_call_completion_to_eval_payload,
        runner_name=default_job_name,
    )

    try:
        with TaskRunSignalGuard(
            controller=runtime,
            writer=writer,
            close_timeout_s=float(args.db_close_timeout_s),
            on_interrupt=flush_partial,
        ):
            try:
                pending = build_pending_attempts(attempt_keys, records, skip_keys=ctx.skip_keys)
                for key, record in pending:
                    prompt = build_simple_tool_call_prompt(record, history_max_chars=history_max_chars)
                    output = run.engine.generate(
                        [prompt],
                        sampling=tool_sampling,
                        batch_size=1,
                        progress_desc=f"ToolCall sample {key.sample_index}",
                        prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES)],
                        prompt_seeds=[sample_repeat_seed(key.sample_index, key.repeat_index, stage=1)],
                    )[0]
                    parse_error: str | None = None
                    decoded_calls: list[dict[str, Any]] = []
                    try:
                        if _looks_like_template_leak(output.text):
                            raise ValueError("decision stage leaked internal template/control tokens")
                        decoded_calls = decode_simple_tool_call_response(output.text)
                    except Exception as exc:  # noqa: BLE001
                        parse_error = str(exc)
                    evaluation = evaluate_simple_tool_calls(record, decoded_calls, parse_error=parse_error)
                    stage = StageRecord(prompt=prompt, completion=output.text, stop_reason=output.finish_reason)
                    payload = SampleRecord(
                        benchmark_name=run.benchmark_name,
                        dataset_split=run.dataset_split,
                        sample_index=key.sample_index,
                        repeat_index=key.repeat_index,
                        pass_index=key.pass_index,
                        stages=[stage],
                        sampling_config=sampling_payload,
                    ).as_payload()
                    payload["agent_result"] = {
                        "reward": float(evaluation.reward),
                        "num_turns": 1,
                        "cost": 0.0,
                        "is_passed": bool(evaluation.is_passed),
                        "error": evaluation.fail_reason or None,
                    }
                    payload["agent_info"] = {
                        **dict(evaluation.details),
                        "fail_reason": evaluation.fail_reason,
                        "cot_mode": CoTMode.COT.value,
                    }
                    payload["agent_trace"] = [
                        {
                            "decision_completion": output.text,
                            "decision_stop_reason": output.finish_reason,
                            "decoded_calls": decoded_calls,
                            "parse_error": parse_error or "",
                        }
                    ]
                    payload["task_id"] = record.task_id
                    payload["domain"] = "function_call"
                    payload["instruction"] = record.instruction
                    payload["metadata"] = dict(record.metadata)
                    writer.enqueue(payload)
            except BaseException:
                runtime.handle_attempt_stage_failure(
                    writer,
                    timeout_s=float(args.db_close_timeout_s),
                    on_after_close=lambda: flush_partial("exception"),
                )
                raise

        completions_payloads, _eval_payloads, metrics = finalize_function_calling_run(
            ctx=ctx,
            completion_to_eval=_simple_tool_call_completion_to_eval_payload,
            model_name=run.model_name,
            avg_k=plan.avg_k,
            timeout_s=float(args.db_close_timeout_s),
            build_score_payload=lambda completions_payloads, _eval_payloads, metrics: make_score_payload(
                run.dataset_slug,
                is_cot=True,
                model_name=run.model_name,
                metrics=metrics,
                samples=len(completions_payloads),
                problems=plan.sample_size,
                task=job_name,
                task_details=build_plan_task_details(plan, cot_mode=CoTMode.COT.value),
                extra={"cot_mode": CoTMode.COT.value, "history_max_chars": history_max_chars},
            ),
        )
    except BaseException as exc:
        if not ctx.runtime.state.is_terminal():
            ctx.runtime.fail_task(error=str(exc))
        raise
    print(f"{default_job_name} done: {len(completions_payloads)} samples")
    return 0


def _normalize_tool_expectation(raw: Any) -> ToolCallExpectation:
    if not isinstance(raw, Mapping):
        return ToolCallExpectation(name="unknown_tool", arguments={}, argument_options={})
    name = str(raw.get("name") or raw.get("tool_name") or raw.get("function_name") or "").strip()
    arguments = raw.get("arguments")
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            arguments = parsed if isinstance(parsed, Mapping) else {}
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, Mapping):
        arguments = {}
    raw_options = raw.get("argument_options")
    argument_options: dict[str, tuple[Any, ...]] = {}
    if isinstance(raw_options, Mapping):
        for key, value in raw_options.items():
            options = tuple(_coerce_list(value) or [value])
            argument_options[str(key)] = options
    for key, value in arguments.items():
        argument_options.setdefault(str(key), (value,))
    return ToolCallExpectation(
        name=name or "unknown_tool",
        arguments=dict(arguments),
        argument_options=argument_options,
    )


def _call_matches_expectation(actual: Mapping[str, Any], expected: ToolCallExpectation) -> tuple[bool, str]:
    actual_name = str(actual.get("name") or "").strip()
    if actual_name != expected.name:
        return False, f"name_mismatch(expected={expected.name}, actual={actual_name})"
    arguments = actual.get("arguments")
    if not isinstance(arguments, Mapping):
        return False, "arguments_not_object"
    actual_arguments = dict(arguments)
    for key, options in expected.argument_options.items():
        if key not in actual_arguments:
            if any(_is_absent_option(option) for option in options):
                continue
            return False, f"missing_argument({key})"
        actual_value = actual_arguments[key]
        if not any(_value_matches(actual_value, option) for option in options):
            return False, f"argument_mismatch({key})"
    for key, value in actual_arguments.items():
        if key not in expected.argument_options and not _is_absent_option(value):
            return False, f"unexpected_argument({key})"
    return True, "ok"


def _value_matches(actual: Any, expected: Any) -> bool:
    if _is_absent_option(expected):
        return _is_absent_option(actual)
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= 1e-9
    if isinstance(actual, str) and not isinstance(expected, str):
        parsed = _try_parse_json_scalar(actual)
        if parsed is not actual:
            return _value_matches(parsed, expected)
    if isinstance(expected, str) and not isinstance(actual, str):
        parsed = _try_parse_json_scalar(expected)
        if parsed is not expected:
            return _value_matches(actual, parsed)
    if isinstance(actual, str) and isinstance(expected, str):
        return normalize_rwkv_text(actual).strip() == normalize_rwkv_text(expected).strip()
    return actual == expected


def _is_absent_option(value: Any) -> bool:
    return value is None or value == "" or value == {} or value == []


def _try_parse_json_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{\"-0123456789tfn":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _expectation_payload(expectation: ToolCallExpectation) -> dict[str, Any]:
    return {
        "name": expectation.name,
        "arguments": dict(expectation.arguments),
        "argument_options": {key: list(value) for key, value in expectation.argument_options.items()},
    }


def _normalize_bfcl_ground_truth_calls(raw: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in _coerce_list(raw):
        if isinstance(item, str):
            name, arguments = _parse_python_call(item)
            calls.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "argument_options": {key: [value] for key, value in arguments.items()},
                }
            )
            continue
        if isinstance(item, Mapping):
            if "name" in item:
                call = _normalize_tool_expectation(item)
                calls.append(_expectation_payload(call))
                continue
            if len(item) != 1:
                continue
            name, argument_options = next(iter(item.items()))
            if not isinstance(argument_options, Mapping):
                argument_options = {}
            canonical_arguments = {
                str(key): _canonical_option_value(_coerce_list(value) or [value])
                for key, value in argument_options.items()
            }
            calls.append(
                {
                    "name": str(name),
                    "arguments": canonical_arguments,
                    "argument_options": {
                        str(key): list(_coerce_list(value) or [value]) for key, value in argument_options.items()
                    },
                }
            )
    return calls


def _canonical_option_value(options: Sequence[Any]) -> Any:
    for option in options:
        if not _is_absent_option(option):
            return option
    return options[0] if options else None


def _parse_python_call(text: str) -> tuple[str, dict[str, Any]]:
    parsed = ast.parse(str(text).strip(), mode="eval")
    if not isinstance(parsed.body, ast.Call):
        raise ValueError(f"BFCL ground-truth expression is not a function call: {text}")
    name = _render_ast_call_name(parsed.body.func)
    arguments: dict[str, Any] = {}
    for keyword in parsed.body.keywords:
        if keyword.arg is None:
            continue
        arguments[keyword.arg] = _literal_from_ast(keyword.value)
    return name, arguments


def _render_ast_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _render_ast_call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _literal_from_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_literal_from_ast(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return [_literal_from_ast(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal_from_ast(key): _literal_from_ast(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal_from_ast(node.operand)
        return -value if isinstance(value, (int, float)) else value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _literal_from_ast(node.left)
        right = _literal_from_ast(node.right)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left / right
    return ast.literal_eval(node)


def _normalize_toolalpaca_golden_answer(raw: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in _coerce_list(raw):
        if not isinstance(item, Mapping):
            continue
        action = str(item.get("Action") or item.get("action") or "").strip()
        action_input = item.get("Action_Input", item.get("action_input", {}))
        arguments: Any = action_input
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments.strip() or "{}")
                arguments = parsed if isinstance(parsed, Mapping) else {}
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, Mapping):
            arguments = {}
        calls.append(
            {
                "name": action,
                "arguments": dict(arguments),
                "argument_options": {key: [value] for key, value in dict(arguments).items()},
            }
        )
    return calls


def _toolalpaca_tools(api_info: Mapping[str, Any]) -> list[dict[str, Any]]:
    descriptions = api_info.get("Function_Description")
    projection = api_info.get("Function_Projection")
    tools: list[dict[str, Any]] = []
    if isinstance(descriptions, Mapping):
        for name, description in descriptions.items():
            name_text = str(name).strip()
            if not name_text or name_text == "components":
                continue
            method = ""
            path = ""
            if isinstance(projection, Mapping):
                projected = projection.get(name)
                if isinstance(projected, Sequence) and not isinstance(projected, (str, bytes, bytearray)):
                    path = str(projected[0]) if len(projected) > 0 else ""
                    method = str(projected[1]) if len(projected) > 1 else ""
            tools.append(
                {
                    "name": name_text,
                    "description": normalize_rwkv_text(str(description or "")),
                    "parameters": _toolalpaca_parameters_from_description(str(description or "")),
                    "metadata": {"path": path, "method": method},
                }
            )
    return tools


def _toolalpaca_parameters_from_description(description: str) -> dict[str, Any]:
    marker = "Parameters:"
    if marker not in description:
        return {"type": "object", "properties": {}, "required": []}
    after = description.split(marker, 1)[1]
    before_output = after.split("\nOutput:", 1)[0].strip()
    try:
        raw_params = json.loads(before_output)
    except json.JSONDecodeError:
        raw_params = {}
    if not isinstance(raw_params, Mapping):
        raw_params = {}
    properties: dict[str, Any] = {}
    required: list[str] = []
    for key, value in raw_params.items():
        description_text = str(value or "")
        value_lower = description_text.lower()
        param_type = "string"
        if value_lower.startswith("integer") or ". integer" in value_lower:
            param_type = "integer"
        elif value_lower.startswith("number") or value_lower.startswith("float") or ". float" in value_lower:
            param_type = "number"
        elif value_lower.startswith("boolean") or ". boolean" in value_lower:
            param_type = "boolean"
        properties[str(key)] = {"type": param_type, "description": description_text}
        if "required." in value_lower:
            required.append(str(key))
    return {"type": "object", "properties": properties, "required": required}


def _render_bfcl_question(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    turns = _coerce_list(raw)
    parts: list[str] = []
    for turn in turns:
        messages = _coerce_list(turn)
        for message in messages:
            if isinstance(message, Mapping):
                role = str(message.get("role") or "user").strip().lower() or "user"
                content = str(message.get("content") or "").strip()
                if content:
                    parts.append(f"{role.title()}: {content}")
            elif str(message or "").strip():
                parts.append(str(message).strip())
    return "\n".join(parts).strip()


def _normalize_tool_schema(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {
            "name": "unknown_tool",
            "description": "",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    function = raw.get("function") if isinstance(raw.get("function"), Mapping) else None
    source = function or raw
    parameters = source.get("parameters") or {"type": "object", "properties": {}, "required": []}
    if not isinstance(parameters, Mapping):
        parameters = {"type": "object", "properties": {}, "required": []}
    parameters = dict(parameters)
    if str(parameters.get("type") or "").lower() == "dict":
        parameters["type"] = "object"
    parameters.setdefault("properties", {})
    parameters.setdefault("required", [])
    return {
        "name": str(source.get("name") or raw.get("name") or "unknown_tool"),
        "description": str(source.get("description") or raw.get("description") or ""),
        "parameters": parameters,
    }


def _render_tool_catalog(tools: Sequence[Mapping[str, Any]]) -> str:
    rendered_tools: list[dict[str, Any]] = []
    for tool in tools:
        parameters = tool.get("parameters")
        if not isinstance(parameters, Mapping):
            parameters = {"type": "object", "properties": {}, "required": []}
        raw_properties = parameters.get("properties")
        rendered_arguments: Any = dict(raw_properties) if isinstance(raw_properties, Mapping) else dict(parameters)
        rendered_schema = json.dumps(rendered_arguments, ensure_ascii=False, sort_keys=True)
        if len(rendered_schema) > _MAX_TOOL_SCHEMA_CHARS:
            rendered_arguments = {
                "_truncated": True,
                "preview": truncate_text(rendered_schema, _MAX_TOOL_SCHEMA_CHARS),
            }
        rendered_tools.append(
            {
                "name": str(tool.get("name") or ""),
                "description": truncate_text(
                    normalize_rwkv_text(str(tool.get("description") or "")),
                    _MAX_TOOL_DESCRIPTION_CHARS,
                ),
                "arguments": rendered_arguments,
            }
        )
    return json.dumps(rendered_tools, ensure_ascii=False, indent=2, sort_keys=True)


def _read_json_or_jsonl_items(path: Path) -> list[Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        if "Extra data" not in str(exc):
            raise
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        return [payload]
    raise ValueError(f"unsupported JSON payload: {path}")


def _coerce_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    return []


def _slug(value: str) -> str:
    rendered = []
    for char in str(value).lower():
        rendered.append(char if char.isalnum() else "_")
    return "_".join(part for part in "".join(rendered).split("_") if part) or "item"


__all__ = [
    "SimpleToolCallEvaluation",
    "SimpleToolCallRecord",
    "ToolCallExpectation",
    "build_simple_tool_call_prompt",
    "decode_simple_tool_call_response",
    "evaluate_simple_tool_calls",
    "load_bfcl_ast_rows_from_sources",
    "load_simple_tool_call_manifest_records",
    "load_toolalpaca_rows_from_source",
    "normalize_simple_tool_call_manifest_row",
    "_run_simple_tool_call",
]
