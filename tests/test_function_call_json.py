from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.eval.datasets.data_loader.function_call import JsonlFunctionCallTaskLoader
from src.eval.evaluators.function_call import FunctionCallPipeline
from src.eval.function_calling.bfcl_exec import evaluate_bfcl_executable_calls
from src.eval.function_calling.toolalpaca_official import (
    execute_toolalpaca_actions,
    local_calls_to_official_actions,
)
from src.eval.metrics.function_call import evaluate_function_call
from src.eval.scheduler.jobs import JOB_CATALOGUE, detect_job_from_dataset


def _simple_tool_call_row() -> dict:
    return {
        "task_id": "multiple_0",
        "instruction": "Calculate a binomial probability.",
        "tools": [
            {
                "name": "calc_binomial_probability",
                "description": "Calculates a binomial probability.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer"},
                        "k": {"type": "integer"},
                        "p": {"type": "float"},
                    },
                    "required": ["n", "k", "p"],
                },
            }
        ],
        "expected_tool_calls": [
            {
                "name": "calc_binomial_probability",
                "arguments": {"n": 20, "k": 5, "p": 1 / 6},
                "argument_options": {"n": [20], "k": [5], "p": [1 / 6]},
            }
        ],
    }


def test_function_call_uses_simple_tool_call_prompt_and_matcher(tmp_path) -> None:
    path = tmp_path / "bfcl_multiple.jsonl"
    path.write_text(json.dumps(_simple_tool_call_row(), ensure_ascii=False) + "\n", encoding="utf-8")
    record = JsonlFunctionCallTaskLoader(path).load()[0]
    pipeline = object.__new__(FunctionCallPipeline)

    prompt = pipeline._make_prompt(record)
    metrics = evaluate_function_call(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "final_answer": (
                    '{"name":"calc_binomial_probability",'
                    '"arguments":{"n":20,"k":5,"p":0.16666666666666666}}'
                ),
            }
        ],
        dataset_path=str(path),
        avg_k=(1,),
    )

    assert record.env == {"type": "simple_tool_call"}
    assert record.scorer["type"] == "simple_tool_call"
    assert record.expected_tool_calls[0]["argument_options"]["p"] == [1 / 6]
    assert prompt.endswith("Assistant: ```json\n")
    assert '"arguments": {' in prompt
    assert '"parameters"' not in prompt
    assert metrics.success_rate == 1.0
    assert metrics.avg_at_k == {"avg@1": 1.0}


def test_function_call_uses_bfcl_exec_scorer_for_exec_rows(tmp_path) -> None:
    path = tmp_path / "bfcl_exec_simple.jsonl"
    row = _simple_tool_call_row()
    row["task_id"] = "exec_simple_0"
    row["expected_tool_calls"] = [
        {
            "name": "calc_binomial_probability",
            "arguments": {"n": 20, "k": 5, "p": 0.6},
            "argument_options": {"n": [20], "k": [5], "p": [0.6]},
        }
    ]
    row["scorer"] = {
        "type": "bfcl_exec",
        "ground_truth": ["calc_binomial_probability(n=20, k=5, p=0.6)"],
        "execution_result_type": ["exact_match"],
    }
    row["metadata"] = {"category": "exec_simple"}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    metrics = evaluate_function_call(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "final_answer": (
                    '{"name":"calc_binomial_probability",'
                    '"arguments":{"n":20,"k":5,"p":0.6}}'
                ),
            }
        ],
        dataset_path=str(path),
        avg_k=(1,),
    )

    assert metrics.success_rate == 1.0
    assert metrics.avg_at_k == {"avg@1": 1.0}
    assert metrics.payloads
    assert metrics.payloads[0]["is_passed"] is True


def test_bfcl_exec_scorer_accepts_bfcl_matrix_argument_names(tmp_path) -> None:
    path = tmp_path / "bfcl_exec_multiple.jsonl"
    row = _simple_tool_call_row()
    row["task_id"] = "exec_multiple_0"
    row["expected_tool_calls"] = [
        {
            "name": "mat_mul",
            "arguments": {"matA": [[1, 2], [3, 4]], "matB": [[5, 6], [7, 8]]},
            "argument_options": {
                "matA": [[[1, 2], [3, 4]]],
                "matB": [[[5, 6], [7, 8]]],
            },
        }
    ]
    row["scorer"] = {
        "type": "bfcl_exec",
        "ground_truth": ["mat_mul(matA=[[1, 2], [3, 4]], matB=[[5, 6], [7, 8]])"],
        "execution_result_type": ["exact_match"],
    }
    row["metadata"] = {"category": "exec_multiple"}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    metrics = evaluate_function_call(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "final_answer": (
                    '{"name":"mat_mul",'
                    '"arguments":{"matA":[[1,2],[3,4]],"matB":[[5,6],[7,8]]}}'
                ),
            }
        ],
        dataset_path=str(path),
        avg_k=(1,),
    )

    assert metrics.success_rate == 1.0
    record = JsonlFunctionCallTaskLoader(path).load()[0]
    result = evaluate_bfcl_executable_calls(
        record,
        [{"name": "mat_mul", "arguments": {"matA": [[1, 2], [3, 4]], "matB": [[5, 6], [7, 8]]}}],
    )
    assert result.details["official_bfcl_exec_source"] == "ShishirPatil/gorilla@28a0f42"
    assert result.details["official_check"]["valid"] is True


def test_bfcl_exec_scorer_does_not_fallback_to_argument_identity(tmp_path) -> None:
    path = tmp_path / "bfcl_exec_simple.jsonl"
    row = _simple_tool_call_row()
    row["task_id"] = "exec_simple_unsupported"
    row["expected_tool_calls"] = [
        {
            "name": "not_an_official_bfcl_function",
            "arguments": {"value": 1},
            "argument_options": {"value": [1]},
        }
    ]
    row["scorer"] = {
        "type": "bfcl_exec",
        "ground_truth": ["not_an_official_bfcl_function(value=1)"],
        "execution_result_type": ["exact_match"],
    }
    row["metadata"] = {"category": "exec_simple"}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    metrics = evaluate_function_call(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "final_answer": '{"name":"not_an_official_bfcl_function","arguments":{"value":1}}',
            }
        ],
        dataset_path=str(path),
        avg_k=(1,),
    )

    assert metrics.success_rate == 0.0
    assert metrics.payloads[0]["fail_reason"] == "bfcl_exec:official_ground_truth_execution_failed"


def test_bfcl_exec_scorer_reconstructs_ground_truth_from_legacy_expected_calls(tmp_path) -> None:
    path = tmp_path / "bfcl_exec_simple_legacy.jsonl"
    row = _simple_tool_call_row()
    row["task_id"] = "exec_simple_legacy"
    row["expected_tool_calls"] = [
        {
            "name": "calculate_density",
            "arguments": {"mass": 50, "volume": 10},
            "argument_options": {"mass": [50], "volume": [10]},
        }
    ]
    row["scorer"] = {"type": "bfcl_exec", "execution_result_type": ["exact_match"]}
    row["metadata"] = {"category": "exec_simple"}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    metrics = evaluate_function_call(
        [
            {
                "sample_index": 0,
                "repeat_index": 0,
                "final_answer": '{"name":"calculate_density","arguments":{"mass":50,"volume":10}}',
            }
        ],
        dataset_path=str(path),
        avg_k=(1,),
    )

    assert metrics.success_rate == 1.0


def test_bfcl_exec_prompt_puts_strict_schema_in_system(tmp_path) -> None:
    path = tmp_path / "bfcl_exec_multiple.jsonl"
    row = _simple_tool_call_row()
    row["task_id"] = "exec_multiple_0"
    row["metadata"] = {"category": "exec_multiple"}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    record = JsonlFunctionCallTaskLoader(path).load()[0]
    pipeline = object.__new__(FunctionCallPipeline)

    prompt = pipeline._make_prompt(record)

    assert 'The JSON shape is {"name":"tool_name","arguments":{...}}.' in prompt
    assert "The arguments value must be a JSON object, not a JSON string." in prompt
    assert "If multiple tool calls are required, return a JSON array" in prompt


def test_toolalpaca_prompt_adds_name_and_argument_constraints(tmp_path) -> None:
    path = tmp_path / "toolalpaca_eval_simulated.jsonl"
    row = _simple_tool_call_row()
    row["task_id"] = "toolalpaca_0"
    row["instruction"] = "Find a wild medium axolotl image."
    row["tools"] = [
        {
            "name": "getRandomAxolotlImage",
            "description": "Retrieve a random image.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "searchAxolotlImages",
            "description": "Search images.\nParameters: {\"color\":\"Required.\"}\nOutput: results",
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {"type": "string"},
                    "size": {"type": "string"},
                    "page": {"type": "integer"},
                },
                "required": ["color"],
            },
        },
    ]
    row["expected_tool_calls"] = [
        {
            "name": "searchAxolotlImages",
            "arguments": {"color": "wild", "size": "medium"},
            "argument_options": {"color": ["wild"], "size": ["medium"]},
        }
    ]
    row["metadata"] = {"source_format": "official_toolalpaca"}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    record = JsonlFunctionCallTaskLoader(path).load()[0]
    pipeline = object.__new__(FunctionCallPipeline)

    prompt = pipeline._make_prompt(record)

    assert "ToolAlpaca selection rules:" in prompt
    assert 'Allowed tool names: ["getRandomAxolotlImage","searchAxolotlImages"].' in prompt
    assert "The name value must exactly copy one allowed tool name" in prompt
    assert "use only argument keys shown" in prompt
    assert "Include every argument whose value is stated or directly implied" in prompt
    assert "return one JSON array item for each required action" in prompt
    assert "Omit optional arguments" not in prompt
    assert "fewest tool calls" not in prompt
    assert '"required": [' in prompt


def test_toolalpaca_description_summary_drops_embedded_parameters(tmp_path) -> None:
    from src.eval.function_calling.simple_tool_call import load_toolalpaca_rows_from_source

    source = tmp_path / "toolalpaca.json"
    source.write_text(
        json.dumps(
            [
                {
                    "Name": "Axolotl",
                    "Instructions": ["Find a wild axolotl image."],
                    "Golden_Answers": [
                        [
                            {
                                "Action": "searchAxolotlImages",
                                "Action_Input": {"color": "wild"},
                            }
                        ]
                    ],
                    "Function_Description": {
                        "searchAxolotlImages": (
                            "Search images.\n"
                            "Parameters: {\"color\":\"string. Required.\"}\n"
                            "Output: matching images"
                        )
                    },
                    "Function_Projection": {"searchAxolotlImages": ["/images", "get"]},
                    "Documentation": json.dumps({"servers": [{"url": "https://example.test"}], "paths": {}}),
                    "NLDocumentation": "searchAxolotlImages: Search images.",
                    "Authentication": {"api_key": "real-key-must-not-be-copied"},
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = load_toolalpaca_rows_from_source(source, dataset_name="toolalpaca_eval_simulated")

    assert rows[0]["tools"][0]["description"] == "Search images."
    assert rows[0]["tools"][0]["parameters"]["required"] == ["color"]
    assert rows[0]["scorer"]["type"] == "toolalpaca_official"
    assert rows[0]["metadata"]["toolalpaca_dataset"] == "simulated"
    assert rows[0]["metadata"]["toolalpaca_function_projection"] == {
        "searchAxolotlImages": ["/images", "get"]
    }
    assert rows[0]["metadata"]["toolalpaca_authentication"] == {"api_key": "***"}


def test_toolalpaca_official_scorer_executes_and_judges_local_json(tmp_path) -> None:
    path = tmp_path / "toolalpaca_eval_real.jsonl"
    row = _simple_tool_call_row()
    row["task_id"] = "toolalpaca_eval_real__weather_000"
    row["instruction"] = "Get current weather in Tokyo."
    row["tools"] = [
        {
            "name": "current_weather",
            "description": "Get current weather.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]
    row["expected_tool_calls"] = [
        {
            "name": "current_weather",
            "arguments": {"query": "Tokyo"},
            "argument_options": {"query": ["Tokyo"]},
        }
    ]
    row["scorer"] = {"type": "toolalpaca_official"}
    row["metadata"] = {
        "source_format": "official_toolalpaca",
        "toolalpaca_dataset": "real",
        "api_name": "Weather",
        "toolalpaca_documentation": json.dumps(
            {
                "servers": [{"url": "https://example.test"}],
                "paths": {
                    "/current": {
                        "get": {
                            "parameters": [
                                {"name": "query", "in": "query", "required": True},
                            ]
                        }
                    }
                },
            }
        ),
        "toolalpaca_nl_documentation": "current_weather: Get current weather.",
        "toolalpaca_function_projection": {"current_weather": ["/current", "get"]},
        "toolalpaca_authentication": {},
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    with (
        patch("src.eval.function_calling.toolalpaca_official.requests.request") as mocked_request,
        patch("src.eval.function_calling.toolalpaca_official.judge_toolalpaca_solution") as mocked_judge,
    ):
        mocked_request.return_value.status_code = 200
        mocked_request.return_value.text = '{"temperature":21}'
        mocked_judge.return_value = {
            "process_correctness": "Yes",
            "final_response_correctness": "Uncertain",
        }
        metrics = evaluate_function_call(
            [
                {
                    "sample_index": 0,
                    "repeat_index": 0,
                    "final_answer": '{"name":"current_weather","arguments":{"query":"Tokyo"}}',
                }
            ],
            dataset_path=str(path),
            avg_k=(1,),
        )

    assert metrics.success_rate == 1.0
    assert metrics.avg_at_k == {"avg@1": 1.0}
    mocked_request.assert_called_once()
    mocked_judge.assert_called_once()


def test_toolalpaca_official_adapter_converts_local_json_calls() -> None:
    actions = local_calls_to_official_actions(
        [
            {
                "name": "PublicHolidayPublicHolidaysV3",
                "arguments": {"year": 2023, "countryCode": "US"},
            }
        ]
    )

    assert actions[0].action == "PublicHolidayPublicHolidaysV3"
    assert actions[0].action_input == {"year": 2023, "countryCode": "US"}


def test_toolalpaca_execution_does_not_inject_redacted_auth_placeholder(tmp_path) -> None:
    path = tmp_path / "toolalpaca_eval_real.jsonl"
    row = _simple_tool_call_row()
    row["task_id"] = "toolalpaca_eval_real__auth_000"
    row["instruction"] = "Call the protected endpoint."
    row["tools"] = [
        {
            "name": "protected_lookup",
            "description": "Protected lookup.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "api_key": {"type": "string"}},
                "required": ["query", "api_key"],
            },
        }
    ]
    row["expected_tool_calls"] = [
        {
            "name": "protected_lookup",
            "arguments": {"query": "alpha"},
            "argument_options": {"query": ["alpha"]},
        }
    ]
    row["scorer"] = {"type": "toolalpaca_official"}
    row["metadata"] = {
        "source_format": "official_toolalpaca",
        "toolalpaca_dataset": "real",
        "api_name": "Protected",
        "toolalpaca_documentation": json.dumps(
            {
                "servers": [{"url": "https://example.test"}],
                "paths": {
                    "/lookup": {
                        "get": {
                            "parameters": [
                                {"name": "query", "in": "query", "required": True},
                                {"name": "api_key", "in": "query", "required": True},
                            ]
                        }
                    }
                },
            }
        ),
        "toolalpaca_function_projection": {"protected_lookup": ["/lookup", "get"]},
        "toolalpaca_authentication": {"api_key": "***"},
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    record = JsonlFunctionCallTaskLoader(path).load()[0]

    steps = execute_toolalpaca_actions(
        record,
        local_calls_to_official_actions([{"name": "protected_lookup", "arguments": {"query": "alpha"}}]),
    )

    assert "Missing required parameters: api_key" in steps[0].observation
    assert "***" not in steps[0].observation


def test_function_call_loader_rejects_legacy_expected_call(tmp_path) -> None:
    path = tmp_path / "legacy_function_call.jsonl"
    row = {
        "task_id": "legacy-1",
        "instruction": "Look up alpha.",
        "tools": [{"name": "lookup", "arguments": {"query": {"type": "string"}}}],
        "expected_call": {"name": "lookup", "arguments": {"query": "alpha"}},
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected_tool_calls"):
        JsonlFunctionCallTaskLoader(path).load()


def test_function_call_loader_rejects_legacy_env_and_scorer(tmp_path) -> None:
    path = tmp_path / "legacy_env.jsonl"
    row = _simple_tool_call_row()
    row["env"] = {"type": "json_function_call"}
    row["scorer"] = {"type": "json_function_call_exact"}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="simple_tool_call env"):
        JsonlFunctionCallTaskLoader(path).load()


def test_function_call_scheduler_exposes_only_bfcl_and_toolalpaca_jobs() -> None:
    assert "function_call" not in JOB_CATALOGUE
    assert JOB_CATALOGUE["function_bfcl_ast"].is_cot is False
    assert JOB_CATALOGUE["function_bfcl_exec"].is_cot is False
    assert JOB_CATALOGUE["function_toolalpaca"].is_cot is False
    assert detect_job_from_dataset("bfcl_exec_multiple_test", is_cot=False) == "function_bfcl_exec"
    assert detect_job_from_dataset("bfcl_multiple_test", is_cot=False) == "function_bfcl_ast"
    assert detect_job_from_dataset("toolalpaca_eval_real_test", is_cot=False) == "function_toolalpaca"
    assert detect_job_from_dataset("bfcl_exec_multiple_test", is_cot=True) is None
