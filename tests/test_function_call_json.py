from __future__ import annotations

import json
from unittest.mock import patch

from src.eval.datasets.data_loader.function_call import JsonlFunctionCallTaskLoader
from src.eval.evaluators.function_call import FunctionCallPipeline
from src.eval.function_calling.one_step.bfcl_exec import evaluate_bfcl_executable_calls
from src.eval.function_calling.one_step.toolalpaca import (
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
                    '{"name":"calc_binomial_probability","arguments":{"n":20,"k":5,"p":0.16666666666666666}}'
                ),
            }
        ],
        dataset_path=str(path),
        avg_k=(1,),
    )

    assert record.env == {"type": "simple_tool_call"}
    assert record.scorer["type"] == "simple_tool_call"
    assert record.expected_tool_calls[0]["argument_options"]["p"] == [1 / 6]
    assert prompt.endswith("Assistant: <think>\n</think>\n```json\n")
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
                "final_answer": ('{"name":"calc_binomial_probability","arguments":{"n":20,"k":5,"p":0.6}}'),
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
                "final_answer": ('{"name":"mat_mul","arguments":{"matA":[[1,2],[3,4]],"matB":[[5,6],[7,8]]}}'),
            }
        ],
        dataset_path=str(path),
        avg_k=(1,),
    )

    assert metrics.success_rate == 1.0
    record = JsonlFunctionCallTaskLoader(path).load()[0]
    result = evaluate_bfcl_executable_calls(
        record,
        [
            {
                "name": "mat_mul",
                "arguments": {"matA": [[1, 2], [3, 4]], "matB": [[5, 6], [7, 8]]},
            }
        ],
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


def test_bfcl_exec_prompt_puts_strict_schema_in_system(tmp_path) -> None:
    path = tmp_path / "bfcl_exec_multiple.jsonl"
    row = _simple_tool_call_row()
    row["task_id"] = "exec_multiple_0"
    row["metadata"] = {"category": "exec_multiple"}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    record = JsonlFunctionCallTaskLoader(path).load()[0]
    pipeline = object.__new__(FunctionCallPipeline)

    prompt = pipeline._make_prompt(record)

    assert "Output JSON schema:" in prompt
    assert "Return exactly one JSON value that validates against the schema." in prompt
    assert "For multiple required tool calls, return a JSON array containing every required call" in prompt
    assert "Do not copy tool schemas, descriptions, type/items/properties/required/default fields" in prompt


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
            "description": 'Search images.\nParameters: {"color":"Required."}\nOutput: results',
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

    assert "Tools:" in prompt
    assert "Output JSON schema:" in prompt
    assert "Use only listed tool names." in prompt
    assert "Each arguments object must contain only final argument values for that tool." in prompt
    assert "Return no prose, no markdown, and no extra text outside the JSON value." in prompt
    assert "Omit optional arguments" not in prompt
    assert "fewest tool calls" not in prompt
    assert '"required": [' in prompt


def test_toolalpaca_description_summary_drops_embedded_parameters(tmp_path) -> None:
    from src.eval.function_calling.one_step.simple_tool_call import (
        load_toolalpaca_rows_from_source,
    )

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
                            'Search images.\nParameters: {"color":"string. Required."}\nOutput: matching images'
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

    assert rows[0]["tools"][0]["description"].startswith("Search images.")
    assert rows[0]["tools"][0]["parameters"]["required"] == ["color"]
    assert rows[0]["scorer"]["type"] == "toolalpaca_official"
    assert rows[0]["metadata"]["toolalpaca_dataset"] == "simulated"
    assert rows[0]["metadata"]["path"] == "/images"
    assert rows[0]["metadata"]["method"] == "get"
    assert rows[0]["metadata"]["api_name"] == "Axolotl"
    assert rows[0]["metadata"]["server_url"] == "https://example.test"
    assert rows[0]["tools"][0]["metadata"]["path"] == "/images"


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
    with patch("src.eval.function_calling.one_step.toolalpaca.requests.request") as mocked_request:
        mocked_request.return_value.status_code = 200
        mocked_request.return_value.text = '{"temperature":21}'
        mocked_request.return_value.headers = {"Content-Type": "application/json"}
        mocked_request.return_value.json.return_value = {"temperature": 21}
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
    assert mocked_request.call_count == 2


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


def test_toolalpaca_execution_does_not_inject_redacted_auth_placeholder(
    tmp_path,
) -> None:
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
                "properties": {
                    "query": {"type": "string"},
                    "api_key": {"type": "string"},
                },
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

    assert "missing_required_arguments(api_key)" in steps[0].observation
    assert "***" not in steps[0].observation


def test_function_call_scheduler_exposes_new_function_call_jobs() -> None:
    from src.eval.scheduler.cli import _resolve_job_list
    from src.eval.scheduler.jobs import JOB_ORDER

    assert JOB_CATALOGUE["function_one_step_bfcl_ast"].is_cot is False
    assert JOB_CATALOGUE["function_one_step_bfcl_exec"].is_cot is False
    assert JOB_CATALOGUE["function_one_step_toolalpaca"].is_cot is False
    assert JOB_CATALOGUE["function_one_step_apibank_l1"].is_cot is False
    assert JOB_CATALOGUE["function_one_step_complexfuncbench_subset"].is_cot is False
    assert JOB_CATALOGUE["function_agent_apibank_l2"].is_cot is False
    assert JOB_CATALOGUE["function_agent_browsecomp_plus"].is_cot is False
    assert "function_one_step_bfcl_ast" in JOB_ORDER
    assert _resolve_job_list(("function_one_step_bfcl_ast",), None, None) == ("function_one_step_bfcl_ast",)
    for job_name in (
        "function_one_step_apibank_l1",
        "function_one_step_complexfuncbench_subset",
        "function_agent_apibank_l2",
        "function_agent_browsecomp_plus",
    ):
        assert JOB_CATALOGUE[job_name].domain == "function_call"
        assert job_name in JOB_ORDER
        assert _resolve_job_list((job_name,), None, None) == (job_name,)
    assert detect_job_from_dataset("apibank_l1_test", is_cot=False) == "function_one_step_apibank_l1"
    assert detect_job_from_dataset("apibank_l2_test", is_cot=False) == "function_agent_apibank_l2"
    assert (
        detect_job_from_dataset("complexfuncbench_subset_test", is_cot=False)
        == "function_one_step_complexfuncbench_subset"
    )
    assert detect_job_from_dataset("browsecomp_plus_test", is_cot=False) == "function_agent_browsecomp_plus"
    assert detect_job_from_dataset("bfcl_exec_multiple_test", is_cot=False) == "function_one_step_bfcl_exec"
    assert detect_job_from_dataset("bfcl_multiple_test", is_cot=False) == "function_one_step_bfcl_ast"
    assert detect_job_from_dataset("toolalpaca_eval_real_test", is_cot=False) == "function_one_step_toolalpaca"
    assert detect_job_from_dataset("bfcl_exec_multiple_test", is_cot=True) is None


def test_function_call_eval_uses_new_one_step_job_names(
    monkeypatch,
) -> None:
    from src.eval.function_calling.one_step.jobs import simple_tool_call_job_name

    monkeypatch.delenv("RWKV_SKILLS_JOB_NAME", raising=False)
    assert simple_tool_call_job_name("bfcl_multiple_test") == "function_one_step_bfcl_ast"

    monkeypatch.setenv("RWKV_SKILLS_JOB_NAME", "function_one_step_bfcl_ast")
    assert simple_tool_call_job_name("bfcl_multiple_test") == "function_one_step_bfcl_ast"

    monkeypatch.setenv("RWKV_SKILLS_JOB_NAME", "function_one_step_bfcl_exec")
    assert simple_tool_call_job_name("bfcl_multiple_test") == "function_one_step_bfcl_ast"
    assert simple_tool_call_job_name("apibank_l1_test") == "function_one_step_apibank_l1"
    assert simple_tool_call_job_name("complexfuncbench_subset_test") == "function_one_step_complexfuncbench_subset"


def test_agent_runner_records_full_trajectory() -> None:
    from src.eval.function_calling.agent.env import AgentObservation, AgentStepResult
    from src.eval.function_calling.agent.runner import run_function_calling_agent

    class FakeEnv:
        def __init__(self) -> None:
            self.actions = []

        def reset(self) -> AgentObservation:
            return AgentObservation("ready", {"source": "fake"})

        def step(self, action) -> AgentStepResult:
            self.actions.append(action)
            return AgentStepResult(
                AgentObservation("done", {"rows": 1}),
                done=True,
                score=1.0,
                success=True,
                details={"ok": True},
            )

    env = FakeEnv()

    def generate_action(events, observation, step):
        assert step == 0
        assert observation.content == "ready"
        assert events[-1]["type"] == "observation"
        return '{"name":"query","arguments":{"sql":"select 1"}}'

    result = run_function_calling_agent(env, generate_action)

    assert result.success is True
    assert result.score == 1.0
    assert env.actions[0].name == "query"
    assert [event["type"] for event in result.events] == [
        "observation",
        "model_output",
        "action",
        "env_result",
        "final_score",
    ]
    assert result.events[-1]["metadata"]["finish_reason"] == "done"


def test_agent_runner_records_parse_error() -> None:
    from src.eval.function_calling.agent.env import AgentObservation
    from src.eval.function_calling.agent.runner import run_function_calling_agent

    class FakeEnv:
        def reset(self) -> AgentObservation:
            return AgentObservation("ready")

        def step(self, action):  # pragma: no cover - parse error should stop first
            raise AssertionError("env.step should not be called")

    result = run_function_calling_agent(FakeEnv(), lambda _events, _observation, _step: "not json")

    assert result.success is False
    assert result.details["finish_reason"] == "parse_error"
    assert result.details["parse_error_count"] == 1
    assert [event["type"] for event in result.events] == [
        "observation",
        "model_output",
        "error",
        "final_score",
    ]


def test_future_official_adapter_boundaries_are_declared(tmp_path) -> None:
    from src.eval.function_calling.agent.adapters.agentbench import (
        AgentBenchAdapterConfig,
        require_agentbench_assets,
    )
    from src.eval.function_calling.agent.adapters.apibank import (
        ApiBankLevel2AdapterConfig,
        require_apibank_level2_assets,
    )
    from src.eval.function_calling.common.action import ToolAction
    from src.eval.function_calling.one_step.apibank import (
        apibank_action_text,
        require_official_apibank_root,
    )

    apibank_root = tmp_path / "api-bank"
    apibank_root.mkdir()
    (apibank_root / "evaluator.py").write_text("", encoding="utf-8")
    (apibank_root / "lv1-lv2-samples" / "level-2-toolsearcher").mkdir(parents=True)
    agentbench_root = tmp_path / "AgentBench"
    (agentbench_root / "src" / "server" / "tasks" / "dbbench").mkdir(parents=True)
    (agentbench_root / "src" / "server" / "tasks" / "knowledgegraph").mkdir(parents=True)

    assert require_official_apibank_root(apibank_root) == apibank_root
    assert require_apibank_level2_assets(ApiBankLevel2AdapterConfig(official_root=apibank_root)) == apibank_root
    assert (
        require_agentbench_assets(AgentBenchAdapterConfig(task="db", official_root=agentbench_root)) == agentbench_root
    )
    assert (
        require_agentbench_assets(AgentBenchAdapterConfig(task="kg", official_root=agentbench_root)) == agentbench_root
    )
    assert apibank_action_text(ToolAction(name="Calculator", arguments={"formula": "1+1"})) == (
        "[Calculator(formula='1+1')]"
    )


def test_apibank_level2_env_runs_expected_trace(monkeypatch, tmp_path) -> None:
    from src.eval.function_calling.agent.adapters import apibank as apibank_agent

    history = [
        {"role": "User", "text": "Can you calculate 1+1?"},
        {
            "role": "API",
            "api_name": "ToolSearcher",
            "param_dict": {"keywords": "calculator"},
            "result": {
                "api_name": "ToolSearcher",
                "input": {"keywords": "calculator"},
                "output": {"name": "Calculator"},
                "exception": None,
            },
        },
        {
            "role": "API",
            "api_name": "Calculator",
            "param_dict": {"formula": "1+1"},
            "result": {
                "api_name": "Calculator",
                "input": {"formula": "1+1"},
                "output": 2,
                "exception": None,
            },
        },
    ]

    class FakeApi:
        def check_api_call_correctness(self, result, ground_truth) -> bool:
            return result == ground_truth

    class FakeToolManager:
        def api_call(self, name, **kwargs):
            return {
                "api_name": name,
                "input": kwargs,
                "output": 2,
                "exception": None,
            }

        def init_tool(self, name):
            return FakeApi()

    root = tmp_path / "api-bank"
    root.mkdir()
    (root / "evaluator.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        apibank_agent,
        "_official_tool_manager",
        lambda _root, **_kwargs: FakeToolManager(),
    )
    row = {
        "task_id": "apibank_l2__fake",
        "env": {"type": "apibank_level2", "official_root": str(root)},
        "metadata": {
            "apibank_official_root": str(root),
            "apibank_history": history,
            "apibank_expected_api_steps": [item for item in history if item["role"] == "API"],
        },
    }

    env = apibank_agent.ApiBankLevel2Env(row)
    observation = env.reset()
    results = [env.step(action) for action in apibank_agent.expected_apibank_level2_actions(row)]

    assert observation.content == "User: Can you calculate 1+1?"
    assert results[0].done is False
    assert results[-1].done is True
    assert results[-1].score == 1.0
    assert results[-1].success is True


def test_complexfuncbench_subset_loader_and_scorer(monkeypatch, tmp_path) -> None:
    from src.eval.datasets.data_loader.function_call import JsonlFunctionCallTaskLoader
    from src.eval.datasets.data_prepper.function_call.complexfuncbench import (
        prepare_complexfuncbench_subset,
    )
    from src.eval.function_calling.one_step.complexfuncbench import (
        evaluate_complexfuncbench_subset_calls,
    )

    source_root = tmp_path / "ComplexFuncBench"
    source_dir = source_root / "data"
    source_dir.mkdir(parents=True)
    source = source_dir / "ComplexFuncBench.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "case-1",
                "functions": [
                    {
                        "name": "SearchHotel",
                        "description": "Search hotels.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}, "adults": {"type": "integer"}},
                            "required": ["city", "adults"],
                        },
                    }
                ],
                "conversations": [
                    {"role": "user", "content": "Find a hotel in Paris for two adults."},
                    {
                        "role": "assistant",
                        "function_call": [
                            {"name": "SearchHotel", "arguments": {"city": "Paris", "adults": 2}}
                        ],
                    },
                    {"role": "tool", "content": [{"hotel": "ok"}]},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RWKV_COMPLEXFUNC_SOURCE_ROOT", str(source_root))

    paths = prepare_complexfuncbench_subset(tmp_path / "out", "test")
    records = JsonlFunctionCallTaskLoader(paths[0]).load()
    record = list(records)[0]
    result = evaluate_complexfuncbench_subset_calls(
        record,
        [{"name": "SearchHotel", "arguments": {"city": "Paris", "adults": 2}}],
    )

    assert record.task_id == "complexfuncbench_subset__case-1"
    assert record.scorer["type"] == "complexfuncbench_subset"
    assert result.is_passed is True
    assert result.details["call_accuracy"] == 1.0


def test_browsecomp_plus_env_exports_official_run_payload() -> None:
    from src.eval.function_calling.agent.adapters.browsecomp_plus import (
        BrowseCompPlusEnv,
        browsecomp_plus_run_from_agent_details,
    )
    from src.eval.function_calling.agent.runner import run_function_calling_agent

    row = {
        "task_id": "browsecomp_plus__q1",
        "instruction": "Who founded Example Corp?",
        "env": {"type": "browsecomp_plus", "query_id": "q1", "k": 1},
        "metadata": {
            "query_id": "q1",
            "query": "Who founded Example Corp?",
            "answer": "Ada Lovelace",
            "browsecomp_plus_documents": [
                {"docid": "42", "text": "Example Corp was founded by Ada Lovelace."}
            ],
        },
    }
    outputs = [
        '{"name":"search","arguments":{"query":"Example Corp founder"}}',
        '{"name":"final_answer","arguments":{"answer":"Ada Lovelace [42]"}}',
    ]

    result = run_function_calling_agent(
        BrowseCompPlusEnv(row),
        lambda _events, _observation, step: outputs[step],
    )
    run_payload = browsecomp_plus_run_from_agent_details(result.details)

    assert result.success is True
    assert run_payload is not None
    assert run_payload["query_id"] == "q1"
    assert run_payload["status"] == "completed"
    assert run_payload["tool_call_counts"] == {"search": 1}
    assert run_payload["retrieved_docids"] == ["42"]
    assert run_payload["result"][-1]["output"] == "Ada Lovelace [42]"


def test_browsecomp_plus_env_uses_bm25_index_searcher(monkeypatch, tmp_path) -> None:
    from src.eval.function_calling.agent.adapters import browsecomp_plus as adapter
    from src.eval.function_calling.common.action import ToolAction

    class FakeSearcher:
        def search(self, query: str, k: int):
            assert query == "Example Corp founder"
            assert k == 1
            return [({"docid": "99", "text": "Example Corp was founded by Ada Lovelace."}, 7.5)]

        def get_document(self, docid: str):
            assert docid == "99"
            return {"docid": "99", "text": "Full document text."}

    index = tmp_path / "bm25"
    index.mkdir()
    (index / "segments_1").write_text("", encoding="utf-8")
    monkeypatch.setattr(adapter, "_get_pyserini_bm25_searcher", lambda _path: FakeSearcher())

    env = adapter.BrowseCompPlusEnv(
        {
            "task_id": "browsecomp_plus__q1",
            "instruction": "Who founded Example Corp?",
            "env": {"type": "browsecomp_plus", "query_id": "q1", "k": 1, "index_path": str(index)},
            "metadata": {"query_id": "q1", "query": "Who founded Example Corp?"},
        }
    )
    env.reset()
    search_result = env.step(ToolAction(name="search", arguments={"query": "Example Corp founder"}))
    document_result = env.step(ToolAction(name="get_document", arguments={"docid": "99"}))

    assert search_result.details["retriever"] == "bm25"
    assert search_result.observation.metadata["retrieved_docids"] == ["99"]
    assert "Ada Lovelace" in search_result.observation.content
    assert "Full document text" in document_result.observation.content


def test_browsecomp_plus_loader_preserves_official_doc_fallback(tmp_path) -> None:
    from src.eval.function_calling.agent.adapters.browsecomp_plus import (
        load_browsecomp_plus_rows_from_decrypted_jsonl,
    )

    source = tmp_path / "browsecomp_plus_decrypted.jsonl"
    source.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "Who founded Example Corp?",
                "answer": "Ada Lovelace",
                "gold_docs": [{"docid": "42", "contents": "Gold document"}],
                "evidence_docs": [{"docid": "43", "text": "Evidence document"}],
                "negative_docs": [{"docid": "44", "text": "Negative document"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_browsecomp_plus_rows_from_decrypted_jsonl(source, official_root=tmp_path)

    documents = rows[0]["metadata"]["browsecomp_plus_documents"]
    assert rows[0]["env"]["index_path"].endswith("indexes/bm25")
    assert [doc["docid"] for doc in documents] == ["42", "43", "44"]


def test_browsecomp_plus_openai_judge_uses_structured_outputs(tmp_path) -> None:
    from src.eval.function_calling.agent.adapters.browsecomp_plus_judge import (
        BrowseCompPlusJudgeConfig,
        evaluate_browsecomp_plus_completions,
    )

    class FakeResponse:
        output_text = json.dumps(
            {
                "extracted_final_answer": "Ada Lovelace",
                "correct_answer": "Ada Lovelace",
                "reasoning": "The extracted answer matches.",
                "correct": "yes",
                "confidence": 90,
            }
        )

    class FakeResponses:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return FakeResponse()

    class FakeClient:
        def __init__(self) -> None:
            self.responses = FakeResponses()

    gt_path = tmp_path / "browsecomp_plus_decrypted.jsonl"
    gt_path.write_text(
        json.dumps({"query_id": "q1", "query": "Who founded Example Corp?", "answer": "Ada Lovelace"})
        + "\n",
        encoding="utf-8",
    )
    qrel_path = tmp_path / "qrel_evidence.txt"
    qrel_path.write_text("q1 0 42 1\n", encoding="utf-8")
    completion = {
        "benchmark_name": "browsecomp_plus",
        "dataset_split": "test",
        "sample_index": 0,
        "repeat_index": 0,
        "prompt1": "prompt",
        "completion1": "completion",
        "final_answer": '{"name":"final_answer","arguments":{"answer":"Ada Lovelace [42]"}}',
        "browsecomp_plus_run": {
            "query_id": "q1",
            "status": "completed",
            "tool_call_counts": {"search": 1},
            "retrieved_docids": ["42"],
            "result": [{"type": "output_text", "output": "Ada Lovelace [42]"}],
        },
    }
    fake_client = FakeClient()

    metrics = evaluate_browsecomp_plus_completions(
        [completion],
        config=BrowseCompPlusJudgeConfig(model="gpt-5.4-mini"),
        ground_truth_path=gt_path,
        qrel_evidence_path=qrel_path,
        eval_dir=tmp_path / "evals",
        client=fake_client,
    )

    assert metrics.accuracy == 1.0
    assert metrics.retrieval_recall == 1.0
    assert metrics.summary["Accuracy (%)"] == 100.0
    assert metrics.payloads[0]["is_passed"] is True
    assert (tmp_path / "evals" / "evaluation_summary.json").exists()
    request = fake_client.responses.calls[0]
    assert request["model"] == "gpt-5.4-mini"
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True
    assert request["text"]["format"]["schema"]["properties"]["correct"]["enum"] == ["yes", "no"]
    assert request["text"]["format"]["schema"]["required"] == [
        "extracted_final_answer",
        "correct_answer",
        "reasoning",
        "correct",
        "confidence",
    ]


def test_browsecomp_plus_judge_reads_benchmark_toml(monkeypatch) -> None:
    from src.eval.benchmark_config import resolve_benchmark_model_config
    from src.eval.function_calling.agent.adapters.browsecomp_plus_judge import (
        BrowseCompPlusJudgeConfig,
    )

    monkeypatch.setenv("BROWSECOMP_PLUS_JUDGE_API_KEY", "test-browsecomp-plus-key")

    config = resolve_benchmark_model_config("browsecomp_plus_test", "rwkv7-test", stage="tool")
    judge = BrowseCompPlusJudgeConfig.from_benchmark_config(config)

    assert config is not None
    assert config.browsecomp_plus_judge is not None
    assert "api_key" not in config.browsecomp_plus_judge
    assert "api_key_env" not in config.browsecomp_plus_judge
    assert judge.api_key == "test-browsecomp-plus-key"
    assert judge.model == "gpt-5.4-mini"
    assert judge.api_mode == "chat"
    assert judge.base_url == "https://api.ablai.top/v1/chat/completions"
    assert judge.max_output_tokens == 1024


def test_browsecomp_plus_openai_judge_uses_chat_completions_endpoint(tmp_path) -> None:
    from src.eval.function_calling.agent.adapters.browsecomp_plus_judge import (
        BrowseCompPlusJudgeConfig,
        evaluate_browsecomp_plus_completions,
    )

    class FakeMessage:
        content = json.dumps(
            {
                "extracted_final_answer": "Ada Lovelace",
                "correct_answer": "Ada Lovelace",
                "reasoning": "The extracted answer matches.",
                "correct": "yes",
                "confidence": 95,
            }
        )

    class FakeChoice:
        message = FakeMessage()

    class FakeCompletion:
        choices = [FakeChoice()]

    class FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return FakeCompletion()

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    gt_path = tmp_path / "browsecomp_plus_decrypted.jsonl"
    gt_path.write_text(
        json.dumps({"query_id": "q1", "query": "Who founded Example Corp?", "answer": "Ada Lovelace"})
        + "\n",
        encoding="utf-8",
    )
    completion = {
        "benchmark_name": "browsecomp_plus",
        "dataset_split": "test",
        "sample_index": 0,
        "repeat_index": 0,
        "prompt1": "prompt",
        "completion1": "completion",
        "browsecomp_plus_run": {
            "query_id": "q1",
            "status": "completed",
            "tool_call_counts": {"search": 1},
            "retrieved_docids": [],
            "result": [{"type": "output_text", "output": "Ada Lovelace"}],
        },
    }
    fake_client = FakeClient()

    metrics = evaluate_browsecomp_plus_completions(
        [completion],
        config=BrowseCompPlusJudgeConfig(
            model="gpt-5.4-mini",
            base_url="https://api.ablai.top/v1/chat/completions",
        ),
        ground_truth_path=gt_path,
        qrel_evidence_path=tmp_path / "missing_qrels.txt",
        client=fake_client,
    )

    assert metrics.accuracy == 1.0
    request = fake_client.chat.completions.calls[0]
    assert request["model"] == "gpt-5.4-mini"
    assert request["messages"][0]["role"] == "user"
    assert request["max_completion_tokens"] == 1024
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["response_format"]["json_schema"]["schema"]["properties"]["correct"]["enum"] == ["yes", "no"]


def test_agent_metrics_use_trajectory_payload() -> None:
    from src.eval.function_calling.agent.scorer import evaluate_function_call_agent

    metrics = evaluate_function_call_agent(
        [
            {
                "benchmark_name": "apibank_l2",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "final_answer": '{"name":"Calculator","arguments":{"formula":"1+1"}}',
                "success": True,
                "official_score": 1.0,
                "stats": {"steps": 2, "tool_calls": 2},
                "agent_details": {
                    "steps": 2,
                    "invalid_action_count": 0,
                    "parse_error_count": 0,
                    "timeout": False,
                    "finish_reason": "done",
                },
                "events": [{"type": "final_score", "metadata": {"success": True}}],
            }
        ]
    )

    assert metrics.success_rate == 1.0
    assert metrics.official_score == 1.0
    assert metrics.avg_steps == 2.0
    assert metrics.invalid_action_rate == 0.0
    assert metrics.timeout_rate == 0.0
    assert metrics.parse_error_rate == 0.0
    assert metrics.payloads and metrics.payloads[0]["is_passed"] is True


def test_apibank_official_scorer_uses_adapter_boundary(monkeypatch, tmp_path) -> None:
    from src.eval.datasets.data_struct.function_call import FunctionCallTaskRecord
    from src.eval.function_calling.one_step import apibank as apibank_adapter

    class FakeApi:
        def check_api_call_correctness(self, result, ground_truth) -> bool:
            return result == ground_truth

    class FakeToolManager:
        def api_call(self, name, **kwargs):
            return {"api_name": name, "input": kwargs, "output": 2, "exception": None}

        def init_tool(self, name):
            return FakeApi()

    monkeypatch.setattr(apibank_adapter, "require_official_apibank_root", lambda _root=None: tmp_path)
    monkeypatch.setattr(
        apibank_adapter,
        "_official_tool_manager",
        lambda _root, **_kwargs: FakeToolManager(),
    )

    record = FunctionCallTaskRecord(
        task_id="apibank_l1__fake__000",
        instruction="calculate",
        expected_tool_calls=[
            {
                "name": "Calculator",
                "arguments": {"formula": "1+1"},
                "argument_options": {"formula": ["1+1"]},
            }
        ],
        scorer={"type": "apibank_official"},
        metadata={
            "source_format": "official_apibank",
            "apibank_official_root": str(tmp_path),
            "apibank_ground_truth_result": {
                "api_name": "Calculator",
                "input": {"formula": "1+1"},
                "output": 2,
                "exception": None,
            },
        },
    )

    result = apibank_adapter.evaluate_apibank_official_calls(
        record,
        [{"name": "Calculator", "arguments": {"formula": "1+1"}}],
    )

    assert result.is_passed is True
    assert result.reward == 1.0
