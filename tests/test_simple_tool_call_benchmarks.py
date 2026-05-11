from __future__ import annotations

from pathlib import Path

from src.eval.datasets.data_prepper.data_manager import available_function_calling_datasets, prepare_dataset
from src.eval.datasets.runtime import read_jsonl_items
from src.eval.function_calling import runner as function_calling_runner
from src.eval.function_calling.simple_tool_call import (
    SimpleToolCallRecord,
    ToolCallExpectation,
    build_simple_tool_call_prompt,
)


def test_simple_tool_call_benchmarks_are_registered() -> None:
    names = set(available_function_calling_datasets())

    assert "bfcl_simple_python" in names
    assert "bfcl_exec_simple" in names
    assert "bfcl_multiple" in names
    assert "bfcl_exec_multiple" in names
    assert "toolalpaca_eval_simulated" in names
    assert "toolalpaca_eval_real" in names


def test_prepare_dataset_materializes_bfcl_small_ast_spec(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "bfcl_data"
    possible_root = source_root / "possible_answer"
    possible_root.mkdir(parents=True)
    question_path = source_root / "BFCL_v4_simple_python.json"
    answer_path = possible_root / "BFCL_v4_simple_python.json"
    question_path.write_text(
        '{"id":"simple_python_0","question":[[{"role":"user","content":"Find the area."}]],'
        '"function":[{"name":"calculate_area","description":"Calculate area",'
        '"parameters":{"type":"dict","properties":{"base":{"type":"integer"},"height":{"type":"integer"}},'
        '"required":["base","height"]}}]}\n',
        encoding="utf-8",
    )
    answer_path.write_text(
        '{"id":"simple_python_0","ground_truth":[{"calculate_area":{"base":[10],"height":[5],"unit":["units",""]}}]}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.bfcl_small.bfcl_small_source_root",
        lambda: source_root,
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("bfcl_simple_python", output_root, "test")

    assert paths == [output_root / "bfcl_simple_python" / "test.jsonl"]
    assert read_jsonl_items(paths[0]) == [
        {
            "task_id": "simple_python_0",
            "instruction": "User: Find the area.",
            "tools": [
                {
                    "name": "calculate_area",
                    "description": "Calculate area",
                    "parameters": {
                        "type": "object",
                        "properties": {"base": {"type": "integer"}, "height": {"type": "integer"}},
                        "required": ["base", "height"],
                    },
                }
            ],
            "expected_tool_calls": [
                {
                    "name": "calculate_area",
                    "arguments": {"base": 10, "height": 5, "unit": "units"},
                    "argument_options": {"base": [10], "height": [5], "unit": ["units", ""]},
                }
            ],
            "metadata": {
                "source_format": "official_bfcl_v4_ast",
                "category": "simple_python",
                "source_path": str(question_path.resolve()),
                "possible_answer_path": str(answer_path.resolve()),
                "execution_result_type": [],
            },
        }
    ]


def test_prepare_dataset_materializes_toolalpaca_spec(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "toolalpaca"
    source_root.mkdir(parents=True)
    source = source_root / "eval_simulated.json"
    source.write_text(
        """[
  {
    "Name": "DemoAPI",
    "Function_Projection": {"lookup": ["/lookup", "get"]},
    "Function_Description": {
      "lookup": "Lookup a value.\\nParameters: {\\"query\\": \\"Required. String. Search query.\\"}\\nOutput: object",
      "components": ""
    },
    "Instructions": ["Look up alpha"],
    "Golden_Answers": [[{"Action": "lookup", "Action_Input": "{\\"query\\": \\"alpha\\"}"}]]
  }
]""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.toolalpaca.toolalpaca_source_root",
        lambda: source_root,
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("toolalpaca_eval_simulated", output_root, "test")

    assert paths == [output_root / "toolalpaca_eval_simulated" / "test.jsonl"]
    assert read_jsonl_items(paths[0]) == [
        {
            "task_id": "toolalpaca_eval_simulated__demoapi_000",
            "instruction": "Look up alpha",
            "tools": [
                {
                    "name": "lookup",
                    "description": (
                        "Lookup a value.\n"
                        'Parameters: {"query": "Required. String. Search query."}\n'
                        "Output: object"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string", "description": "Required. String. Search query."}},
                        "required": ["query"],
                    },
                    "metadata": {"path": "/lookup", "method": "get"},
                }
            ],
            "expected_tool_calls": [
                {
                    "name": "lookup",
                    "arguments": {"query": "alpha"},
                    "argument_options": {"query": ["alpha"]},
                }
            ],
            "metadata": {
                "source_format": "official_toolalpaca",
                "api_name": "DemoAPI",
                "api_index": 0,
                "question_index": 0,
                "source_path": str(source),
            },
        }
    ]


def test_simple_tool_call_prompt_uses_rwkv_json_function_call_shape() -> None:
    record = SimpleToolCallRecord(
        task_id="demo",
        instruction='Translate "Will it rain tomorrow?" into Japanese.',
        tools=(
            {
                "name": "translate_text",
                "description": "Translate text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "target_language": {"type": "string"},
                    },
                    "required": ["text", "target_language"],
                },
            },
        ),
        expected_tool_calls=(
            ToolCallExpectation(
                name="translate_text",
                arguments={"text": "Will it rain tomorrow?", "target_language": "Japanese"},
                argument_options={},
            ),
        ),
        metadata={},
    )

    prompt = build_simple_tool_call_prompt(record, history_max_chars=4000)

    assert prompt.startswith("System: Tools:\n[")
    assert '"name": "translate_text"' in prompt
    assert '"arguments": {' in prompt
    assert '"parameters"' not in prompt
    assert "Return only a JSON function call." in prompt
    assert "Available tools:" not in prompt
    assert '\n\nUser: Translate "Will it rain tomorrow?" into Japanese.\n\nAssistant: ```json\n' in prompt


def test_function_calling_runner_dispatches_simple_tool_call_runner(monkeypatch) -> None:
    called: list[tuple[str, str]] = []
    resolved = function_calling_runner.ResolvedFunctionCallingRun(
        benchmark_kind=function_calling_runner.FunctionCallingBenchmarkKind.BFCL_AST,
        dataset_path=Path("/tmp/bfcl_simple_python_test.jsonl"),
        dataset_slug="bfcl_simple_python_test",
        benchmark_name="bfcl_simple_python",
        dataset_split="test",
        model_name="demo-model",
        engine=None,  # type: ignore[arg-type]
    )

    monkeypatch.setattr(function_calling_runner, "validate_inference_backend_args", lambda _args: None)
    monkeypatch.setattr(function_calling_runner, "_resolve_run", lambda _args: resolved)
    monkeypatch.setattr(
        function_calling_runner,
        "_run_simple_tool_call",
        lambda _args, _run, *, default_job_name, run_context=None: called.append(
            (default_job_name, _run.dataset_slug)
        )
        or 0,
    )

    rc = function_calling_runner.main(["--dataset", "bfcl_simple_python_test.jsonl", "--model-path", "model.pth"])

    assert rc == 0
    assert called == [("function_bfcl_ast", "bfcl_simple_python_test")]
