from __future__ import annotations

from pathlib import Path

from src.eval.datasets.data_prepper.data_manager import available_function_calling_datasets, prepare_dataset
from src.eval.datasets.runtime import read_jsonl_items
from src.eval.function_calling import runner as function_calling_runner
from src.eval.function_calling import bfcl_ast as bfcl_ast_module
from src.eval.function_calling.simple_tool_call import (
    SimpleToolCallRecord,
    ToolCallExpectation,
    build_simple_tool_call_prompt,
    decode_simple_tool_call_response,
    load_simple_tool_call_manifest_records,
    normalize_simple_tool_call_manifest_row,
)
from src.eval.function_calling.toolalpaca_source import (
    _TOOLALPACA_OPTIONAL_KEY,
    _TOOLALPACA_REF_KEY,
    load_toolalpaca_rows_from_source,
)
from src.eval.function_calling.bfcl_exec import (
    BfclExecCallResult,
    BfclExecRecord,
    BfclExecSandbox,
    evaluate_bfcl_exec_calls,
)
from src.eval.function_calling.api_bank import ApiBankCallResult, evaluate_api_bank_calls
from src.eval.function_calling.toolalpaca import ToolAlpacaSandbox, evaluate_toolalpaca_actions


def test_simple_tool_call_benchmarks_are_registered() -> None:
    names = set(available_function_calling_datasets())

    assert "bfcl_simple_python" in names
    assert "bfcl_exec_simple_ast" in names
    assert "bfcl_multiple" in names
    assert "bfcl_exec_multiple_ast" in names
    assert "bfcl_exec_simple" in names
    assert "bfcl_exec_multiple" in names
    assert "bfcl_exec_parallel" in names
    assert "bfcl_exec_parallel_multiple" in names
    assert "apibank_level1" in names
    assert "apibank_level2" in names
    assert "agentbench_db" in names
    assert "agentbench_kg" in names
    assert "toolalpaca_eval_simulated" in names
    assert "toolalpaca_eval_real" in names
    assert "tau3_bench_mock" in names
    assert "tau3_bench_mock_long_context" in names


def test_bfcl_ast_official_root_falls_back_when_manifest_path_is_stale(tmp_path: Path, monkeypatch) -> None:
    fallback_root = tmp_path / "references" / "gorilla" / "berkeley-function-call-leaderboard"
    fallback_root.mkdir(parents=True)
    stale_root = tmp_path / "old-machine" / "berkeley-function-call-leaderboard"
    record = SimpleToolCallRecord(
        task_id="exec_simple_0",
        instruction="User: call the function.",
        tools=(),
        expected_tool_calls=(),
        metadata={"source_format": "official_bfcl_v4_ast", "official_root": str(stale_root)},
    )

    monkeypatch.setattr(bfcl_ast_module, "_repo_default_official_root", lambda: fallback_root)

    assert bfcl_ast_module._record_official_root(record) == str(fallback_root)


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
                "bfcl_official_function": [
                    {
                        "name": "calculate_area",
                        "description": "Calculate area",
                        "parameters": {
                            "type": "dict",
                            "properties": {"base": {"type": "integer"}, "height": {"type": "integer"}},
                            "required": ["base", "height"],
                        },
                    }
                ],
                "bfcl_official_ground_truth": [
                    {"calculate_area": {"base": [10], "height": [5], "unit": ["units", ""]}}
                ],
                "bfcl_official_language": "python",
            },
        }
    ]


def test_api_bank_evaluator_executes_official_style_sandbox() -> None:
    class _FakeSandbox:
        def api_call(self, api_name, arguments):
            assert api_name == "Calculator"
            assert arguments == {"formula": "2+2"}
            return ApiBankCallResult(True, {"output": "4"})

        def check_api_call_correctness(self, api_name, actual, expected):
            assert api_name == "Calculator"
            return actual == expected

    record = SimpleToolCallRecord(
        task_id="api-bank-0",
        instruction="Return next API call",
        tools=(),
        expected_tool_calls=(
            ToolCallExpectation(
                name="Calculator",
                arguments={"formula": "2+2"},
                argument_options={"formula": ("2+2",)},
            ),
        ),
        metadata={"expected_result": {"output": "4"}, "source_dir": "/tmp/api-bank/lv1-lv2-samples/level-1-given-desc"},
    )

    evaluation = evaluate_api_bank_calls(
        record,
        [{"name": "Calculator", "arguments": {"formula": "2+2"}}],
        sandbox=_FakeSandbox(),
    )

    assert evaluation.is_passed is True
    assert evaluation.reward == 1.0


def test_prepare_dataset_materializes_bfcl_exec_ast_spec(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "bfcl_data"
    question_root = source_root / "unused_datasets" / "question"
    possible_root = source_root / "unused_datasets" / "possible_answer"
    question_root.mkdir(parents=True)
    possible_root.mkdir(parents=True)
    question_path = question_root / "BFCL_v4_exec_simple.json"
    answer_path = possible_root / "BFCL_v4_exec_simple.json"
    question_path.write_text(
        '{"id":"exec_simple_0","question":[[{"role":"user","content":"Find the area."}]],'
        '"function":[{"name":"calculate_area","description":"Calculate area",'
        '"parameters":{"type":"dict","properties":{"base":{"type":"integer"},"height":{"type":"integer"}},'
        '"required":["base","height"]}}]}\n',
        encoding="utf-8",
    )
    answer_path.write_text(
        '{"id":"exec_simple_0","ground_truth":[{"calculate_area":{"base":[10],"height":[5]}}],'
        '"execution_result_type":["exact_match"]}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.bfcl_small.bfcl_small_source_root",
        lambda: source_root,
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("bfcl_exec_simple_ast", output_root, "test")

    assert paths == [output_root / "bfcl_exec_simple_ast" / "test.jsonl"]
    [row] = read_jsonl_items(paths[0])
    assert row["task_id"] == "exec_simple_0"
    assert row["metadata"]["category"] == "exec_simple"
    assert row["metadata"]["execution_result_type"] == ["exact_match"]


def test_prepare_dataset_materializes_bfcl_exec_spec(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "bfcl_data"
    question_root = source_root / "unused_datasets" / "question"
    possible_root = source_root / "unused_datasets" / "possible_answer"
    question_root.mkdir(parents=True)
    possible_root.mkdir(parents=True)
    question_path = question_root / "BFCL_v4_exec_simple.json"
    answer_path = possible_root / "BFCL_v4_exec_simple.json"
    question_path.write_text(
        '{"id":"exec_simple_0","question":[[{"role":"user","content":"Find a probability."}]],'
        '"function":[{"name":"calc_binomial_probability","description":"Calculate probability",'
        '"parameters":{"type":"dict","properties":{"n":{"type":"integer"},"k":{"type":"integer"},"p":{"type":"float"}},'
        '"required":["n","k","p"]}}]}\n',
        encoding="utf-8",
    )
    answer_path.write_text(
        '{"id":"exec_simple_0","ground_truth":["calc_binomial_probability(n=20, k=5, p=0.6)"],'
        '"execution_result_type":["exact_match"]}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.bfcl_small.bfcl_small_source_root",
        lambda: source_root,
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("bfcl_exec_simple", output_root, "test")

    assert paths == [output_root / "bfcl_exec_simple" / "test.jsonl"]
    [row] = read_jsonl_items(paths[0])
    assert row["task_id"] == "exec_simple_0"
    assert row["expected_executable_calls"] == ["calc_binomial_probability(n=20, k=5, p=0.6)"]
    assert row["execution_result_type"] == ["exact_match"]
    assert row["metadata"]["source_format"] == "official_bfcl_v4_exec"


def test_prepare_dataset_materializes_bfcl_exec_expression_arguments(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "bfcl_data"
    question_root = source_root / "unused_datasets" / "question"
    possible_root = source_root / "unused_datasets" / "possible_answer"
    question_root.mkdir(parents=True)
    possible_root.mkdir(parents=True)
    (question_root / "BFCL_v4_exec_parallel_multiple.json").write_text(
        '{"id":"exec_parallel_multiple_0","question":[[{"role":"user","content":"Convert money."}]],'
        '"function":[{"name":"convert_currency","description":"Convert currency",'
        '"parameters":{"type":"dict","properties":{"amount":{"type":"float"},'
        '"from_currency":{"type":"string"},"to_currency":{"type":"string"}},'
        '"required":["amount","from_currency","to_currency"]}}]}\n',
        encoding="utf-8",
    )
    (possible_root / "BFCL_v4_exec_parallel_multiple.json").write_text(
        '{"id":"exec_parallel_multiple_0",'
        '"ground_truth":["convert_currency(amount=500*500, from_currency=\'USD\', to_currency=\'EUR\')"],'
        '"execution_result_type":["real_time_match"]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.bfcl_small.bfcl_small_source_root",
        lambda: source_root,
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("bfcl_exec_parallel_multiple", output_root, "test")

    [row] = read_jsonl_items(paths[0])
    assert row["expected_executable_calls"] == [
        "convert_currency(amount=500*500, from_currency='USD', to_currency='EUR')"
    ]
    assert row["expected_tool_calls"][0]["arguments"]["amount"] == 250000


def test_bfcl_exec_evaluator_executes_decoded_and_reference_calls() -> None:
    record = BfclExecRecord(
        task_id="exec_simple_0",
        instruction="Find probability",
        tools=(),
        expected_executable_calls=("calc_binomial_probability(n=20, k=5, p=0.6)",),
        execution_result_type=("exact_match",),
        metadata={},
    )

    evaluation = evaluate_bfcl_exec_calls(
        record,
        [{"name": "calc_binomial_probability", "arguments": {"n": 20, "k": 5, "p": 0.6}}],
    )

    assert evaluation.is_passed is True
    assert evaluation.reward == 1.0


def test_bfcl_exec_evaluator_fails_wrong_executable_result() -> None:
    record = BfclExecRecord(
        task_id="exec_simple_0",
        instruction="Find probability",
        tools=(),
        expected_executable_calls=("calc_binomial_probability(n=20, k=5, p=0.6)",),
        execution_result_type=("exact_match",),
        metadata={},
    )

    evaluation = evaluate_bfcl_exec_calls(
        record,
        [{"name": "calc_binomial_probability", "arguments": {"n": 20, "k": 6, "p": 0.6}}],
    )

    assert evaluation.is_passed is False
    assert "exact_mismatch" in evaluation.fail_reason


def test_bfcl_exec_parallel_evaluator_matches_calls_without_order() -> None:
    record = BfclExecRecord(
        task_id="exec_parallel_multiple_4",
        instruction="Zip code and energy",
        tools=(),
        expected_executable_calls=(
            "get_zipcode_by_ip_address(ip_address='192.168.1.1')",
            "calculate_electrostatic_potential_energy(charge=5.0, voltage=10.0)",
        ),
        execution_result_type=("exact_match", "exact_match"),
        metadata={"category": "exec_parallel_multiple"},
    )

    evaluation = evaluate_bfcl_exec_calls(
        record,
        [
            {"name": "calculate_electrostatic_potential_energy", "arguments": {"charge": 5, "voltage": 10}},
            {"name": "get_zipcode_by_ip_address", "arguments": {"ip_address": "192.168.1.1"}},
        ],
    )

    assert evaluation.is_passed is True


def test_bfcl_exec_sandbox_accepts_official_aliases_and_expression_functions() -> None:
    sandbox = BfclExecSandbox()

    prime_factors = sandbox.execute("get_prime_factors(number=456)")
    derivative_with_lambda = sandbox.execute("estimate_derivative(function='lambda x: x**2', x=5)")
    derivative_expression = sandbox.execute("estimate_derivative(function='x**2', x=5)")
    weather = sandbox.execute("get_weather_data(coordinates={'latitude': 45.4215, 'longitude': -75.6972})")

    assert prime_factors.success is True
    assert prime_factors.result == [2, 2, 2, 3, 19]
    assert derivative_with_lambda.success is True
    assert derivative_expression.success is True
    assert abs(float(derivative_with_lambda.result) - float(derivative_expression.result)) < 1e-6
    assert weather.success is True
    assert isinstance(weather.result, dict)


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
                    "metadata": {"path": "/lookup", "method": "get", "api_name": "DemoAPI"},
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
                "execution_backend": "toolalpaca_simulator",
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
    assert "Output JSON schema:" in prompt
    assert '"oneOf": [' in prompt
    assert "return a JSON array containing every required call" in prompt
    assert "Do not copy tool schemas" in prompt
    assert "Available tools:" not in prompt
    assert '\n\nUser: Translate "Will it rain tomorrow?" into Japanese.\n\nAssistant: ```json\n{' in prompt
    assert "<think>" not in prompt


def test_simple_tool_call_decoder_accepts_arithmetic_literals() -> None:
    decoded = decode_simple_tool_call_response(
        """
{
  "name": "calc_binomial_probability",
  "arguments": {"n": 20, "k": 5, "p": 1 / 6}
}
"""
    )

    assert decoded == [
        {
            "name": "calc_binomial_probability",
            "arguments": {"n": 20, "k": 5, "p": 1 / 6},
        }
    ]


def test_toolalpaca_evaluator_reports_execution_details() -> None:
    record = SimpleToolCallRecord(
        task_id="toolalpaca_eval_simulated__demoapi_000",
        instruction="Look up alpha",
        tools=(),
        expected_tool_calls=(
            ToolCallExpectation(name="lookup", arguments={"query": "alpha"}, argument_options={"query": ("alpha",)}),
        ),
        metadata={},
    )

    evaluation = evaluate_toolalpaca_actions(
        record,
        [{"name": "lookup", "arguments": {"query": "alpha"}}],
        sandbox=ToolAlpacaSandbox(),
    )

    assert evaluation.is_passed is True
    assert evaluation.details["execution_mode"] == "local_toolalpaca_sandbox"
    assert evaluation.details["expected_tool_calls"] == [{"name": "lookup", "arguments": {"query": "alpha"}}]
    assert evaluation.details["decoded_tool_calls"] == [{"name": "lookup", "arguments": {"query": "alpha"}}]
    assert evaluation.details["decoded_execution_results"][0]["Action"] == "lookup"
    assert evaluation.details["decoded_execution_results"][0]["Action_Input"] == {"query": "alpha"}


def test_toolalpaca_source_loader_parses_official_placeholder_references(tmp_path: Path) -> None:
    source = tmp_path / "eval_simulated.json"
    source.write_text(
        """[
  {
    "Name": "DemoAPI",
    "Function_Projection": {
      "searchAnime": ["/anime/search", "get"],
      "getCastAndCrew": ["/anime/{animeId}/cast", "get"],
      "getProfile": ["/profile", "get"]
    },
    "Function_Description": {
      "searchAnime": "Search anime.\\nParameters: {\\"query\\": \\"Required. String.\\"}\\nOutput: object",
      "getCastAndCrew": "Get cast.\\nParameters: {\\"animeId\\": \\"Required. Integer.\\"}\\nOutput: object",
      "getProfile": "Get profile.\\nParameters: {}\\nOutput: object",
      "components": ""
    },
    "Instructions": ["Find cast", "Need details"],
    "Golden_Answers": [
      [
        {"Action": "searchAnime", "Action_Input": "{\\"query\\": \\"Attack on Titan\\"}"},
        {"Action": "getCastAndCrew", "Action_Input": "{\\"animeId\\": ${animeId from searchAnime}}"}
      ],
      [
        {"Action": "getDetails", "Action_Input": "{\\"Question\\": \\"Which profile?\\"}"},
        {"Action": "[Optional]getProfile", "Action_Input": "{}"}
      ]
    ]
  }
]""",
        encoding="utf-8",
    )
    records = [
        normalize_simple_tool_call_manifest_row(row, index=index, source_path=source)
        for index, row in enumerate(
            load_toolalpaca_rows_from_source(source, dataset_name="toolalpaca_eval_simulated")
        )
    ]

    assert records[0].expected_tool_calls[1].arguments == {
        "animeId": {_TOOLALPACA_REF_KEY: "animeId from searchAnime"}
    }
    assert records[1].tools[-1]["name"] == "getDetails"
    assert records[1].expected_tool_calls[1].name == "getProfile"
    assert records[1].expected_tool_calls[1].arguments[_TOOLALPACA_OPTIONAL_KEY] is True


def test_toolalpaca_real_loader_skips_auth_required_apis(tmp_path: Path) -> None:
    source = tmp_path / "eval_real.json"
    source.write_text(
        """[
  {
    "Name": "apilayer weatherstack",
    "Function_Projection": {"current": ["/current", "get"]},
    "Function_Description": {"current": "Weather.\\nParameters: {}\\nOutput: object"},
    "Instructions": ["weather"],
    "Golden_Answers": [[{"Action": "current", "Action_Input": "{}"}]]
  },
  {
    "Name": "Nager.Date",
    "Function_Projection": {"VersionGetVersion": ["/api/v3/Version", "get"]},
    "Function_Description": {"VersionGetVersion": "Version.\\nParameters: {}\\nOutput: object"},
    "Documentation": "{\\"openapi\\": \\"3.0.0\\", \\"servers\\": [{\\"url\\": \\"https://date.nager.at/\\"}], \\"paths\\": {\\"/api/v3/Version\\": {\\"get\\": {\\"responses\\": {\\"200\\": {\\"description\\": \\"ok\\"}}}}}}",
    "Instructions": ["version"],
    "Golden_Answers": [[{"Action": "VersionGetVersion", "Action_Input": "{}"}]]
  }
]""",
        encoding="utf-8",
    )

    rows = load_toolalpaca_rows_from_source(source, dataset_name="toolalpaca_eval_real")

    assert [row["task_id"] for row in rows] == ["toolalpaca_eval_real__nager_date_000"]
    assert rows[0]["metadata"]["execution_backend"] == "toolalpaca_real_http"
    assert rows[0]["metadata"]["api_server_url"] == "https://date.nager.at/"


def test_toolalpaca_sandbox_executes_openapi_style_requests() -> None:
    record = SimpleToolCallRecord(
        task_id="toolalpaca_eval_simulated__demoapi_000",
        instruction="Look up alpha",
        tools=(
            {
                "name": "lookup",
                "description": "Lookup a value",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "page": {"type": "integer"},
                    },
                    "required": ["query"],
                },
                "metadata": {"path": "/lookup/{query}", "method": "get"},
            },
        ),
        expected_tool_calls=(
            ToolCallExpectation(
                name="lookup",
                arguments={"query": "alpha", "page": 1},
                argument_options={},
            ),
        ),
        metadata={},
    )

    evaluation = evaluate_toolalpaca_actions(
        record,
        [{"name": "lookup", "arguments": {"query": "alpha", "page": "1", "ignored": "ok"}}],
        sandbox=ToolAlpacaSandbox(),
    )
    failed = evaluate_toolalpaca_actions(
        record,
        [{"name": "lookup", "arguments": {"query": "beta", "page": 1}}],
        sandbox=ToolAlpacaSandbox(),
    )

    assert evaluation.is_passed is True
    assert evaluation.details["decoded_execution_results"][0]["request"]["path"] == "/lookup/alpha"
    assert evaluation.details["decoded_execution_results"][0]["request"]["query"] == {"page": 1}
    assert failed.is_passed is False
    assert "request_mismatch" in failed.fail_reason


def test_toolalpaca_evaluator_calls_configured_official_simulator(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        text = '{"ok": true, "lookupId": 123}'
        headers = {"Content-Type": "application/json"}

        def json(self):
            return {"ok": True, "lookupId": 123}

    calls: list[dict[str, object]] = []

    def _request(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        return FakeResponse()

    monkeypatch.setenv("TOOLALPACA_SIMULATOR_URL", "http://127.0.0.1:5678")
    monkeypatch.setattr("src.eval.function_calling.toolalpaca.requests.request", _request)
    record = SimpleToolCallRecord(
        task_id="toolalpaca_eval_simulated__demoapi_000",
        instruction="Look up alpha",
        tools=(
            {
                "name": "lookup",
                "description": "Lookup a value",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                "metadata": {"path": "/lookup", "method": "get", "api_name": "DemoAPI"},
            },
        ),
        expected_tool_calls=(
            ToolCallExpectation(name="lookup", arguments={"query": "alpha"}, argument_options={}),
        ),
        metadata={"execution_backend": "toolalpaca_simulator", "api_name": "DemoAPI"},
    )

    evaluation = evaluate_toolalpaca_actions(
        record,
        [{"name": "lookup", "arguments": {"query": "alpha"}}],
    )

    assert evaluation.is_passed is True
    assert evaluation.details["execution_mode"] == "official_toolalpaca_simulator"
    assert [call["url"] for call in calls] == [
        "http://127.0.0.1:5678/DemoAPI/lookup",
        "http://127.0.0.1:5678/DemoAPI/lookup",
    ]
    assert calls[0]["params"] == {"query": "alpha"}


def test_toolalpaca_evaluator_resolves_multi_step_reference_placeholders() -> None:
    record = SimpleToolCallRecord(
        task_id="toolalpaca_eval_simulated__aniapi_000",
        instruction="Find the cast for Attack on Titan",
        tools=(
            {
                "name": "searchAnime",
                "description": "Search anime",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                "metadata": {"path": "/anime/search", "method": "get"},
            },
            {
                "name": "getCastAndCrew",
                "description": "Get cast",
                "parameters": {
                    "type": "object",
                    "properties": {"animeId": {"type": "integer"}},
                    "required": ["animeId"],
                },
                "metadata": {"path": "/anime/{animeId}/cast", "method": "get"},
            },
        ),
        expected_tool_calls=(
            ToolCallExpectation(name="searchAnime", arguments={"query": "Attack on Titan"}, argument_options={}),
            ToolCallExpectation(
                name="getCastAndCrew",
                arguments={"animeId": {_TOOLALPACA_REF_KEY: "animeId from searchAnime"}},
                argument_options={},
            ),
        ),
        metadata={},
    )

    evaluation = evaluate_toolalpaca_actions(
        record,
        [
            {"name": "searchAnime", "arguments": {"query": "Attack on Titan"}},
            {"name": "getCastAndCrew", "arguments": {"animeId": "${animeId from searchAnime}"}},
        ],
        sandbox=ToolAlpacaSandbox(),
    )

    assert evaluation.is_passed is True
    assert evaluation.details["decoded_execution_results"][1]["request"]["path"].startswith("/anime/")


def test_toolalpaca_evaluator_allows_skipping_optional_reference_actions() -> None:
    record = SimpleToolCallRecord(
        task_id="toolalpaca_eval_simulated__auth0_000",
        instruction="Get profile and optionally update it.",
        tools=(
            {
                "name": "getUserProfile",
                "description": "Get profile",
                "parameters": {
                    "type": "object",
                    "properties": {"userId": {"type": "string"}},
                    "required": ["userId"],
                },
                "metadata": {"path": "/users/{userId}", "method": "get"},
            },
            {
                "name": "updateUserProfile",
                "description": "Update profile",
                "parameters": {
                    "type": "object",
                    "properties": {"userId": {"type": "string"}},
                    "required": ["userId"],
                },
                "metadata": {"path": "/users/{userId}", "method": "patch"},
            },
        ),
        expected_tool_calls=(
            ToolCallExpectation(name="getUserProfile", arguments={"userId": "g-user123"}, argument_options={}),
            ToolCallExpectation(
                name="updateUserProfile",
                arguments={"userId": "g-user123", _TOOLALPACA_OPTIONAL_KEY: True},
                argument_options={},
            ),
        ),
        metadata={},
    )

    evaluation = evaluate_toolalpaca_actions(
        record,
        [{"name": "getUserProfile", "arguments": {"userId": "g-user123"}}],
        sandbox=ToolAlpacaSandbox(),
    )

    assert evaluation.is_passed is True
    assert evaluation.details["call_matches"][1]["reason"] == "optional_skipped"


def test_bfcl_exec_evaluator_matches_huge_integer_results_without_float_overflow() -> None:
    class HugeIntSandbox:
        def execute(self, call: str) -> BfclExecCallResult:
            return BfclExecCallResult(call=call, success=True, result=10**400)

    record = BfclExecRecord(
        task_id="huge-int",
        instruction="Return the huge integer.",
        tools=(),
        expected_executable_calls=("huge_integer()",),
        execution_result_type=("exact_match",),
        metadata={},
    )

    evaluation = evaluate_bfcl_exec_calls(
        record,
        [{"name": "huge_integer", "arguments": {}}],
        sandbox=HugeIntSandbox(),  # type: ignore[arg-type]
    )

    assert evaluation.is_passed is True


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
        "_run_bfcl_ast",
        lambda _args, _run, *, run_context=None: called.append(("function_bfcl_ast", _run.dataset_slug)) or 0,
    )

    rc = function_calling_runner.main(["--dataset", "bfcl_simple_python_test.jsonl", "--model-path", "model.pth"])

    assert rc == 0
    assert called == [("function_bfcl_ast", "bfcl_simple_python_test")]
