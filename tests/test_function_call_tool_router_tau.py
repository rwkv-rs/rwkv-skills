from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.eval.datasets.data_struct.function_call import FunctionCallTaskRecord
from src.eval.function_calling.agent.env import AgentObservation
from src.eval.function_calling.agent.pipeline import FunctionCallAgentPipeline
from src.eval.function_calling.long_context_router import (
    LongContextRoutingConfig,
    long_context_routing_config_from_benchmark_config,
    long_context_routing_config_to_payload,
)
from src.eval.function_calling.tool_router import (
    ToolRoutingConfig,
    tool_routing_config_from_benchmark_config,
    tool_routing_config_to_payload,
)


def _tool(name: str, description: str, *properties: str) -> dict[str, object]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {key: {"type": "string"} for key in properties},
            "required": list(properties[:1]),
        },
    }


def test_agent_pipeline_routes_tools_before_prompt_rendering() -> None:
    record = FunctionCallTaskRecord(
        task_id="route_1",
        instruction="Please book a flight from SFO to LAX.",
        tools=[
            _tool("lookup_weather", "Read city weather forecast", "city"),
            _tool("book_flight", "Book an airline ticket", "origin", "destination"),
            _tool("cancel_order", "Cancel a retail order", "order_id"),
        ],
        env={"type": "fake"},
    )
    pipeline = object.__new__(FunctionCallAgentPipeline)
    routes: list[dict[str, object]] = []

    prompt = pipeline._make_prompt(
        record,
        [],
        AgentObservation("Need a flight booking."),
        0,
        tool_routing_config=ToolRoutingConfig(mode="lexical", max_tools=1, trigger_tool_count=1, trigger_catalog_chars=1),
        tool_route_sink=routes,
    )

    assert '"name": "book_flight"' in prompt
    assert '"name": "lookup_weather"' not in prompt
    assert routes[0]["selected_names"] == ["book_flight"]


def test_complexfuncbench_router_keeps_final_answer_control_tool() -> None:
    record = FunctionCallTaskRecord(
        task_id="complex_1",
        instruction="Find a hotel in Paris.",
        tools=[
            _tool("Search_Hotels", "Search hotels with destination id.", "dest_id"),
            _tool("Search_Hotel_Destination", "Find hotel destination by city.", "query"),
            _tool("Get_Seat_Map", "Get aircraft seat map.", "flight_id"),
            _tool("final_answer", "Return the final response.", "answer"),
        ],
        env={"type": "complexfuncbench_official"},
    )
    pipeline = object.__new__(FunctionCallAgentPipeline)

    prompt = pipeline._make_prompt(
        record,
        [],
        AgentObservation("Need a hotel search."),
        0,
        tool_routing_config=ToolRoutingConfig(mode="lexical", max_tools=1, trigger_tool_count=1, trigger_catalog_chars=1),
    )

    assert '"name": "Search_Hotels"' in prompt or '"name": "Search_Hotel_Destination"' in prompt
    assert '"name": "final_answer"' in prompt
    assert '"name": "Get_Seat_Map"' not in prompt


def test_tool_router_config_can_be_read_from_benchmark_toml(monkeypatch, tmp_path: Path) -> None:
    import src.eval.benchmark_config as benchmark_config

    root = tmp_path / "configs"
    root.mkdir()
    (root / "tau2_bench_airline.toml").write_text(
        "\n".join(
            [
                "[default]",
                'tool_router_mode = "lexical"',
                "tool_router_max_tools = 5",
                "tool_router_trigger_tool_count = 8",
                "tool_router_trigger_catalog_chars = 1234",
                "tool_router_context_chars = 2048",
                "tool_router_description_chars = 120",
                'long_context_router_mode = "lexical"',
                "long_context_min_chars = 3000",
                "long_context_chunk_chars = 900",
                "long_context_overlap_lines = 2",
                "long_context_max_evidence_chunks = 3",
                "long_context_max_evidence_chars = 3000",
                "long_context_query_chars = 1000",
                "history_max_chars = 3456",
                "prompt_max_chars = 3072",
                "max_steps = 12",
                "max_tool_errors = 8",
                "decision_max_tokens = 384",
                "max_repeated_tool_calls = 2",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RWKV_BENCHMARK_CONFIG_ROOT", str(root))
    benchmark_config._CONFIG_CACHE.clear()

    config = benchmark_config.resolve_benchmark_model_config("tau2_bench_airline", "rwkv7-test", stage="tool")
    router_config = tool_routing_config_from_benchmark_config(config)

    assert router_config.mode == "lexical"
    assert router_config.max_tools == 5
    assert router_config.trigger_tool_count == 8
    assert router_config.trigger_catalog_chars == 1234
    assert router_config.context_chars == 2048
    assert router_config.description_chars == 120
    long_context_config = long_context_routing_config_from_benchmark_config(
        config,
        fallback_mode=router_config.mode,
    )
    assert long_context_config.mode == "lexical"
    assert long_context_config.min_chars == 3000
    assert long_context_config.chunk_chars == 900
    assert long_context_config.overlap_lines == 2
    assert long_context_config.max_evidence_chunks == 3
    assert long_context_config.max_evidence_chars == 3000
    assert long_context_config.query_chars == 1000
    assert config.history_max_chars == 3456
    assert config.prompt_max_chars == 3072
    assert config.max_steps == 12
    assert config.max_tool_errors == 8
    assert config.decision_max_tokens == 384
    assert config.max_repeated_tool_calls == 2


def test_long_doc_config_aliases_map_to_long_context_fields(monkeypatch, tmp_path: Path) -> None:
    import src.eval.benchmark_config as benchmark_config

    root = tmp_path / "configs"
    root.mkdir()
    (root / "tau2_bench_retail.toml").write_text(
        "\n".join(
            [
                "[default]",
                'long_doc_mode = "lexical"',
                "long_doc_min_chars = 6000",
                "long_doc_max_chars = 1200",
                "long_doc_overlap_lines = 4",
                "long_doc_max_evidence_chunks = 5",
                "long_doc_max_evidence_chars = 7000",
                "long_doc_query_chars = 1500",
                "prompt_max_chars = 3072",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RWKV_BENCHMARK_CONFIG_ROOT", str(root))
    benchmark_config._CONFIG_CACHE.clear()

    config = benchmark_config.resolve_benchmark_model_config("tau2_bench_retail", "rwkv7-test", stage="tool")
    long_context_config = long_context_routing_config_from_benchmark_config(config)

    assert long_context_config.mode == "lexical"
    assert long_context_config.min_chars == 6000
    assert long_context_config.chunk_chars == 1200
    assert long_context_config.overlap_lines == 4
    assert long_context_config.max_evidence_chunks == 5
    assert long_context_config.max_evidence_chars == 7000
    assert long_context_config.query_chars == 1500
    assert config.prompt_max_chars == 3072


def test_tool_router_config_payload_supports_slots_dataclass() -> None:
    config = ToolRoutingConfig(
        mode="lexical",
        max_tools=3,
        trigger_tool_count=4,
        trigger_catalog_chars=500,
        context_chars=1024,
        description_chars=80,
        fallback_to_all_on_empty=False,
    )

    assert not hasattr(config, "__dict__")
    assert tool_routing_config_to_payload(config) == {
        "mode": "lexical",
        "max_tools": 3,
        "trigger_tool_count": 4,
        "trigger_catalog_chars": 500,
        "context_chars": 1024,
        "description_chars": 80,
        "fallback_to_all_on_empty": False,
    }
    json.dumps(tool_routing_config_to_payload(config))


def test_long_context_router_payload_supports_slots_dataclass() -> None:
    config = LongContextRoutingConfig(
        mode="lexical",
        min_chars=3000,
        chunk_chars=900,
        overlap_lines=2,
        max_evidence_chunks=3,
        max_evidence_chars=3000,
        query_chars=1000,
        fallback_to_original_on_empty=False,
    )

    assert not hasattr(config, "__dict__")
    assert long_context_routing_config_to_payload(config) == {
        "mode": "lexical",
        "min_chars": 3000,
        "chunk_chars": 900,
        "overlap_lines": 2,
        "max_evidence_chunks": 3,
        "max_evidence_chars": 3000,
        "query_chars": 1000,
        "fallback_to_original_on_empty": False,
    }
    json.dumps(long_context_routing_config_to_payload(config))


def test_tau_prepper_materializes_base_split_without_mock_registration(monkeypatch, tmp_path: Path) -> None:
    from src.eval.agent_bench.tau_official import TauDomainInfo
    from src.eval.datasets.data_prepper.data_manager import available_function_call_datasets
    from src.eval.datasets.data_prepper.function_call import tau_bench

    monkeypatch.setattr(
        tau_bench,
        "load_tau_tasks",
        lambda *, task_set, domain, split, benchmark_version: [
            {
                "task_id": f"{domain}_1",
                "task_set": task_set,
                "task_split": split,
                "domain": domain,
                "index": 0,
                "instruction": "Book the flight.",
                "task": {"id": f"{domain}_1", "ticket": "Book the flight."},
                "benchmark_version": benchmark_version,
            }
        ],
    )
    monkeypatch.setattr(
        tau_bench,
        "tau_domain_info",
        lambda domain, **_kwargs: TauDomainInfo(
            policy=f"{domain} policy",
            tools=[_tool("book_reservation", "Book a reservation", "user_id")],
        ),
    )
    monkeypatch.setattr(tau_bench, "require_tau_v3_source", lambda _context: None)

    paths = tau_bench.prepare_tau2_bench_airline(tmp_path, "base")
    row = json.loads(paths[0].read_text(encoding="utf-8").strip())
    names = set(available_function_call_datasets())

    assert paths == [tmp_path / "tau2_bench_airline.jsonl"]
    assert row["env"]["type"] == "tau_official"
    assert row["env"]["domain"] == "airline"
    assert row["metadata"]["task_set"] == "airline"
    assert row["metadata"]["task_split"] == "base"
    assert row["metadata"]["env_kwargs"] == {}
    assert row["metadata"]["task"]["ticket"] == "Book the flight."
    assert row["tools"][0]["name"] == "book_reservation"
    assert "max_steps" not in row
    assert "tau3_bench_mock" not in names
    assert "tau3_bench_mock_long_context" not in names


def test_scheduler_exposes_tau2_tau3_jobs() -> None:
    from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path, split_benchmark_and_split
    from src.eval.scheduler.jobs import JOB_CATALOGUE, JOB_ORDER, detect_job_from_dataset

    assert split_benchmark_and_split("tau2_bench_airline") == ("tau2_bench_airline", "")
    assert infer_dataset_slug_from_path("data/tau2_bench_airline/base.jsonl") == "tau2_bench_airline"
    assert infer_dataset_slug_from_path("data/tau2_bench_airline.jsonl") == "tau2_bench_airline"
    assert JOB_CATALOGUE["function_tau2_bench"].domain == "function_call"
    assert JOB_CATALOGUE["function_tau3_bench"].domain == "function_call"
    assert "tau2_bench_airline" in JOB_CATALOGUE["function_tau2_bench"].dataset_slugs
    assert "tau3_bench_banking_knowledge" in JOB_CATALOGUE["function_tau3_bench"].dataset_slugs
    assert "function_tau2_bench" in JOB_ORDER
    assert "function_tau3_bench" in JOB_ORDER
    assert detect_job_from_dataset("tau2_bench_airline", is_cot=False) == "function_tau2_bench"
    assert detect_job_from_dataset("tau3_bench_banking_knowledge", is_cot=False) == "function_tau3_bench"


def test_eval_function_call_agent_routes_tau_to_official_pipeline(monkeypatch, tmp_path: Path) -> None:
    from src.bin import eval_function_call_agent
    from src.infer.sampling import SamplingConfig

    dataset = tmp_path / "tau2_bench_airline.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "task_id": "airline_1",
                "instruction": "Book a flight.",
                "env": {
                    "type": "tau_official",
                    "domain": "airline",
                    "benchmark_version": "tau_v2",
                    "policy": "airline policy",
                },
                "scorer": {"type": "tau_official"},
                "metadata": {
                    "task": {"id": "airline_1", "ticket": "Book a flight."},
                    "benchmark_version": "tau_v2",
                },
                "tools": [_tool("book_reservation", "Book a reservation", "user_id")],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class FakeService:
        def __init__(self) -> None:
            self.completions: list[dict[str, object]] = []

        def get_resume_context(self, **_kwargs):
            captured["force_new_task"] = _kwargs["force_new_task"]
            return SimpleNamespace(completed_keys=set())

        def create_task_from_context(self, **kwargs):
            captured["job_name"] = kwargs["job_name"]
            captured["sampling_config"] = kwargs["sampling_config"]
            return "123"

        def expected_completion_count(self, **_kwargs):
            return 1

        def insert_completion_payload(self, *, payload, task_id):
            assert task_id == "123"
            self.completions.append(payload)

        def list_completion_payloads(self, **_kwargs):
            return list(self.completions)

        def ingest_eval_payloads(self, **kwargs):
            captured["eval_payloads"] = kwargs["payloads"]

        def record_score_payload(self, **kwargs):
            captured["score_payload"] = kwargs["payload"]

        def update_task_status(self, **kwargs):
            captured["task_status"] = kwargs["status"]

        def update_task_session_status(self, **_kwargs):
            pass

        def count_completions(self, **_kwargs):
            return len(self.completions)

    class FakeWriter:
        def __init__(self, *, service, task_id, **_kwargs) -> None:
            self.service = service
            self.task_id = task_id

        def enqueue(self, payload) -> None:
            self.service.insert_completion_payload(payload=payload, task_id=self.task_id)

        def close(self) -> None:
            pass

    class GenericPipelineShouldNotRun:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("generic FunctionCallAgentPipeline should not be used for TAU")

    class FakeTauOfficialPipeline:
        def __init__(self, model_config) -> None:
            captured["model_path"] = model_config.weights_path

        def run(self, **kwargs):
            captured["pipeline"] = "tau_official"
            captured["dataset_name"] = kwargs["dataset_name"]
            captured["options"] = kwargs["options"]
            captured["tool_routing_mode"] = kwargs["tool_routing_config"].mode
            captured["long_context_mode"] = kwargs["long_context_routing_config"].mode
            payload = {
                "benchmark_name": "tau2_bench_airline",
                "dataset_split": "",
                "sample_index": 0,
                "repeat_index": 0,
                "sampling_config": {"stage1": {"max_generate_tokens": 128}},
                "prompt1": "prompt",
                "completion1": "{}",
                "stop_reason1": "done",
                "final_answer": "done",
                "events": [],
                "stats": {"steps": 1, "prompt_chars": 6, "completion_chars": 2},
                "function_call_subtype": "agent",
                "function_call_env_type": "tau_official",
                "function_call_scorer_type": "tau_official",
                "success": True,
                "official_score": 1.0,
                "agent_details": {"score": 1.0, "steps": 1, "finish_reason": "done"},
            }
            kwargs["on_record"](payload)
            return SimpleNamespace(dataset=kwargs["dataset_name"], sample_count=1, payloads=[payload])

    benchmark_config = SimpleNamespace(
        history_max_chars=2345,
        prompt_max_chars=3072,
        max_steps=12,
        max_tool_errors=8,
        decision_max_tokens=384,
        max_repeated_tool_calls=2,
        user_model="user-model",
        user_api_key="user-key",
        user_base_url="https://user.example/v1",
        judge_model="judge-model",
        judge_api_key="judge-key",
        judge_base_url="https://judge.example/v1",
        tool_router_mode="lexical",
        tool_router_max_tools=5,
        tool_router_trigger_tool_count=6,
        tool_router_trigger_catalog_chars=700,
        tool_router_context_chars=1024,
        tool_router_description_chars=80,
        long_context_router_mode=None,
        long_context_min_chars=3000,
        long_context_chunk_chars=900,
        long_context_overlap_lines=2,
        long_context_max_evidence_chunks=3,
        long_context_max_evidence_chars=3000,
        long_context_query_chars=1000,
    )

    fake_service = FakeService()
    monkeypatch.setattr(eval_function_call_agent, "init_orm", lambda _config: None)
    monkeypatch.setattr(eval_function_call_agent, "EvalDbService", lambda: fake_service)
    monkeypatch.setattr(eval_function_call_agent, "CompletionWriteWorker", FakeWriter)
    monkeypatch.setenv("RWKV_TAU_FORCE_NEW_TASK", "1")
    monkeypatch.setattr(eval_function_call_agent, "FunctionCallAgentPipeline", GenericPipelineShouldNotRun)
    monkeypatch.setattr(eval_function_call_agent, "TauOfficialAgentPipeline", FakeTauOfficialPipeline)
    monkeypatch.setattr(eval_function_call_agent, "resolve_or_prepare_dataset", lambda value, **_kwargs: Path(value))
    monkeypatch.setattr(eval_function_call_agent, "export_version_results", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        eval_function_call_agent,
        "resolve_benchmark_model_config",
        lambda *_args, **_kwargs: benchmark_config,
    )
    monkeypatch.setattr(
        eval_function_call_agent,
        "resolve_sampling_config",
        lambda *_args, **_kwargs: SamplingConfig(max_generate_tokens=512),
    )

    exit_code = eval_function_call_agent.main(
        [
            "--model-path",
            "/tmp/rwkv7-test.pth",
            "--dataset",
            str(dataset),
            "--device",
            "cpu",
            "--tool-router-mode",
            "lexical",
        ]
    )

    assert exit_code == 0
    assert captured["pipeline"] == "tau_official"
    assert captured["job_name"] == "function_tau2_bench"
    assert captured["dataset_name"] == "tau2_bench_airline"
    assert captured["tool_routing_mode"] == "lexical"
    assert captured["long_context_mode"] == "lexical"
    assert captured["force_new_task"] is True
    options = captured["options"]
    assert options.max_steps == 12
    assert options.prompt_max_chars == 3072
    assert options.decision_max_tokens == 384
    assert options.user_model == "user-model"
    assert captured["score_payload"]["metrics"]["avg@1"] == 1.0


def test_tau_official_runner_records_tool_routing_payload_for_slots_config(monkeypatch, tmp_path: Path) -> None:
    from src.eval.function_calling.agent import tau_official_runner
    from src.eval.function_calling.agent.tau_official_runner import (
        TauOfficialAgentPipeline,
        TauOfficialRunnerOptions,
    )
    from src.infer.sampling import SamplingConfig

    dataset = tmp_path / "tau2_bench_airline.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "task_id": "airline_1",
                "instruction": "Book a flight.",
                "env": {
                    "type": "tau_official",
                    "domain": "airline",
                    "benchmark_version": "tau_v2",
                },
                "scorer": {"type": "tau_official"},
                "metadata": {
                    "domain": "airline",
                    "task": {"id": "airline_1", "ticket": "Book a flight."},
                    "benchmark_version": "tau_v2",
                },
                "tools": [_tool("book_reservation", "Book a reservation", "user_id")],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class FakeTauOfficialRuntime:
        def __init__(self, *, domain: str) -> None:
            self.domain = domain

    def fake_run_one(self, **kwargs):
        captured["sampling_config"] = kwargs["sampling_config"]
        return {
            "benchmark_name": "tau2_bench_airline",
            "dataset_split": "",
            "sample_index": 0,
            "repeat_index": 0,
            "sampling_config": kwargs["sampling_config"],
            "prompt1": "prompt",
            "completion1": "{}",
            "stop_reason1": "done",
            "final_answer": "done",
            "events": [],
            "stats": {"steps": 1},
            "success": True,
            "official_score": 1.0,
            "agent_details": {},
        }

    monkeypatch.setattr(tau_official_runner, "TauOfficialRuntime", FakeTauOfficialRuntime)
    monkeypatch.setattr(TauOfficialAgentPipeline, "_run_one", fake_run_one)

    pipeline = object.__new__(TauOfficialAgentPipeline)
    result = pipeline.run(
        str(dataset),
        sampling=SamplingConfig(max_generate_tokens=512),
        options=TauOfficialRunnerOptions(user_model="user-model", user_api_key="user-key"),
        dataset_name="tau2_bench_airline",
        tool_routing_config=ToolRoutingConfig(
            mode="lexical",
            max_tools=2,
            trigger_tool_count=3,
            trigger_catalog_chars=400,
            context_chars=1024,
            description_chars=80,
        ),
    )

    assert result.sample_count == 1
    routing_payload = captured["sampling_config"]["tool_routing"]
    assert routing_payload == {
        "mode": "lexical",
        "max_tools": 2,
        "trigger_tool_count": 3,
        "trigger_catalog_chars": 400,
        "context_chars": 1024,
        "description_chars": 80,
        "fallback_to_all_on_empty": True,
    }
    json.dumps(routing_payload)
    long_context_payload = captured["sampling_config"]["long_context_routing"]
    assert long_context_payload["mode"] == "lexical"
    assert long_context_payload["min_chars"] == 6000
    assert long_context_payload["chunk_chars"] == 1000
    assert long_context_payload["max_evidence_chars"] == 6000
    json.dumps(long_context_payload)


def test_tau_official_runner_defaults_match_dedicated_tau_path() -> None:
    from src.eval.function_calling.agent.tau_official_runner import (
        TauOfficialRunnerOptions,
    )

    options = TauOfficialRunnerOptions.from_sources(SimpleNamespace(), None)

    assert options.max_steps == 200
    assert options.max_tool_errors == 10
    assert options.history_max_chars == 16000
    assert options.prompt_max_chars == 24576
    assert options.decision_max_tokens == 1024


def test_tau_official_decision_parser_accepts_common_tool_call_shapes() -> None:
    from src.eval.agent_bench.tau_official import parse_tau_agent_decision

    assert parse_tau_agent_decision(
        '{"tool_calls":[{"function":{"name":"assistant.lookup_order","arguments":"{\\"order_id\\":\\"#A12345678\\"}"}}]}'
    ) == ("lookup_order", {"order_id": "#A12345678"})
    assert parse_tau_agent_decision(
        '{"action":"lookup_order","action_input":{"order_id":"#A12345678"}}'
    ) == ("lookup_order", {"order_id": "#A12345678"})
    assert parse_tau_agent_decision(
        '{"name":"lookup_order","order_id":"#A12345678"}'
    ) == ("lookup_order", {"order_id": "#A12345678"})
    assert parse_tau_agent_decision(
        '{"name":"final_answer","arguments":{"answer":"Done ###STOP###"}}'
    ) == ("respond", {"answer": "Done ###STOP###"})
    assert parse_tau_agent_decision(
        '{"name":"assistant.lookup_order","arguments":{"order_id":"#A12345678"},"id":'
    ) == ("lookup_order", {"order_id": "#A12345678"})


def test_tau_official_agent_auto_compacts_long_context_with_lexical_router() -> None:
    from src.eval.function_calling.agent.tau_official_runner import RWKVTauOfficialAgent

    policy = "\n".join(
        [f"irrelevant policy row {index:03d}" for index in range(30)]
        + ["SPECIAL42 refund evidence: ask for the original payment method before exchange."]
        + [f"policy archive row {index:03d}" for index in range(30)]
    )
    long_tool_output = "\n".join(
        [f"irrelevant tool row {index:03d}" for index in range(30)]
        + ["SPECIAL42 refund evidence: order status is delivered and eligible."]
        + [f"tool archive row {index:03d}" for index in range(30)]
    )

    agent = object.__new__(RWKVTauOfficialAgent)
    agent._tools = [_tool("lookup_order", "Lookup order and refund status", "order_id")]
    agent._tool_routing_config = ToolRoutingConfig(
        mode="lexical",
        max_tools=1,
        trigger_tool_count=1,
        trigger_catalog_chars=1,
    )
    agent._domain_policy = policy
    agent._history_max_chars = 5000
    agent._prompt_max_chars = 6000
    agent._long_context_routing_config = LongContextRoutingConfig(
        mode="lexical",
        min_chars=200,
        chunk_chars=160,
        overlap_lines=1,
        max_evidence_chunks=1,
        max_evidence_chars=260,
        query_chars=500,
    )
    agent._turn_index = 0
    agent.tool_routes = []

    prompt = agent._build_prompt(
        [
            {"role": "user", "content": "Need SPECIAL42 refund evidence."},
            {"role": "assistant", "content": '{"name":"lookup_order","arguments":{"order_id":"SPECIAL42"}}'},
            {"role": "user", "content": long_tool_output},
        ]
    )

    assert "[Long document compacted:" in prompt
    assert "mode=lexical" in prompt
    assert "SPECIAL42 refund evidence: order status is delivered and eligible." in prompt
    assert "SPECIAL42 refund evidence: ask for the original payment method before exchange." in prompt
    assert "irrelevant tool row 000" not in prompt
    trace = agent.tool_routes[0]["long_context"]
    assert trace["mode"] == "lexical"
    assert trace["compacted_message_count"] == 1
    assert trace["policy_compacted"] is True
