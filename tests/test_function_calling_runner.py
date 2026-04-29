from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.eval.evaluating import RunContext, RunMode
from src.eval.function_calling import (
    BfclTaskRecord,
    BfclToolExecutionResult,
    BfclTurn,
    build_bfcl_user_block,
    start_bfcl_runtime,
)
from src.eval.function_calling import bfcl_v3_runner
from src.eval.function_calling.bfcl_v3 import build_bfcl_system_prompt
from src.eval.function_calling import runner as function_calling_runner
from src.eval.function_calling import runner_common
from src.infer.constraints import LiteralChoiceConstraint


def test_function_calling_runner_parser_accepts_benchmark_kind() -> None:
    args = function_calling_runner.parse_args(
        [
            "--dataset",
            "browsecomp_test.jsonl",
            "--benchmark-kind",
            "mcp_bench",
            "--avg-k",
            "1",
            "--model-path",
            "model.pth",
        ]
    )
    assert args.dataset == "browsecomp_test.jsonl"
    assert args.benchmark_kind == "mcp_bench"
    assert args.avg_k == [1.0]


def test_function_calling_runner_resolves_explicit_avg_k_plan() -> None:
    plan = runner_common._resolve_function_calling_plan("bfcl_v3_test", 50, avg_ks=[1.0])

    assert plan.avg_k == 1.0
    assert plan.repeat_count == 1
    assert plan.sample_size == 50


def test_function_calling_runner_rejects_multiple_explicit_avg_k_values() -> None:
    try:
        runner_common._resolve_function_calling_plan("bfcl_v3_test", 50, avg_ks=[1.0, 2.0])
    except ValueError as exc:
        assert "exactly one avg_k override" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected avg_k validation failure")


def test_function_calling_runner_can_infer_benchmark_kind_from_dataset_slug() -> None:
    assert (
        function_calling_runner._infer_benchmark_kind("browsecomp_test.jsonl")
        is function_calling_runner.FunctionCallingBenchmarkKind.BROWSECOMP
    )
    assert (
        function_calling_runner._infer_benchmark_kind("bfcl_v3_test.jsonl")
        is function_calling_runner.FunctionCallingBenchmarkKind.BFCL_V3
    )
    assert (
        function_calling_runner._infer_benchmark_kind("tau2_bench_airline_base.jsonl")
        is function_calling_runner.FunctionCallingBenchmarkKind.TAU2_BENCH
    )


def test_function_calling_runner_main_dispatches_to_internal_implementation(monkeypatch) -> None:
    called: list[str] = []
    resolved = function_calling_runner.ResolvedFunctionCallingRun(
        benchmark_kind=function_calling_runner.FunctionCallingBenchmarkKind.BROWSECOMP,
        dataset_path=Path("/tmp/browsecomp_test.jsonl"),
        dataset_slug="browsecomp_test",
        benchmark_name="browsecomp",
        dataset_split="test",
        model_name="demo-model",
        engine=None,  # type: ignore[arg-type]
    )

    monkeypatch.setattr(function_calling_runner, "validate_inference_backend_args", lambda _args: None)
    monkeypatch.setattr(function_calling_runner, "_resolve_run", lambda _args: resolved)
    monkeypatch.setattr(
        function_calling_runner,
        "_run_browsecomp",
        lambda _args, _run, *, run_context=None: called.append("browsecomp") or 0,
    )

    rc = function_calling_runner.main(["--dataset", "browsecomp_test.jsonl", "--model-path", "model.pth"])

    assert rc == 0
    assert called == ["browsecomp"]


def test_function_calling_runner_main_forwards_explicit_run_context(monkeypatch) -> None:
    captured: dict[str, object] = {}
    resolved = function_calling_runner.ResolvedFunctionCallingRun(
        benchmark_kind=function_calling_runner.FunctionCallingBenchmarkKind.MCP_BENCH,
        dataset_path=Path("/tmp/mcp_bench_test.jsonl"),
        dataset_slug="mcp_bench_test",
        benchmark_name="mcp_bench",
        dataset_split="test",
        model_name="demo-model",
        engine=None,  # type: ignore[arg-type]
    )
    run_context = RunContext(job_name="function_mcp_bench", run_mode=RunMode.RESUME)

    monkeypatch.setattr(function_calling_runner, "validate_inference_backend_args", lambda _args: None)
    monkeypatch.setattr(function_calling_runner, "_resolve_run", lambda _args: resolved)

    def _fake_run(_args, _run, *, run_context=None):
        captured["run_context"] = run_context
        return 0

    monkeypatch.setattr(function_calling_runner, "_run_mcp_bench", _fake_run)

    rc = function_calling_runner.main(
        ["--dataset", "mcp_bench_test.jsonl", "--model-path", "model.pth"],
        run_context=run_context,
    )

    assert rc == 0
    assert captured["run_context"] is run_context


def test_function_calling_runner_main_dispatches_bfcl_v3(monkeypatch) -> None:
    called: list[str] = []
    resolved = function_calling_runner.ResolvedFunctionCallingRun(
        benchmark_kind=function_calling_runner.FunctionCallingBenchmarkKind.BFCL_V3,
        dataset_path=Path("/tmp/bfcl_v3_test.jsonl"),
        dataset_slug="bfcl_v3_test",
        benchmark_name="bfcl_v3",
        dataset_split="test",
        model_name="demo-model",
        engine=None,  # type: ignore[arg-type]
    )

    monkeypatch.setattr(function_calling_runner, "validate_inference_backend_args", lambda _args: None)
    monkeypatch.setattr(function_calling_runner, "_resolve_run", lambda _args: resolved)
    monkeypatch.setattr(
        function_calling_runner,
        "_run_bfcl_v3",
        lambda _args, _run, *, run_context=None: called.append("bfcl_v3") or 0,
    )

    rc = function_calling_runner.main(["--dataset", "bfcl_v3_test.jsonl", "--model-path", "model.pth"])

    assert rc == 0
    assert called == ["bfcl_v3"]


def test_function_calling_runner_detects_template_leak_markers() -> None:
    leaked = (
        "<system message>You are a helpful assistant.</system message>\n"
        "<system message>You are a helpful assistant.</system message>\n"
    )

    assert runner_common._looks_like_template_leak(leaked) is True
    assert runner_common._looks_like_template_leak("Booked flight F1 successfully.") is False


def test_run_bfcl_v3_official_episode_executes_per_turn(monkeypatch) -> None:
    outputs = iter(
        [
            SimpleNamespace(text="reason 1", finish_reason="stop"),
            SimpleNamespace(text="TOOL", finish_reason="stop"),
            SimpleNamespace(text='{"name":"lookup","arguments":{}}', finish_reason="stop"),
            SimpleNamespace(text="reason 2", finish_reason="stop"),
            SimpleNamespace(text="HANDOFF", finish_reason="stop"),
            SimpleNamespace(text='{"name":"final_answer","arguments":{"answer":"done with this turn"}}', finish_reason="stop"),
            SimpleNamespace(text="reason 3", finish_reason="stop"),
            SimpleNamespace(text="HANDOFF", finish_reason="stop"),
            SimpleNamespace(text='{"name":"final_answer","arguments":{"answer":"final answer"}}', finish_reason="stop"),
        ]
    )

    class _FakeEngine:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate(
            self,
            prompts,
            sampling,
            batch_size,
            progress_desc,
            prompt_seeds=None,
            prompt_stop_suffixes=None,
            constraints=None,
            constraint_mode="off",
        ):
            self.calls.append(
                {
                    "prompts": list(prompts),
                    "sampling": sampling,
                    "batch_size": batch_size,
                    "progress_desc": progress_desc,
                    "prompt_seeds": prompt_seeds,
                    "prompt_stop_suffixes": prompt_stop_suffixes,
                    "constraints": constraints,
                    "constraint_mode": constraint_mode,
                }
            )
            return [next(outputs)]

    record = BfclTaskRecord(
        task_id="multi_turn_base_0",
        instruction="Official task",
        tools=(
            {
                "name": "lookup",
                "description": "Lookup state",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        ),
        turns=(
            BfclTurn(messages=({"role": "user", "content": "first"},), ground_truth=("lookup()",)),
            BfclTurn(messages=({"role": "user", "content": "second"},), ground_truth=()),
        ),
        involved_classes=("VehicleControlAPI",),
        initial_state={"VehicleControlAPI": {"fuelLevel": 10}},
        metadata={"official_root": "/tmp/fake"},
    )
    state = bfcl_v3_runner._ActiveBfclEpisode(
        sample_index=0,
        repeat_index=0,
        pass_index=0,
        record=record,
        system_prompt=build_bfcl_system_prompt(record.tools),
        prompt_messages=[],
        active_tools=[dict(tool) for tool in record.tools],
        runtime_state=start_bfcl_runtime(record),
    )
    state.runtime_state.official_model_name = "demo"

    monkeypatch.setattr(
        bfcl_v3_runner,
        "execute_bfcl_official_tool_call",
        lambda *_args, **_kwargs: BfclToolExecutionResult(
            success=True,
            result={"fuelLevel": 12},
            state_snapshot={"VehicleControlAPI": {"fuelLevel": 12}},
            matched_expectation=True,
        ),
    )

    engine = _FakeEngine()
    trace = bfcl_v3_runner._run_bfcl_v3_official_episode(
        state=state,
        run=SimpleNamespace(engine=engine),
        cot_sampling=object(),
        router_sampling=object(),
        tool_sampling=object(),
        ask_sampling=object(),
        handoff_sampling=object(),
        max_steps=4,
        max_tool_errors=2,
        history_max_chars=4000,
    )

    assert state.termination_reason == "agent_stop"
    assert state.runtime_state.decoded_turn_outputs == [[["lookup()"]], []]
    assert state.turn_count == 2
    assert state.step_count == 3
    assert any(entry.get("tool_calls", [{}])[0].get("name") == "lookup" for entry in trace if entry.get("tool_calls"))
    assert any(call["constraint_mode"] == "strict" for call in engine.calls if "Router" in str(call["progress_desc"]))
    assert any(call["constraint_mode"] == "strict" for call in engine.calls if "Tool" in str(call["progress_desc"]))


def test_run_bfcl_generation_step_parses_json_tool_output_after_router() -> None:
    outputs = iter(
        [
            SimpleNamespace(text="<think>Need lookup.</think>", finish_reason="stop"),
            SimpleNamespace(text="TOOL", finish_reason="stop"),
            SimpleNamespace(text='{"name":"lookup","arguments":{"id":"A1"}}', finish_reason="stop"),
        ]
    )

    class _FakeEngine:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate(
            self,
            prompts,
            sampling,
            batch_size,
            progress_desc,
            prompt_seeds=None,
            prompt_stop_suffixes=None,
            constraints=None,
            constraint_mode="off",
        ):
            self.calls.append(
                {
                    "progress_desc": progress_desc,
                    "constraints": constraints,
                    "constraint_mode": constraint_mode,
                }
            )
            return [next(outputs)]

    record = BfclTaskRecord(
        task_id="demo-1",
        instruction="Find A1",
        tools=(
            {
                "name": "lookup",
                "description": "Lookup state",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                    "additionalProperties": False,
                },
            },
        ),
    )
    state = bfcl_v3_runner._start_bfcl_episode(
        sample_index=0,
        repeat_index=0,
        pass_index=0,
        record=record,
    )

    engine = _FakeEngine()
    outcome = bfcl_v3_runner._run_bfcl_generation_step(
        state=state,
        run=SimpleNamespace(engine=engine),
        user_request="Find A1",
        cot_sampling=object(),
        router_sampling=object(),
        tool_sampling=object(),
        ask_sampling=object(),
        handoff_sampling=object(),
        progress_suffix="sample 0 step 1",
        recent_tool_result=None,
        previous_state_snapshot=None,
    )

    assert outcome.ok is True
    assert outcome.action_type == "TOOL"
    assert outcome.tool_call is not None
    assert outcome.tool_call.name == "lookup"
    assert outcome.tool_call.arguments == {"id": "A1"}
    router_call = next(call for call in engine.calls if "Router" in str(call["progress_desc"]))
    tool_call = next(call for call in engine.calls if "Tool" in str(call["progress_desc"]))
    assert router_call["constraint_mode"] == "strict"
    assert tool_call["constraint_mode"] == "strict"
    router_constraint = router_call["constraints"][0]
    assert isinstance(router_constraint, LiteralChoiceConstraint)
    tool_constraint = tool_call["constraints"][0]
    assert tool_constraint.feed_text('{"name":"lookup","arguments":{"id":"A1"}}')
    assert tool_constraint.is_complete()


def test_run_bfcl_generation_step_returns_plain_ask_branch() -> None:
    outputs = iter(
        [
            SimpleNamespace(text="<think>Need a missing id.</think>", finish_reason="stop"),
            SimpleNamespace(text="ASK", finish_reason="stop"),
            SimpleNamespace(text='{"name":"ask_user","arguments":{"question":"Which id should I look up?"}}', finish_reason="stop"),
        ]
    )

    class _FakeEngine:
        def generate(
            self,
            prompts,
            sampling,
            batch_size,
            progress_desc,
            prompt_seeds=None,
            prompt_stop_suffixes=None,
            constraints=None,
            constraint_mode="off",
        ):
            _ = (
                prompts,
                sampling,
                batch_size,
                progress_desc,
                prompt_seeds,
                prompt_stop_suffixes,
                constraints,
                constraint_mode,
            )
            return [next(outputs)]

    record = BfclTaskRecord(task_id="demo-ask", instruction="Find a record")
    state = bfcl_v3_runner._start_bfcl_episode(
        sample_index=0,
        repeat_index=0,
        pass_index=0,
        record=record,
    )

    outcome = bfcl_v3_runner._run_bfcl_generation_step(
        state=state,
        run=SimpleNamespace(engine=_FakeEngine()),
        user_request="Find a record",
        cot_sampling=object(),
        router_sampling=object(),
        tool_sampling=object(),
        ask_sampling=object(),
        handoff_sampling=object(),
        progress_suffix="sample 0 step 1",
        recent_tool_result=None,
        previous_state_snapshot=None,
    )

    assert outcome.ok is True
    assert outcome.action_type == "ASK"
    assert outcome.tool_call is None
    assert outcome.final_answer == "Which id should I look up?"


def test_start_bfcl_episode_wraps_non_official_request_in_rwkv_user_block() -> None:
    record = BfclTaskRecord(
        task_id="demo-0",
        instruction="  Search for A1  ",
        initial_state={"selected": "A1"},
    )

    state = bfcl_v3_runner._start_bfcl_episode(
        sample_index=0,
        repeat_index=0,
        pass_index=0,
        record=record,
    )

    assert state.prompt_messages == [
        {
            "role": "user",
            "content": build_bfcl_user_block("Search for A1"),
        }
    ]
