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
from src.eval.function_calling.rwkv_prompt import RWKV_OFFICIAL_JSON_PROMPT_STYLE


def test_function_calling_runner_parser_accepts_benchmark_kind() -> None:
    args = function_calling_runner.parse_args(
        [
            "--dataset",
            "browsecomp_test.jsonl",
            "--benchmark-kind",
            "mcp_bench",
            "--avg-k",
            "1",
            "--prompt-style",
            "rwkv_official_json",
            "--prompt-max-chars",
            "8192",
            "--long-doc-mode",
            "off",
            "--long-doc-max-chars",
            "512",
            "--long-doc-overlap-lines",
            "2",
            "--long-doc-min-chars",
            "2048",
            "--long-doc-max-evidence-chunks",
            "3",
            "--long-doc-max-evidence-chars",
            "1536",
            "--tool-router-mode",
            "lexical",
            "--tool-router-max-tools",
            "8",
            "--tool-router-trigger-tool-count",
            "10",
            "--tool-router-trigger-catalog-chars",
            "2048",
            "--model-path",
            "model.pth",
        ]
    )
    assert args.dataset == "browsecomp_test.jsonl"
    assert args.benchmark_kind == "mcp_bench"
    assert args.avg_k == [1.0]
    assert args.prompt_style == RWKV_OFFICIAL_JSON_PROMPT_STYLE
    assert args.prompt_max_chars == 8192
    assert args.long_doc_mode == "off"
    assert args.long_doc_max_chars == 512
    assert args.long_doc_overlap_lines == 2
    assert args.long_doc_min_chars == 2048
    assert args.long_doc_max_evidence_chunks == 3
    assert args.long_doc_max_evidence_chars == 1536
    assert args.tool_router_mode == "lexical"
    assert args.tool_router_max_tools == 8
    assert args.tool_router_trigger_tool_count == 10
    assert args.tool_router_trigger_catalog_chars == 2048


def test_function_calling_runner_resolves_explicit_avg_k_plan() -> None:
    plan = runner_common._resolve_function_calling_plan("bfcl_v3_test", 50, avg_ks=[1.0])

    assert plan.avg_k == 1.0
    assert plan.repeat_count == 1
    assert plan.sample_size == 50


def test_function_calling_runner_resolves_configured_avg_k_plan(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "bfcl_simple_python.toml").write_text(
        "[default]\navg_k = [0.5]\nmax_samples = 12\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RWKV_BENCHMARK_CONFIG_ROOT", str(tmp_path))

    plan = runner_common._resolve_function_calling_plan(
        "bfcl_simple_python_test",
        20,
        avg_ks=None,
        model_name="demo-model",
        config_defaults=True,
    )
    sample_limit = runner_common._resolve_function_calling_sample_limit(
        "bfcl_simple_python_test",
        "demo-model",
        max_samples=None,
    )

    assert plan.avg_k == 0.5
    assert plan.repeat_count == 1
    assert plan.sample_size == 10
    assert sample_limit == 12


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
        function_calling_runner._infer_benchmark_kind("complexfuncbench_official_test.jsonl")
        is function_calling_runner.FunctionCallingBenchmarkKind.COMPLEXFUNCBENCH
    )
    assert (
        function_calling_runner._infer_benchmark_kind("longbench_qa_test.jsonl")
        is function_calling_runner.FunctionCallingBenchmarkKind.LONGBENCH
    )
    assert (
        function_calling_runner._infer_benchmark_kind("longbench_qa_balanced_test.jsonl")
        is function_calling_runner.FunctionCallingBenchmarkKind.LONGBENCH
    )
    assert (
        function_calling_runner._infer_benchmark_kind("longcodeqa_test.jsonl")
        is function_calling_runner.FunctionCallingBenchmarkKind.LONGCODEBENCH
    )
    assert (
        function_calling_runner._infer_benchmark_kind("bfcl_v3_test.jsonl")
        is function_calling_runner.FunctionCallingBenchmarkKind.BFCL_V3
    )
    assert (
        function_calling_runner._infer_benchmark_kind("apibank_level1_test.jsonl")
        is function_calling_runner.FunctionCallingBenchmarkKind.API_BANK
    )
    assert (
        function_calling_runner._infer_benchmark_kind("agentbench_db_test.jsonl")
        is function_calling_runner.FunctionCallingBenchmarkKind.AGENTBENCH
    )
    assert (
        function_calling_runner._infer_benchmark_kind("bfcl_simple_python_test.jsonl")
        is function_calling_runner.FunctionCallingBenchmarkKind.BFCL_AST
    )
    assert (
        function_calling_runner._infer_benchmark_kind("bfcl_exec_simple_ast_test.jsonl")
        is function_calling_runner.FunctionCallingBenchmarkKind.BFCL_AST
    )
    assert (
        function_calling_runner._infer_benchmark_kind("bfcl_exec_simple_test.jsonl")
        is function_calling_runner.FunctionCallingBenchmarkKind.BFCL_EXEC
    )
    assert (
        function_calling_runner._infer_benchmark_kind("toolalpaca_eval_simulated_test.jsonl")
        is function_calling_runner.FunctionCallingBenchmarkKind.TOOLALPACA
    )
    assert (
        function_calling_runner._infer_benchmark_kind("complexfuncbench_official_test.jsonl")
        is function_calling_runner.FunctionCallingBenchmarkKind.COMPLEXFUNCBENCH
    )
    assert (
        function_calling_runner._infer_benchmark_kind("tau2_bench_airline_base.jsonl")
        is function_calling_runner.FunctionCallingBenchmarkKind.TAU2_BENCH
    )
    assert (
        function_calling_runner._infer_benchmark_kind("tau3_bench_banking_knowledge_base.jsonl")
        is function_calling_runner.FunctionCallingBenchmarkKind.TAU3_BENCH
    )
    assert (
        function_calling_runner._infer_benchmark_kind("tau3_bench_mock_long_context_base.jsonl")
        is function_calling_runner.FunctionCallingBenchmarkKind.TAU3_BENCH
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


def test_function_calling_runner_main_dispatches_longbench(monkeypatch) -> None:
    called: list[str] = []
    resolved = function_calling_runner.ResolvedFunctionCallingRun(
        benchmark_kind=function_calling_runner.FunctionCallingBenchmarkKind.LONGBENCH,
        dataset_path=Path("/tmp/longbench_qa_test.jsonl"),
        dataset_slug="longbench_qa_test",
        benchmark_name="longbench_qa",
        dataset_split="test",
        model_name="demo-model",
        engine=None,  # type: ignore[arg-type]
    )

    monkeypatch.setattr(function_calling_runner, "validate_inference_backend_args", lambda _args: None)
    monkeypatch.setattr(function_calling_runner, "_resolve_run", lambda _args: resolved)
    monkeypatch.setattr(
        function_calling_runner,
        "_run_longbench",
        lambda _args, _run, *, run_context=None: called.append("longbench") or 0,
    )

    rc = function_calling_runner.main(["--dataset", "longbench_qa_test.jsonl", "--model-path", "model.pth"])

    assert rc == 0
    assert called == ["longbench"]


def test_function_calling_runner_main_dispatches_longcodebench(monkeypatch) -> None:
    called: list[str] = []
    resolved = function_calling_runner.ResolvedFunctionCallingRun(
        benchmark_kind=function_calling_runner.FunctionCallingBenchmarkKind.LONGCODEBENCH,
        dataset_path=Path("/tmp/longcodeqa_test.jsonl"),
        dataset_slug="longcodeqa_test",
        benchmark_name="longcodeqa",
        dataset_split="test",
        model_name="demo-model",
        engine=None,  # type: ignore[arg-type]
    )

    monkeypatch.setattr(function_calling_runner, "validate_inference_backend_args", lambda _args: None)
    monkeypatch.setattr(function_calling_runner, "_resolve_run", lambda _args: resolved)
    monkeypatch.setattr(
        function_calling_runner,
        "_run_longcodebench",
        lambda _args, _run, *, run_context=None: called.append("longcodebench") or 0,
    )

    rc = function_calling_runner.main(["--dataset", "longcodeqa_test.jsonl", "--model-path", "model.pth"])

    assert rc == 0
    assert called == ["longcodebench"]


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


def test_function_calling_runner_main_dispatches_complexfuncbench(monkeypatch) -> None:
    called: list[str] = []
    resolved = function_calling_runner.ResolvedFunctionCallingRun(
        benchmark_kind=function_calling_runner.FunctionCallingBenchmarkKind.COMPLEXFUNCBENCH,
        dataset_path=Path("/tmp/complexfuncbench_official_test.jsonl"),
        dataset_slug="complexfuncbench_official_test",
        benchmark_name="complexfuncbench_official",
        dataset_split="test",
        model_name="demo-model",
        engine=None,  # type: ignore[arg-type]
    )

    monkeypatch.setattr(function_calling_runner, "validate_inference_backend_args", lambda _args: None)
    monkeypatch.setattr(function_calling_runner, "_resolve_run", lambda _args: resolved)
    monkeypatch.setattr(
        function_calling_runner,
        "_run_complexfuncbench",
        lambda _args, _run, *, run_context=None: called.append("complexfuncbench") or 0,
    )

    rc = function_calling_runner.main(["--dataset", "complexfuncbench_official_test.jsonl", "--model-path", "model.pth"])

    assert rc == 0
    assert called == ["complexfuncbench"]


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


def test_function_calling_runner_main_dispatches_simple_tool_call_runner(monkeypatch) -> None:
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


def test_function_calling_runner_main_dispatches_bfcl_exec(monkeypatch) -> None:
    called: list[str] = []
    resolved = function_calling_runner.ResolvedFunctionCallingRun(
        benchmark_kind=function_calling_runner.FunctionCallingBenchmarkKind.BFCL_EXEC,
        dataset_path=Path("/tmp/bfcl_exec_simple_test.jsonl"),
        dataset_slug="bfcl_exec_simple_test",
        benchmark_name="bfcl_exec_simple",
        dataset_split="test",
        model_name="demo-model",
        engine=None,  # type: ignore[arg-type]
    )

    monkeypatch.setattr(function_calling_runner, "validate_inference_backend_args", lambda _args: None)
    monkeypatch.setattr(function_calling_runner, "_resolve_run", lambda _args: resolved)
    monkeypatch.setattr(
        function_calling_runner,
        "_run_bfcl_exec",
        lambda _args, _run, *, run_context=None: called.append(_run.dataset_slug) or 0,
    )

    rc = function_calling_runner.main(["--dataset", "bfcl_exec_simple_test.jsonl", "--model-path", "model.pth"])

    assert rc == 0
    assert called == ["bfcl_exec_simple_test"]


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
            SimpleNamespace(text='{"name":"lookup","arguments":{}}', finish_reason="stop"),
            SimpleNamespace(text='{"name":"final_answer","arguments":{"answer":"done with this turn"}}', finish_reason="stop"),
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
        tool_sampling=object(),
        max_steps=4,
        max_tool_errors=2,
        history_max_chars=4000,
    )

    assert state.termination_reason == "agent_stop"
    assert state.runtime_state.decoded_turn_outputs == [[["lookup()"]], []]
    assert state.turn_count == 2
    assert state.step_count == 3
    assert any(entry.get("tool_calls", [{}])[0].get("name") == "lookup" for entry in trace if entry.get("tool_calls"))
    assert all("Decision" in str(call["progress_desc"]) for call in engine.calls)
    assert all(call["constraint_mode"] == "strict" for call in engine.calls)


def test_run_bfcl_v3_official_episode_keeps_multi_call_json_output(monkeypatch) -> None:
    outputs = iter(
        [
            SimpleNamespace(
                text='[{"name":"lookup","arguments":{}},{"name":"lookup","arguments":{"id":"B2"}}]',
                finish_reason="stop",
            ),
            SimpleNamespace(
                text='{"name":"final_answer","arguments":{"answer":"done"}}',
                finish_reason="stop",
            ),
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
                    "progress_desc": progress_desc,
                    "constraints": constraints,
                    "constraint_mode": constraint_mode,
                }
            )
            return [next(outputs)]

    record = BfclTaskRecord(
        task_id="multi_turn_base_1",
        instruction="Official task",
        tools=(
            {
                "name": "lookup",
                "description": "Lookup state",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        ),
        turns=(BfclTurn(messages=({"role": "user", "content": "first"},), ground_truth=("lookup()", "lookup(id='B2')")),),
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

    monkeypatch.setattr(
        bfcl_v3_runner,
        "execute_bfcl_official_tool_call",
        lambda *_args, **_kwargs: BfclToolExecutionResult(
            success=True,
            result={"ok": True},
            state_snapshot={"ok": True},
            matched_expectation=True,
        ),
    )

    trace = bfcl_v3_runner._run_bfcl_v3_official_episode(
        state=state,
        run=SimpleNamespace(engine=_FakeEngine()),
        tool_sampling=object(),
        max_steps=4,
        max_tool_errors=2,
        history_max_chars=4000,
        prompt_style=RWKV_OFFICIAL_JSON_PROMPT_STYLE,
    )

    assert state.termination_reason == "agent_stop"
    assert state.runtime_state.decoded_turn_outputs == [[["lookup()", "lookup(id='B2')"]]]
    assert trace[0]["tool_calls"] == [
        {"name": "lookup", "arguments": {}},
        {"name": "lookup", "arguments": {"id": "B2"}},
    ]
    assert len(trace[0]["tool_results"]) == 2


def test_run_bfcl_generation_step_uses_official_json_prompt_style() -> None:
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
                    "progress_desc": progress_desc,
                    "prompt_stop_suffixes": prompt_stop_suffixes,
                    "constraints": constraints,
                    "constraint_mode": constraint_mode,
                }
            )
            return [SimpleNamespace(text='{"name":"lookup","arguments":{"id":"A1"}}', finish_reason="stop")]

    record = BfclTaskRecord(
        task_id="demo-official-json",
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
        tool_sampling=object(),
        progress_suffix="sample 0 step 1",
        prompt_style=RWKV_OFFICIAL_JSON_PROMPT_STYLE,
        history_max_chars=4000,
    )

    assert outcome.ok is True
    assert outcome.action_type == "TOOL"
    assert outcome.tool_call is not None
    assert outcome.tool_call.name == "lookup"
    assert len(engine.calls) == 1
    call = engine.calls[0]
    assert "Decision" in str(call["progress_desc"])
    assert call["constraint_mode"] == "strict"
    assert call["prompt_stop_suffixes"] == [list(bfcl_v3_runner.BFCL_DECISION_STOP_SUFFIXES)]
    prompt = str(call["prompts"][0])
    assert prompt.startswith("System: Tools:")
    assert "\n\nUser: Find A1\n\nAssistant: ```json\n{" in prompt
    assert "<think>" not in prompt


def test_run_bfcl_generation_step_returns_plain_ask_branch() -> None:
    outputs = iter(
        [
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
        tool_sampling=object(),
        progress_suffix="sample 0 step 1",
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
