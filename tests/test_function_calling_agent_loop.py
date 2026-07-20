from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.eval.datasets.data_prepper.function_calling.agent_loop import normalize_agent_loop_row
from src.eval.long_doc_evidence import LongDocEvidenceConfig
from src.eval.tasks.function_calling.agent_loop import (
    AgentLoopRecord,
    _decode_agent_loop_calls,
    agent_loop_record_from_row,
    build_agent_loop_prompt,
    run_agent_loop_episode,
)
from src.eval.tasks.function_calling.agent_loop_executors import (
    AgentLoopStepOutcome,
    ExecutorSpec,
    ManifestReplayExecutor,
    ShellSandboxExecutor,
    shell_call_to_command,
    step_outcome_to_function_output,
)
from src.eval.tasks.function_calling.agent_loop_verifiers import (
    VerifierSpec,
    build_agent_loop_verifier,
    parse_rubric_judge_verdict,
    preflight_agent_loop_runtime,
    trace_to_mcp_atlas_transcript,
    widesearch_answer_to_official_row,
)
from src.eval.experiments.parallel_candidate_router.router import (
    ParallelCandidateRouterConfig,
    build_candidate_system_prompt,
    route_parallel_candidate_tool_call,
)

_SEARCH_TOOL = {
    "name": "search",
    "description": "Search the corpus",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
}


def _replay_record() -> AgentLoopRecord:
    return AgentLoopRecord(
        task_id="replay-1",
        instruction="Find who wrote The Left Hand of Darkness and answer.",
        tools=(_SEARCH_TOOL,),
        executor=ExecutorSpec(kind="manifest_replay"),
        verifier=VerifierSpec(kind="expected_tool_calls"),
        expected_tool_calls=(
            {
                "name": "final_answer",
                "arguments": {"answer": "Ursula K. Le Guin"},
                "argument_options": {"answer": ("Ursula K. Le Guin",)},
            },
        ),
        recorded_tool_outputs=(
            {"name": "search", "output": "The Left Hand of Darkness was written by Ursula K. Le Guin."},
        ),
        metadata={"source_benchmark": "widesearch"},
    )


class _FakeEngine:
    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.prompts: list[str] = []

    def generate(self, prompts, **kwargs):
        self.prompts.extend(str(prompt) for prompt in prompts)
        return [SimpleNamespace(text=self._texts.pop(0), finish_reason="stop") for _ in prompts]


def test_agent_loop_episode_replays_tools_and_passes_expected_final_answer() -> None:
    record = _replay_record()
    engine = _FakeEngine(
        [
            '{"name":"search","arguments":{"query":"The Left Hand of Darkness author"}}\n```',
            '{"name":"final_answer","arguments":{"answer":"Ursula K. Le Guin"}}\n```',
        ]
    )
    executor = ManifestReplayExecutor(recorded_tool_outputs=record.recorded_tool_outputs)
    executor.open()
    verifier = build_agent_loop_verifier("expected_tool_calls", SimpleNamespace())

    episode = run_agent_loop_episode(
        record=record,
        engine=engine,
        tool_sampling=object(),
        executor=executor,
        verifier=verifier,
        sample_index=0,
        repeat_index=0,
        pass_index=0,
        max_steps=5,
        max_tool_errors=2,
        history_max_chars=24000,
        max_output_chars=4000,
    )

    assert episode["termination_reason"] == "agent_stop"
    assert episode["final_answer"] == "Ursula K. Le Guin"
    assert episode["verdict"].is_passed is True
    assert episode["num_turns"] == 2

    first_prompt = engine.prompts[0]
    assert first_prompt.startswith("System: Tools:")
    assert "Return only a JSON function call." in first_prompt
    assert '"final_answer"' in first_prompt
    assert first_prompt.rstrip().endswith("Assistant: ```json")

    second_prompt = engine.prompts[1]
    assert "User: Function output:" in second_prompt
    assert "Ursula K. Le Guin" in second_prompt


def test_agent_loop_episode_counts_tool_errors_and_fails() -> None:
    record = _replay_record()
    engine = _FakeEngine(
        [
            '{"name":"search","arguments":{"query":"a"}}\n```',
            '{"name":"lookup_unknown","arguments":{}}\n```',
            '{"name":"lookup_unknown","arguments":{}}\n```',
        ]
    )
    executor = ManifestReplayExecutor(recorded_tool_outputs=record.recorded_tool_outputs)
    executor.open()
    verifier = build_agent_loop_verifier("expected_tool_calls", SimpleNamespace())

    episode = run_agent_loop_episode(
        record=record,
        engine=engine,
        tool_sampling=object(),
        executor=executor,
        verifier=verifier,
        sample_index=0,
        repeat_index=0,
        pass_index=0,
        max_steps=5,
        max_tool_errors=2,
        history_max_chars=24000,
        max_output_chars=4000,
    )

    assert episode["termination_reason"] == "too_many_errors"
    assert episode["verdict"].is_passed is False


def test_agent_loop_prompt_includes_agent_benchmark_policy_and_facts() -> None:
    record = AgentLoopRecord(
        task_id="demo-project",
        instruction="Create the project.",
        tools=(),
        executor=ExecutorSpec(kind="shell_sandbox", config={"backend": "subprocess"}),
        verifier=VerifierSpec(kind="nl2repo_official", config={"official_task_id": "demo-project"}),
        metadata={
            "source_benchmark": "nl2repo",
            "test_commands": ["pytest -q"],
            "test_files": ["tests/test_app.py"],
            "test_case_count": 3,
        },
    )

    prompt = build_agent_loop_prompt(
        record,
        tools=(),
        messages=({"role": "user", "content": record.instruction},),
        history_max_chars=24000,
    )

    assert "NL2Repo workflow" in prompt
    assert "Shell workflow" in prompt
    assert "Known benchmark facts:" in prompt
    assert "test_commands" in prompt
    assert "pytest -q" in prompt
    assert "verifier.official_task_id=demo-project" in prompt


def test_agent_loop_widesearch_prompt_warns_against_duplicate_table_rows() -> None:
    record = AgentLoopRecord(
        task_id="ws_en_001",
        instruction="Return one Markdown table.",
        tools=(),
        executor=ExecutorSpec(kind="web_search", config={}),
        verifier=VerifierSpec(kind="widesearch_official", config={}),
        metadata={"source_benchmark": "widesearch"},
    )

    prompt = build_agent_loop_prompt(
        record,
        tools=(),
        messages=({"role": "user", "content": record.instruction},),
        history_max_chars=24000,
    )

    assert "WideSearch workflow" in prompt
    assert "do not repeat rows" in prompt


def test_agent_loop_decoder_recovers_truncated_final_answer_string_arguments() -> None:
    calls = _decode_agent_loop_calls(
        '{"name": "final_answer", "arguments": "{\\"answer\\": \\"| Brand | Product |\\\\n| A | B"'
    )

    assert calls == [{"name": "final_answer", "arguments": {"answer": "| Brand | Product |\n| A | B"}}]


def test_agent_loop_decoder_does_not_recover_truncated_non_final_tool_call() -> None:
    with pytest.raises(ValueError, match="complete JSON"):
        _decode_agent_loop_calls('{"name": "web_search", "arguments": "{\\"query\\": \\"unfinished"')


def test_agent_loop_episode_can_use_parallel_candidate_router() -> None:
    record = _replay_record()

    class CandidateEngine:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.texts = [
                '{"name":"search","arguments":{"query":"The Left Hand of Darkness author"},"confidence":0.9,"evidence":"need author"}',
                '{"name":"search","arguments":{"query":"The Left Hand of Darkness author"},"confidence":0.95,"evidence":"best candidate"}',
                '{"name":"final_answer","arguments":{"answer":"Ursula K. Le Guin"},"confidence":0.9,"evidence":"tool output names author"}',
                '{"name":"final_answer","arguments":{"answer":"Ursula K. Le Guin"},"confidence":0.95,"evidence":"best candidate"}',
            ]

        def generate(self, prompts, **kwargs):  # noqa: ANN001
            self.calls.append({"prompts": list(prompts), **dict(kwargs)})
            return [
                SimpleNamespace(text=self.texts.pop(0), finish_reason="stop")
                for _prompt in prompts
            ]

    engine = CandidateEngine()
    executor = ManifestReplayExecutor(recorded_tool_outputs=record.recorded_tool_outputs)
    executor.open()
    verifier = build_agent_loop_verifier("expected_tool_calls", SimpleNamespace())

    episode = run_agent_loop_episode(
        record=record,
        engine=engine,
        tool_sampling=object(),
        executor=executor,
        verifier=verifier,
        sample_index=0,
        repeat_index=0,
        pass_index=0,
        max_steps=5,
        max_tool_errors=2,
        history_max_chars=24000,
        max_output_chars=4000,
        candidate_router_mode="parallel",
        candidate_router_config=ParallelCandidateRouterConfig(
            chunk_tools=8,
            batch_size=1,
            include_respond=False,
            ground_identifier_arguments=False,
        ),
    )

    assert episode["termination_reason"] == "agent_stop"
    assert episode["final_answer"] == "Ursula K. Le Guin"
    assert episode["verdict"].is_passed is True
    decision_steps = [item for item in episode["trace"] if item.get("kind") == "decision"]
    assert decision_steps
    assert decision_steps[0]["decision_io"] == "parallel_candidate"
    assert decision_steps[0]["candidate_router"]["mode"] == "parallel_candidate"
    assert any(item.get("kind") == "tool_call" and item.get("name") == "search" for item in episode["trace"])


def test_agent_loop_candidate_router_prompt_does_not_offer_missing_respond_tool() -> None:
    prompt = build_candidate_system_prompt(
        [_SEARCH_TOOL],
        domain_policy="Use final_answer only after the work is complete.",
        domain="nl2repo",
        facts_text=None,
        config=ParallelCandidateRouterConfig(include_respond=False),
    )

    assert "__no_candidate__" in prompt
    assert "Use respond only" not in prompt
    assert "choose respond" not in prompt


def test_parallel_candidate_router_allows_shard_abstention_without_respond() -> None:
    tools = (
        {
            "name": "final_answer",
            "description": "Submit the answer.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
        _SEARCH_TOOL,
    )

    class CandidateEngine:
        def __init__(self) -> None:
            self.texts = [
                '{"name":"__no_candidate__","arguments":{},"confidence":0.1,"evidence":"not complete"}',
                '{"name":"search","arguments":{"query":"repo tests"},"confidence":0.8,"evidence":"need evidence"}',
                '{"name":"search","arguments":{"query":"repo tests"},"confidence":0.9,"evidence":"best real tool"}',
            ]

        def generate(self, prompts, **kwargs):  # noqa: ANN001
            return [SimpleNamespace(text=self.texts.pop(0), finish_reason="stop") for _prompt in prompts]

    route = route_parallel_candidate_tool_call(
        tools=tools,
        messages=({"role": "user", "content": "Inspect the repo and run tests before answering."},),
        domain_policy="Use final_answer only after verification evidence exists.",
        domain="nl2repo",
        facts_text=None,
        engine=CandidateEngine(),
        sampling=object(),
        config=ParallelCandidateRouterConfig(
            chunk_tools=1,
            batch_size=2,
            include_respond=False,
            ground_identifier_arguments=False,
        ),
    )

    assert route.selected is not None
    assert route.selected.name == "search"
    assert [candidate.name for candidate in route.candidates] == ["search"]


def test_parallel_candidate_router_accepts_distinct_context_per_chunk() -> None:
    tools = (_SEARCH_TOOL, _SEARCH_TOOL)

    class CandidateEngine:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompts, **kwargs):  # noqa: ANN001
            self.calls += 1
            if self.calls == 1:
                assert "evidence-marker-a" in prompts[0]
                assert "evidence-marker-b" not in prompts[0]
                assert "evidence-marker-b" in prompts[1]
                assert "evidence-marker-a" not in prompts[1]
                return [
                    SimpleNamespace(
                        text='{"name":"search","arguments":{"query":"alpha"},"confidence":0.8,"evidence":"a"}',
                        finish_reason="stop",
                    ),
                    SimpleNamespace(
                        text='{"name":"search","arguments":{"query":"beta"},"confidence":0.7,"evidence":"b"}',
                        finish_reason="stop",
                    ),
                ]
            return [
                SimpleNamespace(
                    text='{"name":"search","arguments":{"query":"alpha"},"confidence":0.9,"evidence":"best"}',
                    finish_reason="stop",
                )
            ]

    route = route_parallel_candidate_tool_call(
        tools=tools,
        messages=({"role": "user", "content": "shared aggregate context"},),
        messages_by_chunk=(
            ({"role": "user", "content": "evidence-marker-a"},),
            ({"role": "user", "content": "evidence-marker-b"},),
        ),
        domain_policy="Search the relevant evidence shard.",
        domain="test",
        facts_text=None,
        engine=CandidateEngine(),
        sampling=object(),
        config=ParallelCandidateRouterConfig(
            chunk_tools=1,
            include_respond=False,
            ground_identifier_arguments=False,
        ),
    )

    assert route.selected is not None
    assert route.selected.arguments == {"query": "alpha"}


def test_agent_loop_auto_candidate_router_uses_long_metadata_context() -> None:
    record = AgentLoopRecord(
        task_id="metadata-context-1",
        instruction="Answer from the provided context.",
        tools=(_SEARCH_TOOL,),
        executor=ExecutorSpec(kind="manifest_replay"),
        verifier=VerifierSpec(kind="expected_tool_calls"),
        expected_tool_calls=(
            {
                "name": "final_answer",
                "arguments": {"answer": "Ursula K. Le Guin"},
                "argument_options": {"answer": ("Ursula K. Le Guin",)},
            },
        ),
        recorded_tool_outputs=(
            {"name": "search", "output": "The Left Hand of Darkness was written by Ursula K. Le Guin."},
        ),
        metadata={"source_benchmark": "widesearch", "context": "The archive notes. " * 500},
    )

    class CandidateEngine:
        def __init__(self) -> None:
            self.texts = [
                '{"name":"search","arguments":{"query":"The Left Hand of Darkness author"},"confidence":0.9,"evidence":"context asks for answer"}',
                '{"name":"search","arguments":{"query":"The Left Hand of Darkness author"},"confidence":0.95,"evidence":"best candidate"}',
                '{"name":"final_answer","arguments":{"answer":"Ursula K. Le Guin"},"confidence":0.9,"evidence":"tool output names author"}',
                '{"name":"final_answer","arguments":{"answer":"Ursula K. Le Guin"},"confidence":0.95,"evidence":"best candidate"}',
            ]

        def generate(self, prompts, **kwargs):  # noqa: ANN001
            return [SimpleNamespace(text=self.texts.pop(0), finish_reason="stop") for _prompt in prompts]

    executor = ManifestReplayExecutor(recorded_tool_outputs=record.recorded_tool_outputs)
    executor.open()
    verifier = build_agent_loop_verifier("expected_tool_calls", SimpleNamespace())

    episode = run_agent_loop_episode(
        record=record,
        engine=CandidateEngine(),
        tool_sampling=object(),
        executor=executor,
        verifier=verifier,
        sample_index=0,
        repeat_index=0,
        pass_index=0,
        max_steps=5,
        max_tool_errors=2,
        history_max_chars=24000,
        max_output_chars=4000,
        candidate_router_mode="auto",
        candidate_router_config=ParallelCandidateRouterConfig(
            chunk_tools=8,
            batch_size=1,
            context_chars=6000,
            include_respond=False,
            ground_identifier_arguments=False,
        ),
    )

    decision_steps = [item for item in episode["trace"] if item.get("kind") == "decision"]
    assert decision_steps[0]["decision_io"] == "parallel_candidate"


def test_agent_loop_auto_candidate_router_uses_raw_long_instruction_after_compaction() -> None:
    record = AgentLoopRecord(
        task_id="long-instruction-1",
        instruction="Answer from the long task notes.\n" + ("The archive notes. " * 500),
        tools=(_SEARCH_TOOL,),
        executor=ExecutorSpec(kind="manifest_replay"),
        verifier=VerifierSpec(kind="expected_tool_calls"),
        expected_tool_calls=(
            {
                "name": "final_answer",
                "arguments": {"answer": "Ursula K. Le Guin"},
                "argument_options": {"answer": ("Ursula K. Le Guin",)},
            },
        ),
        recorded_tool_outputs=(
            {"name": "search", "output": "The Left Hand of Darkness was written by Ursula K. Le Guin."},
        ),
        metadata={"source_benchmark": "widesearch"},
    )

    class CandidateEngine:
        def __init__(self) -> None:
            self.texts = [
                '{"name":"search","arguments":{"query":"The Left Hand of Darkness author"},"confidence":0.9,"evidence":"long task notes"}',
                '{"name":"search","arguments":{"query":"The Left Hand of Darkness author"},"confidence":0.95,"evidence":"best candidate"}',
                '{"name":"final_answer","arguments":{"answer":"Ursula K. Le Guin"},"confidence":0.9,"evidence":"tool output names author"}',
                '{"name":"final_answer","arguments":{"answer":"Ursula K. Le Guin"},"confidence":0.95,"evidence":"best candidate"}',
            ]

        def generate(self, prompts, **kwargs):  # noqa: ANN001
            return [SimpleNamespace(text=self.texts.pop(0), finish_reason="stop") for _prompt in prompts]

    executor = ManifestReplayExecutor(recorded_tool_outputs=record.recorded_tool_outputs)
    executor.open()
    verifier = build_agent_loop_verifier("expected_tool_calls", SimpleNamespace())

    episode = run_agent_loop_episode(
        record=record,
        engine=CandidateEngine(),
        tool_sampling=object(),
        executor=executor,
        verifier=verifier,
        sample_index=0,
        repeat_index=0,
        pass_index=0,
        max_steps=5,
        max_tool_errors=2,
        history_max_chars=24000,
        max_output_chars=4000,
        long_doc_config=LongDocEvidenceConfig(
            enabled=True,
            mode="lexical",
            max_chunk_chars=500,
            overlap_lines=0,
            min_long_text_chars=1000,
            max_evidence_chunks=2,
            max_evidence_chars=1200,
        ),
        candidate_router_mode="auto",
        candidate_router_config=ParallelCandidateRouterConfig(
            chunk_tools=8,
            batch_size=1,
            context_chars=6000,
            include_respond=False,
            ground_identifier_arguments=False,
        ),
    )

    decision_steps = [item for item in episode["trace"] if item.get("kind") == "decision"]
    assert decision_steps[0]["decision_io"] == "parallel_candidate"
    assert decision_steps[0]["long_doc"]["compacted_message_count"] == 1


def test_step_outcome_to_function_output_truncates() -> None:
    outcome = AgentLoopStepOutcome(ok=True, output="x" * 100, details={"exit_code": 0})
    payload = step_outcome_to_function_output(outcome, max_chars=10)
    assert payload["success"] is True
    assert len(payload["output"]) <= 30  # truncated with ellipsis marker
    assert payload["exit_code"] == 0


def test_shell_call_to_command_converts_calls() -> None:
    assert shell_call_to_command("bash", {"command": "ls -la"}) == "ls -la"
    assert shell_call_to_command("read_file", {"path": "a b.txt"}) == "cat -- 'a b.txt'"
    with pytest.raises(ValueError):
        shell_call_to_command("bash", {})


def test_shell_sandbox_subprocess_round_trip(tmp_path: Path) -> None:
    executor = ShellSandboxExecutor(backend="subprocess", workspace_root=str(tmp_path))
    tools = executor.open()
    assert {tool["name"] for tool in tools} == {"bash", "read_file", "write_file"}
    write = executor.execute("write_file", {"path": "notes/hello.txt", "content": "hi there"})
    assert write.ok is True
    read = executor.execute("read_file", {"path": "notes/hello.txt"})
    assert read.ok is True and "hi there" in str(read.output)
    ran = executor.execute("bash", {"command": "echo done"})
    assert ran.ok is True and "done" in str(ran.output)
    snapshot = executor.snapshot()
    assert snapshot["backend"] == "subprocess"
    executor.close()


def test_shell_sandbox_docker_compose_uses_terminal_bench_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.eval.tasks.function_calling import agent_loop_executors

    compose_file = tmp_path / "docker-compose.yaml"
    compose_file.write_text("services:\n  client:\n    image: ${T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME}\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_run(argv, **kwargs):  # noqa: ANN001
        calls.append({"argv": list(argv), "env": dict(kwargs.get("env") or {})})
        if list(argv)[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="missing")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setenv("RWKV_AGENT_LOOP_DOCKER_BUILD_RETRIES", "1")
    monkeypatch.setattr(agent_loop_executors.subprocess, "run", fake_run)

    executor = ShellSandboxExecutor(
        backend="docker",
        image="rwkv-terminal-bench:compose-demo",
        docker_compose_file=str(compose_file),
    )
    tools = executor.open()
    executor.close()

    assert {tool["name"] for tool in tools} == {"bash", "read_file", "write_file"}
    compose_calls = [call for call in calls if call["argv"][:2] == ["docker", "compose"]]  # type: ignore[index]
    assert any("build" in call["argv"] for call in compose_calls)
    assert any("up" in call["argv"] for call in compose_calls)
    build_env = compose_calls[0]["env"]
    assert build_env["T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME"] == "rwkv-terminal-bench:compose-demo"  # type: ignore[index]
    assert build_env["T_BENCH_TEST_DIR"] == "/tests"  # type: ignore[index]
    assert build_env["T_BENCH_CONTAINER_LOGS_PATH"] == "/logs"  # type: ignore[index]


def test_shell_sandbox_docker_compose_skips_build_for_cached_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.eval.tasks.function_calling import agent_loop_executors

    compose_file = tmp_path / "docker-compose.yaml"
    compose_file.write_text("services:\n  client:\n    image: ${T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME}\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):  # noqa: ANN001, ARG001
        calls.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(agent_loop_executors.subprocess, "run", fake_run)

    executor = ShellSandboxExecutor(
        backend="docker",
        image="rwkv-terminal-bench:cached-demo",
        docker_compose_file=str(compose_file),
    )
    executor.open()
    executor.close()

    compose_calls = [argv for argv in calls if argv[:2] == ["docker", "compose"]]
    assert not any("build" in argv for argv in compose_calls)
    assert any("up" in argv for argv in compose_calls)


def test_preflight_fails_for_unsupported_official_and_missing_assets(monkeypatch) -> None:
    monkeypatch.delenv("RWKV_TERMINAL_BENCH_ROOT", raising=False)
    records = [
        AgentLoopRecord(
            task_id="claw-1",
            instruction="Do the task.",
            tools=(),
            executor=ExecutorSpec(kind="manifest_replay"),
            verifier=VerifierSpec(kind="unsupported_official"),
            metadata={"source_benchmark": "claweval"},
        ),
        AgentLoopRecord(
            task_id="tb-1",
            instruction="Fix the bug.",
            tools=(),
            executor=ExecutorSpec(kind="shell_sandbox", config={"backend": "docker"}),
            verifier=VerifierSpec(kind="terminal_bench_official"),
            metadata={"source_benchmark": "terminal_bench_2_1"},
        ),
    ]
    with pytest.raises(ValueError) as excinfo:
        preflight_agent_loop_runtime(records, SimpleNamespace())
    message = str(excinfo.value)
    assert "claweval" in message
    assert "RWKV_TERMINAL_BENCH_ROOT" in message
    assert "docs/agent_loop.md" in message


def test_terminal_bench_verifier_uses_official_tests_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.eval.tasks.function_calling import agent_loop_verifiers

    root = tmp_path / "terminal-bench"
    task_dir = root / "original-tasks" / "fix-demo"
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True)
    (task_dir / "run-tests.sh").write_text("pytest $TEST_DIR/test_outputs.py\n", encoding="utf-8")
    (tests_dir / "test_outputs.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    monkeypatch.setenv("RWKV_TERMINAL_BENCH_ROOT", str(root))
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):  # noqa: ANN001
        del kwargs
        calls.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="passed", stderr="")

    monkeypatch.setattr(agent_loop_verifiers.subprocess, "run", fake_run)
    verifier = build_agent_loop_verifier("terminal_bench_official", SimpleNamespace())
    record = AgentLoopRecord(
        task_id="fix-demo",
        instruction="Fix it.",
        tools=(),
        executor=ExecutorSpec(kind="shell_sandbox", config={"backend": "docker"}),
        verifier=VerifierSpec(kind="terminal_bench_official", config={"official_task_id": "fix-demo"}),
        metadata={"source_benchmark": "terminal_bench_2_1"},
    )

    verdict = verifier.verify(record, final_answer="", trace=[], executor_snapshot={"container_id": "cid"})

    assert verdict.is_passed is True
    assert ["docker", "cp", str(task_dir / "run-tests.sh"), "cid:/tests/run-tests.sh"] in calls
    assert ["docker", "cp", f"{tests_dir}/.", "cid:/tests/"] in calls
    assert ["docker", "exec", "cid", "bash", "-lc", "bash /tests/run-tests.sh"] in calls


def test_widesearch_and_mcp_atlas_converters() -> None:
    record = _replay_record()
    row = widesearch_answer_to_official_row(record, "Ursula K. Le Guin")
    assert row == {"question_id": "replay-1", "response": "Ursula K. Le Guin", "trial": 0}

    transcript = trace_to_mcp_atlas_transcript(
        record,
        [
            {"kind": "tool_call", "name": "search", "arguments": {"query": "a"}, "output": "result text"},
            {"kind": "final_answer", "answer": "done"},
        ],
        "done",
    )
    assert transcript["task_id"] == "replay-1"
    assert transcript["final_response"] == "done"
    roles = [turn.get("role") for turn in transcript["conversation"]]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_widesearch_official_verifier_uses_official_response_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.eval.tasks.function_calling import agent_loop_verifiers

    official_root = tmp_path / "WideSearch"
    scripts = official_root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "run_infer_and_eval_batching.py").write_text("# official entrypoint\n", encoding="utf-8")
    monkeypatch.setenv("RWKV_WIDESEARCH_OFFICIAL_ROOT", str(official_root))
    monkeypatch.setenv("RWKV_WIDESEARCH_DATA_ROOT", str(tmp_path / "missing-data-root"))
    monkeypatch.delenv("RWKV_WIDESEARCH_EVAL_COMMAND", raising=False)

    record = AgentLoopRecord(
        task_id="ws_en_001",
        instruction="Find rows.",
        tools=(),
        executor=ExecutorSpec(kind="web_search"),
        verifier=VerifierSpec(kind="widesearch_official", config={"pass_threshold": 0.5}),
        metadata={"source_benchmark": "widesearch", "official_task_id": "ws_en_001"},
    )
    seen: dict[str, object] = {}

    def fake_run(command, **kwargs):  # noqa: ANN001
        env = dict(kwargs["env"])
        seen["command"] = command
        seen["env"] = env
        response_path = Path(env["WIDESEARCH_RESPONSE_ROOT"]) / "rwkv_eval_ws_en_001_0_response.json"
        row = json.loads(response_path.read_text(encoding="utf-8").splitlines()[0])
        assert row["instance_id"] == "ws_en_001"
        assert row["response"] == "| name |\n| A |"
        assert row["trial_idx"] == 0
        result_dir = Path(env["WIDESEARCH_RESULT_DIR"])
        result_dir.joinpath("rwkv_eval_ws_en_001_0_eval.json").write_text(
            json.dumps({"score": 0.75}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(agent_loop_verifiers.subprocess, "run", fake_run)
    verifier = build_agent_loop_verifier("widesearch_official", SimpleNamespace())
    assert verifier.preflight([record], SimpleNamespace()) == []

    verdict = verifier.verify(
        record,
        final_answer="| name |\n| A |",
        trace=[{"kind": "tool_call", "name": "web_search", "arguments": {"query": "a"}, "output": "result"}],
        executor_snapshot={},
    )

    assert verdict.is_passed is True
    assert verdict.reward == 0.75
    assert "--response_root" in str(seen["command"])
    assert seen["env"]["WIDESEARCH_INSTANCE_ID"] == "ws_en_001"  # type: ignore[index]
    assert str(official_root) in str(seen["env"]["PYTHONPATH"])  # type: ignore[index]


def test_parse_rubric_judge_verdict_accepts_json_and_fallback() -> None:
    assert parse_rubric_judge_verdict('{"passed": true, "reason": "ok"}') == (True, "ok")
    passed, _reason = parse_rubric_judge_verdict("nonsense text")
    assert passed is False


def test_prepper_normalization_applies_profiles_and_row_overrides() -> None:
    qa_row = {"id": "w-1", "question": "Q?", "answer": "A"}
    normalized = normalize_agent_loop_row(qa_row, dataset_name="widesearch", index=0, source_path="src.jsonl")
    assert normalized["executor"]["kind"] == "web_search"
    assert normalized["verifier"]["kind"] == "widesearch_official"
    assert normalized["metadata"]["source_format"] == "rwkvc_agent_loop"

    rubric_row = {"id": "e-1", "question": "Q?", "answer": "A", "rubrics": ["mentions X", "cites Y"]}
    normalized = normalize_agent_loop_row(rubric_row, dataset_name="e_bench", index=0, source_path="src.jsonl")
    assert normalized["verifier"]["kind"] == "llm_rubric_judge"
    assert normalized["verifier"]["config"]["rubrics"] == ["mentions X", "cites Y"]

    plain_row = {"id": "p-1", "question": "Q?", "answer": "A"}
    normalized = normalize_agent_loop_row(plain_row, dataset_name="prodbench", index=0, source_path="src.jsonl")
    assert normalized["verifier"]["kind"] == "expected_tool_calls"
    assert normalized["expected_tool_calls"][0]["name"] == "final_answer"

    euler_row = {"id": "eu-1", "question": "Project Euler #1?", "answer": "233168"}
    normalized = normalize_agent_loop_row(euler_row, dataset_name="hy_euler_pro", index=0, source_path="src.jsonl")
    assert normalized["executor"]["kind"] == "manifest_replay"
    assert normalized["verifier"]["kind"] == "expected_tool_calls"

    hle_row = {"id": "hle-1", "question": "Expert question?", "answer": "expert answer"}
    normalized = normalize_agent_loop_row(hle_row, dataset_name="hle_with_tools", index=0, source_path="src.jsonl")
    assert normalized["executor"]["kind"] == "web_search"
    assert normalized["verifier"]["kind"] == "llm_rubric_judge"
    assert normalized["verifier"]["config"]["reference_answer"] == "expert answer"

    override_row = {
        "task_id": "tb-1",
        "instruction": "Fix it.",
        "tools": [],
        "answer": "n/a",
        "executor": {"kind": "shell_sandbox", "config": {"backend": "docker", "image": "img"}},
        "verifier": {"kind": "terminal_bench_official", "config": {"official_task_id": "fix-bug"}},
    }
    normalized = normalize_agent_loop_row(override_row, dataset_name="terminal_bench_2_1", index=0, source_path="s")
    assert normalized["executor"]["config"]["image"] == "img"
    assert normalized["verifier"]["config"]["official_task_id"] == "fix-bug"

    record = agent_loop_record_from_row(json.loads(json.dumps(normalized)))
    assert record.executor.kind == "shell_sandbox"
    assert record.verifier.kind == "terminal_bench_official"


def test_agent_loop_prompt_uses_trained_multi_turn_format() -> None:
    record = _replay_record()
    prompt = build_agent_loop_prompt(
        record,
        (*record.tools, {"name": "final_answer", "description": "", "parameters": {"type": "object", "properties": {}}}),
        [
            {"role": "user", "content": record.instruction},
            {"role": "assistant", "content": '{"name":"search","arguments":{"query":"a"}}'},
            {"role": "user", "content": 'Function output:\n{"success":true,"output":"text"}'},
        ],
        history_max_chars=24000,
    )
    assert prompt.startswith("System: Tools:")
    assert "Return only a JSON function call." in prompt
    assert "\n\nUser: Function output:\n" in prompt
    assert 'Assistant: ```json\n{"name":"search"' in prompt
    assert prompt.rstrip().endswith("Assistant: ```json")


def test_repo_tests_official_verifier_runs_task_tests_in_workspace(tmp_path: Path) -> None:
    record = AgentLoopRecord(
        task_id="repo-1",
        instruction="Create hello.txt containing hello.",
        tools=(),
        executor=ExecutorSpec(kind="shell_sandbox", config={"backend": "subprocess"}),
        verifier=VerifierSpec(
            kind="repo_tests_official",
            config={"test_command": "grep -q hello hello.txt"},
        ),
        metadata={"source_benchmark": "nl2repo"},
    )
    executor = ShellSandboxExecutor(backend="subprocess", workspace_root=str(tmp_path))
    executor.open()
    executor.execute("write_file", {"path": "hello.txt", "content": "hello world"})
    verifier = build_agent_loop_verifier("repo_tests_official", SimpleNamespace())
    assert verifier.preflight([record], SimpleNamespace()) == []

    verdict = verifier.verify(record, final_answer="done", trace=[], executor_snapshot=executor.snapshot())
    assert verdict.is_passed is True

    failing = AgentLoopRecord(
        task_id="repo-2",
        instruction="x",
        tools=(),
        executor=record.executor,
        verifier=VerifierSpec(kind="repo_tests_official", config={"test_command": "grep -q missing hello.txt"}),
        metadata={},
    )
    verdict = verifier.verify(failing, final_answer="", trace=[], executor_snapshot=executor.snapshot())
    assert verdict.is_passed is False
    executor.close()


def test_repo_tests_official_preflight_requires_test_command() -> None:
    record = AgentLoopRecord(
        task_id="repo-3",
        instruction="x",
        tools=(),
        executor=ExecutorSpec(kind="shell_sandbox", config={"backend": "subprocess"}),
        verifier=VerifierSpec(kind="repo_tests_official"),
        metadata={},
    )
    verifier = build_agent_loop_verifier("repo_tests_official", SimpleNamespace())
    errors = verifier.preflight([record], SimpleNamespace())
    assert errors and "test_command" in errors[0]


def test_nl2repo_official_verifier_uses_official_post_processor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.eval.tasks.function_calling import agent_loop_verifiers

    official_root = tmp_path / "NL2RepoBench"
    project_dir = official_root / "test_files" / "demo_project"
    project_dir.mkdir(parents=True)
    (official_root / "openhands").mkdir()
    (official_root / "openhands" / "post_processor.py").write_text("# placeholder\n", encoding="utf-8")
    (project_dir / "start.md").write_text("Create demo_project.", encoding="utf-8")
    (project_dir / "test_commands.json").write_text("[]", encoding="utf-8")
    (project_dir / "test_files.json").write_text("[]", encoding="utf-8")
    monkeypatch.setenv("RWKV_NL2REPO_ROOT", str(official_root))
    monkeypatch.setattr(agent_loop_verifiers.shutil, "which", lambda name: "/usr/bin/docker" if name == "docker" else None)

    record = AgentLoopRecord(
        task_id="demo_project",
        instruction="Create demo_project.",
        tools=(),
        executor=ExecutorSpec(kind="shell_sandbox", config={"backend": "subprocess"}),
        verifier=VerifierSpec(kind="nl2repo_official", config={"official_task_id": "demo_project"}),
        metadata={"source_benchmark": "nl2repo"},
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def fake_run(argv, **kwargs):  # noqa: ANN001
        result_path = Path(kwargs["env"]["NL2REPO_RESULT_PATH"])
        result_path.write_text(
            json.dumps(
                {
                    "status": "success",
                    "pytest_results": {"passed": 3, "failed": 0, "errors": 0, "total": 3, "success_rate": 1.0},
                    "log_path": "/tmp/log.log",
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(agent_loop_verifiers.subprocess, "run", fake_run)
    verifier = build_agent_loop_verifier("nl2repo_official", SimpleNamespace())
    assert verifier.preflight([record], SimpleNamespace()) == []

    verdict = verifier.verify(record, final_answer="done", trace=[], executor_snapshot={"workspace": str(workspace)})
    assert verdict.is_passed is True
    assert verdict.reward == 1.0
    assert verdict.details["passed"] == 3


def test_web_search_executor_requires_env_configuration(monkeypatch) -> None:
    from src.eval.tasks.function_calling.agent_loop_executors import (
        WEB_SEARCH_API_KEY_ENV,
        WEB_SEARCH_API_URL_ENV,
        WebSearchExecutor,
    )

    monkeypatch.delenv(WEB_SEARCH_API_URL_ENV, raising=False)
    monkeypatch.delenv(WEB_SEARCH_API_KEY_ENV, raising=False)
    assert WebSearchExecutor.config_error() is not None

    record = AgentLoopRecord(
        task_id="web-1",
        instruction="Search it.",
        tools=(),
        executor=ExecutorSpec(kind="web_search"),
        verifier=VerifierSpec(kind="llm_rubric_judge"),
        metadata={},
    )
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")
    monkeypatch.setenv("JUDGE_API_KEY", "judge-key")
    with pytest.raises(ValueError) as excinfo:
        preflight_agent_loop_runtime([record], SimpleNamespace(judge_model=None, judge_api_key=None, judge_base_url=None))
    assert "RWKV_WEB_SEARCH_API_URL" in str(excinfo.value)

    monkeypatch.setenv(WEB_SEARCH_API_URL_ENV, "https://example.test/search")
    monkeypatch.setenv(WEB_SEARCH_API_KEY_ENV, "k")
    assert WebSearchExecutor.config_error() is None


def test_web_search_preflight_requires_reachable_endpoint(monkeypatch) -> None:
    from src.eval.tasks.function_calling import agent_loop_executors
    from src.eval.tasks.function_calling.agent_loop_executors import (
        WEB_SEARCH_API_KEY_ENV,
        WEB_SEARCH_API_URL_ENV,
    )

    record = AgentLoopRecord(
        task_id="web-1",
        instruction="Search it.",
        tools=(),
        executor=ExecutorSpec(kind="web_search"),
        verifier=VerifierSpec(kind="llm_rubric_judge"),
        metadata={},
    )

    def _raise(*_args: object, **_kwargs: object) -> object:
        raise OSError("connection refused")

    monkeypatch.setenv(WEB_SEARCH_API_URL_ENV, "http://127.0.0.1:18901/search")
    monkeypatch.setenv(WEB_SEARCH_API_KEY_ENV, "k")
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")
    monkeypatch.setenv("JUDGE_API_KEY", "judge-key")
    monkeypatch.setattr(agent_loop_executors.urllib.request, "urlopen", _raise)

    with pytest.raises(ValueError) as excinfo:
        preflight_agent_loop_runtime([record], SimpleNamespace(judge_model=None, judge_api_key=None, judge_base_url=None))

    assert "cannot reach RWKV_WEB_SEARCH_API_URL" in str(excinfo.value)


def test_web_search_preflight_accepts_proxy_health_endpoint(monkeypatch) -> None:
    from src.eval.tasks.function_calling import agent_loop_executors
    from src.eval.tasks.function_calling.agent_loop_executors import (
        WEB_SEARCH_API_KEY_ENV,
        WEB_SEARCH_API_URL_ENV,
    )

    record = AgentLoopRecord(
        task_id="web-1",
        instruction="Search it.",
        tools=(),
        executor=ExecutorSpec(kind="web_search"),
        verifier=VerifierSpec(kind="llm_rubric_judge"),
        metadata={},
    )

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int = -1) -> bytes:
            return b"{"

    def _urlopen(request: object, **_kwargs: object) -> _Response:
        assert request == "http://127.0.0.1:18901/health"
        return _Response()

    monkeypatch.setenv(WEB_SEARCH_API_URL_ENV, "http://127.0.0.1:18901/search")
    monkeypatch.setenv(WEB_SEARCH_API_KEY_ENV, "k")
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")
    monkeypatch.setenv("JUDGE_API_KEY", "judge-key")
    monkeypatch.setattr(agent_loop_executors.urllib.request, "urlopen", _urlopen)

    preflight_agent_loop_runtime([record], SimpleNamespace(judge_model=None, judge_api_key=None, judge_base_url=None))
