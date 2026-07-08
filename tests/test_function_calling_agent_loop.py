from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.eval.datasets.data_prepper.function_calling.agent_loop import normalize_agent_loop_row
from src.eval.tasks.function_calling.agent_loop import (
    AgentLoopRecord,
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
