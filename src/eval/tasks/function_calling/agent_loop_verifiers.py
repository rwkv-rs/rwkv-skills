from __future__ import annotations

"""Verifiers for the generic multi-turn agent-loop benchmark channel.

Sandbox benchmarks are graded by their OFFICIAL verifier (task test scripts,
official eval scripts, official claim-coverage judges) through explicit
subprocess seams. When the official assets, docker, or the judge endpoint are
missing, ``preflight_agent_loop_runtime`` raises before any generation — no
score is ever fabricated. See docs/agent_loop.md for per-benchmark setup.
"""

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from src.eval.env_config import resolve_judge_model_config

if TYPE_CHECKING:
    from src.eval.tasks.function_calling.agent_loop import AgentLoopRecord

_DOCS_HINT = "see docs/agent_loop.md for setup instructions"

TERMINAL_BENCH_ROOT_ENV = "RWKV_TERMINAL_BENCH_ROOT"
WIDESEARCH_ROOT_ENV = "RWKV_WIDESEARCH_OFFICIAL_ROOT"
WIDESEARCH_EVAL_COMMAND_ENV = "RWKV_WIDESEARCH_EVAL_COMMAND"
MCP_ATLAS_ROOT_ENV = "RWKV_MCP_ATLAS_ROOT"
TOOLATHLON_ROOT_ENV = "RWKV_TOOLATHLON_ROOT"

_UNSUPPORTED_OFFICIAL_HINTS: dict[str, str] = {
    "claweval": "clone https://github.com/claw-eval/claw-eval, provision its docker sandbox and grader endpoint",
    "wildclawbench": "clone https://github.com/InternLM/WildClawBench, provision its docker harness and JUDGE_MODEL",
    "skillsbench": "clone https://github.com/benchflow-ai/skillsbench and install the benchflow CLI",
    "apex_agents": "clone https://github.com/Mercor-Intelligence/archipelago and provision its harness",
    "deepswe": "clone https://github.com/datacurve-ai/deep-swe and provision its docker build/test harness",
    "nl2repo": "clone https://github.com/multimodal-art-projection/NL2RepoBench and provision its repo-test harness",
}


@dataclass(frozen=True, slots=True)
class VerifierSpec:
    kind: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentLoopVerdict:
    reward: float
    is_passed: bool
    fail_reason: str
    details: dict[str, Any] = field(default_factory=dict)


class AgentLoopVerifier(Protocol):
    def preflight(self, records: Sequence["AgentLoopRecord"], args: Any) -> list[str]: ...

    def verify(
        self,
        record: "AgentLoopRecord",
        *,
        final_answer: str,
        trace: Sequence[Mapping[str, Any]],
        executor_snapshot: Mapping[str, Any],
    ) -> AgentLoopVerdict: ...


def build_agent_loop_verifier(kind: str, args: Any) -> AgentLoopVerifier:
    if kind == "expected_tool_calls":
        return ExpectedToolCallsVerifier()
    if kind == "llm_rubric_judge":
        return LlmRubricJudgeVerifier(args)
    if kind == "terminal_bench_official":
        return TerminalBenchOfficialVerifier()
    if kind == "repo_tests_official":
        return RepoTestsOfficialVerifier()
    if kind == "widesearch_official":
        return WideSearchOfficialVerifier()
    if kind == "mcp_atlas_official":
        return McpAtlasOfficialVerifier()
    if kind == "toolathlon_official":
        return ToolathlonOfficialVerifier()
    if kind == "unsupported_official":
        return UnsupportedOfficialVerifier()
    raise ValueError(f"unknown agent-loop verifier kind: {kind!r}")


def preflight_agent_loop_runtime(records: Sequence["AgentLoopRecord"], args: Any) -> None:
    """Aggregate executor/verifier preflight failures and raise before generation."""

    errors: list[str] = []
    verifier_kinds = sorted({record.verifier.kind for record in records})
    for kind in verifier_kinds:
        try:
            verifier = build_agent_loop_verifier(kind, args)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        selected = [record for record in records if record.verifier.kind == kind]
        errors.extend(verifier.preflight(selected, args))

    executor_kinds = sorted({record.executor.kind for record in records})
    for kind in executor_kinds:
        if kind == "shell_sandbox":
            backends = {str(record.executor.config.get("backend") or "subprocess") for record in records if record.executor.kind == kind}
            if "docker" in backends and shutil.which("docker") is None:
                errors.append("shell_sandbox docker backend requires the docker CLI on PATH")
        elif kind == "mcp_worker":
            for record in records:
                if record.executor.kind != kind:
                    continue
                runtime_root = str(record.executor.config.get("runtime_root") or "")
                if not runtime_root:
                    errors.append(f"{record.task_id}: mcp_worker executor missing runtime_root; {_DOCS_HINT}")
                elif not (Path(runtime_root).expanduser() / ".venv" / "bin" / "python").is_file():
                    errors.append(f"mcp_worker runtime not provisioned under {runtime_root}; {_DOCS_HINT}")
                break
        elif kind == "web_search":
            from src.eval.tasks.function_calling.agent_loop_executors import WebSearchExecutor

            config_error = WebSearchExecutor.config_error()
            if config_error:
                errors.append(f"{config_error}; {_DOCS_HINT}")
        elif kind != "manifest_replay":
            errors.append(f"unknown agent-loop executor kind: {kind!r}")

    if errors:
        unique = list(dict.fromkeys(errors))
        rendered = "\n  - ".join(unique)
        raise ValueError(f"agent-loop runtime preflight failed:\n  - {rendered}")


class ExpectedToolCallsVerifier:
    """Grades the trace's tool calls + final answer against expected_tool_calls."""

    def preflight(self, records: Sequence["AgentLoopRecord"], args: Any) -> list[str]:
        errors = []
        for record in records:
            if not record.expected_tool_calls:
                errors.append(f"{record.task_id}: expected_tool_calls verifier requires expected_tool_calls rows")
        return errors

    def verify(
        self,
        record: "AgentLoopRecord",
        *,
        final_answer: str,
        trace: Sequence[Mapping[str, Any]],
        executor_snapshot: Mapping[str, Any],
    ) -> AgentLoopVerdict:
        from src.eval.tasks.function_calling.simple_tool_call import (
            SimpleToolCallRecord,
            _normalize_tool_expectation,
            evaluate_simple_tool_calls,
        )

        expectations = tuple(_normalize_tool_expectation(item) for item in record.expected_tool_calls)
        shim = SimpleToolCallRecord(
            task_id=record.task_id,
            instruction=record.instruction,
            tools=tuple(record.tools),
            expected_tool_calls=expectations,
            metadata=dict(record.metadata),
        )
        final_answer_only = all(item.name == "final_answer" for item in expectations)
        if final_answer_only:
            # QA-style rows grade only the submitted answer; intermediate tool
            # calls are the agent's own exploration.
            decoded = (
                [{"name": "final_answer", "arguments": {"answer": final_answer}}] if final_answer else []
            )
        else:
            decoded = [
                {"name": str(step.get("name") or ""), "arguments": dict(step.get("arguments") or {})}
                for step in trace
                if step.get("kind") == "tool_call"
            ]
            if final_answer:
                decoded.append({"name": "final_answer", "arguments": {"answer": final_answer}})
        evaluation = evaluate_simple_tool_calls(shim, decoded)
        return AgentLoopVerdict(
            reward=float(evaluation.reward),
            is_passed=bool(evaluation.is_passed),
            fail_reason=str(evaluation.fail_reason or ""),
            details=dict(evaluation.details),
        )


class LlmRubricJudgeVerifier:
    """LLM judge over the benchmark's official rubrics / expected answer.

    The judge endpoint comes from .env (JUDGE_MODEL / JUDGE_API_KEY /
    JUDGE_BASE_URL) or the --judge-* runner flags.
    """

    def __init__(self, args: Any) -> None:
        self._args = args
        self._config = None

    def _judge_config(self):
        if self._config is None:
            self._config = resolve_judge_model_config(
                model_name=getattr(self._args, "judge_model", None),
                api_key=getattr(self._args, "judge_api_key", None),
                base_url=getattr(self._args, "judge_base_url", None),
            )
        return self._config

    def preflight(self, records: Sequence["AgentLoopRecord"], args: Any) -> list[str]:
        try:
            config = self._judge_config()
        except ValueError as exc:
            return [str(exc)]
        if config is None:
            return [f"llm_rubric_judge verifier requires JUDGE_MODEL + JUDGE_API_KEY in .env; {_DOCS_HINT}"]
        return []

    def verify(
        self,
        record: "AgentLoopRecord",
        *,
        final_answer: str,
        trace: Sequence[Mapping[str, Any]],
        executor_snapshot: Mapping[str, Any],
    ) -> AgentLoopVerdict:
        config = self._judge_config()
        if config is None:
            raise ValueError("llm_rubric_judge verifier requires JUDGE_MODEL + JUDGE_API_KEY")
        rubrics = record.verifier.config.get("rubrics") or record.metadata.get("rubrics") or []
        reference = str(record.verifier.config.get("reference_answer") or record.metadata.get("reference_answer") or "")
        prompt = build_rubric_judge_prompt(
            instruction=record.instruction,
            final_answer=final_answer,
            rubrics=[str(item) for item in rubrics],
            reference_answer=reference,
        )
        verdict_text = _call_openai_judge(config, prompt)
        passed, reason = parse_rubric_judge_verdict(verdict_text)
        return AgentLoopVerdict(
            reward=1.0 if passed else 0.0,
            is_passed=passed,
            fail_reason="" if passed else (reason or "judge rejected the answer"),
            details={"judge_response": verdict_text},
        )


class TerminalBenchOfficialVerifier:
    """Runs the official Terminal-Bench task tests inside the episode container."""

    def preflight(self, records: Sequence["AgentLoopRecord"], args: Any) -> list[str]:
        errors: list[str] = []
        if shutil.which("docker") is None:
            errors.append(f"terminal_bench_official verifier requires the docker CLI; {_DOCS_HINT}")
        root = _official_root(TERMINAL_BENCH_ROOT_ENV)
        if root is None:
            errors.append(
                f"terminal_bench_official verifier requires {TERMINAL_BENCH_ROOT_ENV}=<terminal-bench checkout>; {_DOCS_HINT}"
            )
            return errors
        for record in records:
            task_dir = _terminal_bench_task_dir(root, record)
            if task_dir is None:
                errors.append(
                    f"{record.task_id}: official task dir not found under {root} "
                    f"(set metadata.official_task_id); {_DOCS_HINT}"
                )
                break
        return errors

    def verify(
        self,
        record: "AgentLoopRecord",
        *,
        final_answer: str,
        trace: Sequence[Mapping[str, Any]],
        executor_snapshot: Mapping[str, Any],
    ) -> AgentLoopVerdict:
        container_id = str(executor_snapshot.get("container_id") or "")
        if not container_id:
            raise ValueError("terminal_bench_official verifier requires a docker shell_sandbox executor")
        root = _official_root(TERMINAL_BENCH_ROOT_ENV)
        if root is None:
            raise ValueError(f"missing {TERMINAL_BENCH_ROOT_ENV}")
        task_dir = _terminal_bench_task_dir(root, record)
        if task_dir is None:
            raise ValueError(f"official task dir not found for {record.task_id}")
        test_command = _terminal_bench_test_command(task_dir)
        subprocess.run(
            ["docker", "cp", str(task_dir), f"{container_id}:/official-task"],
            check=True,
            capture_output=True,
            text=True,
        )
        proc = subprocess.run(
            ["docker", "exec", "-w", "/official-task", container_id, "bash", "-lc", test_command],
            capture_output=True,
            text=True,
            timeout=float(record.verifier.config.get("test_timeout_s") or 600.0),
        )
        passed = proc.returncode == 0
        return AgentLoopVerdict(
            reward=1.0 if passed else 0.0,
            is_passed=passed,
            fail_reason="" if passed else f"official tests failed (exit {proc.returncode})",
            details={
                "test_command": test_command,
                "stdout_tail": proc.stdout[-2000:],
                "stderr_tail": proc.stderr[-2000:],
            },
        )


class RepoTestsOfficialVerifier:
    """Runs the benchmark's own programmatic test command (DeepSWE/NL2Repo style).

    These benchmarks ship a per-task test command (e.g. pytest) as the official
    verifier; the command runs against the episode's final workspace/container
    state and the exit code decides pass/fail.
    """

    def preflight(self, records: Sequence["AgentLoopRecord"], args: Any) -> list[str]:
        errors: list[str] = []
        for record in records:
            if not _repo_test_command(record):
                errors.append(
                    f"{record.task_id}: repo_tests_official verifier requires verifier.config.test_command "
                    f"(or metadata.test_command) from the official task definition; {_DOCS_HINT}"
                )
                break
        return errors

    def verify(
        self,
        record: "AgentLoopRecord",
        *,
        final_answer: str,
        trace: Sequence[Mapping[str, Any]],
        executor_snapshot: Mapping[str, Any],
    ) -> AgentLoopVerdict:
        test_command = _repo_test_command(record)
        if not test_command:
            raise ValueError("repo_tests_official verifier requires a test_command")
        timeout_s = float(record.verifier.config.get("test_timeout_s") or 900.0)
        container_id = str(executor_snapshot.get("container_id") or "")
        workspace = str(executor_snapshot.get("workspace") or "")
        if container_id:
            argv = ["docker", "exec", container_id, "bash", "-lc", test_command]
            cwd = None
        elif workspace:
            argv = ["bash", "-lc", test_command]
            cwd = workspace
        else:
            raise ValueError("repo_tests_official verifier requires a shell_sandbox executor snapshot")
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout_s)
        passed = proc.returncode == 0
        return AgentLoopVerdict(
            reward=1.0 if passed else 0.0,
            is_passed=passed,
            fail_reason="" if passed else f"official task tests failed (exit {proc.returncode})",
            details={
                "test_command": test_command,
                "stdout_tail": proc.stdout[-2000:],
                "stderr_tail": proc.stderr[-2000:],
            },
        )


def _repo_test_command(record: "AgentLoopRecord") -> str:
    return str(record.verifier.config.get("test_command") or record.metadata.get("test_command") or "").strip()


class WideSearchOfficialVerifier:
    """Converts final answers into the official WideSearch format and runs the official eval stage."""

    def preflight(self, records: Sequence["AgentLoopRecord"], args: Any) -> list[str]:
        root = _official_root(WIDESEARCH_ROOT_ENV)
        if root is None:
            return [f"widesearch_official verifier requires {WIDESEARCH_ROOT_ENV}=<WideSearch checkout>; {_DOCS_HINT}"]
        if not os.environ.get(WIDESEARCH_EVAL_COMMAND_ENV) and not list(root.glob("scripts/run_infer_and_eval*.py")):
            return [
                f"widesearch_official verifier: official eval script not found under {root}/scripts "
                f"(or set {WIDESEARCH_EVAL_COMMAND_ENV}); {_DOCS_HINT}"
            ]
        return []

    def verify(
        self,
        record: "AgentLoopRecord",
        *,
        final_answer: str,
        trace: Sequence[Mapping[str, Any]],
        executor_snapshot: Mapping[str, Any],
    ) -> AgentLoopVerdict:
        root = _official_root(WIDESEARCH_ROOT_ENV)
        if root is None:
            raise ValueError(f"missing {WIDESEARCH_ROOT_ENV}")
        with tempfile.TemporaryDirectory(prefix="widesearch-eval-") as tmp:
            response_path = Path(tmp) / "responses.jsonl"
            result_dir = Path(tmp) / "results"
            result_dir.mkdir()
            response_path.write_text(
                json.dumps(widesearch_answer_to_official_row(record, final_answer), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            command = os.environ.get(WIDESEARCH_EVAL_COMMAND_ENV) or _default_widesearch_eval_command(root)
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(root),
                env={
                    **os.environ,
                    "WIDESEARCH_RESPONSE_PATH": str(response_path),
                    "WIDESEARCH_RESULT_DIR": str(result_dir),
                },
                capture_output=True,
                text=True,
                timeout=float(record.verifier.config.get("eval_timeout_s") or 1800.0),
            )
            if proc.returncode != 0:
                raise RuntimeError(f"official WideSearch eval failed (exit {proc.returncode}): {proc.stderr[-1000:]}")
            score = _first_score_from_result_dir(result_dir)
            if score is None:
                raise RuntimeError("official WideSearch eval produced no parsable score file")
            passed = score >= float(record.verifier.config.get("pass_threshold") or 1.0)
            return AgentLoopVerdict(
                reward=score,
                is_passed=passed,
                fail_reason="" if passed else f"official score {score:.3f} below threshold",
                details={"official_score": score},
            )


class McpAtlasOfficialVerifier:
    """Runs the official MCP-Atlas claim-coverage judge over the episode transcript."""

    def preflight(self, records: Sequence["AgentLoopRecord"], args: Any) -> list[str]:
        errors: list[str] = []
        root = _official_root(MCP_ATLAS_ROOT_ENV)
        if root is None:
            errors.append(f"mcp_atlas_official verifier requires {MCP_ATLAS_ROOT_ENV}=<mcp-atlas checkout>; {_DOCS_HINT}")
        elif not list(root.rglob("score_claims.py")):
            errors.append(f"mcp_atlas_official verifier: score_claims.py not found under {root}; {_DOCS_HINT}")
        try:
            judge = resolve_judge_model_config(
                model_name=getattr(args, "judge_model", None),
                api_key=getattr(args, "judge_api_key", None),
                base_url=getattr(args, "judge_base_url", None),
            )
        except ValueError as exc:
            errors.append(str(exc))
            judge = None
        if judge is None and not errors:
            errors.append(f"mcp_atlas_official verifier requires JUDGE_MODEL + JUDGE_API_KEY in .env; {_DOCS_HINT}")
        return errors

    def verify(
        self,
        record: "AgentLoopRecord",
        *,
        final_answer: str,
        trace: Sequence[Mapping[str, Any]],
        executor_snapshot: Mapping[str, Any],
    ) -> AgentLoopVerdict:
        root = _official_root(MCP_ATLAS_ROOT_ENV)
        if root is None:
            raise ValueError(f"missing {MCP_ATLAS_ROOT_ENV}")
        scripts = list(root.rglob("score_claims.py"))
        if not scripts:
            raise ValueError(f"score_claims.py not found under {root}")
        judge = resolve_judge_model_config()
        if judge is None:
            raise ValueError("mcp_atlas_official verifier requires JUDGE_MODEL + JUDGE_API_KEY")
        with tempfile.TemporaryDirectory(prefix="mcp-atlas-eval-") as tmp:
            transcript_path = Path(tmp) / "transcript.json"
            output_path = Path(tmp) / "scored.json"
            transcript_path.write_text(
                json.dumps(trace_to_mcp_atlas_transcript(record, trace, final_answer), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            proc = subprocess.run(
                ["python", str(scripts[0]), "--input", str(transcript_path), "--output", str(output_path)],
                cwd=str(scripts[0].parent),
                env={
                    **os.environ,
                    "EVAL_LLM_MODEL": judge.model_name,
                    "EVAL_LLM_API_KEY": judge.api_key,
                    "EVAL_LLM_BASE_URL": judge.base_url or "",
                },
                capture_output=True,
                text=True,
                timeout=float(record.verifier.config.get("eval_timeout_s") or 900.0),
            )
            if proc.returncode != 0:
                raise RuntimeError(f"official MCP-Atlas judge failed (exit {proc.returncode}): {proc.stderr[-1000:]}")
            coverage = _mcp_atlas_coverage(output_path)
            if coverage is None:
                raise RuntimeError("official MCP-Atlas judge produced no parsable coverage")
            threshold = float(record.verifier.config.get("pass_threshold") or 0.5)
            passed = coverage >= threshold
            return AgentLoopVerdict(
                reward=coverage,
                is_passed=passed,
                fail_reason="" if passed else f"claim coverage {coverage:.3f} below {threshold}",
                details={"claim_coverage": coverage},
            )


class ToolathlonOfficialVerifier:
    """Seam for the official Toolathlon per-task evaluators (container-based)."""

    def preflight(self, records: Sequence["AgentLoopRecord"], args: Any) -> list[str]:
        errors: list[str] = []
        root = _official_root(TOOLATHLON_ROOT_ENV)
        if root is None:
            errors.append(f"toolathlon_official verifier requires {TOOLATHLON_ROOT_ENV}=<Toolathlon checkout>; {_DOCS_HINT}")
        if shutil.which("docker") is None and shutil.which("podman") is None:
            errors.append(f"toolathlon_official verifier requires docker or podman; {_DOCS_HINT}")
        return errors

    def verify(
        self,
        record: "AgentLoopRecord",
        *,
        final_answer: str,
        trace: Sequence[Mapping[str, Any]],
        executor_snapshot: Mapping[str, Any],
    ) -> AgentLoopVerdict:
        root = _official_root(TOOLATHLON_ROOT_ENV)
        if root is None:
            raise ValueError(f"missing {TOOLATHLON_ROOT_ENV}")
        evaluator = record.verifier.config.get("evaluator_command")
        if not evaluator:
            raise ValueError(
                "toolathlon_official verifier requires verifier.config.evaluator_command "
                f"(the official per-task evaluator invocation); {_DOCS_HINT}"
            )
        proc = subprocess.run(
            str(evaluator),
            shell=True,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=float(record.verifier.config.get("eval_timeout_s") or 1800.0),
        )
        passed = proc.returncode == 0
        return AgentLoopVerdict(
            reward=1.0 if passed else 0.0,
            is_passed=passed,
            fail_reason="" if passed else f"official evaluator failed (exit {proc.returncode})",
            details={"stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]},
        )


class UnsupportedOfficialVerifier:
    """Hard preflight failure for benchmarks whose official harness is not wired yet."""

    def preflight(self, records: Sequence["AgentLoopRecord"], args: Any) -> list[str]:
        errors: list[str] = []
        for record in records:
            benchmark = str(record.metadata.get("source_benchmark") or record.task_id)
            hint = _UNSUPPORTED_OFFICIAL_HINTS.get(benchmark, "provision the official harness")
            errors.append(
                f"{benchmark}: official verifier integration is not wired yet — {hint}; "
                f"scores are never stubbed ({_DOCS_HINT})"
            )
            break
        return errors

    def verify(self, record: "AgentLoopRecord", **_kwargs: Any) -> AgentLoopVerdict:
        raise ValueError("unsupported_official verifier cannot grade; preflight should have failed the run")


# --- converters (model output / trace -> official verifier input) ---


def widesearch_answer_to_official_row(record: "AgentLoopRecord", final_answer: str) -> dict[str, Any]:
    return {
        "question_id": str(record.metadata.get("official_task_id") or record.task_id),
        "response": final_answer,
        "trial": int(record.metadata.get("trial") or 0),
    }


def trace_to_mcp_atlas_transcript(
    record: "AgentLoopRecord",
    trace: Sequence[Mapping[str, Any]],
    final_answer: str,
) -> dict[str, Any]:
    conversation: list[dict[str, Any]] = [{"role": "user", "content": record.instruction}]
    for step in trace:
        if step.get("kind") == "tool_call":
            conversation.append(
                {
                    "role": "assistant",
                    "tool_calls": [{"name": step.get("name"), "arguments": dict(step.get("arguments") or {})}],
                }
            )
            conversation.append({"role": "tool", "content": step.get("output")})
    conversation.append({"role": "assistant", "content": final_answer})
    return {
        "task_id": str(record.metadata.get("official_task_id") or record.task_id),
        "conversation": conversation,
        "final_response": final_answer,
        "gtfa_claims": record.verifier.config.get("gtfa_claims") or record.metadata.get("gtfa_claims") or [],
    }


def build_rubric_judge_prompt(
    *,
    instruction: str,
    final_answer: str,
    rubrics: Sequence[str],
    reference_answer: str,
) -> str:
    parts = [
        "You are a strict benchmark grader. Judge whether the answer satisfies the task.",
        f"<TASK>\n{instruction}\n</TASK>",
    ]
    if reference_answer:
        parts.append(f"<REFERENCE_ANSWER>\n{reference_answer}\n</REFERENCE_ANSWER>")
    if rubrics:
        rendered = "\n".join(f"- {item}" for item in rubrics)
        parts.append(f"<RUBRICS>\nThe answer passes only if EVERY rubric is satisfied:\n{rendered}\n</RUBRICS>")
    parts.append(f"<ANSWER>\n{final_answer}\n</ANSWER>")
    parts.append('Reply with exactly one JSON object: {"passed": true|false, "reason": "..."}')
    return "\n\n".join(parts)


def parse_rubric_judge_verdict(text: str) -> tuple[bool, str]:
    raw = str(text or "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(raw[start : end + 1])
            return bool(payload.get("passed", False)), str(payload.get("reason") or "")
        except json.JSONDecodeError:
            pass
    lowered = raw.lower()
    return lowered.startswith("true") or '"passed": true' in lowered, raw[:200]


def _call_openai_judge(config: Any, prompt: str) -> str:
    from openai import OpenAI  # pyright: ignore[reportMissingImports]

    client = OpenAI(api_key=config.api_key, base_url=config.base_url or None)
    response = client.chat.completions.create(
        model=config.model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return str(response.choices[0].message.content or "")


def _official_root(env_name: str) -> Path | None:
    raw = os.environ.get(env_name)
    if not raw:
        return None
    root = Path(raw).expanduser()
    return root if root.is_dir() else None


def _terminal_bench_task_dir(root: Path, record: "AgentLoopRecord") -> Path | None:
    task_id = str(record.metadata.get("official_task_id") or record.task_id)
    candidates = (root / "tasks" / task_id, root / task_id)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _terminal_bench_test_command(task_dir: Path) -> str:
    for name in ("run-tests.sh", "run_tests.sh"):
        if (task_dir / name).is_file():
            return f"bash {name}"
    if (task_dir / "tests").is_dir():
        for name in ("test.sh", "run.sh"):
            if (task_dir / "tests" / name).is_file():
                return f"bash tests/{name}"
        return "python -m pytest -x -q tests"
    raise ValueError(f"no official test entrypoint found in {task_dir}")


def _default_widesearch_eval_command(root: Path) -> str:
    scripts = sorted(root.glob("scripts/run_infer_and_eval*.py"))
    if not scripts:
        raise ValueError(f"official WideSearch eval script not found under {root}/scripts")
    return (
        f"python {scripts[0]} --stage eval "
        f"--response_root $WIDESEARCH_RESPONSE_PATH --result_save_root $WIDESEARCH_RESULT_DIR"
    )


def _first_score_from_result_dir(result_dir: Path) -> float | None:
    for path in sorted(result_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping):
            for key in ("score", "accuracy", "f1", "success_rate"):
                value = payload.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
    return None


def _mcp_atlas_coverage(output_path: Path) -> float | None:
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, Mapping):
        for key in ("coverage", "claim_coverage", "score"):
            value = payload.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return None


__all__ = [
    "AgentLoopVerdict",
    "AgentLoopVerifier",
    "MCP_ATLAS_ROOT_ENV",
    "TERMINAL_BENCH_ROOT_ENV",
    "TOOLATHLON_ROOT_ENV",
    "WIDESEARCH_EVAL_COMMAND_ENV",
    "WIDESEARCH_ROOT_ENV",
    "VerifierSpec",
    "build_agent_loop_verifier",
    "build_rubric_judge_prompt",
    "parse_rubric_judge_verdict",
    "preflight_agent_loop_runtime",
    "trace_to_mcp_atlas_transcript",
    "widesearch_answer_to_official_row",
]
