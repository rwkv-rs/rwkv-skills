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
import shlex
import sys
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
NL2REPO_ROOT_ENV = "RWKV_NL2REPO_ROOT"
WIDESEARCH_ROOT_ENV = "RWKV_WIDESEARCH_OFFICIAL_ROOT"
WIDESEARCH_EVAL_COMMAND_ENV = "RWKV_WIDESEARCH_EVAL_COMMAND"
WIDESEARCH_DATA_ROOT_ENV = "RWKV_WIDESEARCH_DATA_ROOT"
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
    if kind == "nl2repo_official":
        return NL2RepoOfficialVerifier()
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
            else:
                health_error = WebSearchExecutor.health_error()
                if health_error:
                    errors.append(f"{health_error}; {_DOCS_HINT}")
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
        self._setup_test_env(container_id, task_dir)
        test_command = "bash /tests/run-tests.sh"
        proc = subprocess.run(
            ["docker", "exec", container_id, "bash", "-lc", test_command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
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

    def _setup_test_env(self, container_id: str, task_dir: Path) -> None:
        run_tests = task_dir / "run-tests.sh"
        if not run_tests.is_file():
            raise ValueError(f"no official run-tests.sh found in {task_dir}")
        subprocess.run(
            ["docker", "exec", container_id, "rm", "-rf", "/tests"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        subprocess.run(
            ["docker", "exec", container_id, "mkdir", "-p", "/tests"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        subprocess.run(
            ["docker", "cp", str(run_tests), f"{container_id}:/tests/run-tests.sh"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        tests_dir = task_dir / "tests"
        if tests_dir.is_dir():
            subprocess.run(
                ["docker", "cp", f"{tests_dir}/.", f"{container_id}:/tests/"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
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
        proc = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
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


class NL2RepoOfficialVerifier:
    """Runs NL2RepoBench's official post-processor against the generated workspace."""

    def preflight(self, records: Sequence["AgentLoopRecord"], args: Any) -> list[str]:
        errors: list[str] = []
        root = _official_root(NL2REPO_ROOT_ENV)
        if root is None:
            errors.append(f"nl2repo_official verifier requires {NL2REPO_ROOT_ENV}=<NL2RepoBench checkout>; {_DOCS_HINT}")
            return errors
        if not (root / "openhands" / "post_processor.py").is_file():
            errors.append(f"nl2repo_official verifier: openhands/post_processor.py not found under {root}; {_DOCS_HINT}")
        if shutil.which("docker") is None:
            errors.append(f"nl2repo_official verifier requires the docker CLI; {_DOCS_HINT}")
        for record in records:
            task_name = _nl2repo_task_name(record)
            if not _nl2repo_task_dir(root, task_name):
                errors.append(f"{record.task_id}: NL2Repo official test files not found for {task_name!r}; {_DOCS_HINT}")
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
        root = _official_root(NL2REPO_ROOT_ENV)
        if root is None:
            raise ValueError(f"missing {NL2REPO_ROOT_ENV}")
        workspace = str(executor_snapshot.get("workspace") or "")
        if not workspace:
            raise ValueError("nl2repo_official verifier requires a subprocess shell_sandbox workspace")
        task_name = _nl2repo_task_name(record)
        if not _nl2repo_task_dir(root, task_name):
            raise ValueError(f"NL2Repo official test files not found for {task_name!r}")
        with tempfile.TemporaryDirectory(prefix="nl2repo-eval-") as tmp:
            result_path = Path(tmp) / "result.json"
            script = _nl2repo_eval_script()
            proc = subprocess.run(
                ["python", "-c", script],
                cwd=str(root),
                env={
                    **os.environ,
                    "NL2REPO_ROOT": str(root),
                    "NL2REPO_TASK": task_name,
                    "NL2REPO_WORKSPACE": workspace,
                    "NL2REPO_RESULT_PATH": str(result_path),
                },
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=float(record.verifier.config.get("test_timeout_s") or 10800.0),
            )
            if proc.returncode != 0:
                return AgentLoopVerdict(
                    reward=0.0,
                    is_passed=False,
                    fail_reason=f"official NL2Repo post-processor failed (exit {proc.returncode})",
                    details={"stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]},
                )
            result = _read_json_mapping(result_path)
        pytest_results = _nl2repo_pytest_results(result)
        total = int(pytest_results.get("total") or 0)
        passed_count = int(pytest_results.get("passed") or 0)
        reward = float(pytest_results.get("success_rate") or ((passed_count / total) if total else 0.0))
        passed = bool(result.get("status") == "success" and total > 0 and passed_count >= total)
        return AgentLoopVerdict(
            reward=reward,
            is_passed=passed,
            fail_reason="" if passed else f"official NL2Repo tests passed {passed_count}/{total}",
            details={
                "official_score": reward,
                "passed": passed_count,
                "total": total,
                "status": result.get("status"),
                "log_path": result.get("log_path"),
            },
        )


class WideSearchOfficialVerifier:
    """Converts final answers into the official WideSearch format and runs the official eval stage."""

    def preflight(self, records: Sequence["AgentLoopRecord"], args: Any) -> list[str]:
        root = _official_root(WIDESEARCH_ROOT_ENV)
        if root is None:
            return [f"widesearch_official verifier requires {WIDESEARCH_ROOT_ENV}=<WideSearch checkout>; {_DOCS_HINT}"]
        data_root = _widesearch_data_root()
        if data_root is not None:
            errors: list[str] = []
            try:
                judge = resolve_judge_model_config(
                    model_name=getattr(args, "judge_model", None),
                    api_key=getattr(args, "judge_api_key", None),
                    base_url=getattr(args, "judge_base_url", None),
                )
            except ValueError as exc:
                errors.append(str(exc))
                judge = None
            if judge is None:
                errors.append(f"widesearch_official direct eval requires JUDGE_MODEL + JUDGE_API_KEY; {_DOCS_HINT}")
            import_error = _widesearch_direct_eval_import_error(root)
            if import_error:
                errors.append(
                    "widesearch_official direct eval cannot import the official eval stack: "
                    f"{import_error}; {_DOCS_HINT}"
                )
            for record in records:
                missing = _widesearch_missing_data_files(data_root, _widesearch_instance_id(record))
                if missing:
                    errors.append(f"{record.task_id}: missing WideSearch official data: {missing}; {_DOCS_HINT}")
                    break
            return errors
        if not os.environ.get(WIDESEARCH_EVAL_COMMAND_ENV) and not list(root.glob("scripts/run_infer_and_eval*.py")):
            return [
                f"widesearch_official verifier: official eval script not found under {root}/scripts "
                f"(or set {WIDESEARCH_EVAL_COMMAND_ENV}); {_DOCS_HINT}"
            ]
        if not os.environ.get(WIDESEARCH_EVAL_COMMAND_ENV) and (root / "src" / "evaluation" / "evaluation.py").is_file():
            import_error = _widesearch_eval_import_error(root)
            if import_error:
                return [
                    "widesearch_official verifier cannot import the official eval stack with the current Python: "
                    f"{import_error}; install WideSearch dependencies or set {WIDESEARCH_EVAL_COMMAND_ENV} "
                    f"to a command that uses the official WideSearch environment; {_DOCS_HINT}"
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
        data_root = _widesearch_data_root()
        if data_root is not None:
            return _verify_widesearch_direct(
                root=root,
                data_root=data_root,
                record=record,
                final_answer=final_answer,
                trace=trace,
            )
        with tempfile.TemporaryDirectory(prefix="widesearch-eval-") as tmp:
            instance_id = _widesearch_instance_id(record)
            trial_idx = int(record.metadata.get("trial") or record.verifier.config.get("trial") or 0)
            model_config_name = str(
                record.verifier.config.get("model_config_name")
                or os.environ.get("WIDESEARCH_MODEL_CONFIG_NAME")
                or "rwkv_eval"
            )
            response_root = Path(tmp) / "responses"
            result_dir = Path(tmp) / "results"
            response_root.mkdir()
            result_dir.mkdir()
            response_path = response_root / _widesearch_response_filename(
                model_config_name=model_config_name,
                instance_id=instance_id,
                trial_idx=trial_idx,
            )
            response_path.write_text(
                json.dumps(
                    _widesearch_official_response_row(
                        instance_id=instance_id,
                        final_answer=final_answer,
                        trace=trace,
                        trial_idx=trial_idx,
                    ),
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            command = os.environ.get(WIDESEARCH_EVAL_COMMAND_ENV) or _default_widesearch_eval_command(root)
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(root),
                env={
                    **os.environ,
                    "PYTHONPATH": _prepend_pythonpath(root),
                    "WIDESEARCH_INSTANCE_ID": instance_id,
                    "WIDESEARCH_MODEL_CONFIG_NAME": model_config_name,
                    "WIDESEARCH_RESPONSE_PATH": str(response_path),
                    "WIDESEARCH_RESPONSE_ROOT": str(response_root),
                    "WIDESEARCH_RESULT_DIR": str(result_dir),
                },
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
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
                encoding="utf-8",
                errors="replace",
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
            encoding="utf-8",
            errors="replace",
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
        "question_id": _widesearch_instance_id(record),
        "response": final_answer,
        "trial": int(record.metadata.get("trial") or 0),
    }


def _widesearch_instance_id(record: "AgentLoopRecord") -> str:
    return str(record.metadata.get("official_task_id") or record.task_id)


def _widesearch_response_filename(*, model_config_name: str, instance_id: str, trial_idx: int) -> str:
    return f"{model_config_name}_{instance_id}_{trial_idx}_response.json"


def _widesearch_official_response_row(
    *,
    instance_id: str,
    final_answer: str,
    trace: Sequence[Mapping[str, Any]],
    trial_idx: int,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for step in trace:
        if step.get("kind") == "tool_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": {
                        "tool": step.get("name"),
                        "arguments": dict(step.get("arguments") or {}),
                    },
                }
            )
            messages.append({"role": "tool", "content": step.get("output")})
    if final_answer:
        messages.append({"role": "assistant", "content": final_answer})
    return {
        "instance_id": instance_id,
        "response": final_answer,
        "messages": messages,
        "trial_idx": trial_idx,
    }


def _widesearch_data_root() -> Path | None:
    raw = os.environ.get(WIDESEARCH_DATA_ROOT_ENV)
    if raw:
        candidate = Path(raw).expanduser()
        return candidate if (candidate / "widesearch.jsonl").is_file() and (candidate / "widesearch_gold").is_dir() else None
    candidates: list[Path] = []
    candidates.append(Path.cwd() / "data" / "widesearch_official")
    for candidate in candidates:
        if (candidate / "widesearch.jsonl").is_file() and (candidate / "widesearch_gold").is_dir():
            return candidate
    return None


def _widesearch_missing_data_files(data_root: Path, instance_id: str) -> str:
    missing: list[str] = []
    if not (data_root / "widesearch.jsonl").is_file():
        missing.append(str(data_root / "widesearch.jsonl"))
    if not (data_root / "widesearch_gold" / f"{instance_id}.csv").is_file():
        missing.append(str(data_root / "widesearch_gold" / f"{instance_id}.csv"))
    return ", ".join(missing)


def _verify_widesearch_direct(
    *,
    root: Path,
    data_root: Path,
    record: "AgentLoopRecord",
    final_answer: str,
    trace: Sequence[Mapping[str, Any]],
) -> AgentLoopVerdict:
    judge = resolve_judge_model_config()
    if judge is None:
        raise ValueError("widesearch_official direct eval requires JUDGE_MODEL + JUDGE_API_KEY")
    instance_id = _widesearch_instance_id(record)
    missing = _widesearch_missing_data_files(data_root, instance_id)
    if missing:
        raise ValueError(f"missing WideSearch official data: {missing}")
    with tempfile.TemporaryDirectory(prefix="widesearch-direct-eval-") as tmp:
        tmp_path = Path(tmp)
        stub_root = tmp_path / "stubs"
        _write_widesearch_stub_modules(stub_root)
        response_path = tmp_path / "response.json"
        result_path = tmp_path / "result.json"
        response_path.write_text(
            json.dumps(
                _widesearch_official_response_row(
                    instance_id=instance_id,
                    final_answer=final_answer,
                    trace=trace,
                    trial_idx=int(record.metadata.get("trial") or record.verifier.config.get("trial") or 0),
                ),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        proc = subprocess.run(
            [sys.executable or "python3", "-c", _widesearch_direct_eval_script()],
            cwd=str(root),
            env={
                **os.environ,
                "PYTHONPATH": _prepend_pythonpath_chain((stub_root, root)),
                "WIDESEARCH_DATA_ROOT": str(data_root),
                "WIDESEARCH_INSTANCE_ID": instance_id,
                "WIDESEARCH_RESPONSE_JSON": str(response_path),
                "WIDESEARCH_RESULT_JSON": str(result_path),
                "WIDESEARCH_JUDGE_MODEL": judge.model_name,
                "WIDESEARCH_JUDGE_API_KEY": judge.api_key,
                "WIDESEARCH_JUDGE_BASE_URL": judge.base_url or "",
            },
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(record.verifier.config.get("eval_timeout_s") or 1800.0),
            check=False,
        )
        if proc.returncode != 0:
            output = (proc.stdout + "\n" + proc.stderr).strip()
            raise RuntimeError(f"official WideSearch direct eval failed (exit {proc.returncode}): {output[-3000:]}")
        result = _read_json_mapping(result_path)
    score = _first_score_from_payload(result)
    if score is None:
        raise RuntimeError("official WideSearch direct eval produced no parsable score")
    passed = score >= float(record.verifier.config.get("pass_threshold") or 1.0)
    return AgentLoopVerdict(
        reward=score,
        is_passed=passed,
        fail_reason="" if passed else f"official score {score:.3f} below threshold",
        details={"official_score": score, "result": result},
    )


def _write_widesearch_stub_modules(stub_root: Path) -> None:
    stub_root.mkdir(parents=True, exist_ok=True)
    pandarallel_root = stub_root / "pandarallel"
    pandarallel_root.mkdir()
    (pandarallel_root / "__init__.py").write_text(
        "class _Pandarallel:\n"
        "    def initialize(self, *args, **kwargs):\n"
        "        return None\n"
        "pandarallel = _Pandarallel()\n",
        encoding="utf-8",
    )
    (stub_root / "dateparser.py").write_text(
        "def parse(value, settings=None):\n"
        "    try:\n"
        "        import pandas as _pd\n"
        "        parsed = _pd.to_datetime(value, errors='coerce')\n"
        "        if _pd.isna(parsed):\n"
        "            return None\n"
        "        return parsed.to_pydatetime()\n"
        "    except Exception:\n"
        "        return None\n",
        encoding="utf-8",
    )
    (stub_root / "volcenginesdkarkruntime.py").write_text(
        "class Ark:\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        raise RuntimeError('Ark is not available in WideSearch direct eval')\n",
        encoding="utf-8",
    )


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


def _prepend_pythonpath(path: Path) -> str:
    current = os.environ.get("PYTHONPATH")
    return f"{path}{os.pathsep}{current}" if current else str(path)


def _prepend_pythonpath_chain(paths: Sequence[Path]) -> str:
    current = os.environ.get("PYTHONPATH")
    rendered = os.pathsep.join(str(path) for path in paths)
    return f"{rendered}{os.pathsep}{current}" if current else rendered


def _widesearch_eval_import_error(root: Path) -> str:
    proc = subprocess.run(
        [
            sys.executable or "python3",
            "-c",
            "import src.evaluation.evaluation",  # noqa: S603 - static import check.
        ],
        cwd=str(root),
        env={**os.environ, "PYTHONPATH": _prepend_pythonpath(root)},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15.0,
        check=False,
    )
    if proc.returncode == 0:
        return ""
    output = (proc.stderr or proc.stdout or "").strip()
    return output[-800:] or f"import check exited {proc.returncode}"


def _widesearch_direct_eval_import_error(root: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="widesearch-import-") as tmp:
        stub_root = Path(tmp) / "stubs"
        _write_widesearch_stub_modules(stub_root)
        proc = subprocess.run(
            [
                sys.executable or "python3",
                "-c",
                "import src.evaluation.data_loader; import src.evaluation.evaluation",
            ],
            cwd=str(root),
            env={**os.environ, "PYTHONPATH": _prepend_pythonpath_chain((stub_root, root))},
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15.0,
            check=False,
        )
    if proc.returncode == 0:
        return ""
    output = (proc.stderr or proc.stdout or "").strip()
    return output[-800:] or f"import check exited {proc.returncode}"


def _widesearch_direct_eval_script() -> str:
    return r'''
import dataclasses
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from openai import OpenAI

from src.evaluation.data_loader import WideSearchQuery, WideSearchResponse
from src.evaluation.evaluation import evaluate_single_query
from src.utils.utils import norm_column
import src.evaluation.metric_utils as metric_utils


def _judge_completion(messages, tools=None, model_config_name="default_eval_config"):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    client = OpenAI(
        api_key=os.environ["WIDESEARCH_JUDGE_API_KEY"],
        base_url=os.environ.get("WIDESEARCH_JUDGE_BASE_URL") or None,
        timeout=300,
        max_retries=2,
    )
    response = client.chat.completions.create(
        model=os.environ["WIDESEARCH_JUDGE_MODEL"],
        messages=messages,
        temperature=0,
    )
    return SimpleNamespace(content=response.choices[0].message.content or "")


metric_utils.llm_completion = _judge_completion

data_root = Path(os.environ["WIDESEARCH_DATA_ROOT"])
instance_id = os.environ["WIDESEARCH_INSTANCE_ID"]
response_payload = json.loads(Path(os.environ["WIDESEARCH_RESPONSE_JSON"]).read_text(encoding="utf-8"))

source_row = None
with (data_root / "widesearch.jsonl").open(encoding="utf-8") as fh:
    for line in fh:
        item = json.loads(line)
        if str(item.get("instance_id") or item.get("official_task_id") or item.get("id")) == instance_id:
            source_row = item
            break
if source_row is None:
    raise SystemExit(f"WideSearch instance not found: {instance_id}")

evaluation = source_row.get("evaluation") or {}
if isinstance(evaluation, str):
    evaluation = json.loads(evaluation)
required = evaluation["required"]
answer = pd.read_csv(data_root / "widesearch_gold" / f"{instance_id}.csv")
answer.columns = [norm_column(col.strip()) for col in answer.columns]
answer = answer[required]
query = WideSearchQuery(
    instance_id=instance_id,
    query=str(source_row.get("query") or source_row.get("question") or ""),
    evaluation=evaluation,
    answer=answer,
    language=str(source_row.get("language") or "en"),
)
response = WideSearchResponse(
    instance_id=instance_id,
    response=str(response_payload.get("response") or ""),
    messages=response_payload.get("messages") or [],
    trial_idx=response_payload.get("trial_idx"),
)
result = evaluate_single_query(
    query,
    response,
    result_save_path=None,
    eval_model_config_name="default_eval_config",
)
Path(os.environ["WIDESEARCH_RESULT_JSON"]).write_text(
    json.dumps(dataclasses.asdict(result), ensure_ascii=False),
    encoding="utf-8",
)
'''


def _nl2repo_task_name(record: "AgentLoopRecord") -> str:
    return str(record.verifier.config.get("official_task_id") or record.metadata.get("official_task_id") or record.task_id)


def _nl2repo_task_dir(root: Path, task_name: str) -> Path | None:
    candidate = root / "test_files" / task_name
    if (
        candidate.is_dir()
        and (candidate / "start.md").is_file()
        and (candidate / "test_commands.json").is_file()
        and (candidate / "test_files.json").is_file()
    ):
        return candidate
    return None


def _nl2repo_eval_script() -> str:
    return r'''
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["NL2REPO_ROOT"])
task_name = os.environ["NL2REPO_TASK"]
workspace = os.environ["NL2REPO_WORKSPACE"]
result_path = Path(os.environ["NL2REPO_RESULT_PATH"])
os.chdir(root)
sys.path.insert(0, str(root))

import test_data_service  # noqa: E402
from openhands.post_processor import post_process_task  # noqa: E402


class _Logger:
    def info(self, message): print(message, flush=True)
    def warning(self, message): print(message, flush=True)
    def error(self, message): print(message, flush=True)
    def debug(self, message): print(message, flush=True)


test_data_service.test_data_list.clear()
test_data_service.read_all_test_data()
test_data = next((item for item in test_data_service.test_data_list if item.proName == task_name), None)
if test_data is None:
    raise SystemExit(f"NL2Repo task not found: {task_name}")

result = post_process_task("rwkv-" + task_name, workspace, test_data, _Logger())
result_path.write_text(json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8")
'''


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _nl2repo_pytest_results(result: Mapping[str, Any]) -> dict[str, Any]:
    raw = result.get("pytest_results")
    if isinstance(raw, Mapping):
        return dict(raw)
    test_results = result.get("test_results")
    if isinstance(test_results, Mapping):
        nested = test_results.get("pytest_results")
        if isinstance(nested, Mapping):
            return dict(nested)
    return {}


def _terminal_bench_task_dir(root: Path, record: "AgentLoopRecord") -> Path | None:
    task_id = str(record.verifier.config.get("official_task_id") or record.metadata.get("official_task_id") or record.task_id)
    candidates = (root / "tasks" / task_id, root / "original-tasks" / task_id, root / task_id)
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
    python_executable = shlex.quote(sys.executable or "python3")
    script_path = shlex.quote(str(scripts[0]))
    return (
        f"{python_executable} {script_path} --stage eval "
        '--response_root "$WIDESEARCH_RESPONSE_ROOT" '
        '--result_save_root "$WIDESEARCH_RESULT_DIR" '
        '--model_config_name "$WIDESEARCH_MODEL_CONFIG_NAME" '
        '--instance_id "$WIDESEARCH_INSTANCE_ID" '
        "--trial_num 1"
    )


def _first_score_from_result_dir(result_dir: Path) -> float | None:
    for path in sorted(result_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping):
            score = _first_score_from_payload(payload)
            if score is not None:
                return score
    return None


def _first_score_from_payload(payload: Mapping[str, Any]) -> float | None:
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
    "NL2REPO_ROOT_ENV",
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
