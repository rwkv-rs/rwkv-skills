from __future__ import annotations

"""Prepare multi-turn agent-loop benchmark manifests.

Rows carry executor/verifier specs so the runner can drive the benchmark's
real environment and grade with its OFFICIAL verifier. Rows without explicit
specs are classified from their format: recorded tool outputs -> manifest
replay, rubric rows -> LLM judge, plain QA rows -> expected final_answer call
(the loop naturally degrades to a single turn, matching official single-turn
benchmarks).
"""

import os
import json
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.eval.benchmark_sources import AGENT_LOOP_BENCHMARK_SOURCES
from src.eval.datasets.data_prepper.function_calling import agent_tool_call as _agent_tool_call
from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.datasets.data_prepper.source_files import (
    load_hf_rows,
    per_dataset_source_env,
    read_source_rows,
    resolve_source_path,
)
from src.eval.datasets.runtime import MaterializingDatasetSpec
from src.eval.scheduler.config import REPO_ROOT

_SOURCE_ROOT_ENV = "RWKV_AGENT_LOOP_SOURCE_ROOT"
_LEGACY_SOURCE_ROOT_ENV = "RWKV_AGENT_BENCHMARKS_ROOT"
_ENV_PREFIX = "RWKV_AGENT_LOOP_SOURCE"
_TERMINAL_BENCH_ROOT_ENV = "RWKV_TERMINAL_BENCH_ROOT"
_NL2REPO_ROOT_ENV = "RWKV_NL2REPO_ROOT"
_DEEPSWE_ROOT_ENV = "RWKV_DEEPSWE_ROOT"
_REQUIRED_FIELDS = ("task_id", "instruction", "tools", "executor", "verifier")

_AGENT_LOOP_SOURCES: dict[str, dict[str, str | None]] = {
    item.benchmark_name: {"source": item.source_url, "source_kind": item.source_kind}
    for item in AGENT_LOOP_BENCHMARK_SOURCES
}
_ALIASES: dict[str, str] = {
    "deep_swe": "deepswe",
    "wide_search": "widesearch",
    "claw_eval": "claweval",
    "hle_tools": "hle_with_tools",
}

_RUBRIC_KEYS = ("rubrics", "rubric", "grading_rubrics", "checklist")
_RECORDED_OUTPUT_KEYS = ("recorded_tool_outputs", "tool_outputs", "tool_results")


@dataclass(frozen=True, slots=True)
class AgentLoopProfile:
    executor_kind: str = "manifest_replay"
    executor_config: dict[str, Any] = field(default_factory=dict)
    verifier_kind: str | None = None  # None -> classify from the row format
    verifier_config: dict[str, Any] = field(default_factory=dict)


_AGENT_LOOP_PROFILES: dict[str, AgentLoopProfile] = {
    # Container/terminal agents graded by official task tests.
    "terminal_bench_2_1": AgentLoopProfile(
        executor_kind="shell_sandbox",
        executor_config={"backend": "docker"},
        verifier_kind="terminal_bench_official",
    ),
    # Repo agents graded by the task's own programmatic test command (pytest etc.).
    "deepswe": AgentLoopProfile(
        executor_kind="shell_sandbox",
        executor_config={"backend": "docker"},
        verifier_kind="repo_tests_official",
    ),
    "nl2repo": AgentLoopProfile(
        executor_kind="shell_sandbox",
        executor_config={"backend": "subprocess"},
        verifier_kind="repo_tests_official",
    ),
    # Search agents: live web search by default; rows may override to replay recorded outputs.
    "widesearch": AgentLoopProfile(executor_kind="web_search", verifier_kind="widesearch_official"),
    "deepsearchqa": AgentLoopProfile(executor_kind="web_search", verifier_kind="llm_rubric_judge"),
    # MCP-server agents graded by their official evaluators.
    "mcp_atlas": AgentLoopProfile(executor_kind="mcp_worker", verifier_kind="mcp_atlas_official"),
    "toolathlon": AgentLoopProfile(executor_kind="mcp_worker", verifier_kind="toolathlon_official"),
    # Docker harnesses whose official graders are not wired yet (preflight-gated).
    "apex_agents": AgentLoopProfile(executor_kind="shell_sandbox", verifier_kind="unsupported_official"),
    "claweval": AgentLoopProfile(executor_kind="shell_sandbox", verifier_kind="unsupported_official"),
    "wildclawbench": AgentLoopProfile(executor_kind="shell_sandbox", verifier_kind="unsupported_official"),
    "skillsbench": AgentLoopProfile(executor_kind="shell_sandbox", verifier_kind="unsupported_official"),
    # Tool-augmented expert QA: rows from HF plus live search tools, judged against reference answers.
    "hle_with_tools": AgentLoopProfile(executor_kind="web_search", verifier_kind="llm_rubric_judge"),
    # Internal benchmarks: classify each row from its format.
    "hy_euler_pro": AgentLoopProfile(),
    "hy_backend_2_0": AgentLoopProfile(),
    "hy_swe_max": AgentLoopProfile(),
    "hy_companybench": AgentLoopProfile(),
    "e_bench": AgentLoopProfile(),
    "prodbench": AgentLoopProfile(),
    "hy_skillsworld": AgentLoopProfile(),
    "hy_finmodelbench": AgentLoopProfile(),
}

_MISSING_PROFILES = set(_AGENT_LOOP_SOURCES) - set(_AGENT_LOOP_PROFILES)
if _MISSING_PROFILES:  # pragma: no cover - guards future benchmark_sources edits
    raise RuntimeError(f"agent-loop benchmarks missing profiles: {sorted(_MISSING_PROFILES)}")


class AgentLoopDatasetSpec(MaterializingDatasetSpec):
    def __init__(self, output_root: Path, split: str, *, name: str) -> None:
        canonical_name = _ALIASES.get(name, name)
        if canonical_name not in _AGENT_LOOP_SOURCES:
            raise ValueError(f"unknown agent-loop dataset alias: {name}")
        super().__init__(
            name,
            output_root,
            split,
            required_fields=_REQUIRED_FIELDS,
            source_kind="rwkvc_agent_loop_source",
        )
        self._canonical_name = canonical_name
        self._source_path: Path | None = None
        self._skipped_rows = 0

    def source_path(self) -> Path:
        if self._source_path is None:
            path = resolve_source_path(
                self._canonical_name,
                self.split,
                env_prefix=_ENV_PREFIX,
                root_envs=(_SOURCE_ROOT_ENV, _LEGACY_SOURCE_ROOT_ENV),
                default_root=REPO_ROOT / "data" / "agent_loop_sources",
            )
            if not path.exists() and self._canonical_name == "terminal_bench_2_1":
                tasks_path = path.parent / "tasks.jsonl"
                if tasks_path.exists():
                    path = tasks_path.resolve()
            self._source_path = path
        return self._source_path

    def download(self) -> None:
        path = self.source_path()
        if not path.exists():
            if self._canonical_name == "terminal_bench_2_1" and _terminal_bench_official_task_dirs():
                return None
            if self._canonical_name == "nl2repo" and _nl2repo_official_project_dirs():
                return None
            if self._canonical_name == "deepswe" and _deepswe_official_task_dirs():
                return None
            if _AGENT_LOOP_SOURCES[self._canonical_name].get("source_kind") == "hf_dataset":
                self._materialize_hf_source(path)
                return None
            env_name = per_dataset_source_env(_ENV_PREFIX, self._canonical_name)
            raise FileNotFoundError(
                f"missing source for {self._canonical_name}:{self.split}: {path}. "
                f"Set {env_name}=<json/jsonl file or dir> or {_SOURCE_ROOT_ENV}=<root>. "
                "See docs/agent_loop.md."
            )

    def load_records(self) -> Iterable[dict[str, Any]]:
        path = self.source_path()
        info = _AGENT_LOOP_SOURCES[self._canonical_name]
        if path.exists():
            rows = read_source_rows(path)
            source_label = str(path)
        elif self._canonical_name == "terminal_bench_2_1" and _terminal_bench_official_task_dirs():
            rows = _terminal_bench_official_rows()
            source_label = str(_terminal_bench_official_root() or path)
        elif self._canonical_name == "nl2repo" and _nl2repo_official_project_dirs():
            rows = _nl2repo_official_rows()
            source_label = str(_nl2repo_official_root() or path)
        elif self._canonical_name == "deepswe" and _deepswe_official_task_dirs():
            rows = _deepswe_official_rows()
            source_label = str(_deepswe_official_root() or path)
        elif info.get("source_kind") == "hf_dataset":
            rows = load_hf_rows(str(info.get("source") or ""), self.split)
            source_label = str(info.get("source") or "")
        else:
            raise FileNotFoundError(path)
        records: list[dict[str, Any]] = []
        skipped_rows = 0
        for index, row in enumerate(rows):
            try:
                records.append(
                    normalize_agent_loop_row(
                        row,
                        dataset_name=self._canonical_name,
                        index=index,
                        source_path=source_label,
                    )
                )
            except ValueError as exc:
                if _is_missing_reference_row(exc):
                    skipped_rows += 1
                    continue
                raise
        self._skipped_rows = skipped_rows
        return records

    def _materialize_hf_source(self, path: Path) -> None:
        info = _AGENT_LOOP_SOURCES[self._canonical_name]
        rows = load_hf_rows(str(info.get("source") or ""), self.split)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for index, row in enumerate(rows):
                try:
                    payload = normalize_agent_loop_row(
                        row,
                        dataset_name=self._canonical_name,
                        index=index,
                        source_path=str(info.get("source") or ""),
                    )
                except ValueError as exc:
                    if _is_missing_reference_row(exc):
                        continue
                    raise
                fh.write(json.dumps(payload, ensure_ascii=False))
                fh.write("\n")
        self._source_path = path.resolve()

    def manifest_extra(self) -> dict[str, Any]:
        info = _AGENT_LOOP_SOURCES[self._canonical_name]
        return {
            "source_path": str(self.source_path()),
            "source": info["source"],
            "source_kind": info["source_kind"],
            "canonical_name": self._canonical_name,
            "input_contract": (
                "JSONL rows may carry explicit executor/verifier specs, recorded tool outputs, "
                "rubrics, or plain question/answer fields; each row is classified by format."
            ),
            "skipped_rows": self._skipped_rows,
        }


def normalize_agent_loop_row(
    row: Mapping[str, Any],
    *,
    dataset_name: str,
    index: int,
    source_path: str | Path,
) -> dict[str, Any]:
    profile = _AGENT_LOOP_PROFILES[dataset_name]

    executor_raw = row.get("executor")
    verifier_raw = row.get("verifier")
    recorded = _recorded_tool_outputs(row)
    rubrics = _rubric_lines(row)

    if isinstance(executor_raw, Mapping) and executor_raw.get("kind"):
        executor = {"kind": str(executor_raw["kind"]), "config": dict(executor_raw.get("config") or {})}
    else:
        executor = {"kind": profile.executor_kind, "config": dict(profile.executor_config)}
        if recorded and profile.executor_kind not in {"shell_sandbox", "mcp_worker"}:
            executor = {"kind": "manifest_replay", "config": {}}

    base = _agent_tool_call._normalize_source_row(  # noqa: SLF001 - shared normalization contract
        row,
        dataset_name=dataset_name,
        index=index,
        source_path=source_path,
    )

    if isinstance(verifier_raw, Mapping) and verifier_raw.get("kind"):
        verifier = {"kind": str(verifier_raw["kind"]), "config": dict(verifier_raw.get("config") or {})}
    elif profile.verifier_kind is not None:
        verifier = {"kind": profile.verifier_kind, "config": dict(profile.verifier_config)}
    elif rubrics:
        verifier = {"kind": "llm_rubric_judge", "config": {"rubrics": rubrics}}
    else:
        verifier = {"kind": "expected_tool_calls", "config": {}}

    if verifier["kind"] == "llm_rubric_judge" and rubrics and "rubrics" not in verifier["config"]:
        verifier["config"]["rubrics"] = rubrics
    if verifier["kind"] == "llm_rubric_judge" and "reference_answer" not in verifier["config"]:
        reference = _reference_answer_from_expected(base.get("expected_tool_calls") or ())
        if reference:
            verifier["config"]["reference_answer"] = reference

    metadata = dict(base.get("metadata") or {})
    metadata["source_format"] = "rwkvc_agent_loop"
    if row.get("official_task_id") or metadata.get("official_task_id"):
        metadata.setdefault("official_task_id", str(row.get("official_task_id") or metadata.get("official_task_id")))

    payload = {
        "task_id": str(base.get("task_id")),
        "instruction": str(base.get("instruction") or ""),
        "system_extra": str(row.get("system_extra") or ""),
        "tools": list(base.get("tools") or []),
        "executor": executor,
        "verifier": verifier,
        "expected_tool_calls": list(base.get("expected_tool_calls") or []),
        "recorded_tool_outputs": recorded,
        "metadata": metadata,
    }
    if dataset_name == "terminal_bench_2_1":
        metadata["official_sandbox_required"] = True
        metadata["official_sandbox_env"] = "RWKV_TERMINAL_BENCH_2_1_SANDBOX_ROOT"
        payload["answer"] = str(row.get("answer") or row.get("expected_answer") or "")
        payload["official_payload"] = dict(row)
    return payload


def _recorded_tool_outputs(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in _RECORDED_OUTPUT_KEYS:
        raw = row.get(key)
        if isinstance(raw, list) and raw:
            return [dict(item) for item in raw if isinstance(item, Mapping)]
    return []


def _rubric_lines(row: Mapping[str, Any]) -> list[str]:
    for key in _RUBRIC_KEYS:
        raw = row.get(key)
        if raw in (None, ""):
            continue
        if isinstance(raw, list):
            lines = [str(item.get("rubric") or item.get("text") or item) if isinstance(item, Mapping) else str(item) for item in raw]
            return [line for line in lines if line.strip()]
        return [str(raw)]
    return []


def _is_missing_reference_row(exc: ValueError) -> bool:
    message = str(exc)
    return "agent tool-call row missing answer/expected_tool_calls" in message


def _reference_answer_from_expected(expected_tool_calls: Iterable[Mapping[str, Any]]) -> str:
    for call in expected_tool_calls:
        if str(call.get("name") or "") == "final_answer":
            arguments = call.get("arguments")
            if isinstance(arguments, Mapping):
                answer = arguments.get("answer")
                if answer not in (None, ""):
                    return str(answer)
    return ""


def _terminal_bench_official_root() -> Path | None:
    raw = os.environ.get(_TERMINAL_BENCH_ROOT_ENV)
    if not raw:
        return None
    root = Path(raw).expanduser()
    return root if root.is_dir() else None


def _terminal_bench_official_task_dirs() -> list[Path]:
    root = _terminal_bench_official_root()
    if root is None:
        return []
    candidates = []
    for parent_name in ("tasks", "original-tasks"):
        parent = root / parent_name
        if parent.is_dir():
            candidates.extend(sorted(path for path in parent.iterdir() if (path / "task.yaml").is_file()))
    return candidates


def _terminal_bench_docker_config(task_dir: Path) -> dict[str, str]:
    compose_file = task_dir / "docker-compose.yaml"
    if not compose_file.is_file():
        compose_file = task_dir / "docker-compose.yml"

    context = task_dir
    dockerfile = task_dir / "Dockerfile"
    if not dockerfile.is_file() and compose_file.is_file():
        compose = _read_yaml_mapping(compose_file)
        services = compose.get("services")
        client = services.get("client") if isinstance(services, Mapping) else None
        build = client.get("build") if isinstance(client, Mapping) else None
        if isinstance(build, str):
            context = (task_dir / build).resolve()
            dockerfile = context / "Dockerfile"
        elif isinstance(build, Mapping):
            raw_context = str(build.get("context") or ".")
            context = (task_dir / raw_context).resolve()
            raw_dockerfile = str(build.get("dockerfile") or "Dockerfile")
            dockerfile = Path(raw_dockerfile)
            if not dockerfile.is_absolute():
                dockerfile = context / dockerfile
    config = {
        "dockerfile_context": str(context),
        "dockerfile_path": str(dockerfile),
    }
    if compose_file.is_file():
        config["docker_compose_file"] = str(compose_file)
    return config


def _terminal_bench_official_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_dir in _terminal_bench_official_task_dirs():
        task_id = task_dir.name
        payload = _read_terminal_bench_task_yaml(task_dir / "task.yaml")
        instruction = str(payload.get("instruction") or payload.get("prompt") or "").strip()
        if not instruction:
            continue
        image = _terminal_bench_image_name(task_id)
        docker_config = _terminal_bench_docker_config(task_dir)
        rows.append(
            {
                "task_id": task_id,
                "instruction": instruction,
                "answer": "__OFFICIAL_TERMINAL_BENCH_TESTS__",
                "tools": [_agent_loop_final_answer_tool()],
                "expected_tool_calls": [
                    {
                        "name": "final_answer",
                        "arguments": {"answer": "__OFFICIAL_TERMINAL_BENCH_TESTS__"},
                        "argument_options": {"answer": ["__OFFICIAL_TERMINAL_BENCH_TESTS__"]},
                    }
                ],
                "executor": {
                    "kind": "shell_sandbox",
                    "config": {
                        "backend": "docker",
                        "image": image,
                        **docker_config,
                        "container_workdir": "/app",
                        "command_timeout_s": float(payload.get("max_agent_timeout_sec") or 900.0),
                    },
                },
                "verifier": {
                    "kind": "terminal_bench_official",
                    "config": {
                        "official_task_id": task_id,
                        "test_timeout_s": float(payload.get("max_test_timeout_sec") or 180.0),
                    },
                },
                "metadata": {
                    "source_benchmark": "terminal_bench_2_1",
                    "official_task_id": task_id,
                    "docker_image": image,
                    **docker_config,
                    "difficulty": payload.get("difficulty"),
                    "category": payload.get("category"),
                    "parser_name": payload.get("parser_name"),
                },
                "official_payload": payload,
            }
        )
    return rows


def _agent_loop_final_answer_tool() -> dict[str, Any]:
    return {
        "name": "final_answer",
        "description": "Submit the final answer for the benchmark item.",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    }


def _read_terminal_bench_task_yaml(path: Path) -> dict[str, Any]:
    return _read_yaml_mapping(path)


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - project env includes PyYAML in practice.
        raise RuntimeError("Terminal-Bench YAML materialization requires PyYAML") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        return {}
    return dict(data)


def _terminal_bench_image_name(task_id: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in task_id).strip("-")
    safe = "-".join(part for part in safe.split("-") if part)
    return f"rwkv-terminal-bench:{safe or 'task'}"


def _nl2repo_official_root() -> Path | None:
    raw = os.environ.get(_NL2REPO_ROOT_ENV)
    if not raw:
        return None
    root = Path(raw).expanduser()
    return root if (root / "test_files").is_dir() else None


def _nl2repo_official_project_dirs() -> list[Path]:
    root = _nl2repo_official_root()
    if root is None:
        return []
    return sorted(
        path
        for path in (root / "test_files").iterdir()
        if path.is_dir()
        and (path / "start.md").is_file()
        and (path / "test_commands.json").is_file()
        and (path / "test_files.json").is_file()
    )


def _nl2repo_official_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for project_dir in _nl2repo_official_project_dirs():
        project = project_dir.name
        instruction = project_dir.joinpath("start.md").read_text(encoding="utf-8").strip()
        test_commands = _read_json_list(project_dir / "test_commands.json")
        test_files = _read_json_list(project_dir / "test_files.json")
        test_case_count = _read_int_file(project_dir / "test_case_count.txt")
        rows.append(
            {
                "task_id": project,
                "instruction": (
                    f"{instruction}\n\n"
                    "Create the complete project in the current workspace using the available shell/file tools. "
                    "When the implementation is ready, call final_answer with a short completion note."
                ),
                "answer": "__OFFICIAL_NL2REPO_TESTS__",
                "tools": [_agent_loop_final_answer_tool()],
                "expected_tool_calls": [
                    {
                        "name": "final_answer",
                        "arguments": {"answer": "__OFFICIAL_NL2REPO_TESTS__"},
                        "argument_options": {"answer": ["__OFFICIAL_NL2REPO_TESTS__"]},
                    }
                ],
                "executor": {
                    "kind": "shell_sandbox",
                    "config": {
                        "backend": "subprocess",
                        "command_timeout_s": 900.0,
                        "max_output_chars": 12000,
                    },
                },
                "verifier": {
                    "kind": "nl2repo_official",
                    "config": {
                        "official_task_id": project,
                        "test_timeout_s": 10800.0,
                    },
                },
                "metadata": {
                    "source_benchmark": "nl2repo",
                    "official_task_id": project,
                    "test_case_count": test_case_count,
                    "test_commands": test_commands,
                    "test_files": test_files,
                    "official_project_dir": str(project_dir),
                },
                "official_payload": {
                    "project": project,
                    "test_case_count": test_case_count,
                    "test_commands": test_commands,
                    "test_files": test_files,
                },
            }
        )
    return rows


def _deepswe_official_root() -> Path | None:
    raw = os.environ.get(_DEEPSWE_ROOT_ENV)
    if not raw:
        return None
    root = Path(raw).expanduser()
    return root if (root / "tasks").is_dir() else None


def _deepswe_official_task_dirs() -> list[Path]:
    root = _deepswe_official_root()
    if root is None:
        return []
    return sorted(
        path
        for path in (root / "tasks").iterdir()
        if path.is_dir()
        and (path / "task.toml").is_file()
        and (path / "instruction.md").is_file()
        and (path / "pre_artifacts.sh").is_file()
        and (path / "tests" / "test.sh").is_file()
        and (path / "tests" / "grader.py").is_file()
    )


def _deepswe_official_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_dir in _deepswe_official_task_dirs():
        task_toml = _read_toml_mapping(task_dir / "task.toml")
        metadata = dict(task_toml.get("metadata") or {})
        environment = dict(task_toml.get("environment") or {})
        verifier = dict(task_toml.get("verifier") or {})
        agent = dict(task_toml.get("agent") or {})
        task_id = str(metadata.get("task_id") or task_dir.name)
        image = str(environment.get("docker_image") or "").strip()
        if not image:
            continue
        instruction = (task_dir / "instruction.md").read_text(encoding="utf-8").strip()
        base_commit = str(metadata.get("base_commit_hash") or "")
        rows.append(
            {
                "task_id": task_id,
                "instruction": instruction,
                "answer": "__OFFICIAL_DEEPSWE_TESTS__",
                "tools": [_agent_loop_final_answer_tool()],
                "expected_tool_calls": [
                    {
                        "name": "final_answer",
                        "arguments": {"answer": "__OFFICIAL_DEEPSWE_TESTS__"},
                        "argument_options": {"answer": ["__OFFICIAL_DEEPSWE_TESTS__"]},
                    }
                ],
                "executor": {
                    "kind": "shell_sandbox",
                    "config": {
                        "backend": "docker",
                        "image": image,
                        "container_workdir": "/app",
                        "command_timeout_s": 900.0,
                        "max_output_chars": 12000,
                        "docker_copy_paths": [
                            {"src": str(task_dir / "tests"), "dst": "/tests"},
                            {"src": str(task_dir / "pre_artifacts.sh"), "dst": "/pre_artifacts.sh"},
                        ],
                        "setup_commands": ["chmod +x /pre_artifacts.sh /tests/test.sh"],
                    },
                },
                "verifier": {
                    "kind": "repo_tests_official",
                    "config": {
                        "test_command": "bash /pre_artifacts.sh && bash /tests/test.sh",
                        "test_timeout_s": float(verifier.get("timeout_sec") or 1800.0),
                    },
                },
                "metadata": {
                    "source_benchmark": "deepswe",
                    "official_task_id": task_id,
                    "display_title": metadata.get("display_title"),
                    "repo": metadata.get("repository_url"),
                    "language": metadata.get("language"),
                    "category": metadata.get("category"),
                    "base_commit_hash": base_commit,
                    "docker_image": image,
                    "official_task_dir": str(task_dir),
                    "agent_timeout_s": float(agent.get("timeout_sec") or 5400.0),
                },
                "official_payload": task_toml,
            }
        )
    return rows


def _read_toml_mapping(path: Path) -> dict[str, Any]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


def _read_json_list(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    raise ValueError(f"expected JSON list in {path}")


def _read_int_file(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


def _register(name: str):
    @FUNCTION_CALLING_REGISTRY.register_spec(name)
    def _prepare(output_root: Path, split: str = "test") -> AgentLoopDatasetSpec:
        return AgentLoopDatasetSpec(output_root, split, name=name)

    return _prepare


for _dataset_name in (*_AGENT_LOOP_SOURCES, *_ALIASES):
    _register(_dataset_name)


__all__ = [
    "AgentLoopDatasetSpec",
    "AgentLoopProfile",
    "normalize_agent_loop_row",
]
