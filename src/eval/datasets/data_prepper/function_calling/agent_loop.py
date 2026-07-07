from __future__ import annotations

"""Prepare multi-turn agent-loop benchmark manifests.

Rows carry executor/verifier specs so the runner can drive the benchmark's
real environment and grade with its OFFICIAL verifier. Rows without explicit
specs are classified from their format: recorded tool outputs -> manifest
replay, rubric rows -> LLM judge, plain QA rows -> expected final_answer call
(the loop naturally degrades to a single turn, matching official single-turn
benchmarks).
"""

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
_ENV_PREFIX = "RWKV_AGENT_LOOP_SOURCE"
_REQUIRED_FIELDS = ("task_id", "instruction", "tools", "executor", "verifier")

_AGENT_LOOP_SOURCES: dict[str, dict[str, str | None]] = {
    item.benchmark_name: {"source": item.source_url, "source_kind": item.source_kind}
    for item in AGENT_LOOP_BENCHMARK_SOURCES
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
    # Search agents: recorded search-tool outputs, official/LLM judging.
    "widesearch": AgentLoopProfile(verifier_kind="widesearch_official"),
    "deepsearchqa": AgentLoopProfile(verifier_kind="llm_rubric_judge"),
    # MCP-server agents graded by their official evaluators.
    "mcp_atlas": AgentLoopProfile(executor_kind="mcp_worker", verifier_kind="mcp_atlas_official"),
    "toolathlon": AgentLoopProfile(executor_kind="mcp_worker", verifier_kind="toolathlon_official"),
    # Docker harnesses whose official graders are not wired yet (preflight-gated).
    "apex_agents": AgentLoopProfile(executor_kind="shell_sandbox", verifier_kind="unsupported_official"),
    "claweval": AgentLoopProfile(executor_kind="shell_sandbox", verifier_kind="unsupported_official"),
    "wildclawbench": AgentLoopProfile(executor_kind="shell_sandbox", verifier_kind="unsupported_official"),
    "skillsbench": AgentLoopProfile(executor_kind="shell_sandbox", verifier_kind="unsupported_official"),
    # Tool-augmented expert QA: recorded/row-provided tools, official-style LLM judging.
    "hle_with_tools": AgentLoopProfile(verifier_kind="llm_rubric_judge"),
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
        if name not in _AGENT_LOOP_SOURCES:
            raise ValueError(f"unknown agent-loop dataset alias: {name}")
        super().__init__(
            name,
            output_root,
            split,
            required_fields=_REQUIRED_FIELDS,
            source_kind="rwkvc_agent_loop_source",
        )
        self._source_path: Path | None = None

    def source_path(self) -> Path:
        if self._source_path is None:
            self._source_path = resolve_source_path(
                self.name,
                self.split,
                env_prefix=_ENV_PREFIX,
                root_envs=(_SOURCE_ROOT_ENV,),
                default_root=REPO_ROOT / "data" / "agent_loop_sources",
            )
        return self._source_path

    def download(self) -> None:
        path = self.source_path()
        if not path.exists():
            if _AGENT_LOOP_SOURCES[self.name].get("source_kind") == "hf_dataset":
                return None
            env_name = per_dataset_source_env(_ENV_PREFIX, self.name)
            raise FileNotFoundError(
                f"missing source for {self.name}:{self.split}: {path}. "
                f"Set {env_name}=<json/jsonl file or dir> or {_SOURCE_ROOT_ENV}=<root>. "
                "See docs/agent_loop.md."
            )

    def load_records(self) -> Iterable[dict[str, Any]]:
        path = self.source_path()
        info = _AGENT_LOOP_SOURCES[self.name]
        if path.exists():
            rows = read_source_rows(path)
            source_label = str(path)
        elif info.get("source_kind") == "hf_dataset":
            rows = load_hf_rows(str(info.get("source") or ""), self.split)
            source_label = str(info.get("source") or "")
        else:
            raise FileNotFoundError(path)
        return [
            normalize_agent_loop_row(row, dataset_name=self.name, index=index, source_path=source_label)
            for index, row in enumerate(rows)
        ]

    def manifest_extra(self) -> dict[str, Any]:
        info = _AGENT_LOOP_SOURCES[self.name]
        return {
            "source_path": str(self.source_path()),
            "source": info["source"],
            "source_kind": info["source_kind"],
            "input_contract": (
                "JSONL rows may carry explicit executor/verifier specs, recorded tool outputs, "
                "rubrics, or plain question/answer fields; each row is classified by format."
            ),
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

    return {
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


def _reference_answer_from_expected(expected_tool_calls: Iterable[Mapping[str, Any]]) -> str:
    for call in expected_tool_calls:
        if str(call.get("name") or "") == "final_answer":
            arguments = call.get("arguments")
            if isinstance(arguments, Mapping):
                answer = arguments.get("answer")
                if answer not in (None, ""):
                    return str(answer)
    return ""


def _register(name: str):
    @FUNCTION_CALLING_REGISTRY.register_spec(name)
    def _prepare(output_root: Path, split: str = "test") -> AgentLoopDatasetSpec:
        return AgentLoopDatasetSpec(output_root, split, name=name)

    return _prepare


for _dataset_name in _AGENT_LOOP_SOURCES:
    _register(_dataset_name)


__all__ = [
    "AgentLoopDatasetSpec",
    "AgentLoopProfile",
    "normalize_agent_loop_row",
]
