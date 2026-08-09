from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.eval.tasks.agent_bench.tasks import (
    TAU_V2_DATA_ROOT,
    load_tau_v2_tasks,
    require_tau_v3_source,
    tau_v2_data_root,
)
from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY

from .common import LocalRowsDatasetSpec

_REQUIRED_FIELDS = ("task_id", "instruction", "task", "benchmark_version")
_TAU3_LIGHT_VERSION = "tau_v3_light"
_MOCK_LIGHT_TASK_IDS = frozenset(
    {
        "create_task_1_with_env_assertions",
        "update_task_with_history_and_env_assertions",
        "impossible_task_1",
    }
)


def _tau_data_root() -> Path:
    if any(
        os.environ.get(name)
        for name in (
            "RWKV_TAU3_DATA_ROOT",
            "TAU3_DATA_ROOT",
            "RWKV_TAU2_DATA_ROOT",
            "TAU2_DATA_DIR",
            "RWKV_TAU3_BENCH_ROOT",
            "TAU3_BENCH_ROOT",
            "RWKV_TAU2_BENCH_ROOT",
            "TAU2_BENCH_ROOT",
        )
    ):
        return tau_v2_data_root()
    return Path(TAU_V2_DATA_ROOT)


def _tau_bench_spec(output_root: Path, *, dataset_name: str, domain: str, split: str) -> LocalRowsDatasetSpec:
    if split != "test":
        raise ValueError(f"{dataset_name} 仅提供 test split")

    def _load() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in load_tau_v2_tasks(domain=domain, split=split):
            payload = dict(row)
            payload["benchmark_version"] = "tau_bench"
            rows.append(payload)
        return rows

    return LocalRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="tau_v2_vendor_manifest",
        required_paths=lambda: (_tau_data_root(),),
        load_local_records=_load,
        extra={"domain": domain, "benchmark_version": "tau_bench"},
    )


def _tau_v2_spec(output_root: Path, *, dataset_name: str, domain: str, split: str) -> LocalRowsDatasetSpec:
    def _load() -> list[dict[str, Any]]:
        return load_tau_v2_tasks(domain=domain, split=split)

    return LocalRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="tau_v2_vendor_manifest",
        required_paths=lambda: (_tau_data_root(),),
        load_local_records=_load,
        extra={"domain": domain, "benchmark_version": "tau_v2"},
    )


def _tau_v3_spec(output_root: Path, *, dataset_name: str, domain: str, split: str) -> LocalRowsDatasetSpec:
    def _load() -> list[dict[str, Any]]:
        require_tau_v3_source(dataset_name)
        rows: list[dict[str, Any]] = []
        for row in load_tau_v2_tasks(domain=domain, split=split):
            payload = dict(row)
            payload["benchmark_version"] = "tau_v3"
            rows.append(payload)
        return rows

    return LocalRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="tau3_official_manifest",
        required_paths=lambda: (_tau_data_root(),),
        load_local_records=_load,
        extra={"domain": domain, "benchmark_version": "tau_v3"},
    )


def _is_tau3_light_split_supported(split: str) -> bool:
    return split in {"base", "test"}


def _tau_v3_light_mock_spec(output_root: Path, *, dataset_name: str, split: str) -> LocalRowsDatasetSpec:
    if not _is_tau3_light_split_supported(split):
        raise ValueError(f"{dataset_name} only provides base/test split aliases")

    def _load() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in load_tau_v2_tasks(domain="mock", split="base"):
            task_id = str(row.get("task_id") or "")
            if task_id not in _MOCK_LIGHT_TASK_IDS:
                continue
            rows.append(_sanitize_tau3_light_mock_row(row, index=len(rows)))
        return rows

    return LocalRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="tau3_lightweight_mock_manifest",
        required_paths=lambda: (_tau_data_root(),),
        load_local_records=_load,
        extra={"domain": "mock", "benchmark_version": _TAU3_LIGHT_VERSION},
    )


def _tau_v3_light_mock_long_context_spec(
    output_root: Path,
    *,
    dataset_name: str,
    split: str,
) -> LocalRowsDatasetSpec:
    if not _is_tau3_light_split_supported(split):
        raise ValueError(f"{dataset_name} only provides base/test split aliases")

    def _load() -> list[dict[str, Any]]:
        return _tau3_mock_long_context_rows()

    return LocalRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="tau3_lightweight_mock_long_context",
        required_paths=lambda: (_tau_data_root(),),
        load_local_records=_load,
        extra={"domain": "mock", "benchmark_version": _TAU3_LIGHT_VERSION},
    )


def _sanitize_tau3_light_mock_row(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    payload = deepcopy(row)
    task = payload.get("task")
    if not isinstance(task, dict):
        task = {}
    task = deepcopy(task)
    criteria = task.get("evaluation_criteria")
    if isinstance(criteria, dict):
        criteria.pop("nl_assertions", None)
        reward_basis = [str(item) for item in (criteria.get("reward_basis") or [])]
        if not reward_basis:
            if criteria.get("env_assertions"):
                reward_basis = ["ENV_ASSERTION"]
            elif criteria.get("actions"):
                reward_basis = ["ACTION"]
        criteria["reward_basis"] = [item for item in reward_basis if item != "NL_ASSERTION"]
    payload.update(
        {
            "task_id": str(task.get("id") or payload.get("task_id") or f"tau3_light_mock_{index}"),
            "domain": "mock",
            "index": int(index),
            "instruction": str(payload.get("instruction") or task.get("ticket") or ""),
            "task": task,
            "benchmark_version": _TAU3_LIGHT_VERSION,
        }
    )
    return payload


def _tau3_mock_long_context_rows() -> list[dict[str, Any]]:
    tasks = [
        _tau3_long_context_create_task(),
        _tau3_long_context_update_task(),
    ]
    return [
        {
            "task_id": str(task["id"]),
            "domain": "mock",
            "index": index,
            "instruction": str(task.get("ticket") or ""),
            "task": task,
            "benchmark_version": _TAU3_LIGHT_VERSION,
        }
        for index, task in enumerate(tasks)
    ]


def _long_archive(*, target_line: str, label: str) -> str:
    before = [
        f"{label} archive row {idx:03d}: unrelated task inventory, no current user request."
        for idx in range(90)
    ]
    after = [
        f"{label} archive appendix {idx:03d}: historical note, ignore unless directly relevant."
        for idx in range(90)
    ]
    return "\n".join(
        [
            "Reference archive for prior support sessions.",
            *before,
            target_line,
            *after,
            "End of reference archive.",
        ]
    )


def _tau3_long_context_create_task() -> dict[str, Any]:
    return {
        "id": "mock_long_context_create_task",
        "description": {
            "purpose": "Lightweight tau3-style long-context tool-use task.",
            "notes": "The current request is buried in a long prior archive before the final user turn.",
        },
        "user_scenario": {
            "persona": "Professional and direct communicator",
            "instructions": "Create one task for the user after reading the current request.",
        },
        "ticket": "Create a task named Important Meeting for user_1.",
        "initial_state": {
            "message_history": [
                {
                    "role": "user",
                    "content": _long_archive(
                        target_line="Current request evidence: user_1 needs a task titled Important Meeting.",
                        label="create",
                    ),
                    "turn_idx": 0,
                },
                {
                    "role": "assistant",
                    "content": "I have loaded the reference archive and will use only the current request.",
                    "turn_idx": 0,
                },
                {
                    "role": "user",
                    "content": "Please create a task titled Important Meeting for user_1 now.",
                    "turn_idx": 1,
                },
            ]
        },
        "evaluation_criteria": {
            "actions": [
                {
                    "action_id": "create_important_meeting",
                    "name": "create_task",
                    "arguments": {"user_id": "user_1", "title": "Important Meeting"},
                    "info": "Create the requested task.",
                }
            ],
            "env_assertions": [
                {
                    "env_type": "assistant",
                    "func_name": "assert_task_status",
                    "arguments": {"task_id": "task_2", "expected_status": "pending"},
                }
            ],
            "reward_basis": ["DB", "ENV_ASSERTION"],
        },
    }


def _tau3_long_context_update_task() -> dict[str, Any]:
    return {
        "id": "mock_long_context_update_task",
        "description": {
            "purpose": "Lightweight tau3-style long-context state update task.",
            "notes": "A previous tool call creates task_2; the current request asks the agent to update it.",
        },
        "user_scenario": {
            "persona": "Professional and direct communicator",
            "instructions": "Continue the task-management conversation and update the existing task.",
        },
        "ticket": "Mark task_2 as completed.",
        "initial_state": {
            "message_history": [
                {
                    "role": "user",
                    "content": "I need to create a task for the project review meeting.",
                    "turn_idx": 0,
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "create_task",
                            "arguments": {
                                "user_id": "user_1",
                                "title": "Project Review",
                                "description": "Review Q4 project status",
                            },
                        }
                    ],
                    "turn_idx": 0,
                },
                {
                    "role": "tool",
                    "id": "call_1",
                    "content": (
                        '{"task_id":"task_2","title":"Project Review",'
                        '"description":"Review Q4 project status","status":"pending"}'
                    ),
                    "turn_idx": 0,
                },
                {
                    "role": "assistant",
                    "content": "The Project Review task was created with ID task_2 and is pending.",
                    "turn_idx": 0,
                },
                {
                    "role": "user",
                    "content": _long_archive(
                        target_line="Current request evidence: task_2 must be marked completed.",
                        label="update",
                    ),
                    "turn_idx": 1,
                },
                {
                    "role": "user",
                    "content": "Please mark task_2 as completed now.",
                    "turn_idx": 2,
                },
            ]
        },
        "evaluation_criteria": {
            "actions": [
                {
                    "action_id": "complete_task_2",
                    "name": "update_task_status",
                    "arguments": {"task_id": "task_2", "status": "completed"},
                    "info": "Update task_2 to completed.",
                }
            ],
            "env_assertions": [
                {
                    "env_type": "assistant",
                    "func_name": "assert_task_status",
                    "arguments": {"task_id": "task_2", "expected_status": "completed"},
                }
            ],
            "reward_basis": ["DB", "ENV_ASSERTION", "ACTION"],
        },
    }


@FUNCTION_CALLING_REGISTRY.register_spec("tau_bench_retail")
def prepare_tau_bench_retail_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    return _tau_bench_spec(output_root, dataset_name="tau_bench_retail", domain="retail", split=split)


@FUNCTION_CALLING_REGISTRY.register_spec("tau_bench_airline")
def prepare_tau_bench_airline_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    return _tau_bench_spec(output_root, dataset_name="tau_bench_airline", domain="airline", split=split)


@FUNCTION_CALLING_REGISTRY.register_spec("tau_bench_telecom")
def prepare_tau_bench_telecom_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    return _tau_bench_spec(output_root, dataset_name="tau_bench_telecom", domain="telecom", split=split)


@FUNCTION_CALLING_REGISTRY.register_spec("tau2_bench_retail")
def prepare_tau2_bench_retail_spec(output_root: Path, split: str = "base") -> LocalRowsDatasetSpec:
    return _tau_v2_spec(output_root, dataset_name="tau2_bench_retail", domain="retail", split=split)


@FUNCTION_CALLING_REGISTRY.register_spec("tau2_bench_airline")
def prepare_tau2_bench_airline_spec(output_root: Path, split: str = "base") -> LocalRowsDatasetSpec:
    return _tau_v2_spec(output_root, dataset_name="tau2_bench_airline", domain="airline", split=split)


@FUNCTION_CALLING_REGISTRY.register_spec("tau2_bench_telecom")
def prepare_tau2_bench_telecom_spec(output_root: Path, split: str = "base") -> LocalRowsDatasetSpec:
    return _tau_v2_spec(output_root, dataset_name="tau2_bench_telecom", domain="telecom", split=split)


@FUNCTION_CALLING_REGISTRY.register_spec("tau3_bench_retail")
def prepare_tau3_bench_retail_spec(output_root: Path, split: str = "base") -> LocalRowsDatasetSpec:
    return _tau_v3_spec(output_root, dataset_name="tau3_bench_retail", domain="retail", split=split)


@FUNCTION_CALLING_REGISTRY.register_spec("tau3_bench_airline")
def prepare_tau3_bench_airline_spec(output_root: Path, split: str = "base") -> LocalRowsDatasetSpec:
    return _tau_v3_spec(output_root, dataset_name="tau3_bench_airline", domain="airline", split=split)


@FUNCTION_CALLING_REGISTRY.register_spec("tau3_bench_telecom")
def prepare_tau3_bench_telecom_spec(output_root: Path, split: str = "base") -> LocalRowsDatasetSpec:
    return _tau_v3_spec(output_root, dataset_name="tau3_bench_telecom", domain="telecom", split=split)


@FUNCTION_CALLING_REGISTRY.register_spec("tau3_bench_banking_knowledge")
def prepare_tau3_bench_banking_knowledge_spec(output_root: Path, split: str = "base") -> LocalRowsDatasetSpec:
    return _tau_v3_spec(
        output_root,
        dataset_name="tau3_bench_banking_knowledge",
        domain="banking_knowledge",
        split=split,
    )


@FUNCTION_CALLING_REGISTRY.register_spec("tau3_bench_mock")
def prepare_tau3_bench_mock_spec(output_root: Path, split: str = "base") -> LocalRowsDatasetSpec:
    return _tau_v3_light_mock_spec(output_root, dataset_name="tau3_bench_mock", split=split)


@FUNCTION_CALLING_REGISTRY.register_spec("tau3_bench_mock_long_context")
def prepare_tau3_bench_mock_long_context_spec(output_root: Path, split: str = "base") -> LocalRowsDatasetSpec:
    return _tau_v3_light_mock_long_context_spec(
        output_root,
        dataset_name="tau3_bench_mock_long_context",
        split=split,
    )


__all__ = [
    "prepare_tau_bench_retail_spec",
    "prepare_tau_bench_airline_spec",
    "prepare_tau_bench_telecom_spec",
    "prepare_tau2_bench_retail_spec",
    "prepare_tau2_bench_airline_spec",
    "prepare_tau2_bench_telecom_spec",
    "prepare_tau3_bench_retail_spec",
    "prepare_tau3_bench_airline_spec",
    "prepare_tau3_bench_telecom_spec",
    "prepare_tau3_bench_banking_knowledge_spec",
    "prepare_tau3_bench_mock_spec",
    "prepare_tau3_bench_mock_long_context_spec",
]
