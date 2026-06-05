from __future__ import annotations

"""Prepare SWE-bench datasets as JSONL for patch generation."""

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.eval.datasets.data_prepper.prepper_registry import CODE_GENERATION_REGISTRY
from src.eval.datasets.runtime import MaterializingDatasetSpec

from ..data_utils import configure_hf_home

_DATASETS = {
    "swe_bench": {
        "hf_dataset": "princeton-nlp/SWE-bench",
        "harness_dataset": "princeton-nlp/SWE-bench",
    },
    "swe_bench_lite": {
        "hf_dataset": "princeton-nlp/SWE-bench_Lite",
        "harness_dataset": "princeton-nlp/SWE-bench_Lite",
    },
    "swe_bench_verified": {
        "hf_dataset": "princeton-nlp/SWE-bench_Verified",
        "harness_dataset": "princeton-nlp/SWE-bench_Verified",
    },
    "swe_bench_lite_oracle": {
        "hf_dataset": "princeton-nlp/SWE-bench_Lite_oracle",
        "harness_dataset": "princeton-nlp/SWE-bench_Lite",
    },
    "swe_bench_lite_bm25_13k": {
        "hf_dataset": "princeton-nlp/SWE-bench_Lite_bm25_13K",
        "harness_dataset": "princeton-nlp/SWE-bench_Lite",
    },
}
_REQUIRED_FIELDS = ("task_id", "prompt", "instance_id")


def _load_swebench_rows(dataset_name: str, split: str) -> list[Mapping[str, Any]]:
    source_override = os.environ.get("RWKV_SKILLS_SWEBENCH_SOURCE", "").strip()
    if source_override:
        return _load_local_rows(Path(source_override).expanduser())

    configure_hf_home()
    try:
        from datasets import load_dataset  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError("Install `datasets` to prepare SWE-bench datasets: pip install datasets") from exc

    dataset = load_dataset(dataset_name, split=split)
    return sorted(dataset, key=lambda item: str(item.get("instance_id", "")))


def _load_local_rows(path: Path) -> list[Mapping[str, Any]]:
    if path.is_dir():
        candidates = sorted(path.glob("*.jsonl")) + sorted(path.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"no JSON/JSONL SWE-bench source files under {path}")
        path = candidates[0]
    if path.suffix.lower() == ".jsonl":
        rows: list[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if raw:
                    rows.append(json.loads(raw))
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping) and isinstance(payload.get("instances"), list):
        return payload["instances"]
    raise ValueError(f"unsupported SWE-bench local source format: {path}")


def _extract_context(row: Mapping[str, Any]) -> str:
    for key in (
        "retrieved_context",
        "context",
        "text",
        "file_context",
        "oracle_context",
        "bm25_context",
        "repo_context",
    ):
        value = row.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(value)
    return ""


def _normalize_swebench_row(
    row: Mapping[str, Any],
    *,
    source_dataset: str,
    harness_dataset: str,
) -> dict[str, Any]:
    instance_id = str(row.get("instance_id") or row.get("task_id") or row.get("id") or "").strip()
    if not instance_id:
        raise ValueError(f"SWE-bench row missing instance_id: {row}")
    problem_statement = str(row.get("problem_statement") or row.get("prompt") or row.get("issue") or "").strip()
    repo = str(row.get("repo") or "").strip()
    base_commit = str(row.get("base_commit") or "").strip()
    hints_text = str(row.get("hints_text") or "").strip()
    retrieved_context = _extract_context(row)
    return {
        "task_id": instance_id,
        "instance_id": instance_id,
        "prompt": problem_statement,
        "repo": repo,
        "base_commit": base_commit,
        "problem_statement": problem_statement,
        "hints_text": hints_text,
        "retrieved_context": retrieved_context,
        "source_dataset": source_dataset,
        "harness_dataset_name": harness_dataset,
        "patch": str(row.get("patch") or ""),
        "test_patch": str(row.get("test_patch") or ""),
        "FAIL_TO_PASS": row.get("FAIL_TO_PASS"),
        "PASS_TO_PASS": row.get("PASS_TO_PASS"),
        "environment_setup_commit": row.get("environment_setup_commit"),
        "difficulty": row.get("difficulty"),
    }


class SweBenchDatasetSpec(MaterializingDatasetSpec):
    def __init__(self, output_root: Path, split: str, *, name: str) -> None:
        if name not in _DATASETS:
            raise ValueError(f"unknown SWE-bench dataset alias: {name}")
        super().__init__(
            name,
            output_root,
            split,
            required_fields=_REQUIRED_FIELDS,
            source_kind="hf_load_dataset",
        )
        self._name = name
        self._spec = _DATASETS[name]

    def download(self) -> None:
        return None

    def load_records(self) -> Iterable[dict[str, Any]]:
        rows = _load_swebench_rows(str(self._spec["hf_dataset"]), self.split)
        return [
            _normalize_swebench_row(
                row,
                source_dataset=str(self._spec["hf_dataset"]),
                harness_dataset=str(self._spec["harness_dataset"]),
            )
            for row in rows
        ]

    def manifest_extra(self) -> dict[str, Any]:
        return {
            "dataset_id": self._spec["hf_dataset"],
            "harness_dataset_name": self._spec["harness_dataset"],
            "source_split": self.split,
        }


def _register(name: str):
    @CODE_GENERATION_REGISTRY.register_spec(name)
    def _prepare(output_root: Path, split: str = "test") -> SweBenchDatasetSpec:
        return SweBenchDatasetSpec(output_root, split, name=name)

    return _prepare


prepare_swe_bench_spec = _register("swe_bench")
prepare_swe_bench_lite_spec = _register("swe_bench_lite")
prepare_swe_bench_verified_spec = _register("swe_bench_verified")
prepare_swe_bench_lite_oracle_spec = _register("swe_bench_lite_oracle")
prepare_swe_bench_lite_bm25_13k_spec = _register("swe_bench_lite_bm25_13k")


__all__ = [
    "SweBenchDatasetSpec",
    "prepare_swe_bench_spec",
    "prepare_swe_bench_lite_spec",
    "prepare_swe_bench_verified_spec",
    "prepare_swe_bench_lite_oracle_spec",
    "prepare_swe_bench_lite_bm25_13k_spec",
]
