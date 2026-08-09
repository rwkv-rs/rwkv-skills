from __future__ import annotations

"""Prepare LiveCodeBench (code generation) datasets as JSONL."""

import os
from pathlib import Path
from typing import Any
from typing import Iterable, Mapping

from ..data_utils import configure_hf_home
from src.eval.datasets.data_prepper.prepper_registry import CODE_GENERATION_REGISTRY
from src.eval.datasets.runtime import MaterializingDatasetSpec

_LITE_DATASET_ID = "livecodebench/code_generation_lite"
_DATASET_CONFIG = "release_latest"
# Pin to the latest known version tag to keep the scheduler reproducible and
# compatible with offline HuggingFace caches.
_DEFAULT_VERSION_TAG = "release_v6"
_REQUIRED_FIELDS = ("task_id", "prompt")


def _iter_livecodebench_records(
    dataset: Iterable[Mapping[str, object]],
    *,
    release_version: str,
    source_dataset: str,
) -> Iterable[dict[str, Any]]:
    for row in dataset:
        payload = dict(row)
        question_id = str(payload.get("question_id", "") or "")
        prompt = payload.get("question_content") or payload.get("question_title") or ""
        payload["task_id"] = question_id
        payload["prompt"] = str(prompt)
        payload.setdefault("question_id", question_id)
        payload["release_version"] = release_version
        payload["source_dataset"] = source_dataset
        yield payload


def _load_livecodebench_rows(split: str, version_tag: str) -> list[Mapping[str, Any]]:
    if split != "test":
        raise ValueError("livecodebench 仅提供 test split")
    configure_hf_home()
    try:
        from datasets import load_dataset  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "需要安装 `datasets` 才能准备 livecodebench 数据集，请运行 `pip install datasets`"
        ) from exc
    resolved_version_tag = (os.environ.get("RWKV_SKILLS_LIVECODEBENCH_VERSION_TAG") or version_tag).strip()
    dataset = load_dataset(
        path=_LITE_DATASET_ID,
        name=_DATASET_CONFIG,
        split=split,
        trust_remote_code=True,
        version_tag=resolved_version_tag,
    )
    return sorted(dataset, key=lambda item: str(item.get("question_id", "")))


class LiveCodeBenchDatasetSpec(MaterializingDatasetSpec):
    def __init__(self, output_root: Path, split: str, *, version_tag: str = _DEFAULT_VERSION_TAG) -> None:
        super().__init__(
            "livecodebench",
            output_root,
            split,
            required_fields=_REQUIRED_FIELDS,
            source_kind="hf_load_dataset",
        )
        self._version_tag = version_tag

    def download(self) -> None:
        return None

    def load_records(self) -> Iterable[dict[str, Any]]:
        rows = _load_livecodebench_rows(self.split, self._version_tag)
        resolved_version_tag = (os.environ.get("RWKV_SKILLS_LIVECODEBENCH_VERSION_TAG") or self._version_tag).strip()
        return list(
            _iter_livecodebench_records(
                rows,
                release_version=resolved_version_tag,
                source_dataset=_LITE_DATASET_ID,
            )
        )

    def manifest_extra(self) -> dict[str, Any]:
        resolved_version_tag = (os.environ.get("RWKV_SKILLS_LIVECODEBENCH_VERSION_TAG") or self._version_tag).strip()
        return {
            "dataset_id": _LITE_DATASET_ID,
            "config": _DATASET_CONFIG,
            "source_split": self.split,
            "version_tag": resolved_version_tag,
        }


@CODE_GENERATION_REGISTRY.register_spec("livecodebench")
def prepare_livecodebench_spec(output_root: Path, split: str = "test") -> LiveCodeBenchDatasetSpec:
    return LiveCodeBenchDatasetSpec(output_root, split)


__all__ = ["prepare_livecodebench_spec"]
