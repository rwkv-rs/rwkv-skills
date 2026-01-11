from __future__ import annotations

"""Prepare LiveCodeBench (code generation) datasets as JSONL."""

import os
from pathlib import Path
from typing import Iterable, Mapping

from datasets import load_dataset

from ..data_utils import configure_hf_home, write_jsonl
from src.eval.datasets.data_prepper.prepper_registry import CODE_GENERATION_REGISTRY

_LITE_DATASET_ID = "livecodebench/code_generation_lite"
_DATASET_CONFIG = "release_latest"
# Pin to the latest known version tag to keep the scheduler reproducible and
# compatible with offline HuggingFace caches.
_DEFAULT_VERSION_TAG = "release_v6"


def _iter_livecodebench_records(
    dataset: Iterable[Mapping[str, object]],
    *,
    release_version: str,
    source_dataset: str,
) -> Iterable[dict]:
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


def _prepare_livecodebench(
    output_root: Path,
    *,
    split: str,
    dataset_name: str,
    version_tag: str,
) -> list[Path]:
    if split != "test":
        raise ValueError(f"{dataset_name} 仅提供 test split")
    configure_hf_home()
    dataset_dir = (output_root / dataset_name).expanduser().resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"

    dataset_id = _LITE_DATASET_ID
    version_tag = (os.environ.get("RWKV_SKILLS_LIVECODEBENCH_VERSION_TAG") or version_tag).strip()
    dataset = load_dataset(
        dataset_id,
        _DATASET_CONFIG,
        split=split,
        version_tag=version_tag,
        trust_remote_code=True,
    )

    rows = sorted(dataset, key=lambda item: str(item.get("question_id", "")))
    write_jsonl(
        target,
        _iter_livecodebench_records(
            rows,
            release_version=version_tag,
            source_dataset=dataset_id,
        ),
    )
    return [target]


@CODE_GENERATION_REGISTRY.register("livecodebench")
def prepare_livecodebench(output_root: Path, split: str = "test") -> list[Path]:
    return _prepare_livecodebench(
        output_root,
        split=split,
        dataset_name="livecodebench",
        version_tag=_DEFAULT_VERSION_TAG,
    )

__all__ = ["prepare_livecodebench"]
