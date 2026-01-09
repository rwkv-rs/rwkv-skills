from __future__ import annotations

"""Prepare LiveCodeBench (code generation) datasets as JSONL."""

from pathlib import Path
from typing import Iterable, Mapping

from datasets import load_dataset

from ..data_utils import configure_hf_home, write_jsonl
from src.eval.datasets.data_prepper.prepper_registry import CODE_GENERATION_REGISTRY

_LITE_DATASET_ID = "livecodebench/code_generation_lite"
_RELEASE_TAGS: dict[str, str] = {
    "livecodebench": "release_latest",
    "livecodebench_v1": "release_v1",
    "livecodebench_v2": "release_v2",
    "livecodebench_v3": "release_v3",
    "livecodebench_v4": "release_v4",
    "livecodebench_v5": "release_v5",
    "livecodebench_v6": "release_v6",
}


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
    release_version: str,
) -> list[Path]:
    if split != "test":
        raise ValueError(f"{dataset_name} 仅提供 test split")
    configure_hf_home()
    dataset_dir = (output_root / dataset_name).expanduser().resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"

    dataset_id = _LITE_DATASET_ID
    dataset = load_dataset(
        dataset_id,
        split=split,
        version_tag=release_version,
        trust_remote_code=True,
    )

    rows = sorted(dataset, key=lambda item: str(item.get("question_id", "")))
    write_jsonl(
        target,
        _iter_livecodebench_records(
            rows,
            release_version=release_version,
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
        release_version=_RELEASE_TAGS["livecodebench"],
    )


@CODE_GENERATION_REGISTRY.register("livecodebench_v1")
def prepare_livecodebench_v1(output_root: Path, split: str = "test") -> list[Path]:
    return _prepare_livecodebench(
        output_root,
        split=split,
        dataset_name="livecodebench_v1",
        release_version=_RELEASE_TAGS["livecodebench_v1"],
    )


@CODE_GENERATION_REGISTRY.register("livecodebench_v2")
def prepare_livecodebench_v2(output_root: Path, split: str = "test") -> list[Path]:
    return _prepare_livecodebench(
        output_root,
        split=split,
        dataset_name="livecodebench_v2",
        release_version=_RELEASE_TAGS["livecodebench_v2"],
    )


@CODE_GENERATION_REGISTRY.register("livecodebench_v3")
def prepare_livecodebench_v3(output_root: Path, split: str = "test") -> list[Path]:
    return _prepare_livecodebench(
        output_root,
        split=split,
        dataset_name="livecodebench_v3",
        release_version=_RELEASE_TAGS["livecodebench_v3"],
    )


@CODE_GENERATION_REGISTRY.register("livecodebench_v4")
def prepare_livecodebench_v4(output_root: Path, split: str = "test") -> list[Path]:
    return _prepare_livecodebench(
        output_root,
        split=split,
        dataset_name="livecodebench_v4",
        release_version=_RELEASE_TAGS["livecodebench_v4"],
    )


@CODE_GENERATION_REGISTRY.register("livecodebench_v5")
def prepare_livecodebench_v5(output_root: Path, split: str = "test") -> list[Path]:
    return _prepare_livecodebench(
        output_root,
        split=split,
        dataset_name="livecodebench_v5",
        release_version=_RELEASE_TAGS["livecodebench_v5"],
    )


@CODE_GENERATION_REGISTRY.register("livecodebench_v6")
def prepare_livecodebench_v6(output_root: Path, split: str = "test") -> list[Path]:
    return _prepare_livecodebench(
        output_root,
        split=split,
        dataset_name="livecodebench_v6",
        release_version=_RELEASE_TAGS["livecodebench_v6"],
    )


__all__ = [
    "prepare_livecodebench",
    "prepare_livecodebench_v1",
    "prepare_livecodebench_v2",
    "prepare_livecodebench_v3",
    "prepare_livecodebench_v4",
    "prepare_livecodebench_v5",
    "prepare_livecodebench_v6",
]
