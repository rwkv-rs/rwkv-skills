from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.tasks.function_calling.longcodebench import (
    LONGCODEBENCH_HF_REPO,
    LONGCODEQA_ARCHIVE,
    load_longcodeqa_rows_from_source,
)

from .common import LocalRowsDatasetSpec, rwkv_rs_datasets_root

_REQUIRED_FIELDS = ("task_id", "prompt", "repo_text", "question", "correct_letter")


def longcodebench_source() -> Path:
    override = (
        os.environ.get("RWKV_LONGCODEQA_SOURCE")
        or os.environ.get("RWKV_LONGCODEBENCH_SOURCE")
        or os.environ.get("LONGCODEBENCH_SOURCE")
    )
    if override:
        return Path(override).expanduser().resolve()
    return (rwkv_rs_datasets_root() / "longcodebench" / LONGCODEQA_ARCHIVE).resolve()


def _load_hf_longcodeqa_rows() -> list[dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("LongCodeBench HF download requires huggingface_hub") from exc

    archive_path = hf_hub_download(
        repo_id=LONGCODEBENCH_HF_REPO,
        filename=LONGCODEQA_ARCHIVE,
        repo_type="dataset",
    )
    return load_longcodeqa_rows_from_source(archive_path)


@FUNCTION_CALLING_REGISTRY.register_spec("longcodeqa")
def prepare_longcodeqa_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    configured_source = longcodebench_source()

    def _load() -> list[dict[str, Any]]:
        source = longcodebench_source()
        if source.exists():
            return load_longcodeqa_rows_from_source(source)
        return _load_hf_longcodeqa_rows()

    def _required_paths() -> tuple[Path, ...]:
        source = longcodebench_source()
        return (source,) if source.exists() else ()

    return LocalRowsDatasetSpec(
        "longcodeqa",
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="local_or_hf_longcodeqa_manifest",
        required_paths=_required_paths,
        load_local_records=_load,
        extra={
            "benchmark": "LongCodeBench",
            "subset": "LongCodeQA",
            "source": "local_or_huggingface",
            "hf_repo": LONGCODEBENCH_HF_REPO,
            "hf_file": LONGCODEQA_ARCHIVE,
            "default_local_source": str(configured_source),
        },
    )


__all__ = [
    "longcodebench_source",
    "prepare_longcodeqa_spec",
]
