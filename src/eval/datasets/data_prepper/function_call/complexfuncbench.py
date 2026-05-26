from __future__ import annotations

import os
from pathlib import Path

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALL_REGISTRY
from src.eval.function_calling.one_step.complexfuncbench import (
    DEFAULT_COMPLEXFUNC_SUBSET_SIZE,
    load_complexfuncbench_subset_rows_from_source,
)
from src.eval.scheduler.config import REPO_ROOT

from ..data_utils import download_file, write_jsonl

_COMPLEXFUNC_HF_URL = (
    "https://huggingface.co/datasets/THUDM/ComplexFuncBench/resolve/main/ComplexFuncBench.jsonl"
)


def complexfuncbench_source_root() -> Path:
    override = (
        os.environ.get("RWKV_COMPLEXFUNC_SOURCE_ROOT")
        or os.environ.get("COMPLEXFUNC_SOURCE_ROOT")
        or os.environ.get("RWKV_COMPLEXFUNCBENCH_SOURCE_ROOT")
        or os.environ.get("COMPLEXFUNCBENCH_SOURCE_ROOT")
    )
    if override:
        return Path(override).expanduser().resolve()
    for candidate in (
        REPO_ROOT / "references" / "ComplexFuncBench",
        REPO_ROOT.parent / "GitHub" / "rwkv-skills" / "references" / "ComplexFuncBench",
        Path("/tmp/rwkv-official-refs/ComplexFuncBench"),
        REPO_ROOT.parent / "ComplexFuncBench",
    ):
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / "references" / "ComplexFuncBench").resolve()


def _source_path(output_root: Path, dataset_name: str) -> Path:
    root = complexfuncbench_source_root()
    for candidate in (
        root / "data" / "ComplexFuncBench.jsonl",
        root / "ComplexFuncBench.jsonl",
    ):
        if candidate.exists():
            return candidate.resolve()
    return download_file(
        _COMPLEXFUNC_HF_URL,
        output_root / "cache" / dataset_name / "ComplexFuncBench.jsonl",
    )


@FUNCTION_CALL_REGISTRY.register("complexfuncbench_subset")
def prepare_complexfuncbench_subset(output_root: Path, split: str = "test") -> list[Path]:
    if split != "test":
        raise ValueError("complexfuncbench_subset only provides test split")
    dataset_name = "complexfuncbench_subset"
    source_path = _source_path(output_root, dataset_name)
    max_rows = int(os.environ.get("RWKV_COMPLEXFUNC_SUBSET_SIZE", str(DEFAULT_COMPLEXFUNC_SUBSET_SIZE)))
    rows = load_complexfuncbench_subset_rows_from_source(
        source_path,
        dataset_name=dataset_name,
        max_rows=max_rows,
    )
    if not rows:
        raise ValueError(f"{source_path} did not yield any ComplexFuncBench subset rows")
    target = output_root / dataset_name / "test.jsonl"
    write_jsonl(target, rows)
    return [target]


__all__ = [
    "complexfuncbench_source_root",
    "prepare_complexfuncbench_subset",
]
