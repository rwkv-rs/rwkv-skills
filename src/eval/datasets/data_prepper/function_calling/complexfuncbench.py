from __future__ import annotations

import os
from pathlib import Path

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.function_calling.complexfuncbench import (
    DEFAULT_COMPLEXFUNC_MAX_ROWS,
    load_complexfuncbench_rows_from_source,
)
from src.eval.scheduler.config import REPO_ROOT

from ..data_utils import download_file, write_jsonl

_COMPLEXFUNC_HF_URL = (
    "https://huggingface.co/datasets/zai-org/ComplexFuncBench/resolve/main/ComplexFuncBench.jsonl"
)


def complexfuncbench_official_root() -> Path | None:
    override = (
        os.environ.get("RWKV_COMPLEXFUNC_OFFICIAL_ROOT")
        or os.environ.get("RWKV_COMPLEXFUNCBENCH_OFFICIAL_ROOT")
    )
    if override:
        return Path(override).expanduser().resolve()
    return None


def complexfuncbench_source_root() -> Path:
    override = (
        os.environ.get("RWKV_COMPLEXFUNC_SOURCE_ROOT")
        or os.environ.get("RWKV_COMPLEXFUNCBENCH_SOURCE_ROOT")
        or os.environ.get("COMPLEXFUNCBENCH_SOURCE_ROOT")
    )
    if override:
        return Path(override).expanduser().resolve()
    for candidate in (
        REPO_ROOT / "references" / "ComplexFuncBench",
        REPO_ROOT.parent / "ComplexFuncBench",
    ):
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / "references" / "ComplexFuncBench").resolve()


def complexfuncbench_source_path(output_root: Path, dataset_name: str) -> Path:
    override = (
        os.environ.get("RWKV_COMPLEXFUNCBENCH_SOURCE")
        or os.environ.get("RWKV_COMPLEXFUNC_SOURCE")
        or os.environ.get("COMPLEXFUNCBENCH_SOURCE")
    )
    if override:
        return Path(override).expanduser().resolve()

    source_root = complexfuncbench_source_root()
    official_root = complexfuncbench_official_root()
    official_candidates = (
        (official_root / "data" / "ComplexFuncBench.jsonl", official_root / "ComplexFuncBench.jsonl")
        if official_root is not None
        else ()
    )
    for candidate in (
        source_root / "data" / "ComplexFuncBench.jsonl",
        source_root / "ComplexFuncBench.jsonl",
        *official_candidates,
        REPO_ROOT / "data" / "cache" / dataset_name / "ComplexFuncBench.jsonl",
        REPO_ROOT / "data" / "cache" / "complexfuncbench_subset" / "ComplexFuncBench.jsonl",
    ):
        if candidate.exists():
            return candidate.resolve()
    return download_file(
        _COMPLEXFUNC_HF_URL,
        output_root / "cache" / dataset_name / "ComplexFuncBench.jsonl",
    )


@FUNCTION_CALLING_REGISTRY.register("complexfuncbench_official")
def prepare_complexfuncbench_official(output_root: Path, split: str = "test") -> list[Path]:
    if split != "test":
        raise ValueError("complexfuncbench_official only provides test split")
    dataset_name = "complexfuncbench_official"
    source_path = complexfuncbench_source_path(output_root, dataset_name)
    official_root = complexfuncbench_official_root()
    max_rows = int(os.environ.get("RWKV_COMPLEXFUNCBENCH_MAX_ROWS", str(DEFAULT_COMPLEXFUNC_MAX_ROWS)))
    response_eval = str(os.environ.get("RWKV_COMPLEXFUNCBENCH_RESPONSE_EVAL", "1")).strip().lower()
    rows = load_complexfuncbench_rows_from_source(
        source_path,
        official_root=official_root,
        dataset_name=dataset_name,
        max_rows=max_rows,
        response_eval=response_eval not in {"0", "false", "no", "off"},
    )
    if not rows:
        raise ValueError(f"{source_path} did not yield any ComplexFuncBench official rows")
    target = output_root / dataset_name / "test.jsonl"
    write_jsonl(target, rows)
    return [target]


__all__ = [
    "complexfuncbench_official_root",
    "complexfuncbench_source_root",
    "complexfuncbench_source_path",
    "prepare_complexfuncbench_official",
]
