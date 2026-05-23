from __future__ import annotations

import os
from pathlib import Path

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALL_REGISTRY
from src.eval.function_calling.one_step.simple_tool_call import (
    load_bfcl_ast_rows_from_sources,
    load_toolalpaca_rows_from_source,
)
from src.eval.scheduler.config import REPO_ROOT

from ..data_utils import download_file, write_jsonl

_BFCL_BASE_URL = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"
_TOOLALPACA_BASE_URL = "https://raw.githubusercontent.com/tangqiaoyu/ToolAlpaca/main/data"

_BFCL_SMALL_CATEGORY_PATHS = {
    "bfcl_simple_python": (
        "simple_python",
        ("BFCL_v4_simple_python.json",),
        ("possible_answer", "BFCL_v4_simple_python.json"),
    ),
    "bfcl_multiple": (
        "multiple",
        ("BFCL_v4_multiple.json",),
        ("possible_answer", "BFCL_v4_multiple.json"),
    ),
    "bfcl_exec_simple": (
        "exec_simple",
        ("unused_datasets", "question", "BFCL_v4_exec_simple.json"),
        ("unused_datasets", "possible_answer", "BFCL_v4_exec_simple.json"),
    ),
    "bfcl_exec_multiple": (
        "exec_multiple",
        ("unused_datasets", "question", "BFCL_v4_exec_multiple.json"),
        ("unused_datasets", "possible_answer", "BFCL_v4_exec_multiple.json"),
    ),
    "bfcl_exec_parallel": (
        "exec_parallel",
        ("unused_datasets", "question", "BFCL_v4_exec_parallel.json"),
        ("unused_datasets", "possible_answer", "BFCL_v4_exec_parallel.json"),
    ),
    "bfcl_exec_parallel_multiple": (
        "exec_parallel_multiple",
        ("unused_datasets", "question", "BFCL_v4_exec_parallel_multiple.json"),
        ("unused_datasets", "possible_answer", "BFCL_v4_exec_parallel_multiple.json"),
    ),
}

_TOOLALPACA_FILES = {
    "toolalpaca_eval_simulated": "eval_simulated.json",
    "toolalpaca_eval_real": "eval_real.json",
}


def bfcl_small_source_root() -> Path:
    override = (
        os.environ.get("RWKV_BFCL_SMALL_SOURCE_ROOT")
        or os.environ.get("RWKV_BFCL_V4_SOURCE_ROOT")
        or os.environ.get("BFCL_V4_SOURCE_ROOT")
    )
    if override:
        return Path(override).expanduser().resolve()
    return _first_existing_root(
        (
            REPO_ROOT / "references" / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data",
            REPO_ROOT.parent
            / "GitHub"
            / "rwkv-skills"
            / "references"
            / "gorilla"
            / "berkeley-function-call-leaderboard"
            / "bfcl_eval"
            / "data",
            REPO_ROOT.parent / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data",
        )
    )


def toolalpaca_source_root() -> Path:
    override = os.environ.get("RWKV_TOOLALPACA_SOURCE_ROOT") or os.environ.get("TOOLALPACA_SOURCE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return _first_existing_root(
        (
            REPO_ROOT / "references" / "ToolAlpaca" / "data",
            REPO_ROOT.parent / "GitHub" / "rwkv-skills" / "references" / "ToolAlpaca" / "data",
            REPO_ROOT.parent / "ToolAlpaca" / "data",
        )
    )


def _first_existing_root(candidates: tuple[Path, ...]) -> Path:
    for path in candidates:
        if path.exists():
            return path.resolve()
    return candidates[0].resolve()


def _cache_path(output_root: Path, dataset_name: str, parts: tuple[str, ...]) -> Path:
    return output_root / "cache" / dataset_name / "__".join(parts)


def _resolve_bfcl_path(
    output_root: Path,
    dataset_name: str,
    parts: tuple[str, ...],
) -> Path:
    local_path = bfcl_small_source_root().joinpath(*parts).resolve()
    if local_path.exists():
        return local_path
    url = f"{_BFCL_BASE_URL}/{'/'.join(parts)}"
    return download_file(url, _cache_path(output_root, dataset_name, parts))


def _resolve_toolalpaca_path(output_root: Path, dataset_name: str, filename: str) -> Path:
    local_path = (toolalpaca_source_root() / filename).resolve()
    if local_path.exists():
        return local_path
    return download_file(
        f"{_TOOLALPACA_BASE_URL}/{filename}",
        _cache_path(output_root, dataset_name, (filename,)),
    )


def _prepare_bfcl_small(output_root: Path, dataset_name: str) -> list[Path]:
    category, question_parts, answer_parts = _BFCL_SMALL_CATEGORY_PATHS[dataset_name]
    question_path = _resolve_bfcl_path(output_root, dataset_name, question_parts)
    answer_path = _resolve_bfcl_path(output_root, dataset_name, answer_parts)
    rows = load_bfcl_ast_rows_from_sources(question_path, answer_path, category=category)
    target = output_root / dataset_name / "test.jsonl"
    write_jsonl(target, rows)
    return [target]


def _prepare_toolalpaca(output_root: Path, dataset_name: str) -> list[Path]:
    source_path = _resolve_toolalpaca_path(output_root, dataset_name, _TOOLALPACA_FILES[dataset_name])
    rows = load_toolalpaca_rows_from_source(source_path, dataset_name=dataset_name)
    target = output_root / dataset_name / "test.jsonl"
    write_jsonl(target, rows)
    return [target]


def _make_bfcl_small_preparer(dataset_name: str):
    def _prepare(output_root: Path, split: str = "test") -> list[Path]:
        if split != "test":
            raise ValueError(f"{dataset_name} only provides test split")
        return _prepare_bfcl_small(output_root, dataset_name)

    return _prepare


def _make_toolalpaca_preparer(dataset_name: str):
    def _prepare(output_root: Path, split: str = "test") -> list[Path]:
        if split != "test":
            raise ValueError(f"{dataset_name} only provides test split")
        return _prepare_toolalpaca(output_root, dataset_name)

    return _prepare


for _dataset_name in _BFCL_SMALL_CATEGORY_PATHS:
    FUNCTION_CALL_REGISTRY.register(_dataset_name)(_make_bfcl_small_preparer(_dataset_name))

for _dataset_name in _TOOLALPACA_FILES:
    FUNCTION_CALL_REGISTRY.register(_dataset_name)(_make_toolalpaca_preparer(_dataset_name))


__all__ = [
    "bfcl_small_source_root",
    "toolalpaca_source_root",
]
