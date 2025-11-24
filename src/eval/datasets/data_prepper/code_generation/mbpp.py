from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping

from evalplus.data import get_mbpp_plus

from ..data_utils import write_jsonl
from src.dataset.data_prepper.prepper_registry import CODE_GENERATION_REGISTRY

_QUESTION_REPLACE = ("    ", "\t")


def _iter_mbpp_records(problems: Mapping[str, dict]) -> Iterable[dict]:
    for task_id, problem in problems.items():
        payload = dict(problem)
        prompt = payload.get("prompt") or payload.get("question") or ""
        if isinstance(prompt, str):
            payload["prompt"] = prompt
            payload["question"] = prompt.replace(*_QUESTION_REPLACE)
        else:
            payload["prompt"] = ""
            payload["question"] = ""
        payload.setdefault("task_id", str(task_id))
        payload.pop("base_input", None)
        payload.pop("plus_input", None)
        yield payload


@CODE_GENERATION_REGISTRY.register("mbpp")
def prepare_mbpp(output_root: Path, split: str = "test") -> list[Path]:
    if split != "test":
        raise ValueError("mbpp 目前仅提供 test split")
    dataset_dir = (output_root / "mbpp").expanduser().resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cache_root = (output_root / "cache").expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    target = dataset_dir / f"{split}.jsonl"
    problems = get_mbpp_plus()
    write_jsonl(target, _iter_mbpp_records(problems))
    return [target]


__all__ = ["prepare_mbpp"]
