from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from typing import Iterable, Mapping

from src.eval.datasets.data_prepper.prepper_registry import CODE_GENERATION_REGISTRY
from src.eval.datasets.runtime import MaterializingDatasetSpec

_QUESTION_REPLACE = ("    ", "\t")
_REQUIRED_FIELDS = ("task_id", "prompt")


def _configure_evalplus_cache(cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    try:
        import evalplus.data.utils as evalplus_data_utils  # pyright: ignore[reportMissingImports]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError("需要安装 `evalplus` 才能准备 MBPP 数据集，请运行 `pip install evalplus`") from exc
    evalplus_data_utils.CACHE_DIR = str(cache_root)


def _load_mbpp_problems(*, plus: bool) -> Mapping[str, dict[str, Any]]:
    try:
        from evalplus.data.mbpp import get_mbpp, get_mbpp_plus  # pyright: ignore[reportMissingImports]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError("需要安装 `evalplus` 才能准备 MBPP 数据集，请运行 `pip install evalplus`") from exc
    return get_mbpp_plus() if plus else get_mbpp()


def _iter_mbpp_records(problems: Mapping[str, dict[str, Any]], *, keep_plus_inputs: bool) -> Iterable[dict[str, Any]]:
    def _normalize_jsonable(value: Any) -> Any:
        if isinstance(value, complex):
            return str(value)
        if isinstance(value, dict):
            return {k: _normalize_jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_normalize_jsonable(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_normalize_jsonable(v) for v in value)
        return value

    for task_id, problem in problems.items():
        payload: dict[str, Any] = _normalize_jsonable(dict(problem))
        prompt = payload.get("prompt") or payload.get("question") or ""
        if isinstance(prompt, str):
            prompt_tab = prompt.replace(*_QUESTION_REPLACE)
            payload["prompt"] = prompt_tab
            payload["question"] = prompt_tab
        else:
            payload["prompt"] = ""
            payload["question"] = ""
        payload.setdefault("task_id", str(task_id))
        if not keep_plus_inputs:
            payload.pop("base_input", None)
            payload.pop("plus_input", None)
        yield payload


class MbppDatasetSpec(MaterializingDatasetSpec):
    def __init__(self, name: str, output_root: Path, split: str, *, keep_plus_inputs: bool) -> None:
        super().__init__(name, output_root, split, required_fields=_REQUIRED_FIELDS, source_kind="evalplus")
        self._keep_plus_inputs = keep_plus_inputs

    def download(self) -> None:
        return None

    def load_records(self) -> Iterable[dict[str, Any]]:
        if self.split != "test":
            raise ValueError(f"{self.name} 目前仅提供 test split")
        _configure_evalplus_cache(self.context.cache_root)
        problems = _load_mbpp_problems(plus=self._keep_plus_inputs)
        return list(_iter_mbpp_records(problems, keep_plus_inputs=self._keep_plus_inputs))

    def manifest_extra(self) -> dict[str, Any]:
        return {"source_split": self.split, "keep_plus_inputs": self._keep_plus_inputs}


@CODE_GENERATION_REGISTRY.register_spec("mbpp")
def prepare_mbpp_spec(output_root: Path, split: str = "test") -> MbppDatasetSpec:
    return MbppDatasetSpec("mbpp", output_root, split, keep_plus_inputs=False)


@CODE_GENERATION_REGISTRY.register_spec("mbpp_plus")
def prepare_mbpp_plus_spec(output_root: Path, split: str = "test") -> MbppDatasetSpec:
    return MbppDatasetSpec("mbpp_plus", output_root, split, keep_plus_inputs=True)


__all__ = ["prepare_mbpp_plus_spec", "prepare_mbpp_spec"]
