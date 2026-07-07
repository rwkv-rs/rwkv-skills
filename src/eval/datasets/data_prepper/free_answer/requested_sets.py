from __future__ import annotations

"""Free-answer preppers for the requested benchmark set (MathArena family,
science QA, rubric/long-context sets, and internal math benchmarks).

Each benchmark's official form is a free response graded by an answer parser
or an LLM judge, so rows normalize to ``problem`` + ``expected_answer``.
Sources resolve local-first (``RWKV_FREE_ANSWER_SOURCE_<NAME>`` or
``RWKV_FREE_ANSWER_SOURCE_ROOT``), then fall back to the official Hugging Face
dataset when one exists.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.eval.benchmark_sources import FREE_ANSWER_BENCHMARK_SOURCES
from src.eval.datasets.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY
from src.eval.datasets.data_prepper.source_files import (
    load_hf_rows,
    per_dataset_source_env,
    read_source_rows,
    resolve_source_path,
)
from src.eval.datasets.runtime import MaterializingDatasetSpec
from src.eval.scheduler.config import REPO_ROOT

_SOURCE_ROOT_ENV = "RWKV_FREE_ANSWER_SOURCE_ROOT"
_ENV_PREFIX = "RWKV_FREE_ANSWER_SOURCE"
_REQUIRED_FIELDS = ("problem", "expected_answer")

_QUESTION_KEYS = ("problem", "question", "prompt", "instruction", "query", "task", "input", "text")
_ANSWER_KEYS = (
    "expected_answer",
    "answer",
    "final_answer",
    "reference_answer",
    "gold_answer",
    "reference_solution",
    "solution",
    "target",
    "output",
)
_CONTEXT_KEYS = ("context", "source_context", "document", "documents", "passage", "passages")
_RUBRIC_KEYS = ("rubrics", "rubric", "grading_rubrics", "checklist")


@dataclass(frozen=True, slots=True)
class RequestedFreeAnswerSpec:
    name: str
    hf_dataset_id: str | None = None
    hf_split: str = "train"


_REQUESTED_FREE_ANSWER_SPECS: dict[str, RequestedFreeAnswerSpec] = {
    spec.name: spec
    for spec in (
        # MathArena family: official grading is automatic final-answer parsing;
        # USAMO 2026 is proof-based and officially judge-graded.
        RequestedFreeAnswerSpec("usamo_2026", hf_dataset_id="MathArena/usamo_2026"),
        RequestedFreeAnswerSpec("matharena_apex", hf_dataset_id="MathArena/apex"),
        RequestedFreeAnswerSpec("arxivmath", hf_dataset_id="MathArena/arxivmath"),
        RequestedFreeAnswerSpec("horizonmath"),
        RequestedFreeAnswerSpec("hy_math"),
        RequestedFreeAnswerSpec("hy_euler_pro"),
        RequestedFreeAnswerSpec("imoanswerbench"),
        # Science QA (officially judge/EED graded).
        RequestedFreeAnswerSpec("phybench", hf_dataset_id="Eureka-Lab/PHYBench"),
        RequestedFreeAnswerSpec("cmt_benchmark"),
        RequestedFreeAnswerSpec("superchem"),
        # Rubric / long-context sets (officially LLM-judge graded).
        RequestedFreeAnswerSpec("cl_bench", hf_dataset_id="tencent/CL-bench", hf_split="test"),
        RequestedFreeAnswerSpec("cl_bench_life", hf_dataset_id="tencent/CL-bench-Life", hf_split="test"),
        RequestedFreeAnswerSpec("aa_lcr", hf_dataset_id="ArtificialAnalysis/AA-LCR", hf_split="test"),
    )
}

_CATALOG_NAMES = {item.benchmark_name for item in FREE_ANSWER_BENCHMARK_SOURCES}
_MISSING_SPECS = _CATALOG_NAMES - set(_REQUESTED_FREE_ANSWER_SPECS) - {"hle"}
if _MISSING_SPECS:  # pragma: no cover - guards future benchmark_sources edits
    raise RuntimeError(f"free-answer benchmarks missing prepper specs: {sorted(_MISSING_SPECS)}")


class RequestedFreeAnswerDatasetSpec(MaterializingDatasetSpec):
    def __init__(self, output_root: Path, split: str, *, name: str) -> None:
        if name not in _REQUESTED_FREE_ANSWER_SPECS:
            raise ValueError(f"unknown requested free-answer dataset: {name}")
        super().__init__(
            name,
            output_root,
            split,
            required_fields=_REQUIRED_FIELDS,
            source_kind="rwkvc_free_answer_source",
        )
        self._spec = _REQUESTED_FREE_ANSWER_SPECS[name]
        self._source_path: Path | None = None

    def source_path(self) -> Path:
        if self._source_path is None:
            self._source_path = resolve_source_path(
                self.name,
                self.split,
                env_prefix=_ENV_PREFIX,
                root_envs=(_SOURCE_ROOT_ENV,),
                default_root=REPO_ROOT / "data" / "free_answer_sources",
            )
        return self._source_path

    def download(self) -> None:
        path = self.source_path()
        if not path.exists() and not self._spec.hf_dataset_id:
            env_name = per_dataset_source_env(_ENV_PREFIX, self.name)
            raise FileNotFoundError(
                f"missing source for {self.name}:{self.split}: {path}. "
                f"Set {env_name}=<json/jsonl file or dir> or {_SOURCE_ROOT_ENV}=<root>."
            )

    def load_records(self) -> Iterable[dict[str, Any]]:
        path = self.source_path()
        if path.exists():
            rows = read_source_rows(path)
            source_label = str(path)
        elif self._spec.hf_dataset_id:
            rows = load_hf_rows(self._spec.hf_dataset_id, self._spec.hf_split)
            source_label = self._spec.hf_dataset_id
        else:
            raise FileNotFoundError(path)
        return [
            normalize_free_answer_row(row, dataset_name=self.name, index=index, source_path=source_label)
            for index, row in enumerate(rows)
        ]

    def manifest_extra(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path()),
            "hf_dataset_id": self._spec.hf_dataset_id,
            "input_contract": "Rows normalize to problem/expected_answer; rubric rows keep rubrics in metadata.",
        }


def normalize_free_answer_row(
    row: Mapping[str, Any],
    *,
    dataset_name: str,
    index: int,
    source_path: str | Path,
) -> dict[str, Any]:
    problem = _problem_text(row)
    if not problem:
        raise ValueError(f"free-answer row missing problem/question: {row}")
    rubrics = _rubric_lines(row)
    expected = _expected_answer_text(row)
    if expected is None:
        if rubrics:
            expected = "\n".join(rubrics)
        else:
            raise ValueError(f"free-answer row missing expected answer: {row}")

    used = {"id", "task_id", "messages", *_QUESTION_KEYS, *_ANSWER_KEYS, *_CONTEXT_KEYS, *_RUBRIC_KEYS}
    payload: dict[str, Any] = {
        "id": _row_id(row, dataset_name, index),
        "problem": problem,
        "expected_answer": expected,
        "source_benchmark": dataset_name,
        "source_path": str(source_path),
    }
    if rubrics:
        payload["rubrics"] = rubrics
    for key, value in row.items():
        if key not in used and key not in payload and _is_json_scalarish(value):
            payload[key] = value
    return payload


def _problem_text(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        rendered = _render_messages(messages)
        if rendered:
            return rendered
    question = _first_present(row, _QUESTION_KEYS)
    if question is None:
        return ""
    question_text = _stringify(question)
    context = _first_present(row, _CONTEXT_KEYS)
    if context not in (None, ""):
        return f"Context:\n{_stringify(context)}\n\nQuestion:\n{question_text}"
    return question_text


def _render_messages(messages: Sequence[Any]) -> str:
    parts: list[str] = []
    for message in messages:
        if isinstance(message, Mapping):
            role = str(message.get("role") or "user").strip().title() or "User"
            content = str(message.get("content") or message.get("text") or "").strip()
            if content and role.lower() != "assistant":
                parts.append(f"{role}: {content}")
        elif str(message or "").strip():
            parts.append(str(message).strip())
    return "\n\n".join(parts).strip()


def _expected_answer_text(row: Mapping[str, Any]) -> str | None:
    answer = _first_present(row, _ANSWER_KEYS)
    if answer is None:
        return None
    return _stringify(answer)


def _rubric_lines(row: Mapping[str, Any]) -> list[str]:
    raw = _first_present(row, _RUBRIC_KEYS)
    if raw in (None, ""):
        return []
    if isinstance(raw, list):
        lines: list[str] = []
        for item in raw:
            if isinstance(item, Mapping):
                text = str(item.get("rubric") or item.get("text") or item.get("description") or "").strip()
                lines.append(text or json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                lines.append(_stringify(item))
        return [line for line in lines if line]
    return [_stringify(raw)]


def _row_id(row: Mapping[str, Any], dataset_name: str, index: int) -> str:
    raw = row.get("id") or row.get("task_id") or row.get("problem_id") or row.get("instance_id")
    if raw not in (None, ""):
        return str(raw)
    return f"{dataset_name}__{index:05d}"


def _first_present(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _is_json_scalarish(value: Any) -> bool:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return True
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _register(name: str):
    @FREE_ANSWER_REGISTRY.register_spec(name)
    def _prepare(output_root: Path, split: str = "test") -> RequestedFreeAnswerDatasetSpec:
        return RequestedFreeAnswerDatasetSpec(output_root, split, name=name)

    return _prepare


for _dataset_name in _REQUESTED_FREE_ANSWER_SPECS:
    _register(_dataset_name)


__all__ = [
    "RequestedFreeAnswerDatasetSpec",
    "normalize_free_answer_row",
]
