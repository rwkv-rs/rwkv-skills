from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping

from ..data_utils import iter_hf_dataset
from src.eval.datasets.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY
from src.eval.datasets.runtime import MaterializingDatasetSpec


_JUDGES_VERDICT_DATASET_ID = "nvidia/judges-verdict"
# HF 上的 `test` split 不含人工标注（annotations 全为 null），而 `train` 才带评分；
# 本 repo 的 `answer-judge_test` 需要 ground-truth judgement，因此映射到 HF 的 train split。
_JUDGES_VERDICT_SOURCE_SPLIT = "train"


def _judgement_text(value: bool) -> str:
    return "Judgement: Yes" if value else "Judgement: No"


def _mean_annotation_score(annotations: object) -> float | None:
    if not isinstance(annotations, list) or not annotations:
        return None
    scores: list[float] = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        raw_score = annotation.get("score")
        if not isinstance(raw_score, int | float | str):
            continue
        try:
            scores.append(float(raw_score))
        except (TypeError, ValueError):
            continue
    if not scores:
        return None
    return sum(scores) / len(scores)


_REQUIRED_FIELDS = ("problem", "expected_answer", "predicted_answer", "expected_judgement")


def _map_answer_judge_row(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    mean_score = _mean_annotation_score(payload.get("annotations"))
    if mean_score is None:
        raise ValueError(
            f"answer-judge: judges-verdict 缺少 annotations/score: item_name={payload.get('item_name')!r}"
        )
    expected_judgement = mean_score > 0.5
    return {
        "problem": str(payload.get("question", "") or ""),
        "expected_answer": str(payload.get("gt_answer", "") or ""),
        "predicted_answer": str(payload.get("gen_answer", "") or ""),
        "expected_judgement": _judgement_text(expected_judgement),
        "comment": f"judges-verdict mean_score={mean_score:.3f}",
        "source": payload.get("dataset_name") or "judges-verdict",
        "source_id": payload.get("item_name"),
        "annotations": payload.get("annotations"),
    }


class AnswerJudgeDatasetSpec(MaterializingDatasetSpec):
    def __init__(self, output_root: Path, split: str) -> None:
        super().__init__("answer-judge", output_root, split, required_fields=_REQUIRED_FIELDS, source_kind="hf_load_dataset")

    def download(self) -> None:
        return None

    def load_records(self) -> Iterable[dict[str, Any]]:
        if self.split != "test":
            raise ValueError("answer-judge 仅提供 test split")
        return [
            _map_answer_judge_row(row)
            for row in iter_hf_dataset(_JUDGES_VERDICT_DATASET_ID, split=_JUDGES_VERDICT_SOURCE_SPLIT)
        ]

    def manifest_extra(self) -> dict[str, Any]:
        return {
            "dataset_id": _JUDGES_VERDICT_DATASET_ID,
            "requested_split": self.split,
            "source_split": _JUDGES_VERDICT_SOURCE_SPLIT,
        }


@FREE_ANSWER_REGISTRY.register_spec("answer_judge")
def prepare_answer_judge_spec(output_root: Path, split: str = "test") -> AnswerJudgeDatasetSpec:
    return AnswerJudgeDatasetSpec(output_root, split)


__all__ = ["prepare_answer_judge_spec"]
