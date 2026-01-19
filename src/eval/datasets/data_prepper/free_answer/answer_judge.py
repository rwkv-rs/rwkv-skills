from __future__ import annotations

from pathlib import Path

from ..data_utils import iter_hf_dataset, write_jsonl
from src.eval.datasets.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY


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
        try:
            scores.append(float(raw_score))
        except (TypeError, ValueError):
            continue
    if not scores:
        return None
    return sum(scores) / len(scores)


@FREE_ANSWER_REGISTRY.register("answer-judge")
def prepare_answer_judge(output_root: Path, split: str = "test") -> list[Path]:
    """answer-judge: judgement benchmark derived from `nvidia/judges-verdict`.

    The HF dataset provides `question` + `gt_answer` + `gen_answer` plus human
    `annotations` (scores in {0, 0.5, 1}). We convert it into a binary verdict:
    `expected_judgement` is "Yes" iff the mean score is strictly > 0.5.
    """

    if split != "test":
        raise ValueError("answer-judge 仅提供 test split")

    rows: list[dict] = []
    for row in iter_hf_dataset(_JUDGES_VERDICT_DATASET_ID, split=_JUDGES_VERDICT_SOURCE_SPLIT):
        payload = dict(row)
        mean_score = _mean_annotation_score(payload.get("annotations"))
        if mean_score is None:
            raise ValueError(
                f"answer-judge: judges-verdict 缺少 annotations/score: item_name={payload.get('item_name')!r}"
            )
        expected_judgement = mean_score > 0.5
        rows.append(
            {
                "problem": payload.get("question", "") or "",
                "expected_answer": payload.get("gt_answer", "") or "",
                "predicted_answer": payload.get("gen_answer", "") or "",
                "expected_judgement": _judgement_text(expected_judgement),
                "comment": f"judges-verdict mean_score={mean_score:.3f}",
                "source": payload.get("dataset_name") or "judges-verdict",
                "source_id": payload.get("item_name"),
                "annotations": payload.get("annotations"),
            }
        )

    dataset_dir = output_root / "answer-judge"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / "test.jsonl"
    write_jsonl(target, rows)
    return [target]
