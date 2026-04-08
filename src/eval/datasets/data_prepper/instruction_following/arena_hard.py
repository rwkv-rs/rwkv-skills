from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

from ..data_utils import dataset_cache_dir, download_file, read_jsonl
from src.eval.datasets.data_prepper.prepper_registry import INSTRUCTION_FOLLOWING_REGISTRY
from src.eval.datasets.runtime import CallableRowsDatasetSpec, DatasetPrepareContext

QUESTIONS_URL = "https://raw.githubusercontent.com/lm-sys/arena-hard-auto/main/data/arena-hard-v0.1/question.jsonl"
BASELINE_URL = (
    "https://raw.githubusercontent.com/lm-sys/arena-hard-auto/main/data/arena-hard-v0.1/model_answer/gpt-4-0314.jsonl"
)


def _load_baseline(path: Path) -> dict[str, str]:
    answers: dict[str, str] = {}
    for row in read_jsonl(path):
        uid = row.get("uid")
        if uid is None:
            continue
        messages = row.get("messages") or []
        answer_text = ""
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "assistant":
                continue
            content = message.get("content", "")
            if isinstance(content, dict):
                answer_text = str(content.get("answer", ""))
            else:
                answer_text = str(content)
            break
        answers[str(uid)] = answer_text
    return answers


def _iter_records(split: str, context: DatasetPrepareContext) -> Iterable[dict]:
    if split != "test":
        raise ValueError("arena-hard 仅提供 test split")

    cache_dir = dataset_cache_dir(context.data_root, "arena_hard")
    questions_path = cache_dir / "question.jsonl"
    baseline_path = cache_dir / "gpt-4-0314.jsonl"
    download_file(QUESTIONS_URL, questions_path)
    download_file(BASELINE_URL, baseline_path)

    baseline_answers = _load_baseline(baseline_path)

    for payload in read_jsonl(questions_path):
        uid = str(payload.get("uid"))
        record = dict(payload)
        record["question"] = record.pop("prompt", "")
        record["baseline_answer"] = baseline_answers.get(uid, "")
        record["answer"] = record["baseline_answer"]
        yield record


@INSTRUCTION_FOLLOWING_REGISTRY.register_spec("arena_hard")
def prepare_arena_hard_spec(output_root: Path, split: str = "test") -> CallableRowsDatasetSpec:
    return CallableRowsDatasetSpec("arena-hard", output_root, split, load_rows=_iter_records, source_kind="url_download")


__all__ = ["prepare_arena_hard_spec"]
