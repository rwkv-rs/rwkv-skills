from __future__ import annotations

import json
from pathlib import Path
from typing import List
from collections.abc import Iterator

from ..data_utils import dataset_cache_dir, download_file, write_jsonl
from src.dataset.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY

DATA_URL = "https://raw.githubusercontent.com/protagolabs/odyssey-math/main/final-odyssey-math-with-levels.jsonl"

ANSWER_ENDINGS = [
    "\\\n\\noindent",
    "\\\n\n\\noindent",
    "\\\n\t\\noindent",
    ".\n\n\\noindent",
    "\n\n\\noindent",
    "\\\n\n  \n\t\\noindent",
    "\\\\ \n\t\\noindent",
    "\\\n\n\t\\noindent",
]


def _strip_problem(text: str) -> str:
    text = text.replace("\\underline{\\hspace{2cm}}", "")
    parts = text.split("\\end{problem}")
    return parts[0].strip() if parts else text.strip()


def _normalize_answer(answer: str) -> str:
    for ending in ANSWER_ENDINGS:
        if answer.endswith(ending):
            answer = answer[: -len(ending)]
            break
    answer = answer.strip().strip("\\")
    if answer.endswith("."):
        answer = answer[:-1]
    return answer.replace("$", "").strip()


def _records(split: str) -> Iterator[dict]:
    if split != "test":
        raise ValueError("math-odyssey 仅提供 test split")
    cache_dir = dataset_cache_dir(Path("data"), "math_odyssey")
    source_path = cache_dir / "math_odyssey.jsonl"
    download_file(DATA_URL, source_path)

    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line_data = json.loads(line)
            key, payload = next(iter(line_data.items()))
            answer = _normalize_answer(payload["answer"])
            yield {
                "problem": _strip_problem(payload["question"]),
                "expected_answer": answer,
                "original_answer": payload["answer"],
                "reference_solution": payload["reasoning"],
                "label": payload["label"],
                "level": payload["level"],
                "id": key,
            }


@FREE_ANSWER_REGISTRY.register("math-odyssey")
def prepare_math_odyssey(output_root: Path, split: str = "test") -> list[Path]:
    dataset_dir = output_root / "math-odyssey"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, _records(split))
    return [target]
