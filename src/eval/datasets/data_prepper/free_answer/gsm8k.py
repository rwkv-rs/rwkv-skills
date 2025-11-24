from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List
from collections.abc import Iterable

from ..data_utils import dataset_cache_dir, download_file, write_jsonl
from src.dataset.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl"


def _parse_answer(answer: str) -> str:
    *_, final = answer.split("####")
    final = final.strip()
    final = final.replace(",", "")
    match = re.search(r"([-+]?[0-9]*\.?[0-9]+)", final)
    return match.group(1) if match else final


def _iter_records(root: Path, split: str, fixes: dict[str, str | int]) -> Iterable[dict]:
    url_split = "test" if split == "test" else "train"
    data_dir = dataset_cache_dir(root, "gsm8k")
    source_path = data_dir / f"{url_split}.jsonl"
    download_file(GSM8K_URL.format(split=url_split), source_path)

    with source_path.open("r", encoding="utf-8") as handle:
        for row in handle:
            payload = json.loads(row)
            question = payload["question"].strip()
            answer = fixes.get(question)
            if answer is None:
                answer = _parse_answer(payload["answer"])
            yield {
                "question": question,
                "answer": str(answer),
                "subject": "gsm8k",
                "reference_solution": payload["answer"],
            }


_FIXES = {
    "Mr. Finnegan has 3 tanks with a capacity of 7000 gallons, 5000 gallons, and 3000 gallons, respectively. If he fills the first tank up to 3/4 full, the second tank with water up to 4/5 of its capacity, and the third tank up to half of its capacity, how many gallons in total are in the tanks?": 10750,
}


@FREE_ANSWER_REGISTRY.register("gsm8k")
def prepare_gsm8k(output_root: Path, split: str = "test") -> list[Path]:
    dataset_dir = output_root / "gsm8k"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, _iter_records(output_root, split, _FIXES))
    return [target]
