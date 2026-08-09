from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY
from src.eval.datasets.runtime import UrlDownloadFile, UrlFilesJsonlDatasetSpec, read_jsonl_items

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl"


def _parse_answer(answer: str) -> str:
    *_, final = answer.split("####")
    final = final.strip().replace(",", "")
    match = re.search(r"([-+]?[0-9]*\.?[0-9]+)", final)
    return match.group(1) if match else final


_FIXES = {
    "Mr. Finnegan has 3 tanks with a capacity of 7000 gallons, 5000 gallons, and 3000 gallons, respectively. If he fills the first tank up to 3/4 full, the second tank with water up to 4/5 of its capacity, and the third tank up to half of its capacity, how many gallons in total are in the tanks?": 10750,
}


def _map_record(payload: dict[str, Any]) -> dict[str, Any]:
    question = str(payload["question"]).strip()
    answer = _FIXES.get(question)
    if answer is None:
        answer = _parse_answer(str(payload["answer"]))
    return {
        "question": question,
        "answer": str(answer),
        "subject": "gsm8k",
        "reference_solution": payload["answer"],
    }


@FREE_ANSWER_REGISTRY.register_spec("gsm8k")
def prepare_gsm8k_spec(output_root: Path, split: str = "test") -> UrlFilesJsonlDatasetSpec:
    url_split = "test" if split == "test" else "train"

    def _load(source_root: Path) -> list[dict[str, Any]]:
        return read_jsonl_items(source_root / f"{url_split}.jsonl", parse_item=_map_record)

    return UrlFilesJsonlDatasetSpec(
        "gsm8k",
        output_root,
        split,
        files=(UrlDownloadFile(Path(f"{url_split}.jsonl"), GSM8K_URL.format(split=url_split)),),
        load_downloaded_records=_load,
        required_fields=("question", "answer"),
    )
