from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from collections.abc import Iterable

try:
    from datasets import load_dataset  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    load_dataset = None  # type: ignore

from ..data_utils import configure_hf_home, write_jsonl
from src.dataset.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY

_SUBCATEGORIES: dict[str, str] = {
    "abstract_algebra": "math",
    "anatomy": "health",
    "astronomy": "physics",
    "business_ethics": "business",
    "clinical_knowledge": "health",
    "college_biology": "biology",
    "college_chemistry": "chemistry",
    "college_computer_science": "computer_science",
    "college_mathematics": "math",
    "college_medicine": "health",
    "college_physics": "physics",
    "computer_security": "computer_science",
    "conceptual_physics": "physics",
    "econometrics": "economics",
    "electrical_engineering": "engineering",
    "elementary_mathematics": "math",
    "formal_logic": "philosophy",
    "global_facts": "other",
    "high_school_biology": "biology",
    "high_school_chemistry": "chemistry",
    "high_school_computer_science": "computer_science",
    "high_school_european_history": "history",
    "high_school_geography": "geography",
    "high_school_government_and_politics": "politics",
    "high_school_macroeconomics": "economics",
    "high_school_mathematics": "math",
    "high_school_microeconomics": "economics",
    "high_school_physics": "physics",
    "high_school_psychology": "psychology",
    "high_school_statistics": "math",
    "high_school_us_history": "history",
    "high_school_world_history": "history",
    "human_aging": "health",
    "human_sexuality": "culture",
    "international_law": "law",
    "jurisprudence": "law",
    "logical_fallacies": "philosophy",
    "machine_learning": "computer_science",
    "management": "business",
    "marketing": "business",
    "medical_genetics": "health",
    "miscellaneous": "other",
    "moral_disputes": "philosophy",
    "moral_scenarios": "philosophy",
    "nutrition": "health",
    "philosophy": "philosophy",
    "prehistory": "history",
    "professional_accounting": "other",
    "professional_law": "law",
    "professional_medicine": "health",
    "professional_psychology": "psychology",
    "public_relations": "politics",
    "security_studies": "politics",
    "sociology": "culture",
    "us_foreign_policy": "politics",
    "virology": "health",
    "world_religions": "philosophy",
}


def _iter_records(split: str) -> Iterable[dict]:
    if split != "test":
        raise ValueError("mmlu-redux 仅提供 test split")
    configure_hf_home()
    if load_dataset is None:  # pragma: no cover - dependency missing
        raise ModuleNotFoundError(
            "需要安装 `datasets` 才能准备 mmlu-redux 数据集，请运行 `pip install datasets`"
        )
    for category in _SUBCATEGORIES:
        dataset = load_dataset("edinburgh-dawg/mmlu-redux-2.0", name=category, split="test")
        subset = _SUBCATEGORIES[category]
        for row in dataset:
            error_type = row.get("error_type")
            if error_type == "ok":
                answer_letter = chr(ord("A") + int(row["answer"]))
            elif error_type == "wrong_groundtruth" and isinstance(row.get("correct_answer"), str):
                answer_letter = row["correct_answer"].strip().upper()
            else:
                continue
            choices = [choice.strip() for choice in row.get("choices", [])]
            payload: dict[str, object] = {
                "question": row.get("question", "").strip(),
                "answer": answer_letter,
                "subject": category,
                "subset": subset,
                "source": row.get("source"),
                "subcategory": category,
                "error_type": error_type,
            }
            for idx, choice in enumerate(choices):
                payload[chr(ord("A") + idx)] = choice
            yield payload


@MULTIPLE_CHOICE_REGISTRY.register("mmlu-redux")
def prepare_mmlu_redux(output_root: Path, split: str = "test") -> list[Path]:
    dataset_dir = output_root / "mmlu-redux"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, _iter_records(split))
    return [target]
