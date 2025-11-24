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
    configure_hf_home()
    if load_dataset is None:  # pragma: no cover - dependency missing
        raise ModuleNotFoundError(
            "需要安装 `datasets` 才能准备 mmlu 数据集，请运行 `pip install datasets`"
        )
    dataset = load_dataset("cais/mmlu", "all", split=split)
    for row in dataset:
        question = row["question"].strip()
        choices = [choice.strip() for choice in row["choices"]]
        answer_idx = int(row["answer"])
        if not (0 <= answer_idx < len(choices)):
            raise ValueError(f"Invalid answer index {answer_idx} for question {row}")
        subject = row.get("subject", "")
        subset = _SUBCATEGORIES.get(subject, "unknown")
        payload: dict[str, object] = {
            "question": question,
            "answer": chr(ord("A") + answer_idx),
            "subject": subject,
            "subset": subset,
        }
        for idx, choice in enumerate(choices):
            payload[chr(ord("A") + idx)] = choice
        yield payload


@MULTIPLE_CHOICE_REGISTRY.register("mmlu")
def prepare_mmlu(output_root: Path, split: str = "test") -> list[Path]:
    dataset_dir = output_root / "mmlu"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, _iter_records(split))
    return [target]
