from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterator, Sequence

from ..data_utils import dataset_cache_dir, download_file
from src.eval.datasets.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY
from src.eval.datasets.runtime import CallableRowsDatasetSpec, DatasetPrepareContext

DATA_URL = (
    "https://huggingface.co/datasets/qintongli/GSM-Plus/resolve/main/data/test-00000-of-00001.jsonl?download=true"
)
DEFAULT_CATEGORIES: Sequence[str] = (
    "adding_operation",
    "critical_thinking",
    "digit_expansion",
    "distraction_insertion",
    "integer-decimal-fraction_conversion",
    "numerical_substitution",
    "problem_understanding",
    "reversing_operation",
)


def _load_cleaning_rules() -> dict[str, set[int]]:
    resource_path = Path(__file__).with_suffix("").parent / "resources" / "gsm_plus_cleaned_indexes.json"
    data = json.loads(resource_path.read_text())
    return {key: set(indices) for key, indices in data.items()}


def _records(
    split: str,
    context: DatasetPrepareContext,
    categories: Sequence[str],
    cleaning: str,
) -> Iterator[dict]:
    if split != "test":
        raise ValueError("gsm-plus 仅提供 test split")
    cache_dir = dataset_cache_dir(context.data_root, "gsm_plus")
    raw_path = cache_dir / "gsm_plus_test.jsonl"
    download_file(DATA_URL, raw_path)

    with raw_path.open("r", encoding="utf-8") as handle:
        data = [json.loads(line) for line in handle]

    cleaning_rules = _load_cleaning_rules()
    if cleaning == "none":
        valid_indices = set(range(len(data)))
    else:
        if cleaning not in cleaning_rules:
            raise ValueError(f"Unknown cleaning level: {cleaning}")
        valid_indices = cleaning_rules[cleaning]
    category_set = {cat.replace(" ", "_") for cat in categories}

    for idx, payload in enumerate(data):
        if idx not in valid_indices:
            continue
        perturbation = payload["perturbation_type"].replace(" ", "_")
        if perturbation not in category_set:
            continue
        expected_answer = payload.get("answer") or payload.get("expected_answer")
        if expected_answer == "None":
            expected_answer = "insufficient"
        entry = {
            "problem": payload["question"],
            "expected_answer": expected_answer,
            "reference_solution": payload.get("solution") or payload.get("reference_solution"),
            "perturbation_type": perturbation,
            "source_index": idx,
        }
        # include remaining metadata except duplicates
        for key, value in payload.items():
            if key in entry or key in {"question", "answer", "expected_answer", "solution", "reference_solution"}:
                continue
            entry[key] = value
        yield entry


@FREE_ANSWER_REGISTRY.register_spec("gsm_plus")
def prepare_gsm_plus_spec(
    output_root: Path,
    split: str = "test",
    *,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    cleaning: str = "light",
) -> CallableRowsDatasetSpec:
    return CallableRowsDatasetSpec(
        "gsm-plus",
        output_root,
        split,
        load_rows=lambda resolved_split, context, _categories=tuple(categories), _cleaning=cleaning: _records(
            resolved_split,
            context,
            _categories,
            _cleaning,
        ),
        source_kind="url_download",
        manifest_extra_factory=lambda resolved_split, _categories=tuple(categories), _cleaning=cleaning: {
            "source_split": resolved_split,
            "categories": list(_categories),
            "cleaning": _cleaning,
        },
    )
