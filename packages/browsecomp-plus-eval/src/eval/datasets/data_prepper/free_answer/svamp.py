from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterator

from ..data_utils import dataset_cache_dir, download_file
from src.eval.datasets.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY
from src.eval.datasets.runtime import CallableRowsDatasetSpec, DatasetPrepareContext

DATA_URL = "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json"


def _records(split: str, context: DatasetPrepareContext) -> Iterator[dict]:
    if split != "test":
        raise ValueError("svamp 仅提供 test split")
    cache_dir = dataset_cache_dir(context.data_root, "svamp")
    source_path = cache_dir / "SVAMP.json"
    download_file(DATA_URL, source_path)

    with source_path.open("r", encoding="utf-8") as handle:
        original = json.load(handle)
        for entry in original:
            answer = entry["Answer"]
            if isinstance(answer, (int, float)) and int(answer) == answer:
                answer = int(answer)
            yield {
                "problem": entry["Body"].rstrip(".") + ". " + entry["Question"],
                "expected_answer": answer,
                "reference_equation": entry["Equation"],
            }


@FREE_ANSWER_REGISTRY.register_spec("svamp")
def prepare_svamp_spec(output_root: Path, split: str = "test") -> CallableRowsDatasetSpec:
    return CallableRowsDatasetSpec("svamp", output_root, split, load_rows=_records, source_kind="url_download")
