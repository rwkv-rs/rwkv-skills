from __future__ import annotations

from pathlib import Path

from ..data_utils import load_qwen_dataset
from src.eval.datasets.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY
from src.eval.datasets.runtime import CallableRowsDatasetSpec


@FREE_ANSWER_REGISTRY.register_spec("gaokao2023en")
def prepare_gaokao2023en_spec(output_root: Path, split: str = "test") -> CallableRowsDatasetSpec:
    return CallableRowsDatasetSpec(
        "gaokao2023en",
        output_root,
        split,
        load_rows=lambda resolved_split, context: load_qwen_dataset(
            "gaokao2023en",
            resolved_split,
            data_root=context.data_root,
        ),
        source_kind="qwen_dataset",
    )
