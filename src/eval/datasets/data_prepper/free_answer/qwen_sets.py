from __future__ import annotations

from pathlib import Path

from ..data_utils import load_qwen_dataset
from src.eval.datasets.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY
from src.eval.datasets.runtime import CallableRowsDatasetSpec


def _register(dataset: str, name: str, split: str = "test") -> None:
    @FREE_ANSWER_REGISTRY.register_spec(name)
    def _prepare_spec(output_root: Path, split: str = split) -> CallableRowsDatasetSpec:  # type: ignore[override]
        return CallableRowsDatasetSpec(
            name,
            output_root,
            split,
            load_rows=lambda resolved_split, context, _dataset=dataset: load_qwen_dataset(
                _dataset,
                resolved_split,
                data_root=context.data_root,
            ),
            source_kind="qwen_dataset",
            manifest_extra_factory=lambda resolved_split, _dataset=dataset: {
                "dataset": _dataset,
                "source_split": resolved_split,
            },
        )


_register("math", "hendrycks_math")
_register("college_math", "college_math")
_register("minerva_math", "minerva_math")
_register("olympiadbench", "olympiadbench")
_register("amc23", "amc23")
