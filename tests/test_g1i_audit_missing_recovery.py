from ops.g1i_strict46.run_audit_missing import (
    audit_cell_to_dataset,
    missing_datasets,
)


def test_audit_cell_to_scheduler_dataset_mapping() -> None:
    assert audit_cell_to_dataset("mmlu__test") == "mmlu"
    assert audit_cell_to_dataset("commonsense_qa__validation") == "commonsense_qa"
    assert audit_cell_to_dataset("simpleqa__test") == "simpleqa"
    assert audit_cell_to_dataset("gpqa__diamond") == "gpqa_diamond"
    assert audit_cell_to_dataset("gpqa__main") == "gpqa_main"
    assert audit_cell_to_dataset("gpqa__extended") == "gpqa_extended"


def test_missing_datasets_only_uses_rejected_cells_for_selected_model() -> None:
    audit = {
        "models": {
            "model-a": {
                "missing_cells": [
                    {"benchmark": "mmlu__test"},
                    {"benchmark": "gpqa__main"},
                    {"benchmark": "mmlu__test"},
                ]
            },
            "model-b": {"missing_cells": [{"benchmark": "svamp__test"}]},
        }
    }

    assert missing_datasets(audit, "model-a") == ["gpqa_main", "mmlu"]

