from __future__ import annotations

import json

from src.eval.datasets.data_loader.free_answer import JsonlFreeAnswerLoader
from src.eval.datasets.data_loader.multiple_choice import JsonlMultipleChoiceLoader


def test_free_answer_loader_extracts_context_alias_outside_metadata(tmp_path) -> None:
    dataset_path = tmp_path / "free.jsonl"
    rows = [
        {
            "question": "Q?",
            "answer": "A",
            "source_context": "CTX",
            "topic": "logic",
            "id": "row-1",
        },
        {
            "problem": "Context:\nold\n\nQuestion:\nlegacy?",
            "answer": "B",
        },
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    records = list(JsonlFreeAnswerLoader(dataset_path).load())

    assert records[0].question == "Q?"
    assert records[0].answer == "A"
    assert records[0].context == "CTX"
    assert records[0].subject == "logic"
    assert records[0].metadata == {"id": "row-1"}
    assert records[1].question == "Context:\nold\n\nQuestion:\nlegacy?"
    assert records[1].context is None


def test_multiple_choice_loader_extracts_context_alias_outside_metadata(tmp_path) -> None:
    dataset_path = tmp_path / "mc.jsonl"
    row = {
        "question": "Pick one.",
        "choices": ["red", "blue"],
        "answer": 1,
        "document": {"hint": "blue"},
        "subject": "colors",
        "id": "row-1",
    }
    dataset_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    record = list(JsonlMultipleChoiceLoader(dataset_path).load())[0]

    assert record.question == "Pick one."
    assert record.choices == ["red", "blue"]
    assert record.answer_index == 1
    assert record.context == '{"hint": "blue"}'
    assert record.subject == "colors"
    assert record.metadata == {"id": "row-1"}
