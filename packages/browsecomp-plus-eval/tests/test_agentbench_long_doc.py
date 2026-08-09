from __future__ import annotations

from src.eval.tasks.function_calling.agentbench import build_agentbench_prompt
from src.eval.long_doc_evidence import LongDocEvidenceConfig


def test_agentbench_prompt_compacts_long_document_messages() -> None:
    long_document = "\n".join(
        [f"noise row {index:03d} ignore" for index in range(80)]
        + ["passport ABC123 holder Alice verified"]
        + [f"archive row {index:03d} ignore" for index in range(80)]
    )

    prompt = build_agentbench_prompt(
        [
            {"role": "system", "content": "System seed."},
            {"role": "user", "content": long_document},
            {"role": "user", "content": "Which holder matches passport ABC123?"},
        ],
        [
            {
                "type": "function",
                "function": {
                    "name": "final_answer",
                    "description": "Answer the user.",
                    "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}},
                },
            }
        ],
        history_max_chars=12000,
        allow_final_answer_text=False,
        prompt_max_chars=5000,
        long_doc_config=LongDocEvidenceConfig(
            max_chunk_chars=240,
            overlap_lines=1,
            min_long_text_chars=400,
            max_evidence_chunks=1,
            max_evidence_chars=320,
        ),
    )

    assert "Long document compacted" in prompt
    assert "passport ABC123 holder Alice verified" in prompt
    assert "Which holder matches passport ABC123?" in prompt
    assert "noise row 000 ignore" not in prompt
    assert len(prompt) <= 5000


def test_agentbench_prompt_requires_complete_object_arguments() -> None:
    prompt = build_agentbench_prompt(
        [{"role": "system", "content": "Operate the database."}, {"role": "user", "content": "Run the SQL."}],
        [
            {
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "description": "Execute SQL.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ],
        history_max_chars=4000,
        allow_final_answer_text=False,
    )

    assert "The `arguments` value must be a JSON object" in prompt
    assert "Do not abbreviate, summarize, or replace argument values with ellipses" in prompt
    assert '{"name":"execute_sql","arguments":{"query":"SELECT 1"}}' in prompt
