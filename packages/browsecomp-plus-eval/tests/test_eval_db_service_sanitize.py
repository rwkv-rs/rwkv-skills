from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.db.eval_db_service import EvalDbService


class Directory:
    def __str__(self) -> str:
        return "/tmp/bfcl-support"


@dataclass
class DemoDataclass:
    path: Path
    marker: object


def test_sanitize_json_text_converts_non_json_objects_recursively() -> None:
    payload = {
        "text": "ok\x00",
        "details": {
            "directory": Directory(),
            "dataclass": DemoDataclass(path=Path("/tmp/a"), marker=Directory()),
        },
    }

    sanitized = EvalDbService._sanitize_json_text(payload)

    assert sanitized["text"] == "ok"
    assert sanitized["details"]["directory"] == "/tmp/bfcl-support"
    assert sanitized["details"]["dataclass"]["path"] == "/tmp/a"
    assert sanitized["details"]["dataclass"]["marker"] == "/tmp/bfcl-support"
    json.dumps(sanitized)


def test_completion_context_preserves_browsecomp_plus_run_docids() -> None:
    docids = [str(index) for index in range(80)]
    payload = {
        "prompt1": "prompt",
        "completion1": "completion",
        "stop_reason1": "stop",
        "sampling_config": {},
        "browsecomp_plus_run": {
            "query_id": "q1",
            "status": "completed",
            "retrieved_docids": docids,
            "tool_call_counts": {"search": 12},
            "result": [{"type": "output_text", "output": "answer"}],
        },
    }

    context = EvalDbService._build_completion_context(payload)

    assert context["browsecomp_plus_run"]["retrieved_docids"] == docids
    assert context["browsecomp_plus_run"]["result"] == [{"type": "output_text", "output": "answer"}]


def test_completion_context_preserves_long_doc_trace() -> None:
    payload = {
        "prompt1": "prompt",
        "completion1": "completion",
        "stop_reason1": "stop",
        "sampling_config": {},
        "long_doc": {"mode": "lexical", "compacted": True, "selected_chunk_ids": [2]},
        "long_context": {"long_doc": {"enabled": True}},
    }

    context = EvalDbService._build_completion_context(payload)

    assert context["long_doc"] == {"mode": "lexical", "compacted": True, "selected_chunk_ids": [2]}
    assert context["long_context"] == {"long_doc": {"enabled": True}}
