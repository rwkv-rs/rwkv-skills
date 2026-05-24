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
