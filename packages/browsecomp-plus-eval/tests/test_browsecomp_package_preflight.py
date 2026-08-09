from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_preflight_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "preflight_browsecomp_plus.py"
    spec = importlib.util.spec_from_file_location("browsecomp_package_preflight", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_count_valid_queries_reports_duplicates(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    rows = [
        {"query_id": "q1", "query": "one", "answer": "a"},
        {"query_id": "q1", "query": "duplicate", "answer": "b"},
        {"query_id": "q2", "query": "two", "answer": "c"},
        {"query_id": "ignored", "query": "", "answer": "d"},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    module = _load_preflight_module()

    assert module.count_valid_queries(source) == (3, 1)
