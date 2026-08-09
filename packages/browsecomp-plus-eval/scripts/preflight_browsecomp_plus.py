#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def count_valid_queries(source_path: Path) -> tuple[int, int]:
    rows = 0
    duplicate_ids = 0
    seen: set[str] = set()
    with source_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            payload = json.loads(line)
            query = str(payload.get("query") or payload.get("question") or "").strip()
            if not query:
                continue
            query_id = str(payload.get("query_id") or payload.get("id") or index)
            duplicate_ids += int(query_id in seen)
            seen.add(query_id)
            rows += 1
    return rows, duplicate_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline BrowseComp-Plus asset preflight; no DB or endpoint access")
    parser.add_argument("--root", type=Path, required=True, help="Official BrowseComp-Plus directory")
    parser.add_argument("--expected-rows", type=int, default=830)
    parser.add_argument("--skip-row-count", action="store_true", help="Skip reading the large decrypted JSONL")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    required = {
        "source": root / "data" / "browsecomp_plus_decrypted.jsonl",
        "bm25_index": root / "indexes" / "bm25",
        "qrel_evidence": root / "topics-qrels" / "qrel_evidence.txt",
        "official_evaluator": root / "scripts_evaluation" / "evaluate_run.py",
    }
    missing = [f"{name}={path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise SystemExit("Missing official assets: " + "; ".join(missing))
    if not any(required["bm25_index"].iterdir()):
        raise SystemExit(f"BM25 index is empty: {required['bm25_index']}")

    result: dict[str, object] = {"root": str(root), "assets_ok": True}
    if not args.skip_row_count:
        rows, duplicate_ids = count_valid_queries(required["source"])
        if rows != args.expected_rows or duplicate_ids:
            raise SystemExit(
                f"Dataset integrity failed: rows={rows}, expected={args.expected_rows}, duplicate_ids={duplicate_ids}"
            )
        result.update({"rows": rows, "duplicate_query_ids": duplicate_ids})
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
