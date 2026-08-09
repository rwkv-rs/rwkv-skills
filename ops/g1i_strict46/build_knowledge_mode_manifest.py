#!/usr/bin/env python3
"""Build the exact G1h/G1i Knowledge CoT+NoCoT reuse/retest manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _replay_map(report: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    database = str(report.get("database") or "")
    return {
        (database, int(row["task_id"])): row
        for row in report.get("tasks", [])
        if "task_id" in row
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--replay", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    replay_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for path in args.replay:
        replay_rows.update(_replay_map(json.loads(path.read_text(encoding="utf-8"))))

    cells = []
    for cell in inventory.get("cells", []):
        latest = cell.get("latest")
        replay = None
        if isinstance(latest, dict):
            replay = replay_rows.get((str(latest.get("database") or ""), int(latest["task_id"])))
        reusable = bool(
            replay
            and (
                replay.get("strict_reuse_eligible")
                or replay.get("replay_eligible_except_cutoff")
            )
        )
        architecture = "G1h" if "g1h" in str(cell["model_name"]) else "G1i"
        size = next(
            size for size in ("1.5B", "2.9B", "7.2B", "13.3B")
            if size.lower().replace("b", "b") in str(cell["model_name"]).lower()
        )
        cells.append(
            {
                "architecture": architecture,
                "size": size,
                "model_name": cell["model_name"],
                "benchmark": cell["benchmark"],
                "mode": cell["mode"],
                "action": "reuse_recomputed" if reusable else "retest",
                "source_database": latest.get("database") if isinstance(latest, dict) else None,
                "source_task_id": int(latest["task_id"]) if isinstance(latest, dict) else None,
                "recomputed_score": replay.get("recomputed_score") if replay else None,
                "reasons": replay.get("reasons", ["missing_score_and_completions"]) if replay else ["missing_score_and_completions"],
            }
        )

    counts = Counter((row["architecture"], row["size"], row["mode"], row["action"]) for row in cells)
    payload = {
        "target_cells": len(cells),
        "reusable_cells": sum(row["action"] == "reuse_recomputed" for row in cells),
        "retest_cells": sum(row["action"] == "retest" for row in cells),
        "distribution": [
            {
                "architecture": key[0],
                "size": key[1],
                "mode": key[2],
                "action": key[3],
                "count": count,
            }
            for key, count in sorted(counts.items())
        ],
        "retest_datasets_by_architecture_size_mode": {
            f"{architecture}|{size}|{mode}": sorted(
                row["benchmark"]
                for row in cells
                if row["architecture"] == architecture
                and row["size"] == size
                and row["mode"] == mode
                and row["action"] == "retest"
            )
            for architecture in ("G1h", "G1i")
            for size in ("1.5B", "2.9B", "7.2B", "13.3B")
            for mode in ("no_cot", "cot")
        },
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "target_cells": payload["target_cells"],
                "reusable_cells": payload["reusable_cells"],
                "retest_cells": payload["retest_cells"],
                "distribution": payload["distribution"],
                "retest_datasets_by_architecture_size_mode": {
                    key: value
                    for key, value in payload[
                        "retest_datasets_by_architecture_size_mode"
                    ].items()
                    if value
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
